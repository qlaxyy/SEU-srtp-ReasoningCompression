"""Shared helpers for ASC pair generation and steering-vector extraction."""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_REASONING_EFFORT = "none"
DEFAULT_DEEPSEEK_THINKING = "disabled"

DEEPSEEK_SYSTEM_PROMPT = (
    "You are an expert math reasoner. Produce concise, math-focused reasoning. "
    "Avoid repeated verification, long explanations, and unnecessary prose. "
    "Always include the final answer in \\boxed{}."
)

DEEPSEEK_LONG_SYSTEM_PROMPT = (
    "You are an expert math reasoner. Produce a detailed, natural-language "
    "chain-of-thought style solution with enough intermediate reasoning. "
    "Always include the final answer in \\boxed{}."
)

DEEPSEEK_LONG_TEMPLATE = (
    "Solve the following problem step by step. Give a detailed solution with "
    "clear reasoning.\n\nProblem:\n{problem}"
)

DEEPSEEK_SHORT_TEMPLATE = (
    "Solve the following problem step by step, but keep the reasoning as short "
    "and symbolic as possible. Use minimal English.\n\nProblem:\n{problem}"
)

LAYER_HINTS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 20,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 20,
    "Qwen/QwQ-32B": 57,
}


def is_qwen3_model_name(name: str | None) -> bool:
    return bool(name) and "qwen3" in name.lower()


def is_qwen3_model(
    model_name: str | None,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> bool:
    if is_qwen3_model_name(model_name):
        return True
    if model is not None:
        if is_qwen3_model_name(getattr(model, "name_or_path", None)):
            return True
        config = getattr(model, "config", None)
        if is_qwen3_model_name(getattr(config, "name_or_path", None)):
            return True
    if tokenizer is not None and is_qwen3_model_name(getattr(tokenizer, "name_or_path", None)):
        return True
    return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def problem_from_row(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "input", "prompt"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError(f"Could not find a problem field in row keys={list(row)}")


def load_jsonl_problems(dataset_path: str, num_samples: int, seed: int) -> list[str]:
    rows = read_jsonl(dataset_path)
    problems = [problem_from_row(row) for row in rows]
    if num_samples > 0 and num_samples < len(problems):
        rng = random.Random(seed)
        indices = list(range(len(problems)))
        rng.shuffle(indices)
        problems = [problems[i] for i in indices[:num_samples]]
    return problems


def parse_max_memory(text: str | None) -> dict[int | str, str] | None:
    if not text:
        return None
    memory: dict[int | str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("--max_memory must look like '0:20GiB,1:20GiB'")
        key, value = item.split(":", 1)
        key = key.strip()
        memory[int(key) if key.isdigit() else key] = value.strip()
    return memory


def get_input_device(model: Any) -> torch.device:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for key in ("model.embed_tokens", "model.tok_embeddings", "transformer.wte"):
            device = model.hf_device_map.get(key)
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def summarize_device_map(device_map: dict[str, Any]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for device in device_map.values():
        key = str(device)
        summary[key] = summary.get(key, 0) + 1
    return summary


def iter_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_deepseek_client(api_key: str, base_url: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "DeepSeek API mode requires the openai package. Install it with "
            "`pip install openai` or add it to your environment."
        ) from exc
    return OpenAI(api_key=api_key, base_url=base_url)


def deepseek_request_options(reasoning_effort: str, thinking: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if reasoning_effort and reasoning_effort != "none":
        options["reasoning_effort"] = reasoning_effort
    if thinking and thinking != "none":
        options["extra_body"] = {"thinking": {"type": thinking}}
    return options


def generate_short_with_deepseek(
    client: Any,
    problem: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    reasoning_effort: str,
    thinking: str,
    include_reasoning: bool,
    retries: int,
) -> tuple[str, int | None]:
    prompt = DEEPSEEK_SHORT_TEMPLATE.format(problem=problem)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": DEEPSEEK_SYSTEM_PROMPT + "\n\n" + prompt,
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **deepseek_request_options(reasoning_effort, thinking),
            )
            message = response.choices[0].message
            content = (getattr(message, "content", None) or "").strip()
            reasoning = (getattr(message, "reasoning_content", None) or "").strip()
            text = (reasoning + "\n" + content).strip() if include_reasoning and reasoning else content
            tokens = None
            if getattr(response, "usage", None) is not None:
                tokens = getattr(response.usage, "completion_tokens", None)
            return text, tokens
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            wait = min(2**attempt, 30) + random.random()
            print(f"[warn] DeepSeek API failed: {exc!r}; retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"DeepSeek API failed after {retries + 1} attempts: {last_exc!r}")


def generate_with_deepseek(
    client: Any,
    problem: str,
    model_name: str,
    system_prompt: str,
    user_template: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    reasoning_effort: str,
    thinking: str,
    include_reasoning: bool,
    retries: int,
) -> tuple[str, int | None]:
    prompt = user_template.format(problem=problem)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": system_prompt + "\n\n" + prompt,
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **deepseek_request_options(reasoning_effort, thinking),
            )
            message = response.choices[0].message
            content = (getattr(message, "content", None) or "").strip()
            reasoning = (getattr(message, "reasoning_content", None) or "").strip()
            text = (reasoning + "\n" + content).strip() if include_reasoning and reasoning else content
            tokens = None
            if getattr(response, "usage", None) is not None:
                tokens = getattr(response.usage, "completion_tokens", None)
            return text, tokens
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            wait = min(2**attempt, 30) + random.random()
            print(f"[warn] DeepSeek API failed: {exc!r}; retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"DeepSeek API failed after {retries + 1} attempts: {last_exc!r}")


def make_prompt(
    tokenizer: Any,
    problem: str,
    mode: str,
    use_chat_template: bool,
    qwen3_enable_thinking: bool = False,
    add_no_think_tag: bool = False,
) -> str:
    is_qwen3 = is_qwen3_model(None, tokenizer=tokenizer)
    if mode == "long":
        if is_qwen3:
            content = (
                f"Question: {problem}\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
        else:
            content = f"Question: {problem}\nLet's think step by step."
    elif mode == "short":
        content = (
            f"Question: {problem}\n"
            "Solve step by step, but keep the reasoning concise and math-focused. "
            "Avoid repeated verification and unnecessary prose. Put the final "
            "answer in \\boxed{}."
        )
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

    if add_no_think_tag and not qwen3_enable_thinking:
        content = "/no_think\n" + content

    if use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("Tokenizer has no chat_template; omit --use_chat_template.")
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=qwen3_enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return content


@torch.no_grad()
def generate_many(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    input_device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    repetition_penalty: float,
) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    token_counts: list[int] = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for batch in iter_batches(prompts, batch_size):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).to(input_device)
            prompt_len = inputs["input_ids"].shape[-1]
            generation_kwargs: dict[str, Any] = dict(inputs)
            generation_kwargs.update(
                {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                }
            )
            if top_k is not None and top_k > 0:
                generation_kwargs["top_k"] = top_k
            if min_p is not None:
                generation_kwargs["min_p"] = min_p

            try:
                outputs = model.generate(**generation_kwargs)
            except TypeError as exc:
                if "min_p" in generation_kwargs and "min_p" in str(exc):
                    generation_kwargs.pop("min_p", None)
                    outputs = model.generate(**generation_kwargs)
                else:
                    raise
            new_tokens = outputs[:, prompt_len:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            texts.extend(decoded)
            token_counts.extend((new_tokens != tokenizer.pad_token_id).sum(dim=1).tolist())
    finally:
        tokenizer.padding_side = old_padding_side
    return texts, [int(count) for count in token_counts]


def generate_deepseek_parallel(
    problems: list[str],
    workers: int,
    generator: Any,
) -> tuple[list[str], list[int | None]]:
    if workers <= 1 or len(problems) <= 1:
        outputs = [generator(problem) for problem in problems]
        return [item[0] for item in outputs], [item[1] for item in outputs]

    texts: list[str | None] = [None] * len(problems)
    token_counts: list[int | None] = [None] * len(problems)
    with ThreadPoolExecutor(max_workers=min(workers, len(problems))) as executor:
        future_to_index = {
            executor.submit(generator, problem): idx
            for idx, problem in enumerate(problems)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            text, tokens = future.result()
            texts[idx] = text
            token_counts[idx] = tokens
    return [str(text) for text in texts], token_counts


def generate_pairs(
    model: Any,
    tokenizer: Any,
    problems: list[str],
    output_path: str,
    input_device: torch.device,
    generation_batch_size: int,
    deepseek_workers: int,
    use_chat_template: bool,
    qwen3_enable_thinking: bool,
    add_no_think_tag: bool,
    long_source: str,
    short_source: str,
    deepseek_client: Any | None,
    deepseek_model: str,
    deepseek_max_tokens: int,
    deepseek_temperature: float,
    deepseek_reasoning_effort: str,
    deepseek_thinking: str,
    deepseek_include_reasoning: bool,
    deepseek_retries: int,
    max_new_tokens_long: int,
    max_new_tokens_short: int,
    temperature_long: float,
    temperature_short: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    repetition_penalty: float,
    save_interval: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    next_save = save_interval if save_interval > 0 else 0

    for problem_batch in tqdm(
        iter_batches(problems, generation_batch_size),
        desc="Generating CoT pair batches",
    ):
        if long_source == "target_model":
            long_prompts = [
                make_prompt(
                    tokenizer,
                    problem,
                    "long",
                    use_chat_template,
                    qwen3_enable_thinking=qwen3_enable_thinking,
                    add_no_think_tag=add_no_think_tag,
                )
                for problem in problem_batch
            ]
            long_cots, long_tokens = generate_many(
                model,
                tokenizer,
                long_prompts,
                input_device,
                batch_size=generation_batch_size,
                max_new_tokens=max_new_tokens_long,
                temperature=temperature_long,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )
        elif long_source == "deepseek_api":
            if deepseek_client is None:
                raise ValueError("deepseek_api long source requires a DeepSeek client.")
            long_prompts = [
                DEEPSEEK_LONG_TEMPLATE.format(problem=problem)
                for problem in problem_batch
            ]
            long_cots, long_tokens = generate_deepseek_parallel(
                problem_batch,
                workers=deepseek_workers,
                generator=lambda problem: generate_with_deepseek(
                    client=deepseek_client,
                    problem=problem,
                    model_name=deepseek_model,
                    system_prompt=DEEPSEEK_LONG_SYSTEM_PROMPT,
                    user_template=DEEPSEEK_LONG_TEMPLATE,
                    max_tokens=max_new_tokens_long,
                    temperature=temperature_long,
                    top_p=top_p,
                    reasoning_effort=deepseek_reasoning_effort,
                    thinking=deepseek_thinking,
                    include_reasoning=True,
                    retries=deepseek_retries,
                ),
            )
        else:
            raise ValueError(f"Unknown long_source: {long_source}")

        if short_source == "target_model":
            short_prompts = [
                make_prompt(
                    tokenizer,
                    problem,
                    "short",
                    use_chat_template,
                    qwen3_enable_thinking=qwen3_enable_thinking,
                    add_no_think_tag=add_no_think_tag,
                )
                for problem in problem_batch
            ]
            short_cots, short_tokens = generate_many(
                model,
                tokenizer,
                short_prompts,
                input_device,
                batch_size=generation_batch_size,
                max_new_tokens=max_new_tokens_short,
                temperature=temperature_short,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )
        elif short_source == "deepseek_api":
            if deepseek_client is None:
                raise ValueError("deepseek_api short source requires a DeepSeek client.")
            short_prompts = [
                DEEPSEEK_SHORT_TEMPLATE.format(problem=problem)
                for problem in problem_batch
            ]
            short_cots, short_tokens = generate_deepseek_parallel(
                problem_batch,
                workers=deepseek_workers,
                generator=lambda problem: generate_short_with_deepseek(
                    client=deepseek_client,
                    problem=problem,
                    model_name=deepseek_model,
                    max_tokens=deepseek_max_tokens,
                    temperature=deepseek_temperature,
                    top_p=top_p,
                    reasoning_effort=deepseek_reasoning_effort,
                    thinking=deepseek_thinking,
                    include_reasoning=deepseek_include_reasoning,
                    retries=deepseek_retries,
                ),
            )
        else:
            raise ValueError(f"Unknown short_source: {short_source}")

        for index, problem in enumerate(problem_batch):
            long_length_capped = (
                long_tokens[index] is not None
                and long_tokens[index] >= max_new_tokens_long
            )
            short_length_capped = (
                short_tokens[index] is not None
                and short_tokens[index] >= max_new_tokens_short
            )
            pairs.append(
                {
                    "problem": problem,
                    "long_source": long_source,
                    "short_source": short_source,
                    "long_prompt": long_prompts[index],
                    "short_prompt": short_prompts[index],
                    "long_cot": long_cots[index],
                    "short_cot": short_cots[index],
                    "long_tokens": long_tokens[index],
                    "short_tokens": short_tokens[index],
                    "long_length_capped": long_length_capped,
                    "short_length_capped": short_length_capped,
                }
            )

        if next_save and len(pairs) >= next_save:
            write_json(output, pairs)
            while next_save and len(pairs) >= next_save:
                next_save += save_interval

    write_json(output, pairs)
    return pairs


def normalize_pair(row: dict[str, Any]) -> tuple[str, str, str]:
    problem = problem_from_row(row)
    short = (
        row.get("short_cot")
        or row.get("short_answer")
        or row.get("concise_cot")
        or row.get("answer_short")
    )
    long = (
        row.get("long_cot")
        or row.get("long_answer")
        or row.get("verbose_cot")
        or row.get("answer_long")
    )
    if short is None and "answer" in row and "long_cot" not in row:
        short = row["answer"]
    if short is None or long is None:
        raise ValueError(
            "Pair rows must contain problem plus short_cot/long_cot "
            f"fields; got keys={list(row)}"
        )
    return str(problem), str(short), str(long)


def append_cot_to_prompt(prompt: str, cot: str) -> str:
    prompt = str(prompt)
    cot = str(cot)
    if prompt.endswith(("\n", " ", "\t")):
        return prompt + cot
    return prompt + "\n" + cot


def pair_texts_for_activation(row: dict[str, Any]) -> tuple[str, str]:
    _problem, short_cot, long_cot = normalize_pair(row)
    long_prompt = row.get("long_prompt")
    if not long_prompt:
        raise ValueError(
            "Pair rows must contain long_prompt so short/long activations use "
            "the same target-model prompt prefix."
        )
    return (
        append_cot_to_prompt(str(long_prompt), short_cot),
        append_cot_to_prompt(str(long_prompt), long_cot),
    )


def get_transformer_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise AttributeError("Could not find transformer layers on this model.")


def first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return first_tensor(output[0])
    raise TypeError(f"Unsupported layer output type: {type(output)!r}")


@torch.no_grad()
def last_token_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    input_device: torch.device,
    layer_index: int,
    max_input_tokens: int,
    batch_size: int,
) -> torch.Tensor:
    activations: list[torch.Tensor] = []
    old_truncation_side = tokenizer.truncation_side
    old_padding_side = tokenizer.padding_side
    layers = get_transformer_layers(model)
    if layer_index >= len(layers):
        raise IndexError(
            f"layer_index={layer_index} is out of range for {len(layers)} layers"
        )

    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    try:
        for batch in tqdm(iter_batches(texts, batch_size), desc="Extracting activations"):
            captured: list[torch.Tensor] = []

            def capture_hook(_module: Any, _inputs: Any, output: Any) -> None:
                captured.append(first_tensor(output)[:, -1].detach().float().cpu())

            handle = layers[layer_index].register_forward_hook(capture_hook)
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
                return_token_type_ids=False,
            ).to(input_device)
            try:
                base_model = getattr(model, "model", None)
                if base_model is not None:
                    base_model(**inputs, use_cache=False)
                else:
                    model(**inputs, use_cache=False)
            finally:
                handle.remove()
            if not captured:
                raise RuntimeError(
                    f"Layer hook did not capture activations for layer {layer_index}."
                )
            activations.append(captured[0])
    finally:
        tokenizer.truncation_side = old_truncation_side
        tokenizer.padding_side = old_padding_side

    if not activations:
        raise RuntimeError("No activations extracted.")
    return torch.cat(activations, dim=0)


def extract_vectors(
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    input_device: torch.device,
    layer_index: int,
    max_input_tokens: int,
    activation_batch_size: int,
    direction: str,
) -> torch.Tensor:
    short_texts: list[str] = []
    long_texts: list[str] = []
    for row in pairs:
        short_text, long_text = pair_texts_for_activation(row)
        short_texts.append(short_text)
        long_texts.append(long_text)

    short_acts = last_token_activations(
        model,
        tokenizer,
        short_texts,
        input_device,
        layer_index,
        max_input_tokens,
        activation_batch_size,
    )
    long_acts = last_token_activations(
        model,
        tokenizer,
        long_texts,
        input_device,
        layer_index,
        max_input_tokens,
        activation_batch_size,
    )
    if direction == "short_minus_long":
        return short_acts - long_acts
    if direction == "long_minus_short":
        return long_acts - short_acts
    raise ValueError(f"Unknown direction: {direction}")


def resolve_layer_index(model: Any, model_name: str, layer_index: str) -> int:
    if layer_index != "auto":
        return int(layer_index)

    for key, value in LAYER_HINTS.items():
        if model_name == key or key in model_name or model_name in key:
            return value

    raise ValueError(
        "Could not infer layer index for this model. For new models, pass "
        "--layer_index explicitly and keep it consistent across vector extraction, "
        "gamma extraction, and evaluation."
    )


def load_model_and_tokenizer(args: Any) -> tuple[Any, Any, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "torch_dtype": args.dtype,
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    if args.attn_impl != "auto":
        kwargs["attn_implementation"] = args.attn_impl
    max_memory = parse_max_memory(args.max_memory)
    if max_memory is not None:
        kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs).eval()
    input_device = get_input_device(model)
    return model, tokenizer, input_device
