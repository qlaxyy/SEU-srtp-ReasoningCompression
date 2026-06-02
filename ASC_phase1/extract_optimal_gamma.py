"""
Final practical ASC gamma extractor.

This script implements the strategy selected from the Qwen/Llama experiments:

    gamma = ratio * median_hidden_norm / ||raw_steering_vector||

The default ratio is 0.2. The returned gamma is the raw-vector gamma used by
eval_asc_paper.py / eval_asc_dual_gpu_grid.py via --gamma_override.

Examples:

  python extract_optimal_gamma.py \
    --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --pairs_path pairs/qwen7b_math_train_deepseek_pairs_100_checked.json

  python extract_optimal_gamma.py \
    --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset gsm8k \
    --num_cal 50 \
    --output_path results/gamma_llama_hidden_ratio_02.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_PAIRS_PATH = "pairs/qwen7b_math_train_deepseek_pairs_100_checked.json"
DEFAULT_STEERING_VECTOR = "vectors/steering_vectors_qwen7b_math_train_deepseek_checked.pt"
DEFAULT_OUTPUT_PATH = "results/gamma_qwen7b_math_train_deepseek_checked_hidden_norm.json"


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "steering_vector_path": "./vectors/steering_vectors_qwen7b.pt",
        "layer_index": 20,
        "paper_gamma": 0.275,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "steering_vector_path": "./vectors/steering_vectors_llama8b.pt",
        "layer_index": 20,
        "paper_gamma": 0.46,
    },
    "Qwen/QwQ-32B": {
        "steering_vector_path": "./vectors/steering_vectors_qwq32b.pt",
        "layer_index": 57,
        "paper_gamma": 0.50,
    },
}

MODEL_CONFIGS["/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"] = (
    MODEL_CONFIGS["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]
)
MODEL_CONFIGS["/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"] = (
    MODEL_CONFIGS["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
)
MODEL_CONFIGS["/root/autodl-tmp/Qwen/QwQ-32B"] = MODEL_CONFIGS["Qwen/QwQ-32B"]

PAIR_STEERING_VECTOR_DEFAULTS: dict[str, str] = {
    "qwen7b_math_train_deepseek_pairs_100_checked": DEFAULT_STEERING_VECTOR,
    "qwen7b_math_train_deepseek_pairs_100_checked.json": DEFAULT_STEERING_VECTOR,
}


def resolve_model_config(model_name: str) -> dict[str, Any]:
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    for key, config in MODEL_CONFIGS.items():
        if model_name in key or key in model_name:
            return config
    return {}


def infer_steering_vector_from_pairs(pairs_path: str | Path | None) -> str | None:
    if pairs_path is None:
        return None
    path = Path(pairs_path)
    return (
        PAIR_STEERING_VECTOR_DEFAULTS.get(path.name)
        or PAIR_STEERING_VECTOR_DEFAULTS.get(path.stem)
    )


def torch_dtype_from_arg(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_raw_steering_vector(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    obj = safe_torch_load(path, map_location="cpu")
    if isinstance(obj, dict):
        tensors = [value for value in obj.values() if torch.is_tensor(value)]
        if not tensors:
            raise ValueError(f"No tensor found in steering vector file: {path}")
        obj = tensors[0]
    if not torch.is_tensor(obj):
        raise TypeError(f"Expected tensor steering vector, got {type(obj)!r}")

    raw = obj.detach().float()
    stats: dict[str, Any] = {
        "path": str(path),
        "raw_shape": list(raw.shape),
        "raw_dtype": str(obj.dtype),
    }
    if raw.ndim == 2:
        row_norms = raw.norm(dim=1).cpu().numpy()
        stats.update(
            {
                "raw_vectors": int(raw.shape[0]),
                "hidden_size": int(raw.shape[1]),
                "row_l2_mean": float(np.mean(row_norms)),
                "row_l2_std": float(np.std(row_norms)),
                "row_l2_min": float(np.min(row_norms)),
                "row_l2_max": float(np.max(row_norms)),
                "used_vector_rule": "mean(dim=0)",
            }
        )
        vec = raw.mean(dim=0)
    elif raw.ndim == 1:
        stats.update(
            {
                "raw_vectors": 1,
                "hidden_size": int(raw.shape[0]),
                "used_vector_rule": "as-is",
            }
        )
        vec = raw
    else:
        raise ValueError(f"Expected 1D or 2D steering tensor, got shape {tuple(raw.shape)}")

    norm = float(vec.norm().item())
    if norm < 1e-12:
        raise RuntimeError("Steering vector norm is approximately zero.")
    stats.update(
        {
            "used_shape": list(vec.shape),
            "used_l2_norm": norm,
            "used_l1_norm": float(vec.abs().sum().item()),
            "used_linf_norm": float(vec.abs().max().item()),
            "used_mean": float(vec.mean().item()),
            "used_std": float(vec.std().item()),
            "used_min": float(vec.min().item()),
            "used_max": float(vec.max().item()),
        }
    )
    return vec, stats


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


def read_json_or_jsonl(path: str | Path) -> Any:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")
    if text[0] in "[{":
        return json.loads(text)
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def problem_from_row(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "input", "prompt"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError(f"Could not find problem/question field in row keys={list(row)}")


def normalize_pair_field(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return None


def load_pair_calibration_texts(
    pairs_path: str | Path,
    source: str,
) -> list[str]:
    payload = read_json_or_jsonl(pairs_path)
    if isinstance(payload, dict):
        if "pairs" in payload:
            payload = payload["pairs"]
        elif "data" in payload:
            payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError("--pairs_path must contain a JSON list or JSONL rows.")
    texts: list[str] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        problem = problem_from_row(row)
        if source == "question":
            texts.append(problem)
            continue
        cot = normalize_pair_field(
            row,
            (
                "long_cot",
                "long_answer",
                "verbose_cot",
                "answer_long",
            )
            if source == "long"
            else (
                "short_cot",
                "short_answer",
                "concise_cot",
                "answer_short",
            ),
        )
        if cot is None:
            raise ValueError(
                f"--pair_calibration_source {source!r} requires {source}_cot "
                f"fields; got keys={list(row)}"
            )
        long_prompt = row.get("long_prompt")
        if not long_prompt:
            raise ValueError(
                "Pair rows must contain long_prompt so calibration uses the "
                "same target-model prompt prefix as vector extraction."
            )
        prompt = str(long_prompt)
        texts.append(
            prompt + str(cot)
            if prompt.endswith(("\n", " ", "\t"))
            else prompt + "\n" + str(cot)
        )
    if not texts:
        raise RuntimeError(f"No calibration texts found in {pairs_path}")
    return texts


def load_questions(dataset: str, local_data_path: str | None) -> list[str]:
    if local_data_path is None:
        if dataset == "gsm8k":
            local_data_path = "./datasets/gsm8k/test.jsonl"
        elif dataset == "math":
            local_data_path = "./datasets/math/test.jsonl"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    questions: list[str] = []
    for row in read_jsonl(local_data_path):
        question = row.get("question") or row.get("problem") or row.get("input")
        if question:
            questions.append(str(question))
    if not questions:
        raise RuntimeError(f"No questions found in {local_data_path}")
    return questions


def build_prompt(problem: str, mode: str, tokenizer: AutoTokenizer | None = None) -> str:
    if mode == "raw":
        return problem
    if mode == "paper_cot":
        return f"Question: {problem}\nLet's think step by step."
    if mode == "paper_boxed_cot":
        return (
            f"Question: {problem}\n"
            "Let's think step by step. Put the final answer in \\boxed{}."
        )
    if mode == "deepseek":
        if tokenizer is None:
            raise ValueError("deepseek mode requires tokenizer")
        instruction = (
            "Reason step by step to solve the following math problem. Put your "
            f"final answer inside \\boxed{{}} at the end of your reasoning.\n"
            f"Problem: {problem}\n\nOkay, let's break this down step by step:"
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        ) + "\n"
    if mode == "chat_paper_cot":
        if tokenizer is None:
            raise ValueError("chat_paper_cot mode requires tokenizer")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Question: {problem}\nLet's think step by step."}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"Unknown prompt_mode: {mode}")


def load_tokenizer(model_name: str, mode: str) -> AutoTokenizer:
    if mode not in {"auto", "fast", "slow"}:
        raise ValueError(f"Unknown tokenizer_mode: {mode}")

    attempts = [True, False] if mode == "auto" else [mode == "fast"]
    errors: list[str] = []
    for use_fast in attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=use_fast,
            )
            print(f"  tokenizer:      {'fast' if use_fast else 'slow'}")
            return tokenizer
        except Exception as exc:
            errors.append(f"{'fast' if use_fast else 'slow'} tokenizer failed: {exc}")

    raise RuntimeError("Could not load tokenizer.\n" + "\n".join(errors))


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
        value = value.strip()
        memory[int(key) if key.isdigit() else key] = value
    return memory


def model_input_device(model: torch.nn.Module) -> torch.device:
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
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def compute_hidden_norm_stats(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    layer_index: int,
    batch_size: int,
    max_input_tokens: int,
) -> dict[str, Any]:
    input_device = model_input_device(model)
    layers = get_transformer_layers(model)
    if layer_index >= len(layers):
        raise IndexError(
            f"layer_index={layer_index} is out of range for {len(layers)} layers"
        )
    norms: list[float] = []
    old_truncation_side = tokenizer.truncation_side
    old_padding_side = tokenizer.padding_side
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="Measuring hidden norms"):
            batch_prompts = prompts[start : start + batch_size]
            captured: list[torch.Tensor] = []

            def capture_hook(_module: Any, _inputs: Any, output: Any) -> None:
                captured.append(first_tensor(output)[:, -1].detach().float().cpu())

            handle = layers[layer_index].register_forward_hook(capture_hook)
            inputs = tokenizer(
                batch_prompts,
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
            hidden = captured[0]
            norms.extend(float(value) for value in hidden.norm(dim=-1).detach().cpu().tolist())
    finally:
        tokenizer.truncation_side = old_truncation_side
        tokenizer.padding_side = old_padding_side

    arr = np.asarray(norms, dtype=np.float64)
    return {
        "num_samples": int(len(norms)),
        "hidden_norm_mean": float(np.mean(arr)),
        "hidden_norm_median": float(np.median(arr)),
        "hidden_norm_p25": float(np.percentile(arr, 25)),
        "hidden_norm_p75": float(np.percentile(arr, 75)),
        "hidden_norm_min": float(np.min(arr)),
        "hidden_norm_max": float(np.max(arr)),
    }


def parse_ratios(text: str) -> list[float]:
    ratios = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not ratios:
        raise ValueError("At least one ratio is required.")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("Ratios must be positive.")
    return ratios


def parse_float_list(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def read_paper_kl_candidates(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload = read_json_or_jsonl(path)
    rows: list[dict[str, Any]] = []
    selected = payload.get("selected") if isinstance(payload, dict) else None
    if isinstance(selected, dict) and "gamma_for_raw_vector" in selected:
        rows.append(
            {
                "method": f"paper_kl_selected_eps_{selected.get('epsilon')}",
                "gamma_for_raw_vector": float(selected["gamma_for_raw_vector"]),
                "epsilon": selected.get("epsilon"),
            }
        )
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if isinstance(candidates, list):
        for row in candidates:
            if isinstance(row, dict) and "gamma_for_raw_vector" in row:
                rows.append(
                    {
                        "method": f"paper_kl_eps_{row.get('epsilon')}",
                        "gamma_for_raw_vector": float(row["gamma_for_raw_vector"]),
                        "epsilon": row.get("epsilon"),
                    }
                )
    deduped: dict[float, dict[str, Any]] = {}
    for row in rows:
        deduped[round(float(row["gamma_for_raw_vector"]), 12)] = row
    return list(deduped.values())


def gamma_csv(values: list[float]) -> str:
    deduped = sorted({round(float(value), 10) for value in values})
    return ",".join(f"{value:.10g}" for value in deduped)


def make_fine_grid(center: float, radius: float, step: float) -> list[float]:
    if radius <= 0 or step <= 0:
        return []
    values: list[float] = []
    current = max(0.0, center - radius)
    end = center + radius
    while current <= end + step * 0.5:
        values.append(round(current, 10))
        current += step
    values.append(round(center, 10))
    return sorted({value for value in values if value >= 0})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract practical ASC gamma with hidden-norm normalized calibration."
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math"])
    parser.add_argument("--local_data_path", type=str, default=None)
    parser.add_argument(
        "--pairs_path",
        type=str,
        default=DEFAULT_PAIRS_PATH,
        help="Use the same pair file that produced the steering vector for calibration.",
    )
    parser.add_argument(
        "--pair_calibration_source",
        type=str,
        default="long",
        choices=["long", "short", "question"],
        help="When --pairs_path is set, use question+long_cot by default.",
    )
    parser.add_argument("--steering_vector", type=str, default=None)
    parser.add_argument("--layer_index", type=int, default=None)
    parser.add_argument("--num_cal", type=int, default=None)
    parser.add_argument("--ratios", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--select_ratio", type=float, default=0.2)
    parser.add_argument("--paper_kl_json", type=str, default=None)
    parser.add_argument(
        "--include_paper_kl_in_grid",
        action="store_true",
        help=(
            "Also include strict paper-KL gamma values in candidate_gammas_csv. "
            "They are usually tiny and are kept as diagnostics by default."
        ),
    )
    parser.add_argument(
        "--extra_gammas",
        type=str,
        default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5",
        help="Extra gamma values to include in candidate_gammas_csv for grid evaluation.",
    )
    parser.add_argument(
        "--fine_radius",
        type=float,
        default=0.08,
        help="Radius around the selected hidden-ratio gamma for the fine-grid suggestion.",
    )
    parser.add_argument(
        "--fine_step",
        type=float,
        default=0.02,
        help="Step size for the fine-grid suggestion around the selected gamma.",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="paper_cot",
        choices=["raw", "paper_cot", "paper_boxed_cot", "deepseek", "chat_paper_cot"],
    )
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_memory", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="flash_attention_2",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--tokenizer_mode",
        type=str,
        default="auto",
        choices=["auto", "fast", "slow"],
        help="auto tries fast tokenizer first and falls back to slow tokenizer.",
    )
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    if args.num_cal is not None and args.num_cal <= 0:
        raise ValueError("--num_cal must be positive when provided.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = resolve_model_config(args.model_name)
    steering_vector_path = (
        args.steering_vector
        or infer_steering_vector_from_pairs(args.pairs_path)
        or config.get("steering_vector_path")
    )
    layer_index = args.layer_index if args.layer_index is not None else config.get("layer_index")
    if steering_vector_path is None:
        raise ValueError("Could not infer steering vector path. Pass --steering_vector.")
    if layer_index is None:
        raise ValueError("Could not infer layer index. Pass --layer_index.")

    print("ASC hidden-norm gamma extraction")
    print(f"  model:           {args.model_name}")
    if args.pairs_path:
        print(f"  pairs:           {args.pairs_path}")
    else:
        print(f"  dataset:         {args.dataset}")
    print(f"  prompt_mode:     {args.prompt_mode}")
    if args.pairs_path:
        print("  text mode:       long_prompt + cot")
    print(f"  steering vector: {steering_vector_path}")
    print(f"  layer index:     {layer_index}")
    print(f"  ratios:          {args.ratios}")

    tokenizer = load_tokenizer(args.model_name, args.tokenizer_mode)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_from_arg(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": True,
        "attn_implementation": args.attn_impl,
    }
    max_memory = parse_max_memory(args.max_memory)
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).eval()

    steering_vector, vector_stats = load_raw_steering_vector(steering_vector_path)
    if args.pairs_path:
        calibration_texts = load_pair_calibration_texts(
            args.pairs_path,
            args.pair_calibration_source,
        )
        rng = random.Random(args.seed)
        rng.shuffle(calibration_texts)
        if args.num_cal is not None:
            calibration_texts = calibration_texts[: args.num_cal]
        if args.pair_calibration_source == "question":
            prompts = [
                build_prompt(question, args.prompt_mode, tokenizer)
                for question in calibration_texts
            ]
        else:
            prompts = calibration_texts
    else:
        questions = load_questions(args.dataset, args.local_data_path)
        rng = random.Random(args.seed)
        rng.shuffle(questions)
        if args.num_cal is not None:
            questions = questions[: args.num_cal]
        prompts = [build_prompt(question, args.prompt_mode, tokenizer) for question in questions]

    hidden_stats = compute_hidden_norm_stats(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        layer_index=int(layer_index),
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
    )

    steering_norm = float(steering_vector.norm().item())
    hidden_ref = float(hidden_stats["hidden_norm_median"])
    candidates = []
    ratios = parse_ratios(args.ratios)
    for ratio in ratios:
        perturb_l2 = ratio * hidden_ref
        gamma = perturb_l2 / steering_norm
        candidates.append(
            {
                "method": f"hidden_norm_ratio_{ratio:g}",
                "ratio": float(ratio),
                "gamma_for_raw_vector": float(gamma),
                "gamma_for_unit_vector": float(perturb_l2),
                "steering_perturb_l2": float(perturb_l2),
                "formula": "gamma = ratio * hidden_norm_median / steering_vector_l2_norm",
            }
        )

    selected = min(candidates, key=lambda row: abs(float(row["ratio"]) - args.select_ratio))
    paper_kl_candidates = read_paper_kl_candidates(args.paper_kl_json)
    grid_values = parse_float_list(args.extra_gammas)
    grid_values.extend(float(row["gamma_for_raw_vector"]) for row in candidates)
    if args.include_paper_kl_in_grid:
        grid_values.extend(float(row["gamma_for_raw_vector"]) for row in paper_kl_candidates)
    fine_grid_values = make_fine_grid(
        center=float(selected["gamma_for_raw_vector"]),
        radius=args.fine_radius,
        step=args.fine_step,
    )
    strict_kl_values = [float(row["gamma_for_raw_vector"]) for row in paper_kl_candidates]
    report = {
        "selection_type": "hidden_norm_normalized_gamma",
        "selected": selected,
        "candidates": candidates,
        "paper_kl_candidates": paper_kl_candidates,
        "grid_suggestion": {
            "note": (
                "Default grid excludes strict paper-KL values because they are "
                "diagnostic safety bounds and are often orders of magnitude "
                "smaller than practical compression gammas."
            ),
            "candidate_gammas": sorted({round(float(value), 10) for value in grid_values}),
            "candidate_gammas_csv": gamma_csv(grid_values),
            "fine_candidate_gammas": fine_grid_values,
            "fine_candidate_gammas_csv": gamma_csv(fine_grid_values),
            "strict_paper_kl_gammas": sorted(
                {round(float(value), 12) for value in strict_kl_values}
            ),
            "strict_paper_kl_gammas_csv": gamma_csv(strict_kl_values),
        },
        "config": {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "local_data_path": args.local_data_path,
            "pairs_path": args.pairs_path,
            "pair_calibration_source": args.pair_calibration_source if args.pairs_path else None,
            "text_mode": "long_prompt+cot" if args.pairs_path else None,
            "prompt_mode": args.prompt_mode,
            "num_cal": args.num_cal,
            "ratios": args.ratios,
            "select_ratio": args.select_ratio,
            "paper_kl_json": args.paper_kl_json,
            "include_paper_kl_in_grid": args.include_paper_kl_in_grid,
            "extra_gammas": args.extra_gammas,
            "fine_radius": args.fine_radius,
            "fine_step": args.fine_step,
            "seed": args.seed,
            "layer_index_zero_based": int(layer_index),
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "device_map": args.device_map,
            "max_memory": args.max_memory,
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "tokenizer_mode": args.tokenizer_mode,
            "paper_gamma": config.get("paper_gamma"),
        },
        "steering_vector_stats": vector_stats,
        "hidden_norm_stats": hidden_stats,
        "note": (
            "gamma_for_raw_vector is the value to pass to eval_asc_paper.py "
            "or eval_asc_dual_gpu_grid.py as --gamma_override."
        ),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nRESULT")
    for row in candidates:
        print(
            f"  {row['method']}: gamma={row['gamma_for_raw_vector']:.10g}, "
            f"perturb_l2={row['steering_perturb_l2']:.6f}"
        )
    print(f"  selected gamma_override: {selected['gamma_for_raw_vector']:.10g}")
    print(f"  grid candidate_gammas: {report['grid_suggestion']['candidate_gammas_csv']}")
    print(f"  fine grid candidate_gammas: {report['grid_suggestion']['fine_candidate_gammas_csv']}")
    if strict_kl_values:
        print(
            "  strict paper-KL diagnostic gammas: "
            f"{report['grid_suggestion']['strict_paper_kl_gammas_csv']}"
        )
    print(f"  saved: {output_path}")


if __name__ == "__main__":
    main()
