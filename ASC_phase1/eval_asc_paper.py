"""
Unified paper-aligned ASC evaluator.

This is now the single evaluation entrypoint for both one-GPU and two-GPU runs.

Key behavior:
  - GPU layout is auto-selected by visible CUDA devices unless overridden.
  - A single gamma can be evaluated with the old --steering/--gamma_override flow.
  - Multiple gammas can be evaluated with --candidate_gammas.
  - Original paper models can use MODEL_CONFIGS defaults.
  - New models must pass steering vector, layer, and gamma explicitly.
  - gamma=0 means ordinary CoT generation with no steering hook.
  - Each completed gamma is written immediately to its own JSON file.

Examples:
  python eval_asc_paper.py --dataset gsm8k --limit 100

  python eval_asc_paper.py --dataset gsm8k --steering --gamma_override 0.126 \
      --limit 100 --output_path results/paper_qwen_gsm8k_asc_0126.json

  python eval_asc_paper.py --dataset math --limit 200 \
      --candidate_gammas 0,0.1,0.2,0.3 \
      --per_gamma_output_dir results/new_vector_qwen_math_200 \
      --steering_vector_path vectors/steering_vectors_qwen7b_math_train_deepseek_checked.pt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_utils import compare_answers
from answer_utils import extract_all_answers
from answer_utils import extract_answer
from answer_utils import extract_ground_truth


MODEL_CONFIGS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "gamma": 0.27,
        "steering_vector_path": "./vectors/steering_vectors_qwen7b.pt",
        "layer_index": 20,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "gamma": 0.46,
        "steering_vector_path": "./vectors/steering_vectors_llama8b.pt",
        "layer_index": 20,
    },
    "Qwen/QwQ-32B": {
        "gamma": 0.5,
        "steering_vector_path": "./vectors/steering_vectors_qwq32b.pt",
        "layer_index": 57,
    },
}

MODEL_CONFIGS["/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"] = (
    MODEL_CONFIGS["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]
)
MODEL_CONFIGS["/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"] = (
    MODEL_CONFIGS["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
)
MODEL_CONFIGS["/root/autodl-tmp/Qwen/QwQ-32B"] = MODEL_CONFIGS["Qwen/QwQ-32B"]


def is_qwen3_model_name(name: str | None) -> bool:
    return bool(name) and "qwen3" in name.lower()


def is_qwen3_model(model_name: str | None) -> bool:
    return is_qwen3_model_name(model_name)


def cli_has_arg(flag: str, argv: list[str]) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_config(model_name: str) -> tuple[str, dict[str, Any]]:
    if model_name not in MODEL_CONFIGS:
        for key in MODEL_CONFIGS:
            if model_name in key or key in model_name:
                model_name = key
                break
    return model_name, MODEL_CONFIGS.get(model_name, {})


def build_paper_prompt(
    problem: str,
    mode: str,
    tokenizer=None,
    qwen3_enable_thinking: bool = False,
    add_no_think_tag: bool = False,
) -> str:
    if mode == "raw":
        return problem
    if mode == "paper_cot":
        return f"Question: {problem}\nLet's think step by step."
    if mode == "paper_boxed_cot":
        return (
            f"Question: {problem}\n"
            "Let's think step by step. Put the final answer in \\boxed{}."
        )
    if mode in {"chat_paper_cot", "chat_boxed_cot"}:
        if tokenizer is None:
            raise ValueError(f"{mode} requires tokenizer")
        if mode == "chat_paper_cot":
            content = f"Question: {problem}\nLet's think step by step."
        else:
            content = (
                f"Question: {problem}\n"
                "Let's think step by step. Put the final answer in \\boxed{}."
            )
        if qwen3_enable_thinking and is_qwen3_model_name(getattr(tokenizer, "name_or_path", "")):
            content = (
                f"Question: {problem}\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
        if add_no_think_tag and not qwen3_enable_thinking:
            content = "/no_think\n" + content
        messages = [{"role": "user", "content": content}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=qwen3_enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    raise ValueError(f"Unknown prompt_mode: {mode}")


def parse_prediction(
    generation: str,
    gt: str,
    dataset: str,
    use_our_eval: bool,
) -> tuple[str, bool]:
    if use_our_eval:
        pred = extract_answer(generation)
        return pred, compare_answers(pred, gt)

    preds = extract_all_answers(generation)
    seen = set()
    preds = [p for p in preds if not (p in seen or seen.add(p))]
    if not preds:
        return "", False

    for pred in reversed(preds):
        if compare_answers(pred, gt):
            return pred, True

    if dataset == "math" and len(preds) > 1:
        pred = ",".join(sorted(preds))
        gt_norm = ",".join(sorted([p.strip() for p in gt.split(",")]))
    else:
        pred = preds[-1]
        gt_norm = gt
    return pred, compare_answers(pred, gt_norm)


def has_repetition_artifact(generation: str) -> bool:
    """Detect obvious post-answer template loops without affecting correctness."""

    final_answer_hits = len(
        re.findall(r"(?:final\s+answer|\\boxed\s*\{)", generation, flags=re.I)
    )
    if final_answer_hits >= 8:
        return True

    lines = [line.strip() for line in generation.splitlines() if line.strip()]
    if not lines:
        return False
    counts: dict[str, int] = {}
    for line in lines:
        if len(line) < 8:
            continue
        counts[line] = counts.get(line, 0) + 1
        if counts[line] >= 6:
            return True
    return False


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_max_memory(text: str | None) -> dict[int | str, str] | None:
    if not text:
        return None
    memory: dict[int | str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("--max_memory must look like '0:22GiB,1:22GiB'")
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        memory[int(key) if key.isdigit() else key] = value
    return memory


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


def parse_candidate_gammas(
    text: str | None,
    gamma_min: float,
    gamma_max: float,
    gamma_steps: int,
    include_zero: bool,
) -> list[float]:
    if text:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    else:
        if gamma_steps < 2:
            values = [float(gamma_max)]
        else:
            values = np.linspace(gamma_min, gamma_max, gamma_steps).tolist()

    if include_zero and all(abs(value) > 1e-12 for value in values):
        values = [0.0] + values

    deduped: list[float] = []
    seen: set[float] = set()
    for value in values:
        rounded = round(float(value), 10)
        if rounded not in seen:
            deduped.append(rounded)
            seen.add(rounded)
    return deduped


def resolve_candidate_gammas(args: argparse.Namespace, cfg: dict[str, Any]) -> list[float]:
    if args.candidate_gammas is not None:
        gamma_max = args.gamma_max if args.gamma_max is not None else float(cfg.get("gamma", 1.0))
        return parse_candidate_gammas(
            args.candidate_gammas,
            gamma_min=args.gamma_min,
            gamma_max=gamma_max,
            gamma_steps=args.gamma_steps,
            include_zero=args.include_zero,
        )

    if args.gamma_max is not None:
        return parse_candidate_gammas(
            None,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            gamma_steps=args.gamma_steps,
            include_zero=args.include_zero,
        )

    gamma = args.gamma_override if args.steering and args.gamma_override is not None else 0.0
    if args.steering and args.gamma_override is None:
        if "gamma" not in cfg:
            raise ValueError(
                "This model has no default gamma. Pass --gamma_override or "
                "--candidate_gammas explicitly."
            )
        gamma = cfg["gamma"]
    return [round(float(gamma), 10)]


def resolve_device_map(args: argparse.Namespace) -> Any:
    if args.device_map is not None:
        return args.device_map

    if not torch.cuda.is_available():
        return None

    visible_gpus = torch.cuda.device_count()
    if args.num_gpus == "auto":
        requested_gpus = min(visible_gpus, 2)
    else:
        requested_gpus = int(args.num_gpus)

    if requested_gpus >= 2 and visible_gpus >= 2:
        return "balanced"
    return {"": 0}


def get_transformer_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise AttributeError("Could not find transformer layers on this model.")


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


def load_samples(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.local_data_path is None:
        args.local_data_path = (
            "./datasets/gsm8k/test.jsonl"
            if args.dataset == "gsm8k"
            else "./datasets/math/test.jsonl"
        )

    samples = []
    for item in read_jsonl(args.local_data_path):
        question = item["question"] if args.dataset == "gsm8k" else item["problem"]
        samples.append(
            {
                "question": question,
                "gt_answer": extract_ground_truth(args.dataset, item),
            }
        )
    if args.limit > 0:
        samples = samples[: args.limit]
    return samples


def load_steering_vector(path: str) -> torch.Tensor:
    try:
        steering_obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        steering_obj = torch.load(path, map_location="cpu")
    if isinstance(steering_obj, dict):
        tensors = [value for value in steering_obj.values() if torch.is_tensor(value)]
        if not tensors:
            raise ValueError(f"No tensor found in {path}")
        steering_obj = tensors[0]
    if not torch.is_tensor(steering_obj):
        raise TypeError(f"Expected tensor steering vector, got {type(steering_obj)!r}")
    steering_vec_cpu = steering_obj.detach().float().cpu()
    if steering_vec_cpu.ndim == 2:
        steering_vec_cpu = steering_vec_cpu.mean(dim=0)
    elif steering_vec_cpu.ndim != 1:
        raise ValueError(f"Unexpected steering vector shape: {tuple(steering_vec_cpu.shape)}")
    return steering_vec_cpu


def make_cached_steering_hook(steering_vec_cpu: torch.Tensor, gamma: float):
    cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def add_steer(_, __, output):
        hidden = output[0]
        target = hidden[:, -1, :]
        key = (str(target.device), target.dtype)
        steering_vec = cache.get(key)
        if steering_vec is None:
            steering_vec = steering_vec_cpu.to(device=target.device, dtype=target.dtype)
            cache[key] = steering_vec
        hidden[:, -1, :] = target - gamma * steering_vec
        return (hidden, *output[1:])

    return add_steer


@torch.no_grad()
def run_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    repetition_penalty: float,
    input_device: torch.device,
) -> tuple[list[str], list[int]]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    ).to(input_device)
    prompt_len = inputs["input_ids"].shape[1]

    autocast_enabled = input_device.type == "cuda"
    autocast_device_type = "cuda" if input_device.type == "cuda" else "cpu"
    with torch.amp.autocast(
        dtype=torch.bfloat16,
        device_type=autocast_device_type,
        enabled=autocast_enabled,
    ):
        generation_kwargs: dict[str, Any] = dict(inputs)
        generation_kwargs.update(
            {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "top_p": top_p,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
                "repetition_penalty": repetition_penalty,
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

    generations = tokenizer.batch_decode(
        outputs[:, prompt_len:], skip_special_tokens=True
    )
    token_counts = (
        outputs[:, prompt_len:] != tokenizer.pad_token_id
    ).sum(dim=1).tolist()
    return generations, token_counts


def model_alias_from_name(model_name: str) -> str:
    lower = model_name.lower()
    if "qwen3" in lower:
        alias = Path(model_name.rstrip("/\\")).name.lower()
        alias = re.sub(r"[^a-z0-9]+", "_", alias).strip("_")
        return alias or "qwen3"
    if "qwq" in lower:
        return "qwq"
    if "qwen" in lower:
        return "qwen"
    if "llama" in lower:
        return "llama"
    alias = Path(model_name.rstrip("/\\")).name.lower()
    alias = re.sub(r"[^a-z0-9]+", "_", alias).strip("_")
    return alias or "model"


def gamma_file_tag(gamma: float) -> str:
    if abs(gamma) < 1e-12:
        return "000"
    scaled = gamma * 100
    rounded = round(scaled)
    if abs(scaled - rounded) < 1e-8:
        return f"{int(rounded):03d}"
    return f"{gamma:.6g}".replace("-", "m").replace(".", "p")


def per_gamma_filename(
    model_alias: str,
    dataset: str,
    gamma: float,
    total: int,
) -> str:
    return f"paper_{model_alias}_{dataset}_asc_{gamma_file_tag(gamma)}_{total}.json"


def write_json_atomic(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(output_path)


def build_output_report(
    args: argparse.Namespace,
    gamma: float,
    metrics: dict[str, Any],
    run_metrics: list[dict[str, Any]],
    details: list[dict[str, Any]],
    candidate_gammas: list[float],
) -> dict[str, Any]:
    config = vars(args).copy()
    config.update(
        {
            "steering": abs(gamma) > 1e-12,
            "gamma_override": float(gamma),
            "candidate_gammas": candidate_gammas,
            "resolved_device_map": str(args.resolved_device_map),
        }
    )
    return {
        "config": config,
        "metrics": metrics,
        "run_metrics": run_metrics,
        "detailed_results": details,
    }


def evaluate_gamma(
    gamma: float,
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, str]],
    args: argparse.Namespace,
    input_device: torch.device,
    steering_vec_cpu: torch.Tensor | None,
    layer_index: int,
) -> dict[str, Any]:
    handle = None
    if abs(gamma) > 1e-12:
        if steering_vec_cpu is None:
            raise ValueError("Nonzero gamma requires a steering vector.")
        layers = get_transformer_layers(model)
        handle = layers[layer_index].register_forward_hook(
            make_cached_steering_hook(steering_vec_cpu, gamma)
        )

    total = len(samples)
    run_metrics: list[dict[str, Any]] = []
    last_details: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []

    try:
        for run_idx in range(args.num_runs):
            set_seed(args.seed + run_idx)
            correct = 0
            total_tokens = 0
            total_time = 0.0
            length_capped = 0
            repetition_artifacts = 0
            details: list[dict[str, Any]] = []

            desc = f"gamma={gamma:.6g}"
            if args.num_runs > 1:
                desc += f" run={run_idx + 1}/{args.num_runs}"
            pbar = tqdm(range(0, total, args.batch_size), desc=desc)

            for i in pbar:
                batch = samples[i : i + args.batch_size]
                prompts = [
                    build_paper_prompt(
                        row["question"],
                        args.prompt_mode,
                        tokenizer,
                        qwen3_enable_thinking=args.qwen3_enable_thinking,
                        add_no_think_tag=args.qwen3_add_no_think_tag,
                    )
                    for row in batch
                ]
                gt_answers = [row["gt_answer"] for row in batch]

                t0 = time.time()
                generations, token_counts = run_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    repetition_penalty=args.repetition_penalty,
                    input_device=input_device,
                )
                elapsed = time.time() - t0
                total_time += elapsed

                for prompt, gen, token_count, gt in zip(
                    prompts,
                    generations,
                    token_counts,
                    gt_answers,
                ):
                    is_length_capped = token_count >= args.max_new_tokens
                    repetition_artifact = has_repetition_artifact(gen)
                    pred, is_correct = parse_prediction(
                        gen,
                        gt,
                        args.dataset,
                        args.use_our_eval,
                    )
                    correct += int(is_correct)
                    total_tokens += token_count
                    length_capped += int(is_length_capped)
                    repetition_artifacts += int(repetition_artifact)

                    if args.save_failures and not is_correct:
                        failure_cases.append(
                            {
                                "question": prompt[:300],
                                "model_output": gen[-500:],
                                "full_output": gen,
                                "pred_answer": pred,
                                "gt_answer": gt,
                                "gamma": float(gamma),
                                "run": run_idx + 1,
                            }
                        )

                    if args.save_details != "none":
                        details.append(
                            {
                                "question": prompt[:200],
                                "model_output": gen,
                                "pred_answer": pred,
                                "gt_answer": gt,
                                "correct": is_correct,
                                "tokens": token_count,
                                "length_capped": is_length_capped,
                                "repetition_artifact": repetition_artifact,
                            }
                        )

                done = min(i + args.batch_size, total)
                pbar.set_postfix(
                    {
                        "acc": f"{correct}/{done}",
                        "tok/s": f"{sum(token_counts) / max(elapsed, 0.001):.0f}",
                    }
                )

            run_metrics.append(
                {
                    "run": run_idx + 1,
                    "accuracy": correct / total,
                    "correct": correct,
                    "total": total,
                    "avg_tokens": total_tokens / total,
                    "avg_time_sec": total_time / total,
                    "length_capped_rate": length_capped / total,
                    "repetition_artifact_rate": repetition_artifacts / total,
                }
            )
            last_details = details
    finally:
        if handle is not None:
            handle.remove()

    accuracies = [row["accuracy"] for row in run_metrics]
    avg_tokens = [row["avg_tokens"] for row in run_metrics]
    avg_times = [row["avg_time_sec"] for row in run_metrics]
    length_capped_rates = [row["length_capped_rate"] for row in run_metrics]
    repetition_artifact_rates = [
        row["repetition_artifact_rate"] for row in run_metrics
    ]
    correct_mean = int(round(float(np.mean([row["correct"] for row in run_metrics]))))

    return {
        "gamma": float(gamma),
        "accuracy": float(np.mean(accuracies)),
        "correct": correct_mean,
        "total": int(total),
        "avg_tokens": float(np.mean(avg_tokens)),
        "avg_time_sec": float(np.mean(avg_times)),
        "length_capped_rate": float(np.mean(length_capped_rates)),
        "repetition_artifact_rate": float(np.mean(repetition_artifact_rates)),
        "accuracy_std": float(np.std(accuracies)) if len(accuracies) > 1 else 0.0,
        "run_metrics": run_metrics,
        "detailed_results": last_details,
        "failure_cases": failure_cases,
    }


def strip_details_for_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"detailed_results", "failure_cases", "run_metrics"}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified paper-aligned ASC evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math"])
    parser.add_argument("--local_data_path", type=str, default=None)
    parser.add_argument("--steering", action="store_true")
    parser.add_argument("--gamma_override", type=float, default=None)
    parser.add_argument("--candidate_gammas", type=str, default=None)
    parser.add_argument("--gamma_min", type=float, default=0.0)
    parser.add_argument("--gamma_max", type=float, default=None)
    parser.add_argument("--gamma_steps", type=int, default=6)
    parser.add_argument(
        "--include_zero",
        action="store_true",
        default=False,
        help="Add gamma=0 to generated/explicit gamma lists if missing.",
    )
    parser.add_argument(
        "--steering_vector_path",
        "--steering_vector",
        type=str,
        default=None,
        help=(
            "Override steering vector path. Defaults exist only for the original "
            "Qwen-7B/Llama-8B/QwQ-32B models."
        ),
    )
    parser.add_argument("--layer_index", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="paper_cot",
        choices=[
            "raw",
            "paper_cot",
            "paper_boxed_cot",
            "chat_paper_cot",
            "chat_boxed_cot",
        ],
    )
    parser.add_argument("--output_path", type=str, default="./eval_asc_paper_results.json")
    parser.add_argument(
        "--per_gamma_output_dir",
        type=str,
        default="results",
        help="Directory for one full JSON file per gamma value.",
    )
    parser.add_argument(
        "--file_model_alias",
        type=str,
        default=None,
        help="Optional filename model tag, e.g. qwen or llama.",
    )
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--qwen3_enable_thinking",
        dest="qwen3_enable_thinking",
        action="store_true",
    )
    parser.add_argument(
        "--no-qwen3_enable_thinking",
        dest="qwen3_enable_thinking",
        action="store_false",
    )
    parser.set_defaults(qwen3_enable_thinking=False)
    parser.add_argument(
        "--disable_qwen3_auto_defaults",
        action="store_true",
        help="Skip Qwen3-specific default overrides.",
    )
    parser.add_argument("--save_failures", action="store_true")
    parser.add_argument(
        "--save_details",
        choices=["all", "none"],
        default="all",
        help="all saves every sample's full model_output/CoT in each JSON.",
    )
    parser.add_argument(
        "--use_our_eval",
        action="store_true",
        help="Compatibility flag; uses answer_utils.py either way.",
    )
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2")
    parser.add_argument(
        "--num_gpus",
        type=str,
        default="auto",
        choices=["auto", "1", "2"],
        help="auto uses 2 visible GPUs when available, otherwise 1.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Override auto GPU layout, e.g. auto, balanced, balanced_low_0.",
    )
    parser.add_argument(
        "--max_memory",
        type=str,
        default=None,
        help="Optional, e.g. '0:22GiB,1:22GiB,cpu:80GiB'.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--write_grid_summary",
        action="store_true",
        help="Optional compatibility summary. Per-gamma files are always saved first.",
    )
    parser.add_argument(
        "--early_stop_accuracy",
        type=float,
        default=None,
        help="Optional: stop after consecutive nonzero gammas below this accuracy.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Consecutive low-accuracy nonzero gammas before stopping. 0 disables.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Deprecated; completed gamma files are saved immediately.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = sys.argv[1:]

    if is_qwen3_model(args.model_name) and not args.disable_qwen3_auto_defaults:
        if not cli_has_arg("--prompt_mode", argv):
            args.prompt_mode = "chat_boxed_cot"
        if not (
            cli_has_arg("--qwen3_enable_thinking", argv)
            or cli_has_arg("--no-qwen3_enable_thinking", argv)
        ):
            args.qwen3_enable_thinking = True
        if not cli_has_arg("--max_new_tokens", argv):
            args.max_new_tokens = 4096 if args.dataset == "gsm8k" else 8192
        if not cli_has_arg("--temperature", argv):
            args.temperature = 0.6
        if not cli_has_arg("--top_p", argv):
            args.top_p = 0.95
        if not cli_has_arg("--top_k", argv):
            args.top_k = 20
        if not cli_has_arg("--min_p", argv):
            args.min_p = 0.0

    if args.max_new_tokens <= 0:
        args.max_new_tokens = 8192 if args.dataset == "math" else 4096
    if args.num_runs <= 0:
        raise ValueError("--num_runs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    cudnn.benchmark = True

    args.model_name, cfg = resolve_model_config(args.model_name)
    if args.steering_vector_path is None:
        args.steering_vector_path = cfg.get("steering_vector_path")
    if args.layer_index is None:
        args.layer_index = cfg.get("layer_index")

    candidate_gammas = resolve_candidate_gammas(args, cfg)
    multi_gamma = len(candidate_gammas) > 1
    needs_steering_vector = any(abs(gamma) > 1e-12 for gamma in candidate_gammas)
    if needs_steering_vector and args.steering_vector_path is None:
        raise ValueError(
            "Nonzero gamma requires a steering vector. For new models, pass "
            "--steering_vector_path explicitly. The built-in defaults are only "
            "for the original Qwen-7B/Llama-8B/QwQ-32B models."
        )
    if needs_steering_vector and args.layer_index is None:
        raise ValueError(
            "Nonzero gamma requires --layer_index. For new models, use the same "
            "layer used by extract_steering_vector.py."
        )
    model_alias = args.file_model_alias or model_alias_from_name(args.model_name)

    args.resolved_device_map = resolve_device_map(args)
    args.qwen3_add_no_think_tag = is_qwen3_model(args.model_name)

    print("ASC paper evaluation")
    print(f"  model:          {args.model_name}")
    print(f"  dataset:        {args.dataset}")
    print(f"  limit:          {args.limit}")
    print(f"  prompt:         {args.prompt_mode}")
    if is_qwen3_model(args.model_name):
        print(f"  qwen3 thinking: {args.qwen3_enable_thinking}")
    print(
        "  sampling:       "
        f"temperature={args.temperature}, top_p={args.top_p}, "
        f"top_k={args.top_k}, min_p={args.min_p}"
    )
    print(f"  batch_size:     {args.batch_size}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  device_map:     {args.resolved_device_map}")
    print("  gammas:         " + ", ".join(f"{gamma:.6g}" for gamma in candidate_gammas))
    if needs_steering_vector:
        print(f"  vector:         {args.steering_vector_path}")
        print(f"  layer:          {args.layer_index}")
    print(
        "  output:         "
        + (args.per_gamma_output_dir if multi_gamma else args.output_path)
    )

    if torch.cuda.is_available():
        print(f"  cuda devices:   {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"    cuda:{idx}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")
    else:
        print("  [warn] CUDA is not available.")

    print(f"[1/4] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[2/4] Loading model")
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_from_arg(args.dtype),
        "attn_implementation": args.attn_impl,
        "trust_remote_code": True,
    }
    if args.resolved_device_map is not None:
        model_kwargs["device_map"] = args.resolved_device_map
    max_memory = parse_max_memory(args.max_memory)
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).eval()
    input_device = get_input_device(model)
    print(f"  input device:   {input_device}")
    if hasattr(model, "hf_device_map"):
        print(f"  hf_device_map summary: {summarize_device_map(model.hf_device_map)}")

    print("[3/4] Loading samples")
    samples = load_samples(args)
    if not samples:
        raise ValueError("No samples loaded for evaluation.")
    print(f"  samples:        {len(samples)}")

    steering_vec_cpu = None
    if needs_steering_vector:
        print("[4/4] Loading steering vector and running gammas")
        steering_vec_cpu = load_steering_vector(args.steering_vector_path)
        print(f"  vector norm:    {float(steering_vec_cpu.norm().item()):.6f}")
    else:
        print("[4/4] Running gammas")

    per_gamma_output_dir = Path(args.per_gamma_output_dir)
    per_gamma_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_path)
    gamma_results: list[dict[str, Any]] = []
    low_accuracy_streak = 0
    stopped_early = False
    stop_reason = ""

    try:
        for gamma in candidate_gammas:
            row = evaluate_gamma(
                gamma=gamma,
                model=model,
                tokenizer=tokenizer,
                samples=samples,
                args=args,
                input_device=input_device,
                steering_vec_cpu=steering_vec_cpu,
                layer_index=args.layer_index,
            )

            details = row.pop("detailed_results")
            failure_cases = row.pop("failure_cases")

            metrics = {
                "accuracy_mean": row["accuracy"],
                "accuracy_std": row["accuracy_std"],
                "avg_tokens": row["avg_tokens"],
                "avg_time_per_sample": row["avg_time_sec"],
                "length_capped_rate": row["length_capped_rate"],
                "repetition_artifact_rate": row["repetition_artifact_rate"],
                "max_new_tokens": args.max_new_tokens,
            }

            if multi_gamma:
                gamma_path = per_gamma_output_dir / per_gamma_filename(
                    model_alias=model_alias,
                    dataset=args.dataset,
                    gamma=row["gamma"],
                    total=row["total"],
                )
            else:
                gamma_path = output_path

            row["per_gamma_file"] = str(gamma_path)
            report = build_output_report(
                args=args,
                gamma=row["gamma"],
                metrics=metrics,
                run_metrics=row["run_metrics"],
                details=details,
                candidate_gammas=candidate_gammas,
            )
            write_json_atomic(gamma_path, report)

            if args.save_failures and failure_cases:
                fail_path = Path(str(gamma_path).replace(".json", "_failures.json"))
                write_json_atomic(fail_path, failure_cases)

            gamma_results.append(row)
            print(
                f"  gamma={gamma:.6g}: acc={row['accuracy']:.4f}, "
                f"tokens={row['avg_tokens']:.1f}, time={row['avg_time_sec']:.2f}s, "
                f"repeat={row['repetition_artifact_rate']:.2%}"
            )
            print(f"    saved: {gamma_path}")

            if (
                args.early_stop_accuracy is not None
                and args.early_stop_patience > 0
                and gamma > 0
            ):
                if row["accuracy"] < args.early_stop_accuracy:
                    low_accuracy_streak += 1
                else:
                    low_accuracy_streak = 0
                if low_accuracy_streak >= args.early_stop_patience:
                    stopped_early = True
                    stop_reason = (
                        f"{low_accuracy_streak} consecutive nonzero gammas below "
                        f"{args.early_stop_accuracy:.4f}"
                    )
                    print(f"  [early stop] {stop_reason}")
                    break

            if args.write_grid_summary and multi_gamma:
                partial = {
                    "config": vars(args),
                    "candidate_gammas": candidate_gammas,
                    "completed": [strip_details_for_summary(item) for item in gamma_results],
                    "stopped_early": stopped_early,
                    "stop_reason": stop_reason,
                }
                write_json_atomic(output_path, partial)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Completed gamma files above are already saved.")

    print("\n" + "=" * 72)
    print("RESULT")
    for row in gamma_results:
        print(
            f"  gamma={row['gamma']:.6g} | acc={row['accuracy']:.4f} "
            f"({row['correct']}/{row['total']}) | tokens={row['avg_tokens']:.1f} "
            f"| time={row['avg_time_sec']:.2f}s "
            f"| repeat={row['repetition_artifact_rate']:.2%}"
        )
    if stopped_early:
        print(f"  stopped early:  {stop_reason}")
    print("=" * 72)

    if args.write_grid_summary and multi_gamma:
        output = {
            "config": vars(args),
            "candidate_gammas": candidate_gammas,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "gamma_results": [strip_details_for_summary(row) for row in gamma_results],
        }
        write_json_atomic(output_path, output)
        print(f"\nSaved optional grid summary to {output_path}")


if __name__ == "__main__":
    main()
