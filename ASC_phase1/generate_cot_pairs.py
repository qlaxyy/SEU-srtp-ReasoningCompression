"""
Generate long/short CoT pairs from a local JSONL dataset.

This script only does the first half of the steering-vector pipeline:
  1. Read problems from a local JSONL file.
  2. Generate long and short CoTs.
  3. Save pairs to JSON for manual filtering.

After deleting invalid pairs, run extract_steering_vector.py on the checked file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

import asc_steering_utils as utils


DEFAULT_DEEPSEEK_API_KEY = ""


def torch_dtype_from_arg(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_deepseek_api_key(cli_key: str | None) -> str:
    key = cli_key or os.environ.get("DEEPSEEK_API_KEY") or DEFAULT_DEEPSEEK_API_KEY
    if not key:
        raise ValueError(
            "DeepSeek API key is missing. Set DEEPSEEK_API_KEY, pass "
            "--deepseek_api_key, or fill DEFAULT_DEEPSEEK_API_KEY near the top "
            "of generate_cot_pairs.py."
        )
    return key


def cli_has_arg(flag: str, argv: list[str]) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate long/short CoT pairs from a local JSONL dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/math/train.jsonl",
        help="Local JSONL file. No online dataset loading is performed.",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_pairs_path",
        type=str,
        default="pairs/qwen7b_math_train_deepseek_pairs_100.json",
    )
    parser.add_argument(
        "--long_source",
        choices=["target_model", "deepseek_api"],
        default="target_model",
    )
    parser.add_argument(
        "--short_source",
        choices=["target_model", "deepseek_api"],
        default="deepseek_api",
    )
    parser.add_argument("--use_chat_template", action="store_true")
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
    parser.add_argument(
        "--qwen3_no_think_tag",
        dest="qwen3_no_think_tag",
        action="store_true",
        help="Prefix /no_think in Qwen3 chat prompts when thinking is disabled.",
    )
    parser.add_argument(
        "--no-qwen3_no_think_tag",
        dest="qwen3_no_think_tag",
        action="store_false",
    )
    parser.set_defaults(qwen3_no_think_tag=True)
    parser.add_argument("--max_new_tokens_long", type=int, default=4096)
    parser.add_argument("--max_new_tokens_short", type=int, default=2048)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--temperature_long", type=float, default=0.7)
    parser.add_argument("--temperature_short", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--deepseek_api_key", type=str, default=None)
    parser.add_argument(
        "--deepseek_base_url",
        type=str,
        default=utils.DEFAULT_DEEPSEEK_BASE_URL,
    )
    parser.add_argument("--deepseek_model", type=str, default=utils.DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--deepseek_max_tokens", type=int, default=2048)
    parser.add_argument("--deepseek_temperature", type=float, default=0.0)
    parser.add_argument(
        "--deepseek_reasoning_effort",
        type=str,
        default=utils.DEFAULT_DEEPSEEK_REASONING_EFFORT,
    )
    parser.add_argument(
        "--deepseek_thinking",
        type=str,
        choices=["disabled", "enabled", "none"],
        default=utils.DEFAULT_DEEPSEEK_THINKING,
    )
    parser.add_argument("--deepseek_retries", type=int, default=5)
    parser.add_argument("--deepseek_workers", type=int, default=4)
    parser.add_argument(
        "--include_deepseek_reasoning",
        action="store_true",
        help="Also include reasoning_content when DeepSeek returns it.",
    )
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_memory", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = sys.argv[1:]

    if utils.is_qwen3_model(args.model_name) and not args.disable_qwen3_auto_defaults:
        args.use_chat_template = True
        if not (
            cli_has_arg("--qwen3_enable_thinking", argv)
            or cli_has_arg("--no-qwen3_enable_thinking", argv)
        ):
            args.qwen3_enable_thinking = True
        if not cli_has_arg("--max_new_tokens_long", argv):
            args.max_new_tokens_long = 8192
        if not cli_has_arg("--max_new_tokens_short", argv):
            args.max_new_tokens_short = 2048
        if not cli_has_arg("--temperature_long", argv):
            args.temperature_long = 0.6
        if not cli_has_arg("--temperature_short", argv):
            args.temperature_short = 0.6
        if not cli_has_arg("--top_p", argv):
            args.top_p = 0.95
        if not cli_has_arg("--top_k", argv):
            args.top_k = 20
        if not cli_has_arg("--min_p", argv):
            args.min_p = 0.0

    if args.deepseek_thinking == "disabled" and args.deepseek_reasoning_effort != "none":
        raise ValueError(
            "DeepSeek API does not allow thinking=disabled together with "
            "reasoning_effort. Use --deepseek_reasoning_effort none."
        )
    if args.num_samples == 0 or args.num_samples < -1:
        raise ValueError("--num_samples must be positive or -1 for all rows.")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Local dataset not found: {dataset_path}. Please check --dataset_path."
        )

    utils.set_seed(args.seed)
    args.dtype = torch_dtype_from_arg(args.dtype)

    model = None
    tokenizer = None
    input_device = torch.device("cpu")
    needs_target_model = (
        args.long_source == "target_model" or args.short_source == "target_model"
    )
    if needs_target_model:
        model, tokenizer, input_device = utils.load_model_and_tokenizer(args)
        print(f"input device: {input_device}")
        if hasattr(model, "hf_device_map"):
            print(f"hf_device_map summary: {utils.summarize_device_map(model.hf_device_map)}")

    deepseek_client = None
    uses_deepseek = (
        args.long_source == "deepseek_api" or args.short_source == "deepseek_api"
    )
    if uses_deepseek:
        api_key = resolve_deepseek_api_key(args.deepseek_api_key)
        deepseek_client = utils.build_deepseek_client(api_key, args.deepseek_base_url)

    problems = utils.load_jsonl_problems(
        dataset_path=str(dataset_path),
        num_samples=args.num_samples,
        seed=args.seed,
    )
    if not problems:
        raise ValueError(f"No problems loaded from {dataset_path}")

    print("Generating CoT pairs")
    print(f"  dataset:     {dataset_path}")
    print(f"  samples:     {len(problems)}")
    print(f"  long source: {args.long_source}")
    print(f"  short source:{args.short_source}")
    if utils.is_qwen3_model(args.model_name):
        print(f"  qwen3 thinking: {args.qwen3_enable_thinking}")
    print(
        "  sampling:    "
        f"temperature_long={args.temperature_long}, "
        f"temperature_short={args.temperature_short}, "
        f"top_p={args.top_p}, top_k={args.top_k}, min_p={args.min_p}"
    )
    print(f"  output:      {args.output_pairs_path}")

    add_no_think_tag = args.qwen3_no_think_tag and utils.is_qwen3_model(
        args.model_name, model=model, tokenizer=tokenizer
    )

    pairs = utils.generate_pairs(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        output_path=args.output_pairs_path,
        input_device=input_device,
        generation_batch_size=args.generation_batch_size,
        deepseek_workers=args.deepseek_workers,
        use_chat_template=args.use_chat_template,
        qwen3_enable_thinking=args.qwen3_enable_thinking,
        add_no_think_tag=add_no_think_tag,
        long_source=args.long_source,
        short_source=args.short_source,
        deepseek_client=deepseek_client,
        deepseek_model=args.deepseek_model,
        deepseek_max_tokens=args.deepseek_max_tokens,
        deepseek_temperature=args.deepseek_temperature,
        deepseek_reasoning_effort=args.deepseek_reasoning_effort,
        deepseek_thinking=args.deepseek_thinking,
        deepseek_include_reasoning=args.include_deepseek_reasoning,
        deepseek_retries=args.deepseek_retries,
        max_new_tokens_long=args.max_new_tokens_long,
        max_new_tokens_short=args.max_new_tokens_short,
        temperature_long=args.temperature_long,
        temperature_short=args.temperature_short,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        save_interval=args.save_interval,
    )

    metadata = {
        "model_name": args.model_name if needs_target_model else None,
        "dataset_path": str(dataset_path),
        "num_pairs": len(pairs),
        "seed": args.seed,
        "long_source": args.long_source,
        "short_source": args.short_source,
        "use_chat_template": args.use_chat_template,
        "qwen3_enable_thinking": args.qwen3_enable_thinking,
        "max_new_tokens_long": args.max_new_tokens_long,
        "max_new_tokens_short": args.max_new_tokens_short,
        "temperature_long": args.temperature_long,
        "temperature_short": args.temperature_short,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "deepseek_model": args.deepseek_model if uses_deepseek else None,
        "deepseek_reasoning_effort": (
            args.deepseek_reasoning_effort if uses_deepseek else None
        ),
        "deepseek_thinking": args.deepseek_thinking if uses_deepseek else None,
        "output_pairs_path": args.output_pairs_path,
        "created_at_unix": time.time(),
    }
    utils.write_json(str(args.output_pairs_path) + ".metadata.json", metadata)

    print("\nSaved CoT pairs")
    print(f"  path:    {args.output_pairs_path}")
    print(f"  samples: {len(pairs)}")
    print("  next:    manually remove invalid rows, then run extract_steering_vector.py")


if __name__ == "__main__":
    main()
