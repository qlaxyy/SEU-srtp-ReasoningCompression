"""
Extract ASC steering vectors from a checked long/short CoT pair file.

This script only does the second half of the pipeline:
  1. Read a manually filtered pairs JSON file.
  2. Extract target-layer activations for long_prompt+short_cot and long_prompt+long_cot.
  3. Save short-minus-long vectors to a .pt file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

import asc_steering_utils as utils


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ASC steering vectors from checked CoT pairs."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--output_vector_path", type=str, required=True)
    parser.add_argument("--layer_index", type=str, default="auto")
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--activation_batch_size", type=int, default=4)
    parser.add_argument(
        "--direction",
        choices=["short_minus_long", "long_minus_short"],
        default="short_minus_long",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    utils.set_seed(args.seed)
    args.dtype = torch_dtype_from_arg(args.dtype)

    pairs_path = Path(args.pairs_path)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    pairs = utils.read_json(pairs_path)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("--pairs_path must contain a non-empty JSON list.")

    model, tokenizer, input_device = utils.load_model_and_tokenizer(args)
    layer_index = utils.resolve_layer_index(model, args.model_name, args.layer_index)

    print("Extracting steering vectors")
    print(f"  model:       {args.model_name}")
    print(f"  pairs:       {pairs_path}")
    print(f"  samples:     {len(pairs)}")
    print(f"  layer:       {layer_index}")
    print(f"  direction:   {args.direction}")
    print("  text mode:   long_prompt + cot")
    print(f"  input device:{input_device}")
    if hasattr(model, "hf_device_map"):
        print(f"  hf_device_map summary: {utils.summarize_device_map(model.hf_device_map)}")

    vectors = utils.extract_vectors(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        input_device=input_device,
        layer_index=layer_index,
        max_input_tokens=args.max_input_tokens,
        activation_batch_size=args.activation_batch_size,
        direction=args.direction,
    )

    output_vector = Path(args.output_vector_path)
    output_vector.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, output_vector)

    norms = vectors.norm(dim=1)
    first_pair = pairs[0]
    metadata = {
        "model_name": args.model_name,
        "layer_index": layer_index,
        "direction": args.direction,
        "text_mode": "long_prompt+cot",
        "num_vectors": int(vectors.shape[0]),
        "hidden_size": int(vectors.shape[1]),
        "vector_norm_mean": float(norms.mean().item()),
        "vector_norm_median": float(norms.median().item()),
        "vector_norm_min": float(norms.min().item()),
        "vector_norm_max": float(norms.max().item()),
        "mean_vector_norm": float(vectors.mean(dim=0).norm().item()),
        "pairs_path": str(pairs_path),
        "long_source": str(first_pair.get("long_source") or "unknown"),
        "short_source": str(first_pair.get("short_source") or "unknown"),
        "activation_batch_size": args.activation_batch_size,
        "output_vector_path": str(output_vector),
        "created_at_unix": time.time(),
    }
    utils.write_json(output_vector.with_suffix(output_vector.suffix + ".metadata.json"), metadata)

    print("\nSaved steering vectors")
    print(f"  path:             {output_vector}")
    print(f"  samples:          {vectors.shape[0]}")
    print(f"  hidden size:      {vectors.shape[1]}")
    print(f"  mean vector norm: {metadata['mean_vector_norm']:.6f}")


if __name__ == "__main__":
    main()
