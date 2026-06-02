"""
Download EleutherAI/hendrycks_math to a local JSONL file.

After this, generate_cot_pairs.py can read the local JSONL without pulling the
Hugging Face dataset again.

Example:

  python download_datasets.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def parse_subjects(text: str) -> list[str]:
    if text == "all":
        return SUBJECTS
    subjects = [part.strip() for part in text.split(",") if part.strip()]
    if not subjects:
        raise ValueError("No subjects selected.")
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Hendrycks-MATH as JSONL.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subjects", type=str, default="all")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional HuggingFace dataset cache directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/math/train.jsonl",
    )
    args = parser.parse_args()

    selected_subjects = parse_subjects(args.subjects)
    datasets = []
    for subject in selected_subjects:
        dataset = load_dataset(
            "EleutherAI/hendrycks_math",
            name=subject,
            split=args.split,
            cache_dir=args.cache_dir,
        )
        datasets.append(dataset.map(lambda row, s=subject: {**row, "subject": s}))

    merged = concatenate_datasets(datasets)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in merged:
            file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    print(f"Saved {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main()
