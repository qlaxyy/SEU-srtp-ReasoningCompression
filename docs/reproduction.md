# Reproduction guide

## Scope of this release

The inference kernels, adapters, model-specific vectors and controller parameters come from the completed online experiments. A portable CLI now replaces private machine paths and deployment wrappers. The release has CPU validation and a verified 2,166-file upstream runtime snapshot; a fresh GPU benchmark of the repackaged launcher is still required. Historical scores are not newly measured release scores.

Use an editable installation from a Git checkout (`pip install -e .`). The repository-relative assets and runtime are intentional; building a standalone PyPI wheel is outside this release.

## Environment

Use a dedicated Linux x86-64 Python 3.12 environment and an NVIDIA GPU. The original runs used a 24 GB RTX 4090 D. The reproducible runtime is tied to the EasySteer vLLM fork, not a general adapter for arbitrary vLLM versions.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/install_gpu.sh
```

The installer fetches the pinned vLLM code, overlays the 12 historical modified/new files, installs the small bundled EasySteer library, and checks the Python source hashes. It uses the vLLM 0.26.0 CUDA 12.9 precompiled wheel with the recorded compatibility commit. It does not install a driver or download model weights. A compatible CUDA driver is required. `VLLM_PRECOMPILED_WHEEL_LOCATION` can point to an already downloaded wheel.

The original runtime used PyTorch 2.11.0+cu129, Triton 3.6.0 and Transformers 5.16.1. The separate CPU grading dependencies preserve the older ReBalance parser. Avoid upgrading these packages during a reproduction run.

Download the relevant official model from [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) or [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B). The release contains hashes for the exact configuration, tokenizer and weight files used historically. The launcher verifies them before inference. If a newer remote snapshot differs, select the matching older model snapshot instead of silently bypassing the check.

## Datasets

```bash
python scripts/prepare_data.py --dataset math500
python scripts/prepare_data.py --dataset gsm8k
python scripts/prepare_data.py --dataset amc23
```

The script keeps benchmark questions and gold answers in separate files. It validates the complete question set against historical normalized hashes, then restores the historical order. Optional `--revision` fixes the Hugging Face dataset revision. The downloaded dataset fingerprint and selected revision are recorded locally.

AIME25 is intentionally not auto-downloaded: the historical project text is partially edited/normalized. Supply a local question file matching `configs/datasets/aime2025_identities.json` to reproduce that text. A different public revision can be evaluated with `--allow-data-variant`, but must be reported as a new dataset variant. Full question/answer files are not redistributed in this repository.

Custom input can be JSONL or a JSON list:

```json
{"problem": "What is 3 + 4?", "source": "example"}
```

Use `--dataset custom` for such inputs. The public question hash is SHA-256 over the NFKC-normalized text with all whitespace removed. Fitting question identities are checked for overlap. Only `problem`, identity, index and source fields enter generation; gold and unrelated columns are discarded.

## Generation

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-7B

python -m reasoning_compression.cli \
  --model 7b --dataset gsm8k --method full --seed 42 \
  --input data/gsm8k/questions.json --model-path "$MODEL_PATH" \
  --output outputs/7b_gsm8k_full_s42 --execute
```

Omit `--execute`, or specify `--dry-run`, to inspect the plan without loading PyTorch, a model or a GPU. A dry-run without `--input` checks configuration only; with `--input`, it also checks the question set. `--max-samples N` selects the first N questions after the full input validation and labels the run as a subset.

| Method | ReBalance | L27 penalty | Answer vector |
|---|---|---|---|
| `unsteered` | Off | Off | Off |
| `rebalance` | On | Off | Off |
| `l27` | On | Fixed `ln 2` | Off |
| `dynamic32` | On | Calibrated, maximum 32 | Off |
| `full` | On | Calibrated, maximum 32 | SRQ/PCA16, β=0.25 |

Shared decoding: seed 42 by default, temperature 0.7, top-p 0.95, BF16, maximum 16,000 new tokens, TP=1, no prefix cache and no speculative decoding. Each request receives the selected seed, as in the historical protocol.

| Model | Context | Concurrent sequences | Batched tokens | GPU memory fraction | Chunked prefill | Async |
|---|---:|---:|---:|---:|---|---|
| 1.5B | 32768 | 256 | 32768 | 0.90 | Off | On |
| 7B MATH/AMC/AIME/custom | 17920 | 32 | 4096 | 0.95 | On | Off |
| 7B GSM8K | 17920 | 64 | 4096 | 0.95 | On | Off |

New public-CLI comparisons use the same selected engine settings for all methods. The historical result table retains its original baseline settings; these are not silently relabeled as new matched comparisons.

Batch helper, which defaults to dry-run:

```bash
MODEL=7b DATASET=math500 bash scripts/run_benchmarks.sh
MODEL=7b DATASET=math500 ACTION=--execute MODEL_PATH="$MODEL_PATH" \
  bash scripts/run_benchmarks.sh
```

Set `METHODS="dynamic32 full"` if baselines are already available under the intended protocol. The helper never reruns or overwrites an existing output directory. An interrupted run retains partial JSONL and failure evidence but is not scored as a complete result. Resume across processes is not implemented: use a new output directory for a fresh full run. This avoids pretending a changed batch continuation is identical to the original trajectory.

## Scoring and aggregation

Use a separate CPU environment so the legacy grading dependencies do not alter the serving stack:

```bash
python3.12 -m venv .venv-grade
.venv-grade/bin/python -m pip install -r requirements-grading.txt
.venv-grade/bin/python scripts/check_grader.py
.venv-grade/bin/python scripts/grade.py \
  --run outputs/7b_gsm8k_full_s42 --gold data/gsm8k/gold.json
.venv-grade/bin/python scripts/summarize.py --runs outputs --output outputs/summary.csv
```

The grader preserves ReBalance's answer extractor and mathematical equivalence code. It extracts from the entire generated output using the historical convention; it is not an LLM judge. Linux is the reference scoring platform; the legacy symbolic grader uses child processes and timeouts. Run grading through the saved script, not an interactive stdin snippet, so spawn-based platforms can import the child process correctly. Incomplete outputs, duplicate identities, inconsistent token counts and missing gold are rejected. Truncated but completed model outputs remain in the denominator.

The summarizer compares only runs with matching model, question-file identity and seed. It reports accuracy difference versus single ReBalance and token reduction versus unsteered. Missing baselines leave comparison cells empty. Duplicate baseline runs require selecting a narrower directory.

## Audit files and engineering gates

`RUN.json` records Git commit, dirty-tree status, source hashes, environment, command and plan. `job/` holds immutable copied assets and sanitized input. `generation/` holds streamed predictions, timings, audits and completion/failure markers. `SUMMARY.json` is created only after successful scoring; source output hashes are retained.

Combined-method runs first execute synthetic forced-token checks for phase isolation, exact off/zero restoration and nonzero answer-logit effects. 7B also checks replay-state restoration and independent BF16 addition. These diagnostics do not demonstrate benchmark improvement. If a gate fails, inspect the environment/source hashes and failure report; do not disable it and report the run as a successful reproduction.

Compiler caches are per output directory. One process at a time obtains a local lock for the selected `CUDA_VISIBLE_DEVICES` set. TP, speculative decoding and prefix-cache extensions need their own correctness validation.
