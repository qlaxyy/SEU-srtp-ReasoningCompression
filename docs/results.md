# Historical results and interpretation

These records were collected before the open-source packaging. They are not fresh GPU measurements of the public launcher. `results/historical_results.csv` contains the unrounded correct counts and total output-token counts from the project's final reports.

## Counting and comparison

- MATH-500: 500 questions; GSM8K: 1,319; AMC23: 40; AIME25: 30.
- Output tokens include reasoning, answer and generated delimiters. Prompt tokens are excluded. Wrong answers and cap hits remain in all totals.
- `compression = 100 × (1 − T_method / T_unsteered)`.
- Accuracy is compared with **single ReBalance**, meaning the project's self-calibrated ReBalance implementation without L27 or answer guidance.
- Seed 42, temperature 0.7, top-p 0.95, output cap 16,000. These are single-seed exploratory results, not SOTA claims or multi-seed significance results.

## Answer-vector ablation

Within the later dynamic32 configuration:

| Model | Dataset | Dynamic32 correct | Full correct | Dynamic32 total tokens | Full total tokens |
|---|---|---:|---:|---:|---:|
| 1.5B | MATH-500 | 405/500 | 399/500 | 1,337,243 | 1,469,014 |
| 1.5B | GSM8K | 1019/1319 | 1024/1319 | 886,535 | 937,813 |
| 7B | MATH-500 | 449/500 | 458/500 | 1,309,532 | 1,320,138 |
| 7B | GSM8K | 1176/1319 | 1196/1319 | 1,058,823 | 1,056,250 |

The answer direction helps the recorded 7B accuracy in both datasets and the 1.5B GSM8K accuracy. It hurts 1.5B MATH-500 accuracy and increases total tokens in three of these four comparisons. It therefore cannot be described as universally beneficial or universally token-saving.

## Historical protocol differences

The MATH/GSM historical baselines were generated in earlier batches. For 7B, earlier MATH settings used context 17,408, max sequences 64, batched tokens 2,048, memory fraction 0.92; the final combined run used 17,920 / 32 / 4,096 / 0.95. Earlier GSM settings used 17,408 / 32 / 2,048 / 0.92; the final run used 17,920 / 64 / 4,096 / 0.95. One earlier 7B MATH baseline was completed after an interruption with its already completed answers retained. These facts limit causal and wall-clock comparisons; they do not change the full-set denominators.

Numerical paths under batching/BF16 can change sampled outputs even when a semantic intervention has not yet fired. Different reasoning lengths between answer-vector groups alone do not demonstrate that the vector compressed earlier reasoning. A new matched-protocol comparison is supported by the public CLI and should be reported separately from this historical table.

AMC23 and AIME25 comparisons used their recorded matched model protocols. AIME25 is the project's saved text variant: its reference answers match the public set, but not every problem statement is text-identical. The release publishes normalized identities and refuses to silently treat a different dataset revision as the same experiment.

## Provenance and coverage

The four MATH/GSM rows come from the September 29 stage-results handoff and the completed 1.5B/7B online evaluation reports. AMC23 and AIME25 rows come from their October 1 final reports. Release preparation reuses those recorded aggregate values and does not regrade or regenerate model outputs.

Only these completed, audited result groups are included. Other pilot, interrupted, partial or differently scored experiments are not merged into the main table. Large raw outputs and internal deployment logs remain outside the public repository.
