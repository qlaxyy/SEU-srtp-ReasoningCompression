# Calibration and asset provenance

The release includes ready-to-use reasoning vectors, answer vectors and lexical tables. It does not redistribute full training traces, hidden-state matrices or paid SRQ annotations. Benchmark inference needs no labeling API and makes no network API calls to a judge.

To inspect or rebuild the lexical table, see `configs/l27_terms.json` and run `python scripts/compile_lexicon.py --model-path "$MODEL_PATH" --output outputs/opening.npz --compare assets/1p5b/opening.npz`. This loads only the tokenizer, compiles on CPU, and checks array equality against the selected frozen table. Select the matching model's asset when comparing; NPZ container hashes alone can differ because of archive metadata.

## ReBalance assets

`assets/<model>/original_vector.pt` and `original_fit.json` are the project's frozen `auto-code-v2` adaptation: mixed step labels, raw vector norm, question-group layer selection and fitted controller parameters. Decoder output layers are zero-based 20 for 1.5B and 21 for 7B; the corresponding hidden-state tuple indices are 21 and 22.

The method field in each fit file records the adaptation choices, including the LDA midpoint sign, lower-bound construction and curve scale. These assets are intended for the exact model/tokenizer hashes shipped alongside them. Full raw training data regeneration is not automated in this release. ReBalance's upstream repository remains the reference for generating calibration trajectories and extracting its direction.

## Dynamic penalty scale

Input is an NPZ with:

- `strengths`: one-dimensional, finite normalized negative coefficients in `(0,1]`, computed as `clip(alpha / min(low_val_1, low_val_2), 0, 1)`;
- `question_ids`: aligned normalized question SHA-256 strings. Multiple steps can share a question ID.

Use training-only completed reasoning boundaries, excluding final `</think>` closure. Positive-coefficient boundaries are excluded. The original training subset had 300 questions, each with at least one negative boundary. The fitter gives every question equal total weight and returns `rho = m/(1-m)` at the weighted median `m`.

```bash
python scripts/fit_assets.py --kind penalty \
  --input data/training_negative_states.npz --output outputs/new_penalty_fit
```

The script reproduces the calibration formula; it cannot reconstruct missing training states from the test outputs. Published `rho` values are already available in `penalty_calibration.json`. New assets go to a separate output directory and do not replace the frozen defaults.

## Answer vector

The original labeling pool contained 340 training trajectories; 271 were selected (235 positive and 36 negative) using the project's existing SRQ selection. The release stores selected question identities, class counts and final vector hashes. The raw annotations and model-specific answer activations remain outside this public package. In particular, positive/negative selection is not simply equivalent to correct/incorrect final answer.

To refit from your own training annotations, collect answer-content mean hidden states at the intended model layer, using the same teacher-forcing convention for both classes. The historical vectors used unsteered teacher forcing over saved intervened text, not the original online KV state. For 7B, the texts and labels came from the 1.5B collection, but activations were extracted with 7B.

Input NPZ:

- `features`: `[n_trajectories, hidden_size]` mean answer-state matrix;
- `positive`: Boolean vector of the same length, after applying your declared SRQ selection; `False` identifies retained negative samples, not discarded/unknown ones;
- `question_ids`: aligned normalized training question SHA-256 strings.

```bash
python scripts/fit_assets.py --kind answer --rank 16 \
  --input data/selected_training_answer_states.npz --output outputs/new_answer_fit
```

`fit_answer_vector` implements the same centered Cartesian-difference covariance and norm-restored mean projection as the historical recipe. The release CPU test compares it with an explicitly materialized Cartesian-difference PCA on a small matrix. This verifies the algebra, not answer-generation quality.

The fitter rejects overlap with the published evaluation identities. It does not solicit paid labels or infer a new selection threshold. Reproducing the exact paid-label selection from scratch requires the original annotations/selection protocol; providing the final inference vector is not a claim that those raw annotations are included. If labels, texts, model, pooling positions or selection rule change, treat the vector as a new experiment and keep the frozen one for comparison.
