# Method and implementation

## What is inherited and what is added

ReBalance supplies the reasoning-state direction and an adaptive coefficient. The released parameters are the project's self-calibrated, automatically selected-layer adaptation, not the numerical parameters of the ReBalance paper. EasySteer supplies steering payloads and decoder hooks, while its vLLM fork supplies batched execution. Our additions coordinate a state-dependent lexical penalty and a separate answer vector with those mechanisms.

The reasoning vector is applied at the original step boundary. A tokenizer piece containing `ĊĊ` identifies a double-newline boundary. Its hidden state predicts the next token: visually, the action sits before the first token of the next step, rather than after an already observed whole step. The controller uses statistics of the completed prefix. It is not an oracle for mathematical correctness.

The pinned implementation uses confidence and variance statistics in its coefficient computation. Offline mixed labels also involve reflection-keyword signals. The implementation and self-calibration details differ from a literal transcription of the paper's displayed `sign · B · tanh` equation; the authoritative runnable controller is `runtime/overlays/vllm/vllm/steer_vectors/rebalance.py`.

## Calibrated dynamic lexical penalty

Let `α` be the existing coefficient and `b = min(low_val_1, low_val_2) < 0` its negative reference bound. Define

\[
s=\operatorname{clip}(\alpha/b,0,1),\quad
f_\rho(s)=\frac{s}{s+\rho(1-s)},\quad
k(s)=1+(K-1)f_\rho(s),\qquad K=32.
\]

For candidates matched by the L27 automaton at an eligible reasoning-step opening,

\[
z_j' = z_j-k(s)\ln 2.
\]

Other candidate logits remain unchanged. Eligibility requires a finite negative ReBalance coefficient, valid completed-prefill/decode state, reasoning phase, and an open lexical step prefix. A compiled finite-state matcher handles multi-token phrases. The penalty happens after the LM head and before temperature/top-p sampling. Raw model confidence is retained for the ReBalance update, so the penalty does not become its own confidence input.

The 27 base words/phrases come from ReBalance's labeling lexicon and are listed in `configs/l27_terms.json`. This project adapts the lexicon into a streamed, step-opening candidate penalty; it does not claim to have invented the word list. `reasoning_compression/lexicon.py` provides the matching/compilation code, including case, word boundaries and light morphological variants.

For a penalized candidate `i` and an unpenalized candidate `j`, before top-p truncation their softmax odds ratio changes by `2^(-k/T)`. It does not mean the final sampled probability is divided by two. Numerical endpoint handling deliberately retains the historical BF16 arithmetic.

To obtain `ρ`, training question `q` contributes total weight one across its negative boundary states. If `m` is the resulting weighted median of `s`, set `ρ=m/(1-m)`, giving `fρ(m)=1/2`. This establishes a training-derived scale convention; it does not fit a probability of correctness or optimize test-set accuracy.

| Model | Decoder layer (zero based) | Hidden size | ρ |
|---|---:|---:|---:|
| 1.5B | 20 | 1536 | 0.4787751735381186 |
| 7B | 21 | 3584 | 0.35783045814796643 |

The released curve uses a training-only fit on 300 question identities. Its source states were a CPU projection of offline calibration traces; the online lexical-opportunity distribution need not equal that distribution.

## SRQ/PCA16 answer direction

SRQ labels were produced for 340 training trajectories, of which 294 had correct final answers and 46 did not. The existing selection retained 235 positive and 36 negative trajectories. Hidden states are mean-pooled over final-answer content. The 7B vector reuses those 1.5B-generated texts and labels but extracts its own 7B hidden states; vectors cannot be exchanged across models.

Let positive/negative answer-state matrices be `P,N`, and `d = mean(P) - mean(N)`. The covariance of all centered Cartesian positive-minus-negative differences equals the sum of the two within-class population covariances. Compute its top 16 orthonormal eigenvectors `U16`, and use

\[
d_A=\|d\|\frac{U_{16}U_{16}^{\mathsf T}d}{\|U_{16}U_{16}^{\mathsf T}d\|},\qquad
h'=h+0.25d_A.
\]

The CPU implementation computes an SVD of stacked, class-centered states, avoiding materializing all 8,460 pair differences. This is the project's low-dimensional mean-contrast interpretation, with explicit pairing and norm conventions. It is different from selecting the first principal component alone. PCA is used for extraction; runtime injection remains additive.

The generated `</think>` input turns on answer injection to predict the first answer token. It continues for answer tokens and stops on a new `<think>` or EOS. Prompt tokens do not trigger the phase. The reasoning KV history is retained; the final benchmark is a fresh online generation, not a continuation from saved unsteered reasoning. The direction can influence conversion to the final answer but cannot rewrite already generated reasoning.

## Serving details

- Request state follows request IDs through batch reordering, not the row index from a previous batch.
- 1.5B retains the historical asynchronous, non-chunked-prefill configuration. Unsupported preemption stops the run.
- 7B uses synchronous chunked prefill with restoration of phase, lexical and controller state after KV preemption. Historical answer-token injections are replayed.
- The 7B additive operator stores the scaled direction in BF16 before addition and uses an opaque Triton/custom-op boundary. Independent eager arithmetic is an engineering reference.
- Before full combined-method evaluation, zero-strength restoration and nonzero downstream-logit effects are checked on synthetic forced-token fixtures. Those fixtures are never counted as benchmark outputs. 7B additionally checks forced preemption in both phases.

Each method runs in its own process. Core runtime files retain some earlier opt-in research branches for source identity, but the public CLI exposes only the five documented methods. Hybrid early termination and conditional mixture routing are not enabled.
