# Activation Steering for Chain-of-Thought Compression

Seyedarmin Azizi∗ University of Southern California seyedarm@usc.edu Erfan Baghaei Potraghloo University of Southern California baghaeip@usc.edu Massoud Pedram University of Southern California pedram@usc.edu

## Abstract

Large language models (LLMs) excel at complex reasoning when they include intermediate steps, known as chains of thought (CoTs). However, these rationales are often overly verbose, even for simple problems, leading to wasted context, increased latency, and higher energy consumption. We observe that verbose, English-heavy CoTs and concise, math-centric CoTs occupy distinct regions in the model’s residual-stream activation space. By extracting and injecting a steering vec- tor to transition between these modes, we can reliably shift generation toward more concise reasoning, effectively compressing CoTs without retraining. We formalize this approach as Activation-Steered Compression (ASC), an inference-time tech- nique that shortens reasoning traces by directly modifying hidden representations. In addition, we provide a theoretical analysis of the impact of ASC on the output distribution, derived from a closed-form KL-divergence-bounded constraint to regulate steering strength. Using only 50 paired verbose and concise examples, ASC achieves up to 67.43% reduction in CoT length on MATH500 and GSM8K datasets, while maintaining accuracy across 7B, 8B, and 32B parameter models. As a training-free method, ASC introduces negligible runtime overhead and, on MATH500, delivers an average 2.73× speedup in end-to-end reasoning wall-clock time on an 8B model. This makes ASC a practical and efficient tool for stream- lining the deployment of reasoning-capable LLMs in latency- or cost-sensitive settings. The code is available at https://github.com/ArminAzizi98/ASC.

## 1 Introduction

Explicit reasoning traces, commonly known as chains of thought (CoTs), significantly enhance the per- formance of LLMs on multi-step tasks such as mathematical problem solving, logical inference, and program synthesis [29, 8, 26]. However, this advantage often comes with the drawback of generating unnecessarily lengthy and verbose rationales [5, 31]. This verbosity not only increases computational costs by producing more tokens and consuming additional energy, but also risks impairing perfor- mance through overthinking - where the model introduces redundant steps, multiple self-verifications, and variations [5]. This can lead to performance degradation [27]. Furthermore, lengthy CoTs pose challenges for deployment in latency-sensitive or resource-constrained environments [7]. In this paper, we ask: Can we compress chains of thought without retraining, by manipulating the model’s hidden representations at inference time? We answer the question affirmatively with Activation-Steered Compression (ASC). Our key observation is that internal representations of ∗ Corresponding author Preprint. arXiv:2507.04742v2 [cs.AI]

Jul 2025 verbose, natural-language CoTs and their concise, math-centric counterparts occupy distinct regions in the model activation space. To evaluate this hypothesis, we sample questions from the MATH500 [13] and GSM8K [8] benchmarks and use two open-weight reasoning models: DeepSeek-Distill-Qwen- 7B and DeepSeek-Distill-LLaMA-8B. For each sample, we generate two variants of the CoT: (1) a verbose reasoning chain produced by the model itself under standard prompting and (2) a concise reasoning produced by GPT-4o prompted to minimize natural language verbosity and maximize math-centric reasoning. We feed each input independently into the model and extract residual stream activations, that is, the outputs of the transformer block, in a predetermined layer (e.g., layer 21 in our experiments). A two-dimensional t-SNE projection [24] of these activations, shown in Figure 2, reveals a clear separation between the two reasoning styles. This separation motivates the construction of a steering vector, which is a direction in the activation space that shifts the model’s reasoning towards the concise response generation mode. By extracting this vector from a small calibration set and injecting it during generation, we guide the model to focus on essential steps, reducing verbosity while preserving accuracy. An example of such a pair of responses is shown in Figure 1. Why Activation Steering for CoT Compression? Existing methods for compressing CoT reason- ing can be broadly categorized into three approaches: (i) retraining-based methods that fine-tune models to produce shorter rationales, using techniques such as knowledge distillation [34] or em- bedding reasoning within compact latent tokens [16]; (ii) prompt-engineering strategies that employ carefully designed instructions to encourage models to “reason briefly,” utilize contrastive demonstrations, or favor symbolic sketches over verbose prose [2, 33]; and (iii) heuristic early-exit mechanisms that halt generation once a confidence or entropy threshold is reached, trading com- pleteness for speed [32]. Activation steering offers an intriguing and effective middle ground. It is lightweight, requires only the addition of a single vector during inference, and directly reshapes hidden representations to enable on-the-fly compression. Because it does not involve updating the model parameters, this method is deployment-agnostic, making it equally applicable to both open-source and closed-source checkpoints. Moreover, it is orthogonal and compatible with the three categories of CoT compression described above. Finally, steering aligns well with interpretability: by nudging hidden states toward the sub-manifold of focused and concise reasoning, it exposes a controllable axis linking latent representations to computational efficiency.

**Figure 1: A representative pair of verbose vs. concise CoTs used for generating the steering vector.**

**Figure 2: t-SNE visualization of residual stream representations for long (verbose) and short (concise) CoT responses across two datasets and two models.**

Steering involves not only the direction of modifications in the activation space but also selecting the appropriate scale of the steering vector. If the scale is too small, the intervention has little effect; If too large, the model’s output distribution can change unpredictably or even collapse. Previous approaches typically choose this scale heuristically, often by grid search or visual inspection. In contrast, we introduce a principled method for calibrating the steering strength by explicitly bounding the KL divergence between the original and steered output distributions. Our analysis provides a closed-form formula that accounts for both the local sensitivity and curvature of the model’s output relative to the intervention, allowing reliable and provably controlled distribution shifts. Our key contributions are: 1. We conceptualize CoT verbosity as a latent, steerable dimension of model behavior, re- framing rationale compression as a problem of representation-level control rather than output-level post-processing. 2. We propose Activation-Steered Compression (ASC), a training-free framework that uses linear activation injections to reliably shorten CoTs. A single steering vector consistently shifts generations from verbose natural language to concise reasoning chains. ASC is orthogonal to existing compression methods (e.g., early-exit or token pruning) and can be composed with them to further enhance efficiency. 3. We provide the first theoretical framework for safe activation steering by deriving a closed- form scaling rule that bounds the KL divergence at the model output. Our formulation accounts for both the local Jacobian and second-order curvature, enabling principled control over the distributional shift. 4. We conducted extensive experiments in various reasoning tasks and model sizes (7B, 8B, and 32B), showing that ASC reduces CoT length by up to 67.43% on MATH500 and GSM8K without accuracy degradation. On MATH500, ASC delivers an average 2.73× inference speedup on an 8B reasoning model, representing, to our knowledge, the largest efficiency gains achieved by a purely inference time intervention.

## 2 Background

We place ASC at the intersection of research on CoT prompting, representation engineering, and the computational economics of long-context decoding. Chain-of-thought (CoT) prompting improves multi-step reasoning by encouraging language models to articulate intermediate steps, often using signals such as “Let’s think step by step” [28]. Several enhancements have refined this approach: self-consistency[25] samples multiple rationales and selects the response supported by the majority; tree-of-thought [33] performs look-ahead search across branching reasoning paths; and program-of-thought [6] converts natural language reasoning into executable code. Although effective, these methods often increase the output length significantly. A recent study [5] showed that o1-style reasoning models frequently produce excessively long CoTs — even for simple questions like “What is 2 + 3?”— due to redundant computations, unnecessary self-verification, and lingering explanations. We term these inefficiencies verbosity, which we aim to address directly through inference-time activation-level intervention. Activation Steering and Representation Engineering Linear activation editing has emerged as a lightweight alternative to fine-tuning. Activation Addition (ActAdd) demonstrates that adding a direc- tion corresponding to “<|positive sentiment|>” can change the tone of the output [23]. Reference[1] formalizes the approach as representation engineering, defining vectors as basis elements in a con- trollable subspace. The applications now span style transfer [12], factual correction [18], and gender debiasing [17]. However, to our knowledge, no previous work targets efficiency metrics such as token count. Efficiency Challenges in Processing Long Sequences Standard decoder-only transformers scale the inference computation with sequence length quadratically. Empirical profiling on an A100 NVIDIA GPU shows that halving a sequence from 8k to 4k tokens reduces latency by ~40% and energy by ~35%. Compression, therefore, offers a direct lever for green AI and a cost-effective deployment [21].

## 3 Related Work

Previous work tackles the CoT efficiency gap primarily through methods that require additional train- ing: knowledge distillation schemes that learn concise rationales [34], latent token approaches that embed reasoning in compact vectors [16], token-level pruning with supervised objectives such as TO- KENSKIP [30], reinforcement-learning-based trajectory shortening exemplified by THINKPRUNE [15], and latent-reasoning optimization frameworks that fine-tune internal deliberation steps [3]. While effective, these techniques incur considerable computational cost or architectural modifications. In contrast, we propose a training-free, inference-time method that compresses CoTs by directly manipulating hidden representations, retaining the accuracy benefits of reasoning traces without the overhead of retraining. Chain of drafts (CoD) [31] and the approach of [20] reduce verbosity by embedding explicit length constraints in the prompt. CoD instructs the model to “think step by step” but keep the each draft to at most five words, whereas [20] limits the final answer to a user-specified number of sentences to create inference-time interventions. Although such heuristics can shorten outputs, they assume that the model will faithfully obey length directives, a behavior that recent studies show is unreliable for reasoning-oriented LLMs [11]. The closest work to ours is SEAL [4], which constructs its steering vector by manually labeling the thought segments as execution, reflection, or transition, and then damping the latter two segment types. In contrast, (i) we learn a single verbosity axis from paired VERBOSE–vs.–CONCISE CoTs without any manual labels, (ii) rely solely on off-the-shelf prompts to generate training pairs, and (iii) obtain a domain-agnostic vector that generalizes across reasoning tasks. Therefore, our method provides a taxonomy-free, training-free complement to SEAL’s category-based calibration.

## 4 Activation-Steered Compression

Motivated by the goal of improving CoT efficiency through manipulation of the model’s activation space, we introduce Activation-Steered Compression (ASC)—a method that shifts the model’s hidden representations toward the subspace associated with concise, math-centric chains of thought. The method is summarized in Figure 3. First, we randomly sample 50 calibration samples from target dataset (in our case we have focused on MATH500 [13] and from GSM8K [8]). For each question $q_i$ in the calibration set, we obtain:

**Figure 3: Steering vector extraction and application using pairs of concise and verbose CoTs.**

- Verbose CoT $l_i$ – generated by the target model with standard CoT prompting [29].
- Concise CoT $s_i$ – produced by GPT-4o instructed to use concise math-centric reasoning with minimal English.

We denote the output of the transformer block in layer $\ell$ as the residual stream of layer $\ell$, and use $h_\ell$ to refer to it. Formally, $h_\ell$ is a matrix of shape $T \times d$, where $T$ is the number of tokens in the input sequence and $d$ is the hidden dimension of the model. With a slight abuse of notation, we write $h_\ell(s)$ to denote the residual stream in layer $\ell$ when a string $s$ is fed into the model.

Following this notation, we feed the concatenated input `[question + CoT]` into the target model and extract the residual-stream activations corresponding to the final token in the input sequence. Specifically, we obtain $h_\ell(q_i \oplus l_i)[-1]$ and $h_\ell(q_i \oplus s_i)[-1]$ at a selected layer $\ell$, corresponding to the verbose and concise CoTs, respectively.

The steering vector is then computed as the average difference between these final-token activations across $N$ examples:
$$
v_\ell = \frac{1}{N}\sum_{i=1}^N \left(h_\ell(q_i \oplus s_i)[-1] - h_\ell(q_i \oplus l_i)[-1]\right).
$$

$v_\ell$ is the desired steering direction for shifting the long and verbose CoT toward a more compact CoT. At inference time, given a new question and the $i$-th generated token $x_i$, we modify the residual stream by injecting the steering vector $v_\ell$ into layer $\ell$ during each decoding step, until an end-of-sequence delimiter is emitted. Specifically, for each decoding step $i$, we apply:
$$
h_\ell(x_i) \leftarrow h_\ell(x_i) + \gamma v_\ell,\qquad \forall i \in [1,\mathrm{decoding\_steps}].
$$
Here, $\gamma$ is a hyperparameter that controls the injection strength of the steering vector. If $\gamma$ is too large, it can significantly distort the residual stream distribution, leading to degenerate or incoherent output. In contrast, if $\gamma$ is too small, the steering effect becomes negligible.

In the remainder of this section, we analyze the effect of the scaling parameter $\gamma$ on the model output distribution from a theoretical perspective. We derive a safe upper bound on $\gamma$ that guarantees that the output divergence remains within a user-specified threshold. For simplicity, we now drop the layer index $\ell$ throughout the analysis. We steer hidden activations by adding a direction $v$ at layer $\ell$, and choose the scale $\gamma$ so that the resulting output distribution remains close to the unsteered model. Formally, letting $z$ and $\tilde z$ denote the pre-softmax logits before and after steering, we constrain the forward KL divergence:
$$
\mathrm{KL}(\mathrm{softmax}(z)\,\|\,\mathrm{softmax}(\tilde z)) \le \epsilon,
$$
where $\epsilon$ is a user-specified divergence budget (we use $\epsilon = 10^{-3}$ in practice).

$\oplus$ is the string concatenation operator.

The full derivation, deferred to Appendix A.1, decomposes the logit shift into a linear component $\gamma Wv$ and a curvature-dependent remainder, where $W$ is the Jacobian $J(\cdot)$ of the logit map with respect to the activations of layer $\ell$. Under a mild smoothness condition with constant $L$ as the upper bound of directional curvature, we derive a provable upper bound of KL that is quadratic, cubic, and quartic in $\gamma$. Specifically, defining
$$
a := \|Wv\|_2,\qquad L := \sup_{t\in[0,\gamma]} \frac{\|J(h + tv) - J(h)\|_2}{t},
$$
we obtain a closed-form scale $\gamma_{\max}$ that ensures $\mathrm{KL} \le \epsilon$. The expression includes a curvature-aware safety factor:
$$
\gamma_{\max} = \max\left\{0,\ \left(1 - \frac{L\gamma_{\mathrm{raw}}}{4a}\right)\gamma_{\mathrm{raw}}\right\},
$$
where $\gamma_{\mathrm{raw}} = (a/L)\cdot x$ and $x$ is determined by solving the dimensionless cubic equation
$$
x^3 + x^2 - \frac{4\epsilon L^2}{a^4} = 0.
$$
All constants are explicit; no additional hyperparameters are introduced. In practice, we estimate the two scale parameters $a$ and $L$ on the small calibration set (50 hidden states). For each hidden state, we evaluate one Jacobian–vector product with the chosen steering direction and record its Euclidean norm; the median of these norms is taken as our estimate of $a$. To obtain $L$, we compute a single Hessian–vector product along the same direction at each calibration point, collect the resulting norms, and take their 95th percentile. Both JVP and HVP operations are one-line calls in modern autodiff frameworks, so the entire procedure runs in a few seconds even on large-scale models. All proofs, derivations, and bounds appear in Appendix A.1. We adopt this calibrated $\gamma_{\max}$ in all experiments to control distributional shift while preserving the intended compression effect of each steering vector.

## 5 Experiments

This section presents our experimental results demonstrating that ASC effectively reduces the length of CoT reasoning while maintaining or improving task performance. We begin by describing our experimental setup in section 5.1, followed by the main results in section 5.2.

### 5.1 Experimental Setup

Models, Datasets, and Baselines. We evaluate ASC on several recent open-source reasoning models: DeepSeek-R1-Distill-LLaMA-8B [10], DeepSeek-R1-Distill-Qwen-7B [9], and QwQ-32B [22]. The evaluation is performed on multiple reasoning benchmarks, including MATH-500 [14] and GSM8K [8]. As baselines, we compare ASC against vanilla CoT prompting (no steering), CoD [31], DEER [32], TCC [19], and SEAL [4], a recent method for compressed reasoning that uses steering vectors. Implementation Details. For all experiments, we use the decoding hyperparameters $\mathrm{temperature}=0.7$, $\mathrm{top\_p}=0.9$, and $\mathrm{repetition\_penalty}=1.1$; all other settings follow the default configurations of the respective models. The evaluation datasets are accessed through the Hugging Face datasets library. Experiments are conducted on NVIDIA A6000 GPUs, using PyTorch version 2.5.1+cu124 and the transformers library version 4.50.1. The hyperparameters related to steering, namely the steering strength $\gamma$ and the layer index used to extract and apply the steering vector, are detailed in Appendix C.

### 5.2 Main Results

Table 1 presents the performance of ASC compared to baseline CoT compression techniques. On the DeepSeek-R1-Distill-LLaMA-8B model, ASC reduces CoT length by up to 61.2% without any loss in accuracy, outperforming prior methods in compression effectiveness. On the same model and the GSM8K dataset, ASC achieves a compression rate of 67.43%, while also slightly improving answer accuracy by 0.2%, matching or exceeding the performance of the vanilla CoT baseline. On MATH500, ASC achieves a 33.8% reduction in CoT length, again outperforming all baselines while maintaining equivalent accuracy. On the larger QwQ-32B model, ASC compresses CoTs by 50.7% and 45.7% on MATH500 and GSM8K, respectively. Notably, on MATH500, it also yields a 0.4% accuracy improvement over the vanilla CoT. Upon inspection, we find that the high token count in some model responses arises

primarily from either examples exceeding their token budget or exhibiting excessive branching and thought switching during generation. This aligns with the observations of [27], who show that LLMs similar to o1 tend to generate longer responses when frequently switching between reasoning paths without deeply pursuing any one. This behavior, termed under-thinking, often manifests itself as verbose outputs filled with abandoned or partially developed reasoning trajectories. Among the models evaluated, QwQ-32B appears particularly susceptible to this issue. On the challenging MATH500 benchmark, ASC mitigates this behavior by promoting concise, linear reasoning and earlier halting, thereby suppressing extraneous chains of thought. Qualitative examples that illustrate this suppression are provided in Appendix B, where ASC responses exhibit significantly fewer thought changes than their vanilla CoT counterparts. In summary, across all models and datasets, ASC consistently achieves the highest CoT compression while preserving the final answer accuracy.

**Figure 4: Speed comparison of CoT, CoD, and ASC on MATH500 dataset.**

Since one of the primary goals of CoT compression is to reduce end-to-end response latency, we measure the average generation time for three models—DeepSeek-R1-Distill-LLaMA-8B, DeepSeek-R1-Distill-Qwen-7B, and QwQ-32B—on the MATH500 dataset. Latency is measured on an NVIDIA A6000 GPU. We then compute and report the inverse latency (i.e., generation speed) for three decoding strategies: standard CoT, Chain-of-Drafts (CoD), and our proposed ASC, as shown in Figure 4. The results indicate that ASC improves the generation speed of CoT-based reasoning by up to 2.73×, with no loss in answer accuracy.

**Table 1: Performance comparison of CoT, TCC, DEER, CoD, SEAL and ASC on reasoning tasks.**

| Model | Method | MATH500 Acc. (%) ↑ | MATH500 Tokens ↓ | GSM8K Acc. (%) ↑ | GSM8K Tokens ↓ |
|---|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Qwen-7B | CoT | 88.8 | 3984 | 88.6 | 1080 |
| DeepSeek-R1-Distill-Qwen-7B | TCC | 89.2 | 3864 | 88.0 | 892 |
| DeepSeek-R1-Distill-Qwen-7B | DEER | 89.8 | 2143 | 90.6 | 917 |
| DeepSeek-R1-Distill-Qwen-7B | SEAL | 89.4 | 2661 | 88.4 | 811 |
| DeepSeek-R1-Distill-Qwen-7B | CoD | 88.2 | 1852 | 87.9 | 550 |
| DeepSeek-R1-Distill-Qwen-7B | ASC | 89.0 | 1543 | 88.6 | 536 |
| DeepSeek-R1-Distill-LLaMA-8B | CoT | 89.2 | 3554 | 89.1 | 2610 |
| DeepSeek-R1-Distill-LLaMA-8B | DEER | 89.2 | 2830 | 89.3 | 2124 |
| DeepSeek-R1-Distill-LLaMA-8B | CoD | 88.8 | 3028 | 89.1 | 914 |
| DeepSeek-R1-Distill-LLaMA-8B | ASC | 89.2 | 2353 | 89.3 | 850 |
| QwQ-32B | CoT | 93.8 | 4508 | 96.5 | 1530 |
| QwQ-32B | TCC | 94.4 | 4315 | 95.8 | 1348 |
| QwQ-32B | DEER | 94.6 | 3316 | 96.3 | 977 |
| QwQ-32B | CoD | 93.8 | 3400 | 96.2 | 1116 |
| QwQ-32B | ASC | 94.2 | 2222 | 96.4 | 830 |

## 6 Discussion and Ablations

Cross-Task Generalization. To investigate whether CoT verbosity is consistently reflected in the model’s representation space, we examine the alignment of ASC steering vectors extracted from different reasoning tasks. Specifically, we analyze whether steering vectors derived from one dataset generalize to another. We conduct this study using the DeepSeek-R1-Distill-Qwen-7B model and two benchmarks: GSM8K and MATH500. Following the ASC methodology, we independently compute steering vectors for each dataset using 50 paired examples. We then assess the cosine similarity between the two vectors to quantify their alignment. In addition, we evaluate cross-task generalization by applying each dataset’s steering vector to compress CoTs in the other dataset, measuring both length reduction and accuracy retention.

The results are presented in Table 2. First, the cosine similarity between the two steering vectors is 0.92, indicating strong alignment in the vectors from verbose to concise CoTs in MATH500 and GSM8K. Second, the performance of cross-dataset steering matches closely that of in-dataset vectors. Although there is a slight drop in accuracy and a slight increase in token count, ASC with cross-dataset steering still outperforms the vanilla CoT baseline (Table 1). These findings suggest that verbosity reduction occupies a largely shared latent direction across reasoning tasks, supporting our initial hypothesis that CoT efficiency can generally be attributed to the latent representations of the model.

**Table 2: Performance of ASC on MATH500 and GSM8K using dataset-specific vs. cross-dataset steering vectors.**

| Dataset | Steering Vector Source | Accuracy (%) | CoT Tokens |
|---|---|---:|---:|
| MATH500 | MATH500 (in-dataset) | 89.0 | 1543 |
| MATH500 | GSM8K (cross-dataset) | 88.8 | 1631 |
| GSM8K | GSM8K (in-dataset) | 88.6 | 536 |
| GSM8K | MATH500 (cross-dataset) | 88.4 | 611 |

**Figure 5: Effect of steering strength $\gamma$ on CoT compression and answer accuracy for the DeepSeek-R1-Distill-Qwen-7B model on the MATH500 dataset.**

Effect of Steering Strength $\gamma$. The steering strength $\gamma$ is a critical hyperparameter in ASC, as it directly influences both the degree of CoT compression and the quality of the generated output. To analyze its effect, we use the DeepSeek-R1-Distill-Qwen-7B model on the MATH500 dataset and perform a sweep over a range of $\gamma$ values. The sweep begins at $\gamma=0$ (i.e., no steering) and gradually increases until the steering induces noticeable compression along with a significant drop in answer accuracy at $\gamma=0.5$. The results are shown in Figure 5, highlighting the trade-off between CoT compression and answer accuracy as the steering strength $\gamma$ increases. For small values of $\gamma$, increasing the strength yields substantial reductions in CoT length with minimal impact on accuracy. However, beyond a certain point, further increases in $\gamma$ lead to significant accuracy degradation despite continued compression. Notably, the value of $\gamma$ selected by ASC—computed via the KL-divergence–constrained scaling described in Section 4—closely aligns with the empirical breakpoint where performance begins to degrade. This supports the theoretical grounding of our method for setting steering strength.

## 7 Conclusion

We introduce Activation-Steered Compression (ASC), a training-free method for reducing the ver- bosity of Chain-of-Thought (CoT) reasoning in large language models by manipulating internal representations at inference time. By leveraging steering vectors derived from paired verbose and concise rationales, ASC effectively compresses CoTs without sacrificing accuracy. We further con- tribute a closed-form, KL-constrained scaling framework for principled control of steering strength, and provide empirical evidence that verbosity lies along a shared latent direction across tasks. ASC complements existing CoT compression techniques and requires no retraining, and overall advances the efficiency and practicality of LLM-based reasoning by showing that conciseness is not only desirable but also steerable via the internal geometry of the model.

## References

- [1] Sam Burns et al. An introduction to representation engineering: Activation steering. Alignment
Forum, 2024.

- [2] Andy Chen et al. Contrastive chain-of-thought prompting. arXiv preprint arXiv:2310.02306,
2023.

- [3] Haolin Chen, Yihao Feng, Zuxin Liu, Weiran Yao, Akshara Prabhakar, Shelby Heinecke,
Ricky Ho, Phil L. Mui, Silvio Savarese, Caiming Xiong, and Huan Wang. Language models are hidden reasoners: Unlocking latent reasoning capabilities via self-rewarding. In Interna- tional Conference on Learning Representations (ICLR), under review, 2025. OpenReview ID 4Po8d9GAfQ.

- [4] Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, and Zhangyang Wang. Seal: Steer-
able reasoning calibration of large language models for free. arXiv preprint arXiv:2504.07986, 2025.

- [5] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song,
Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do not think that much for 2+3=? on the overthinking of o1-like llms, 2025. URL https://arxiv.org/abs/2412.21187.

- [6] Zhuosheng Chen, Aston Zhang, Mu Li, and Alex Smola. Program-of-thought prompting:
Efficient reasoning with small language models. arXiv preprint arXiv:2305.10601, 2023.

- [7] Jeffrey Cheng and Benjamin Van Durme. Compressed chain of thought: Efficient reasoning
through dense representations. arXiv preprint arXiv:2412.13171, 2024.

- [8] Karl Cobbe et al. Training verifiers to solve math word problems. arXiv preprint
arXiv:2110.14168, 2021.

- [9] DeepSeek-AI. Deepseek-r1-distill-qwen-7b. https://huggingface.co/deepseek-ai/
DeepSeek-R1-Distill-Qwen-7B, 2025.

- [10] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
learning, 2025. URL https://arxiv.org/abs/2501.12948.

- [11] Tingchen Fu, Jiawei Gu, Yafu Li, Xiaoye Qu, and Yu Cheng. Scaling reasoning, losing control:
Evaluating instruction following in large reasoning models. arXiv preprint arXiv:2505.14810, 2025.

- [12] Aviv Haviv, Sagie Benaim, Asaf Noy, and Lior Wolf. Style steering via activation injection in
large language models. arXiv preprint arXiv:2403.00555, 2024.

- [13] Dan Hendrycks, Steven Basart, Nicholas Carlini, Jacob Steinhardt, and Dawn Song. Measuring
mathematical problem solving with the math dataset. In International Conference on Machine Learning (ICML), 2021.

- [14] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn
Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS Datasets and Benchmarks Track, 2021.

- [15] Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang.
Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv:2504.01296, 2025.

- [16] Yuchen Li et al. Uncovering latent chain of thought vectors in language models. arXiv preprint
arXiv:2409.14026, 2024.

- [17] Xuezhe Liang, Haoming Jiang, and Graham Neubig. Manipulating large language models with
representation editing for fairness. arXiv preprint arXiv:2311.01543, 2023.

- [18] Kevin Meng, Eric Mitchell, David Bau, and Percy Liang. Locating and editing factual associa-
tions in gpt. Advances in Neural Information Processing Systems (NeurIPS), 2023.

- [19] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi,
Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

- [20] Alessandro Stolfo, Vidhisha Balachandran, Safoora Yousefi, Eric Horvitz, and Besmira Nushi.
Improving instruction-following in language models through activation steering. arXiv preprint arXiv:2410.12877, 2024.

- [21] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for
deep learning in nlp. In Proc. of ACL, 2019.

- [22] Alibaba Qwen Team. Qwq-32b: A 32 b reasoning model from the qwen series. https:
//huggingface.co/Qwen/QwQ-32B, 2025. Apache 2.0 licensed, open-weight; competitive reasoning performance.

- [23] Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan José Vázquez, Ulisse
Mini, and Monte MacDiarmid. Steering language models with activation engineering. OpenRe- view, 2023. URL: https://openreview.net/forum?id=2XBPdPIcFK.

- [24] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine
learning research, 9(11), 2008.

- [25] Xuezhi Wang, Jason Wei, Jingshu Liu, Dale Schuurmans, Denny Zhou, and Quoc Le.
Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2023.

- [26] Xuezhi Wang et al. Self-consistency improves chain of thought reasoning in language models.
ICLR, 2023.

- [27] Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian
Yu, Juntao Li, Zhuosheng Zhang, et al. Thoughts are all over the place: On the underthinking of o1-like llms. arXiv preprint arXiv:2501.18585, 2025.

- [28] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Quoc V. Le, and
Ed Chi. Chain-of-thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022.

- [29] Jason Wei et al. Chain-of-thought prompting elicits reasoning in large language models.
NeurIPS, 2022.

- [30] Heming Xia, Yongqi Li, Chak Tou Leong, Wenjie Wang, and Wenjie Li. Tokenskip: Controllable
chain-of-thought compression in llms. arXiv preprint arXiv:2502.12067, 2025.

- [31] Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. Chain of draft: Thinking faster by
writing less, 2025. URL https://arxiv.org/abs/2502.18600.

- [32] Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Zheng
Lin, Li Cao, and Weiping Wang. Dynamic early exit in reasoning models. arXiv preprint arXiv:2504.15895, 2025.

- [33] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate reasoning via chain of thought. arXiv preprint arXiv:2305.10601, 2023.

- [34] Mingyuan Zhang et al. Compressed chain of thought: Efficient reasoning through dense
contemplation tokens. EMNLP, 2024.

## Appendix A KL-Constrained Scaling of Steering Vectors

### A.1 Bounding the Distributional Shift of Additive Steering

We study the output–distribution shift incurred when an additive steering update is applied to the hidden state at layer $\ell$ of a language model. For an activation vector $h \in \mathbb{R}^d$ we form
$$
\tilde h := h + \gamma v,\qquad \|v\|_2 = 1,
$$
to analyze how large the Kullback–Leibler (KL) divergence between the pre- and post-steering output distributions can become. Throughout, let $F_{\ell\to\text{logit}} : \mathbb{R}^d \to \mathbb{R}^m$ denote the sub-network that maps layer-$\ell$ activations to the pre-softmax logits. All vector norms $\|\cdot\|_2$ and operator-2 norms are Euclidean; they coincide when the argument is a vector.

Notation for higher-order derivatives. The Jacobian of $F_{\ell\to\text{logit}}$ at $h$ is the matrix
$$
J(h) := \nabla_h F_{\ell\to\text{logit}}(h) \in \mathbb{R}^{m\times d},
$$
whose $j$-th row is $(\nabla_h F_j(h))^\top$. The Hessian of a scalar component is the usual matrix of second partials. For a unit vector $a$ we abbreviate directional Hessian evaluation
$$
\nabla_h^2 F_{\ell\to\text{logit}}(h)[a,a]
:= (\nabla_h^2 F_1(h)[a,a], \dots, \nabla_h^2 F_m(h)[a,a])^\top \in \mathbb{R}^m.
$$

### A.1.1 A smoothness assumption

Assumption 1. There exists a constant $L>0$ such that for every unit direction $v$ and every $t \in [0,\gamma]$,
$$
\|J(h + tv) - J(h)\|_2 \le Lt.
$$

Implication. Assumption 1 is stronger than merely requiring bounded second derivatives. In fact, according to the mean value theorem for vector-valued Lipschitz maps, $J$ is differentiable almost everywhere and its derivative (the third-order tensor of second partials) has the operator norm at most $L$. Contracting this tensor twice with the same unit vector $v$ yields
$$
\|\nabla_h^2 F_{\ell\to\text{logit}}(h + \tau v)[v,v]\|_2 \le L,\qquad \forall \tau \in [0,\gamma]. \tag{1}
$$
because $\|H[v,v]\|_2 \le \|H\|_{\mathrm{op}}\|v\|_2^2 = \|H\|_{\mathrm{op}}$. Thus Assumption 1 implies–though it is not equivalent to–a uniform bound on the directional Hessian.

### A.1.2 Local linearization with a controlled remainder

Define $z := F_{\ell\to\text{logit}}(h)$ and $W := J(h) \in \mathbb{R}^{m\times d}$. By the fundamental theorem of calculus and Eq. (1), the steered logits decompose as
$$
\tilde z = F_{\ell\to\text{logit}}(h + \gamma v). \tag{2}
$$
$$
\tilde z = z + \delta + r(\gamma),\qquad
\delta := \gamma Wv,\qquad
r(\gamma) := \int_0^\gamma (\gamma-s)\,\nabla_h^2 F_{\ell\to\text{logit}}(h + s v)[v,v]\,ds. \tag{3}
$$
The linear component is $\delta=\gamma Wv$, while the remainder obeys
$$
\|r(\gamma)\|_2 \le \frac{1}{2}L\gamma^2. \tag{4}
$$

### A.1.3 KL divergence as a Bregman divergence

Let $g(x) = \log\sum_{i=1}^m e^{x_i}$ and denote $p=\mathrm{softmax}(z)$ and $\tilde p=\mathrm{softmax}(\tilde z)$. For the log-partition function $g$, the Bregman divergence is
$$
D_g(\tilde z, z) = g(\tilde z) - g(z) - \langle \nabla g(z), \tilde z - z \rangle = \mathrm{KL}(p\|\tilde p). \tag{5}
$$
Thus, the classical forward KL direction appears.

### A.1.4 Integral representation and spectral bound

Using the integral representation of a Bregman divergence for twice-differentiable convex $g$ we obtain
$$
\mathrm{KL}(p\|\tilde p)
= \int_0^1 (1-t)\,(\tilde z - z)^\top \nabla^2 g\!\left(z + t(\tilde z - z)\right)\,(\tilde z - z)\,dt. \tag{6}
$$
Because $\nabla^2 g(x)$ equals the Fisher information matrix $F(x)=\mathrm{diag}(p)-pp^\top$, whose largest eigenvalue never exceeds $1/2$, and the factor $(1-t)$ integrates to $1/2$, we have
$$
\mathrm{KL}(p\|\tilde p) \le \frac{1}{4}\|\tilde z - z\|_2^2. \tag{7}
$$
This constant $1/4$ is tight for our purposes.

### A.1.5 Putting the pieces together

With $\tilde z - z = \delta + r(\gamma)$ and the triangle inequality,
$$
\|\tilde z - z\|_2^2 \le \big(\|\delta\|_2 + \|r(\gamma)\|_2\big)^2. \tag{8}
$$
$$
\|\tilde z - z\|_2^2 \le \|\delta\|_2^2 + 2\|\delta\|_2\|r(\gamma)\|_2 + \|r(\gamma)\|_2^2. \tag{9}
$$
Invoking (4) and $\|\delta\|_2=\gamma a$ with $a:=\|Wv\|_2$, we derive from (7) the corrected steering bound:
$$
\mathrm{KL}(p\|\tilde p) \le \frac{1}{4}\gamma^2 a^2 + \frac{1}{4}La\gamma^3 + \frac{1}{16}L^2\gamma^4. \tag{10}
$$

Safe $\gamma$ budget with a curvature safety factor. Fix a target divergence $\epsilon>0$. Ignoring the last term in (10) yields the cubic inequality
$$
\frac{1}{4}a^2\gamma^2 + \frac{1}{4}La\gamma^3 \le \epsilon.
$$
Set $x := (L\gamma)/a$ (dimensionless) and $\beta := 4\epsilon L^2/a^4$. The inequality becomes $x^3 + x^2 - \beta \le 0$, whose unique positive root solves
$$
x^3 + x^2 - \beta = 0.
$$
Writing the depressed cubic
$$
\left(x + \frac{1}{3}\right)^3 - \frac{1}{3}\left(x + \frac{1}{3}\right) + \left(\frac{2}{27} - \beta\right) = 0,
$$
and setting $p=-1/3$, $q=2/27-\beta$, $\Delta = (q/2)^2 + (p/3)^3$, the real Cardano root is
$$
x = \sqrt[3]{-\frac{q}{2} + \sqrt{\Delta}} + \sqrt[3]{-\frac{q}{2} - \sqrt{\Delta}} - \frac{1}{3}. \tag{11}
$$
Numerically, this expression is unambiguous if one takes the real branch of each cube root. Finally,
$$
\gamma_{\mathrm{raw}} = \frac{a}{L}x. \tag{12}
$$

Degenerate direction $a=0$. If $a=0$ (the steering vector lies in the null-space of $W$) the quadratic and cubic terms vanish; retaining the quartic term in (10) gives $\frac{L^2}{16}\gamma^4 \le \epsilon$ and hence $\gamma \le (16\epsilon)^{1/4}/\sqrt{L}$. We therefore set
$$
\gamma_{\mathrm{raw}} =
\begin{cases}
\dfrac{a}{L}x, & a>0,\\\\
\dfrac{(16\epsilon)^{1/4}}{\sqrt{L}}, & a=0.
\end{cases}
$$

Curvature safety factor. Because the quartic term in (10) is strictly positive, $\gamma_{\mathrm{raw}}$ is slightly optimistic when $L\gamma$ is not negligible relative to $a$. We therefore define the final scale
$$
\gamma_{\max} = \max\left\{0,\ \left(1 - \frac{L\gamma_{\mathrm{raw}}}{4a}\right)\gamma_{\mathrm{raw}}\right\}. \tag{13}
$$
with the convention $L\gamma_{\mathrm{raw}}/a := 0$ when $a=0$. One checks directly that substituting $\gamma=\gamma_{\max}$ into (10) yields $\mathrm{KL}\le\epsilon$ whenever $x<4$, which covers all practical regimes. When $L\to 0$ the safety factor tends to 1 and Eq. (13) continuously reduces to the familiar local-linear scale $\gamma_{\max} = 2\sqrt{\epsilon}/a$.

## Appendix B Qualitative Results

In this section, we present illustrative examples from the MATH500 dataset, comparing standard Chain-of-Thought (CoT) responses with those produced by ASC.

**Figure 6: Qualitative Example for comparing ASC response against vanilla CoT response.**

**Figure 7: Qualitative Example for comparing ASC response against vanilla CoT response.**

## Appendix C Steering Hyperparameters

**Table 3: Hyperparameters for three different models.**

| Model | $\gamma$ | Layer Index |
|---|---:|---:|
| DeepSeek-Distill-Qwen-7B | 0.275 | 21 |
| DeepSeek-Distill-LLaMA-8B | 0.46 | 21 |
| QwQ-32B | 0.50 | 57 |

Table 3 summarizes the hyperparameters used for steering in our reasoning models. The steering strength $\gamma$ is selected based on $\gamma_{\max}$, as derived in Section 4, and the choice of layer index is determined empirically. Early layers are avoided because representations are still underdeveloped,

while injecting at the final layers has limited impact due to diminished transformation capacity. Therefore, we select a mid-layer range where representations are sufficiently structured yet still amenable to effective steering. This middle ground provides a practical trade-off between steerability and representational richness.

