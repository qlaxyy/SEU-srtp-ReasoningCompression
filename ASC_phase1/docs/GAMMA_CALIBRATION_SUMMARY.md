# Gamma 选择说明

## 当前结论

用脚本给出候选 gamma，再用实际生成评测选出可用 gamma。


论文 KL 公式控制的是固定上下文下的一步 next-token 分布变化。它有理论意义，但不等价于完整生成任务上的“最佳 gamma”。实际评测关心的是整段答案是否正确、CoT 是否变短、推理是否崩掉。这几件事不能只靠单步 KL 保证。

因此，当前最终选择 gamma 的依据还是：

```text
正确率不明显下降，同时 token 数下降明显。
```

例如 Qwen3-8B 在官方 thinking-mode 设置下，GSM8K 目前更好的点是：

```text
gamma=0.65
accuracy=96.00%
avg_tokens=1467.00
token compression=23.1%
```

继续加到 `0.75/0.90` 仍然能压缩，但正确率下降，而且 `0.75` 的 token 数还高于 `0.65`，所以当前不如 `0.65`。

## 两个脚本的定位

### `extract_optimal_gamma.py`

这是实用脚本。

它根据 steering vector 对 hidden state 的扰动幅度，给出一组候选 gamma。它不证明最优，只是帮我们缩小搜索范围。

示例：

```bash
python extract_optimal_gamma.py \
  --model_name /root/autodl-tmp/Qwen/Qwen3-8B \
  --dataset math \
  --pairs_path pairs/qwen3_8b_math_train_deepseek_pairs_50_checked.json \
  --pair_calibration_source long \
  --steering_vector vectors/steering_vectors_qwen3_8b_math_train_deepseek_checked_layer24.pt \
  --layer_index 24 \
  --ratios 0.05,0.1,0.15,0.2,0.25,0.3 \
  --select_ratio 0.2 \
  --batch_size 4 \
  --max_input_tokens 8192 \
  --attn_impl flash_attention_2 \
  --output_path results/gamma_qwen3_8b_math_train_deepseek_checked_hidden_norm_layer24.json
```

输出里主要看：

```text
selected.gamma_for_raw_vector
grid_suggestion.candidate_gammas_csv
grid_suggestion.fine_candidate_gammas_csv
```

这些值只是候选，后面还要跑 `eval_asc_paper.py`。

### `extract_paper_kl_gamma.py`

这是诊断脚本。

它尝试按论文 KL 上界公式求一个保守 gamma。这个脚本可以作为“我们尝试工程化复现论文推导”的证据，但当前不建议把它作为最终 gamma 选择器。

示例：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python extract_paper_kl_gamma.py \
  --model_name /root/autodl-tmp/Qwen/Qwen3-8B \
  --pairs_path pairs/qwen3_8b_math_train_deepseek_pairs_50_checked.json \
  --steering_vector vectors/steering_vectors_qwen3_8b_math_train_deepseek_checked_layer24.pt \
  --layer_index 24 \
  --epsilon 1.0 \
  --attn_impl sdpa \
  --curvature_method finite_diff \
  --finite_diff_delta 1.5 \
  --center_logits \
  --max_input_tokens 8192 \
  --output_path results/gamma_qwen3_8b_math_train_deepseek_checked_paper_kl_layer24.json
```

注意：这个结果只能说明局部 KL 意义下的保守扰动范围，不能直接说明完整生成任务上哪个 gamma 最好。

## 最终评测命令

gamma 候选出来后，还是要直接跑任务。

以 Qwen3-8B / GSM8K 为例：

```bash
python eval_asc_paper.py \
  --model_name /root/autodl-tmp/Qwen/Qwen3-8B \
  --dataset gsm8k \
  --limit 200 \
  --prompt_mode chat_boxed_cot \
  --qwen3_enable_thinking \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --min_p 0.0 \
  --max_new_tokens 4096 \
  --candidate_gammas 0,0.46,0.55,0.65,0.75,0.9 \
  --steering_vector_path vectors/steering_vectors_qwen3_8b_math_train_deepseek_checked_layer24.pt \
  --layer_index 24 \
  --batch_size 24 \
  --attn_impl flash_attention_2 \
  --per_gamma_output_dir results/qwen3_8b_gsm8k_200_layer24
```

生成、提向量、正式评测优先使用 `flash_attention_2`。只有在当前环境的 flash-attn 和 torch/CUDA 不兼容，或者做 KL 曲率诊断这类数值敏感步骤时，再临时换成 `sdpa`。
