# ASC：Activation Steering 压缩推理实验

本仓库用于复现与扩展 ASC（Activation Steering for Chain-of-Thought Compression）方向的阶段性实验。核心目标是：在尽量保持数学推理正确率的同时，减少 reasoning 模型生成的 CoT token 数。

当前整理后的主要代码入口是：

```text
ASC_phase1/
```

结果汇总入口是：

```text
results/RESULT_SUMMARY.md
```

## 当前阶段结论

当前已经完成的主要工作：

- 整理出相对独立的 ASC 实验管线；
- 支持下载/读取本地 GSM8K 与 MATH 数据；
- 支持生成 long/short CoT pairs；
- 支持人工剔除无效 pairs 后提取 steering vector；
- 支持在 CoT baseline 与 ASC 不同 gamma 下做评测；
- 支持 Qwen3-8B thinking mode 的官方推荐采样参数；
- 尝试实现论文 KL gamma 公式，并记录其工程困难。

当前比较有价值的阶段性结果如下。表中的压缩率均相对于同模型、同数据集的 CoT baseline 计算。

| Model | Dataset | Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---|---|---:|---:|---:|---:|
| Qwen3-8B | GSM8K | Self-extracted vector | 0.65 | 96.00% | 1467.00 | 23.1% |
| DeepSeek-R1-Distill-Qwen-7B | GSM8K | Self-extracted vector | 0.30 | 90.50% | 499.41 | 30.3% |
| DeepSeek-R1-Distill-Qwen-7B | MATH | Self-extracted vector | 0.30 | 88.00% | 1639.63 | 30.3% |
| DeepSeek-R1-Distill-Qwen-7B | MATH | Author vector | 0.27 | 88.00% | 1774.40 | 24.5% |
| DeepSeek-R1-Distill-Llama-8B | MATH | Author vector | 0.47 | 86.00% | 1810.80 | 26.3% |

这些结果说明：原作者向量和自己重新提取的新向量都能在部分模型/数据集上带来有效压缩；其中新向量在 DeepSeek-R1-Distill-Qwen-7B 的 GSM8K 与 MATH 上表现比较稳定，Qwen3-8B 在官方 thinking-mode 设置下也已经出现可继续推进的压缩信号。

完整结果见：

```text
results/RESULT_SUMMARY.md
```

## 仓库结构

```text
ASC_phase1/
  answer_utils.py                 # 答案提取与正确性判断
  asc_steering_utils.py            # pairs 生成与 activation 提取公共工具
  download_datasets.py             # 下载/整理 GSM8K 与 MATH
  generate_cot_pairs.py            # 生成 long/short CoT pairs
  extract_steering_vector.py       # 从清洗后的 pairs 提取 steering vector
  extract_optimal_gamma.py         # 基于 hidden norm 给出 gamma 搜索范围
  extract_paper_kl_gamma.py        # 论文 KL gamma 诊断的工程实现尝试
  eval_asc_paper.py                # 统一评测脚本
  测试.md                          # 三个实验层次的完整命令清单
  docs/
    GAMMA_CALIBRATION_SUMMARY.md   # gamma 方法总结
    ASC_PAPER_FULL.md              # ASC 论文 Markdown 阅读版
    全复现.md                      # 复现说明补充

results/
  RESULT_SUMMARY.md                # GitHub 展示版结果总表
  
legacy/
  author_code/                     # 作者公开的原始脚本，仅作对照
```

根目录下仍保留了一些早期探索脚本和结果文件。后续正式使用时，优先查看 `ASC_phase1/`。

## 核心实验流程

完整流程分为四步：

```text
1. 下载/准备数据集
2. 生成 long/short CoT pairs
3. 人工检查 pairs，删除无效、截断、复读样本
4. 提取 steering vector，并在评测脚本中测试不同 gamma
```

最重要的规则：

```text
提取引导向量、提取 gamma、最终评测必须使用同一个 layer_index。
```

prompt 也要按模型口径固定：

```text
DeepSeek-R1-Distill-Qwen-7B / DeepSeek-R1-Distill-Llama-8B / QwQ-32B: paper_cot
Qwen3-8B: chat_boxed_cot + qwen3_enable_thinking
```

换到全新模型时，不要直接照抄这里的 prompt。先看模型官网或模型卡，确认是否需要 chat template、thinking mode，以及数学评测推荐的输出格式。

例如 Qwen3-8B 当前推荐使用：

```text
--layer_index 24
```

## 关键脚本

| 脚本 | 作用 |
|---|---|
| `ASC_phase1/download_datasets.py` | 下载/整理 GSM8K 与 MATH 数据 |
| `ASC_phase1/generate_cot_pairs.py` | 生成 long/short CoT pairs |
| `ASC_phase1/extract_steering_vector.py` | 根据 checked pairs 提取 steering vector |
| `ASC_phase1/extract_optimal_gamma.py` | 用 hidden norm ratio 估计 gamma 搜索范围 |
| `ASC_phase1/extract_paper_kl_gamma.py` | 尝试按论文 KL 上界公式求 gamma |
| `ASC_phase1/eval_asc_paper.py` | 评测 CoT baseline 与 ASC gamma |
| `ASC_phase1/测试.md` | 原模型、新向量、新模型三种层次的命令清单 |

## Qwen3-8B 当前推荐评测口径

Qwen3-8B 使用官方 thinking-mode 推荐参数：

```text
prompt_mode=chat_boxed_cot
thinking=True
temperature=0.6
top_p=0.95
top_k=20
min_p=0.0
```

当前阶段统一使用 transformers 路径评测。vLLM 虽然可以加速 `gamma=0` baseline，但会引入额外依赖兼容风险，暂不放入主流程。

```bash
python eval_asc_paper.py \
  --model_name /root/autodl-tmp/Qwen/Qwen3-8B \
  --dataset gsm8k \
  --limit 200 \
  --candidate_gammas 0,0.46,0.55,0.65,0.75,0.9 \
  --prompt_mode chat_boxed_cot \
  --qwen3_enable_thinking \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --min_p 0.0 \
  --max_new_tokens 4096 \
  --steering_vector_path vectors/steering_vectors_qwen3_8b_math_train_deepseek_checked_layer24.pt \
  --layer_index 24 \
  --batch_size 24 \
  --attn_impl flash_attention_2 \
  --per_gamma_output_dir results/qwen3_8b_gsm8k_200_layer24
```

更多命令见：

```text
ASC_phase1/测试.md
```

## Gamma 相关说明

目前有两类 gamma 辅助脚本：

```text
extract_optimal_gamma.py
```

这是经验尺度方法，根据 steering perturbation 占 hidden norm 的比例给出候选 gamma。它不是自动最优，只用于缩小搜索范围。

```text
extract_paper_kl_gamma.py
```

这是按论文 KL 上界思路自行实现的诊断脚本。论文给出了理论形式，但公开代码没有完整工程实现；当前实现可用于分析和记录，但不作为最终自动 gamma 选择依据。

当前实践结论是：gamma 仍主要通过任务评测选择，即看准确率与 token 数的折中。

## 已知问题

第二阶段计划中的“论文 KL gamma 自动求解”目前没有完全成功。主要原因是：

- 论文给出了理论推导，但没有提供完整工程实现；
- 对大模型长上下文做 JVP/HVP 或有限差分时显存压力很大；
- 理论得到的 gamma 与实际任务最优 gamma 之间仍有明显差距；
- 这也导致后续“动态 gamma 校准机制”暂时无法可靠推进。

因此当前阶段更稳妥的做法是：

```text
用 hidden norm 方法给出候选范围，再用 GSM8K/MATH 实测选择 gamma。
```

## 下一步

后续工作主要有三个方向：

1. **自动清洗 long/short CoT pairs**

   当前 pairs 仍依赖人工筛选。后续可以复用已有的答案提取与比对逻辑，自动判断 long/short CoT 是否给出正确答案，并剔除无答案、答案错误、长度截断或明显异常的样本。

2. **转换自动 gamma 获取思路**

   现有论文 KL gamma 公式在工程实现上仍不稳定。后续可以从“不偏离推理轨迹”的角度重新设计自动 gamma 获取方法：不是只追求更大的扰动，而是约束 steering 后的模型状态仍沿着合理推理轨迹前进。

3. **研究加速推理与 ASC 注入的结合**

   当前 `gamma=0` baseline 可以使用 vLLM 加速，但非零 gamma 仍依赖 transformers 的中间层 hook。后续需要研究 activation steering 的引导向量注入是否能在 vLLM 或类似高吞吐推理框架中实现，从而减少 ASC 评测与部署成本。
