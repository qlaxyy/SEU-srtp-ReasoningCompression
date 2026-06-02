# ASC Result Summary

## Reading Notes

- `CoT` 表示不使用 ASC 的普通 chain-of-thought baseline。
- `Author vector` 表示作者原始代码/配置中提供的 steering vector。
- `Self-extracted vector` 表示本项目重新生成 long/short CoT pairs 后提取的新 steering vector。
- `Token compression` 按同模型、同数据集的 CoT baseline 计算：

```text
Token compression = (CoT avg tokens - ASC avg tokens) / CoT avg tokens
```

## Main Findings

- Qwen3-8B 在官方 thinking-mode 口径下，`Self-extracted vector + gamma=0.65` 可以在 GSM8K 上保持 `96.00%` 正确率，并把平均输出从 `1908.30` tokens 降到 `1467.00` tokens，压缩约 `23.1%`。
- DeepSeek-R1-Distill-Qwen-7B 上的新向量在 GSM8K 和 MATH 上都能带来稳定压缩信号，其中 GSM8K 上可达到约 `30%+` token 压缩。
- 原作者向量在 DeepSeek-R1-Distill-Qwen-7B/GSM8K 与 DeepSeek-R1-Distill-Qwen-7B/MATH 上仍然有效，但新向量在部分设置下可以支持更高 gamma。
- DeepSeek-R1-Distill-Llama-8B 结果显示 ASC 也能压缩 token，但高 gamma 更容易损伤正确率，因此需要更谨慎选择 gamma。

## Qwen3-8B / GSM8K

官方 thinking-mode 口径：

```text
thinking=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0
```

| Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---:|---:|---:|---:|
| CoT | CoT | 95.00% | 1908.30 | 0.0% |
| Self-extracted vector | 0.20 | 97.50% | 1669.20 | 12.5% |
| Self-extracted vector | 0.29 | 95.50% | 1612.00 | 15.5% |
| Self-extracted vector | 0.38 | 96.00% | 1628.50 | 14.7% |
| Self-extracted vector | 0.46 | 97.00% | 1559.10 | 18.3% |
| Self-extracted vector | 0.55 | 96.00% | 1521.00 | 20.3% |
| Self-extracted vector | 0.65 | 96.00% | 1467.00 | 23.1% |
| Self-extracted vector | 0.75 | 95.00% | 1513.90 | 20.7% |
| Self-extracted vector | 0.90 | 94.50% | 1470.90 | 22.9% |

## DeepSeek-R1-Distill-Qwen-7B / GSM8K

| Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---:|---:|---:|---:|
| CoT | CoT | 91.25% | 716.12 | 0.0% |
| Author vector | 0.10 | 92.25% | 632.54 | 11.7% |
| Author vector | 0.20 | 88.50% | 548.65 | 23.4% |
| Author vector | 0.24 | 90.00% | 534.64 | 25.3% |
| Self-extracted vector | 0.30 | 90.50% | 499.41 | 30.3% |
| Self-extracted vector | 0.35 | 86.75% | 480.99 | 32.8% |
| Self-extracted vector | 0.40 | 86.50% | 485.53 | 32.2% |

## DeepSeek-R1-Distill-Qwen-7B / MATH

| Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---:|---:|---:|---:|
| CoT | CoT | 86.00% | 2351.38 | 0.0% |
| Author vector | 0.27 | 88.00% | 1774.40 | 24.5% |
| Self-extracted vector | 0.20 | 86.00% | 1623.40 | 31.0% |
| Self-extracted vector | 0.30 | 88.00% | 1639.63 | 30.3% |

## DeepSeek-R1-Distill-Llama-8B / GSM8K

| Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---:|---:|---:|---:|
| CoT | CoT | 90.00% | 853.58 | 0.0% |
| Author vector | 0.20 | 85.20% | 700.02 | 18.0% |

## DeepSeek-R1-Distill-Llama-8B / MATH

| Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---:|---:|---:|---:|
| CoT | CoT | 88.00% | 2457.16 | 0.0% |
| Author vector | 0.20 | 92.00% | 2281.86 | 7.1% |
| Author vector | 0.47 | 86.00% | 1810.80 | 26.3% |
