"""
严格按照 ASC 论文 KL 上界推导求解 gamma。

这个脚本刻意保持“纯净”：
  - 只使用已经准备好的长/短 CoT pairs。
  - 不做 gamma 网格搜索。
  - 不生成完整答案。
  - 不根据正确率、token 数或经验 KL 调参。
  - 不依赖其他 gamma 校准脚本。

论文流程对应关系：
  1. 对每个问题 q_i，已经有长链 l_i 和短链 s_i。
  2. steering vector 已经由 h(q_i+s_i)[-1] - h(q_i+l_i)[-1] 得到。
  3. 本脚本读取该 steering vector，并归一化成单位方向 v。
  4. 在同一批 pairs 的 question + long_cot 上估计：
       a_i = ||J(h_i)v||_2
       L_i = ||d^2F(h_i)[v,v]||_2
  5. 取 a = median(a_i)，L = percentile_95(L_i)。
  6. 按论文三次方程求 gamma_raw_unit，再乘曲率安全因子得到 gamma_unit。
  7. 将单位向量尺度 gamma_unit 换算成评测代码使用的 raw-vector gamma：
       gamma_for_raw_vector = gamma_unit / ||v_raw||_2

输出里的 selected.gamma_for_raw_vector 可以直接传给：
  eval_asc_custom_vector.py --gamma_override
  eval_asc_dual_gpu_grid.py 中对应 gamma 参数
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# 0. 固定配置
# ---------------------------------------------------------------------------

DEFAULT_EPSILON = 1.0
DEFAULT_MODEL_NAME = "/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_PAIRS_PATH = "pairs/qwen7b_math_train_deepseek_pairs_100_checked.json"
DEFAULT_STEERING_VECTOR = "vectors/steering_vectors_qwen7b_math_train_deepseek_checked.pt"
DEFAULT_OUTPUT_PATH = "results/gamma_qwen7b_math_train_deepseek_checked_paper_kl_centered_fd_8192_eps1.json"

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen": {
        "layer_index": 20,
        "steering_vector_path": "./vectors/steering_vectors_qwen7b.pt",
        "paper_reference_gamma": 0.275,
    },
    "llama": {
        "layer_index": 20,
        "steering_vector_path": "./vectors/steering_vectors_llama8b.pt",
        "paper_reference_gamma": 0.46,
    },
    "qwq": {
        "layer_index": 20,
        "steering_vector_path": "./vectors/steering_vectors_qwq32b.pt",
        "paper_reference_gamma": 0.50,
    },
}

PAIR_STEERING_VECTOR_DEFAULTS: dict[str, str] = {
    "qwen7b_math_train_deepseek_pairs_100_checked": DEFAULT_STEERING_VECTOR,
    "qwen7b_math_train_deepseek_pairs_100_checked.json": DEFAULT_STEERING_VECTOR,
}


@dataclass
class PairRecord:
    """一条校准样本。

    论文意义：
      problem = q_i
      long_cot = l_i，由目标模型生成的冗长思维链
      short_cot = s_i，由强模型或人工策略生成的短思维链

    本脚本求 a/L 时只使用 question + long_cot，因为它是目标模型的自然
    冗长推理状态，也就是 ASC 要从中“推开”的基准状态。
    """

    problem: str
    long_cot: str
    short_cot: str
    long_prompt: str | None = None


@dataclass
class CalibrationStats:
    """论文 KL 公式需要的校准统计量。"""

    num_pairs_available: int
    num_requested: int | None
    num_used: int
    steering_vector_norm: float
    a_median: float
    a_mean: float
    a_min: float
    a_max: float
    l_percentile: float
    l_value: float
    l_median: float
    l_mean: float
    l_min: float
    l_max: float
    curvature_method: str
    logit_centering: bool


@dataclass
class GammaResult:
    """最终论文 gamma 结果。

    gamma_unit_vector:
      论文公式中的 gamma，作用在单位向量 v 上。

    gamma_for_raw_vector:
      作者 generate.py / 你的 eval 脚本实际使用 raw vector，所以需要把
      gamma_unit_vector 除以 raw vector 的 L2 norm。
    """

    epsilon: float
    gamma_unit_vector: float
    gamma_for_raw_vector: float
    gamma_for_eval_asc: float
    gamma_raw_unit_before_safety: float
    safety_factor: float
    cubic_root_x: float
    beta: float
    kl_upper_at_gamma_unit: float
    paper_condition_x_lt_4: bool
    safety_clamped_to_zero: bool
    method: str


@dataclass
class TailContext:
    """从第 ell 层输出继续跑到 logits 所需的上下文。

    论文里的 F_{ell -> logit}(h) 只应该包含目标层之后的子网络。
    因此我们先正常前向一次，缓存：
      - 第 ell 层输出的完整序列 hidden states；
      - 后续层需要复用的 attention_mask / position_ids / rotary embedding 等参数。

    之后 fn(h_last) 只替换最后 token 的 hidden state，并从 ell+1 层继续前向。
    """

    hidden_after_layer: torch.Tensor
    common_args: tuple[Any, ...]
    common_kwargs: dict[str, Any]


# ---------------------------------------------------------------------------
# 1. 参数、文件读取、模型路径推断
# ---------------------------------------------------------------------------


def resolve_model_config(model_name: str) -> dict[str, Any]:
    """根据模型名粗略推断默认 layer 和旧 steering vector 路径。

    如果你传入了 --steering_vector 和 --layer_index，这里的默认值不会生效。
    """

    name = model_name.lower()
    if "qwen3" in name:
        return {}
    if "qwen" in name and "32" not in name and "qwq" not in name:
        return MODEL_CONFIGS["qwen"]
    if "llama" in name:
        return MODEL_CONFIGS["llama"]
    if "qwq" in name or "32b" in name:
        return MODEL_CONFIGS["qwq"]
    return {}


def infer_steering_vector_from_pairs(pairs_path: str | Path | None) -> str | None:
    """Infer the steering vector used by known checked pair files."""

    if pairs_path is None:
        return None
    path = Path(pairs_path)
    return (
        PAIR_STEERING_VECTOR_DEFAULTS.get(path.name)
        or PAIR_STEERING_VECTOR_DEFAULTS.get(path.stem)
    )


def parse_max_memory(text: str | None) -> dict[int | str, str] | None:
    """解析 transformers 的 max_memory 参数。

    示例：
      --max_memory 0:20GiB,1:20GiB
    """

    if not text:
        return None
    result: dict[int | str, str] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        result[int(key) if key.isdigit() else key] = value.strip()
    return result


def torch_dtype_from_arg(name: str) -> Any:
    """把命令行 dtype 字符串转换成 torch dtype。"""

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def read_json_or_jsonl(path: str | Path) -> Any:
    """读取 JSON 或 JSONL。

    pairs 文件可能是：
      1. 一个 JSON list；
      2. 一个包含 pairs/data/samples 字段的 JSON object；
      3. 每行一个样本的 JSONL。
    """

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {file_path}")
    if text[0] in "[{":
        return json.loads(text)

    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {file_path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    """保存结果报告。"""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def problem_from_row(row: dict[str, Any]) -> str:
    """从不同格式的样本里取题目文本。"""

    for key in ("problem", "question", "input", "prompt"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError(f"找不到 problem/question 字段，当前字段为：{list(row)}")


def first_nonempty_field(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """从多个候选字段中取第一个非空值。"""

    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return None


def load_pairs(pairs_path: str | Path) -> list[PairRecord]:
    """读取长短 CoT pairs，并自动统计有效样本数。

    这就是你说的“我懒得传 --num_cal 80”：脚本会自己数。
    如果你不传 --num_cal，它会默认使用 pairs 文件里的全部有效 pairs。
    """

    payload = read_json_or_jsonl(pairs_path)
    if isinstance(payload, dict):
        payload = payload.get("pairs") or payload.get("data") or payload.get("samples")
    if not isinstance(payload, list):
        raise ValueError("--pairs_path 必须是 JSON list 或 JSONL。")

    pairs: list[PairRecord] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        problem = problem_from_row(row)
        long_cot = first_nonempty_field(
            row,
            ("long_cot", "long_answer", "long_response", "verbose_cot"),
        )
        short_cot = first_nonempty_field(
            row,
            ("short_cot", "short_answer", "short_response", "concise_cot"),
        )
        if not long_cot or not short_cot:
            continue
        long_prompt = first_nonempty_field(row, ("long_prompt",))
        pairs.append(
            PairRecord(
                problem=problem,
                long_cot=long_cot,
                short_cot=short_cot,
                long_prompt=long_prompt,
            )
        )

    if not pairs:
        raise ValueError(f"No usable long/short pairs found in {pairs_path}")
    return pairs


def select_calibration_pairs(
    pairs: list[PairRecord],
    num_cal: int | None,
    seed: int,
) -> list[PairRecord]:
    """确定用于估计 a/L 的 pairs。

    - num_cal 不传：使用全部有效 pairs。
    - num_cal 传入：随机打乱后取指定数量。
    """

    selected = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(selected)
    if num_cal is None:
        return selected
    return selected[: min(num_cal, len(selected))]


def append_cot_to_prompt(prompt: str, cot: str) -> str:
    if prompt.endswith(("\n", " ", "\t")):
        return prompt + cot
    return prompt + "\n" + cot


def build_long_cot_text(pair: PairRecord) -> str:
    """构造论文中估计 a/L 的输入：long_prompt + long_cot。"""

    if not pair.long_prompt:
        raise ValueError(
            "Pair rows must contain long_prompt so calibration uses the same "
            "target-model prompt prefix as vector extraction."
        )
    return append_cot_to_prompt(pair.long_prompt, pair.long_cot)


def load_tokenizer(model_name: str, tokenizer_mode: str) -> AutoTokenizer:
    """加载 tokenizer。

    一些 Llama 本地目录缺 tokenizer.json 时，fast tokenizer 可能报错；
    tokenizer_mode=auto 会自动 fallback 到 slow tokenizer。
    """

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if tokenizer_mode == "fast":
        kwargs["use_fast"] = True
    elif tokenizer_mode == "slow":
        kwargs["use_fast"] = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    except Exception:
        if tokenizer_mode != "auto":
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def load_steering_vector(path: str | Path) -> torch.Tensor:
    """读取 raw steering vector。

    作者代码保存的是多条差分向量，推理时：
      steering_vec = torch.load(...).mean(dim=0)

    所以这里保持同样约定：
      - 如果是 [N, hidden]，就 mean(dim=0)。
      - 如果已经是 [hidden]，就直接使用。

    注意：这里不归一化；归一化只在估计论文 a/L 时临时使用。
    """

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        tensors = [value for value in obj.values() if torch.is_tensor(value)]
        if not tensors:
            raise ValueError(f"No tensor found in steering vector file: {path}")
        obj = tensors[0]

    if not torch.is_tensor(obj):
        raise TypeError(f"Expected tensor steering vector, got {type(obj)!r}")

    vec = obj.detach().float().cpu()
    if vec.ndim == 2:
        vec = vec.mean(dim=0)
    elif vec.ndim != 1:
        raise ValueError(f"Unexpected steering vector shape: {tuple(vec.shape)}")
    return vec


# ---------------------------------------------------------------------------
# 2. 模型前向与隐藏状态替换工具
# ---------------------------------------------------------------------------


def configure_attention_for_derivatives(attn_impl: str) -> None:
    """为 JVP/HVP 配置注意力实现。

    fused attention，例如 flash_attention_2，常常没有二阶导实现。
    因此严格论文 KL 校准默认使用 eager。
    """

    if attn_impl != "eager" or not torch.cuda.is_available():
        return
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """定位 decoder 层列表。

    Qwen/Llama 通常是 model.model.layers。
    这里保留几个常见 fallback，避免换模型时直接崩掉。
    """

    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for base_name, layer_name in candidates:
        base = getattr(model, base_name, None)
        layers = getattr(base, layer_name, None) if base is not None else None
        if layers is not None:
            return layers
    raise AttributeError("无法找到 decoder layers，请检查模型结构。")


def first_tensor(output: Any) -> torch.Tensor:
    """从 layer output 中取 hidden states tensor。"""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return first_tensor(output[0])
    raise TypeError(f"Unsupported layer output type: {type(output)!r}")


def module_device(module: torch.nn.Module) -> torch.device:
    """找一个模块所在设备。"""

    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_input_device(model: torch.nn.Module) -> torch.device:
    """找到输入 token 应该放在哪张卡。

    使用 device_map 切模型时，输入通常放 embedding 所在设备。
    """

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for key in ("model.embed_tokens", "transformer.wte", "model.layers.0"):
            value = device_map.get(key)
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value not in {"cpu", "disk"}:
                return torch.device(value)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(obj: Any, device: torch.device) -> Any:
    """递归移动 tensor 到指定设备。

    device_map 切模型时，不同 decoder layer 可能在不同 GPU 上。我们手动跑
    tail subnetwork，因此每进入一层前都要把 hidden states 和复用参数移动到
    该层所在设备。
    """

    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    return obj


@torch.no_grad()
def capture_tail_context(
    model: AutoModelForCausalLM,
    inputs: dict[str, torch.Tensor],
    layers: torch.nn.ModuleList,
    layer_index: int,
) -> TailContext:
    """静态前向一次，缓存论文 F_{ell -> logit} 需要的上下文。

    对应论文符号：
      h_i = h^ell(q_i + l_i)[-1]

    与旧实现不同，这里不是之后再重跑完整模型。我们只缓存第 ell 层输出的
    完整序列，以及 ell+1 层收到的公共参数；后续 fn(h) 会直接从 ell+1
    层继续跑到 logits。这更接近论文里“从第 ell 层到 logits 的子网络”。
    """

    captured_hidden: list[torch.Tensor] = []
    captured_tail_args: list[tuple[Any, ...]] = []
    captured_tail_kwargs: list[dict[str, Any]] = []

    def capture_layer_output(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        captured_hidden.append(first_tensor(output).detach())

    def capture_next_layer_input(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        # args[0] 是 hidden_states，本脚本会手动传入替换后的 hidden。
        # 剩下的 attention_mask / position_ids / position_embeddings 等复用。
        captured_tail_args.append(tuple(args[1:]))
        tail_kwargs = dict(kwargs)
        tail_kwargs.pop("hidden_states", None)
        tail_kwargs["use_cache"] = False
        captured_tail_kwargs.append(tail_kwargs)

    handles = [layers[layer_index].register_forward_hook(capture_layer_output)]
    if layer_index + 1 < len(layers):
        handles.append(
            layers[layer_index + 1].register_forward_pre_hook(
                capture_next_layer_input,
                with_kwargs=True,
            )
        )

    try:
        base_model = getattr(model, "model", None)
        if base_model is not None:
            base_model(**inputs, use_cache=False)
        else:
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if not captured_hidden:
        raise RuntimeError(f"Layer {layer_index} hook did not capture hidden state.")

    common_args: tuple[Any, ...] = ()
    common_kwargs: dict[str, Any] = {"use_cache": False}
    if layer_index + 1 < len(layers):
        if not captured_tail_kwargs:
            raise RuntimeError(f"Layer {layer_index + 1} pre-hook did not capture tail kwargs.")
        common_args = captured_tail_args[0]
        common_kwargs = captured_tail_kwargs[0]

    return TailContext(
        hidden_after_layer=captured_hidden[0],
        common_args=common_args,
        common_kwargs=common_kwargs,
    )


def apply_final_norm_and_head(
    model: AutoModelForCausalLM,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """把最后一层 hidden states 映射成 next-token logits。

    Qwen/Llama 结构通常是：
      decoder layers -> final norm -> lm_head
    """

    base_model = getattr(model, "model", None)
    norm = getattr(base_model, "norm", None) if base_model is not None else None
    if norm is None:
        norm = getattr(model, "norm", None)
    if norm is not None:
        hidden_states = hidden_states.to(module_device(norm))
        hidden_states = norm(hidden_states)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise AttributeError("无法找到 lm_head，不能从 hidden states 映射到 logits。")

    last_hidden = hidden_states[:, -1:, :].to(module_device(lm_head))
    logits = lm_head(last_hidden)
    return logits[0, -1, :].float()


def tail_logits_from_hidden(
    model: AutoModelForCausalLM,
    layers: torch.nn.ModuleList,
    layer_index: int,
    context: TailContext,
    new_last_hidden: torch.Tensor,
) -> torch.Tensor:
    """实现论文中的 F_{ell -> logit}(h)。

    输入 new_last_hidden 是第 ell 层最后 token 的 hidden state。函数会：
      1. 在缓存的完整序列中只替换最后 token；
      2. 从第 ell+1 层继续前向；
      3. 经过 final norm 和 lm_head 得到 next-token logits。

    这样求 JVP/HVP 时，梯度图只覆盖 tail subnetwork，显存和论文定义都更合理。
    """

    hidden_states = context.hidden_after_layer.to(new_last_hidden.device)
    prefix = hidden_states[:, :-1, :]
    last = new_last_hidden.to(device=hidden_states.device, dtype=hidden_states.dtype)
    hidden_states = torch.cat([prefix, last.view(1, 1, -1)], dim=1)

    for idx in range(layer_index + 1, len(layers)):
        layer = layers[idx]
        layer_dev = module_device(layer)
        hidden_states = hidden_states.to(layer_dev)
        common_args = move_to_device(context.common_args, layer_dev)
        common_kwargs = move_to_device(context.common_kwargs, layer_dev)
        common_kwargs["use_cache"] = False
        output = layer(hidden_states, *common_args, **common_kwargs)
        hidden_states = first_tensor(output)

    return apply_final_norm_and_head(model, hidden_states)


# ---------------------------------------------------------------------------
# 3. 估计论文里的 a 和 L
# ---------------------------------------------------------------------------


def finite_difference_a_l(
    fn: Any,
    h: torch.Tensor,
    v: torch.Tensor,
    delta: float,
    center_logits: bool,
) -> tuple[float, float]:
    """用中心差分估计 a 和 L。

    这是 fallback 方法，用于 fused attention 或某些模型不支持 HVP 的情况。

    一阶：
      Jv ≈ [F(h+δv)-F(h-δv)] / (2δ)

    二阶方向曲率：
      H[v,v] ≈ [F(h+δv)-2F(h)+F(h-δv)] / δ²
    """

    h0 = h.detach()
    v0 = v.detach()
    with torch.no_grad():
        z0 = fn(h0)
        z_plus = fn(h0 + delta * v0)
        z_minus = fn(h0 - delta * v0)
        jvp_vec = (z_plus - z_minus) / (2.0 * delta)
        hvp_vec = (z_plus - 2.0 * z0 + z_minus) / (delta**2)
    return logit_delta_norm(jvp_vec, center_logits), logit_delta_norm(hvp_vec, center_logits)


def logit_delta_norm(vec: torch.Tensor, center_logits: bool) -> float:
    """计算 logits 扰动范数。

    softmax 对所有词表 logits 同加一个常数不敏感，因此默认先去掉均值
    再取 L2 norm，避免把不影响 KL 的 common-mode shift 算进 a/L。
    """

    value = vec.float()
    if center_logits:
        value = value - value.mean()
    return float(value.norm().detach().cpu().item())


def resolve_curvature_method(curvature_method: str) -> str:
    """把 auto 固定成一种方法，避免每个样本混用不同曲率估计。"""

    if curvature_method == "auto":
        return "jvp_fd"
    return curvature_method


def estimate_one_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: torch.nn.ModuleList,
    layer_index: int,
    text: str,
    unit_vector_cpu: torch.Tensor,
    max_input_tokens: int,
    curvature_method: str,
    finite_diff_delta: float,
    center_logits: bool,
) -> tuple[float, float, str]:
    """对一个样本估计 a_i 和 L_i。

    输入 text 是 question + long_cot。

    论文定义：
      a_i = ||J(h_i)v||_2
      L_i ≈ ||d²F(h_i)[v,v]||_2

    这里的 v 必须是单位向量，因为论文推导默认 ||v||_2 = 1。
    """

    input_device = model_input_device(model)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        return_token_type_ids=False,
    ).to(input_device)

    context = capture_tail_context(model, inputs, layers, layer_index)
    h0 = context.hidden_after_layer[0, -1, :].detach().float()
    unit_vector = unit_vector_cpu.to(device=h0.device, dtype=torch.float32)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return tail_logits_from_hidden(model, layers, layer_index, context, x)

    if curvature_method == "finite_diff":
        a_value, l_value = finite_difference_a_l(
            fn,
            h0,
            unit_vector,
            finite_diff_delta,
            center_logits,
        )
        return a_value, l_value, "finite_diff"

    h = h0.requires_grad_(True)
    with torch.enable_grad():
        try:
            _, jvp_vec = torch.autograd.functional.jvp(fn, h, unit_vector)
            a_value = logit_delta_norm(jvp_vec, center_logits)
            del jvp_vec
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        if curvature_method == "jvp_fd":
            # 这是默认工程实现：a 用精确 JVP，L 用中心差分方向曲率。
            # 原因是精确 HVP 需要保留二阶计算图，在 7B/8B 长序列上非常容易
            # OOM；中心差分仍然估计同一个 d²F(h)[v,v]，只是数值近似。
            _a_fd, l_value = finite_difference_a_l(
                fn,
                h0,
                unit_vector,
                finite_diff_delta,
                center_logits,
            )
            return a_value, l_value, "jvp_fd"

        try:
            # F 是向量值 logits 函数。
            # H[v,v] 可以通过“JVP 的 JVP”得到，而不是对 logits.sum() 做标量 HVP。
            def jvp_fn(x: torch.Tensor) -> torch.Tensor:
                _, inner_jvp = torch.autograd.functional.jvp(
                    fn,
                    x,
                    unit_vector,
                    create_graph=True,
                )
                return inner_jvp

            _, hvp_vec = torch.autograd.functional.jvp(jvp_fn, h, unit_vector) #hvp_vec = 在方向 unit_vector 上的 Hessian 向量积
            l_value = logit_delta_norm(hvp_vec, center_logits) # .norm() = 计算该向量的范数（L2范数）
            return a_value, l_value, "hvp"
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise


def estimate_a_and_l(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_index: int,
    steering_vec_raw: torch.Tensor,
    calibration_pairs: list[PairRecord],
    num_pairs_available: int,
    num_requested: int | None,
    max_input_tokens: int,
    l_percentile: float,
    curvature_method: str,
    finite_diff_delta: float,
    center_logits: bool,
) -> tuple[CalibrationStats, list[float], list[float]]:
    """在校准 pairs 上估计论文公式里的 a 和 L。

    统计方式：
      a = median_i ||J(h_i)v||_2
      L = percentile_i ||d²F(h_i)[v,v]||_2
    """

    layers = get_decoder_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(f"layer_index={layer_index} out of range; model has {len(layers)} layers.")

    # 只对 hidden state 求导，不训练模型参数。
    for param in model.parameters():
        param.requires_grad_(False)

    raw_norm = float(steering_vec_raw.norm().item())
    if raw_norm <= 0:
        raise ValueError("steering vector norm is zero.")
    unit_vector_cpu = steering_vec_raw / raw_norm
    effective_curvature_method = resolve_curvature_method(curvature_method)

    a_samples: list[float] = []
    l_samples: list[float] = []
    methods: list[str] = []

    for pair in tqdm(calibration_pairs, desc="Calibrating paper KL gamma"):
        text = build_long_cot_text(pair)
        try:
            a_value, l_value, method = estimate_one_sample(
                model=model,
                tokenizer=tokenizer,
                layers=layers,
                layer_index=layer_index,
                text=text,
                unit_vector_cpu=unit_vector_cpu,
                max_input_tokens=max_input_tokens,
                curvature_method=effective_curvature_method,
                finite_diff_delta=finite_diff_delta,
                center_logits=center_logits,
            )
        except Exception as exc:
            print(f"  [skip] calibration sample failed: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if math.isfinite(a_value) and math.isfinite(l_value):
            a_samples.append(a_value)
            l_samples.append(l_value)
            methods.append(method)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(a_samples) < 3:
        raise RuntimeError(f"Too few successful calibration samples: {len(a_samples)}")

    a_arr = np.asarray(a_samples, dtype=np.float64)
    l_arr = np.asarray(l_samples, dtype=np.float64)
    if len(set(methods)) > 1:
        raise RuntimeError(f"Mixed curvature methods are not allowed: {sorted(set(methods))}")
    method_summary = methods[0]

    return (
        CalibrationStats(
            num_pairs_available=num_pairs_available,
            num_requested=num_requested,
            num_used=len(a_samples),
            steering_vector_norm=raw_norm,
            a_median=float(np.median(a_arr)),
            a_mean=float(np.mean(a_arr)),
            a_min=float(np.min(a_arr)),
            a_max=float(np.max(a_arr)),
            l_percentile=float(l_percentile),
            l_value=float(np.percentile(l_arr, l_percentile)),
            l_median=float(np.median(l_arr)),
            l_mean=float(np.mean(l_arr)),
            l_min=float(np.min(l_arr)),
            l_max=float(np.max(l_arr)),
            curvature_method=method_summary,
            logit_centering=bool(center_logits),
        ),
        a_samples,
        l_samples,
    )


# ---------------------------------------------------------------------------
# 4. 按论文公式求解 gamma
# ---------------------------------------------------------------------------


def kl_upper_bound(gamma: float, a_value: float, l_value: float) -> float:
    """论文 Appendix A.1 的 KL 四次上界。

    KL <= 1/4 γ²a² + 1/4 Laγ³ + 1/16 L²γ⁴

    论文最终求闭式 gamma 时会先忽略四次项，然后再加安全因子。
    这里保留该函数只是为了报告 selected gamma 代入完整上界后的数值。
    """

    return (
        0.25 * (gamma**2) * (a_value**2)
        + 0.25 * l_value * a_value * (gamma**3)
        + 0.0625 * (l_value**2) * (gamma**4)
    )


def positive_cubic_root(beta: float) -> float:
    """求论文中的正实根 x。

    论文令：
      x = Lγ/a
      beta = 4εL²/a⁴

    忽略四次项后得到三次方程：
      x³ + x² - beta = 0
    """

    p = -1.0 / 3.0
    q = 2.0 / 27.0 - beta
    delta = (q / 2.0) ** 2 + (p / 3.0) ** 3

    if delta >= 0:
        x = float(
            np.cbrt(-q / 2.0 + math.sqrt(delta))
            + np.cbrt(-q / 2.0 - math.sqrt(delta))
            - 1.0 / 3.0
        )
    else:
        phi = math.acos((-q / 2.0) / math.sqrt(-((p / 3.0) ** 3)))
        roots = [
            2.0 * math.sqrt(-p / 3.0) * math.cos(phi / 3.0) - 1.0 / 3.0,
            2.0 * math.sqrt(-p / 3.0) * math.cos((phi + 2.0 * math.pi) / 3.0) - 1.0 / 3.0,
            2.0 * math.sqrt(-p / 3.0) * math.cos((phi + 4.0 * math.pi) / 3.0) - 1.0 / 3.0,
        ]
        positives = [value for value in roots if value > 0]
        if not positives:
            raise RuntimeError(f"No positive cubic root for beta={beta}")
        x = float(max(positives))

    if x <= 0:
        raise RuntimeError(f"No positive cubic root for beta={beta}")
    return x


def solve_paper_gamma(
    a_value: float,
    l_value: float,
    epsilon: float,
    steering_norm: float,
) -> GammaResult:
    """严格按论文三次方程 + 曲率安全因子求 gamma。

    论文步骤：
      1. beta = 4εL²/a⁴
      2. 解 x³ + x² - beta = 0
      3. gamma_raw_unit = (a/L)x
      4. gamma_unit = (1 - L*gamma_raw_unit/(4a)) * gamma_raw_unit
      5. gamma_for_raw_vector = gamma_unit / ||v_raw||_2
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if steering_norm <= 0:
        raise ValueError("steering_norm must be positive.")
    if a_value < 1e-12 and l_value < 1e-12:
        raise RuntimeError("Both a and L are approximately zero; gamma is unconstrained.")

    if l_value < 1e-12:
        beta = 0.0
        x = 0.0
        gamma_raw_unit = 2.0 * math.sqrt(epsilon) / max(a_value, 1e-12)
        safety = 1.0
    elif a_value < 1e-12:
        beta = float("inf")
        x = float("inf")
        gamma_raw_unit = (16.0 * epsilon) ** 0.25 / math.sqrt(l_value)
        safety = 1.0
    else:
        beta = 4.0 * epsilon * (l_value**2) / (a_value**4)
        x = positive_cubic_root(beta)
        gamma_raw_unit = (a_value / l_value) * x
        safety = max(0.0, 1.0 - (l_value * gamma_raw_unit) / (4.0 * a_value))

    gamma_unit = max(0.0, safety * gamma_raw_unit)
    gamma_raw_vector = gamma_unit / steering_norm

    return GammaResult(
        epsilon=float(epsilon),
        gamma_unit_vector=float(gamma_unit),
        gamma_for_raw_vector=float(gamma_raw_vector),
        gamma_for_eval_asc=float(gamma_raw_vector),
        gamma_raw_unit_before_safety=float(gamma_raw_unit),
        safety_factor=float(safety),
        cubic_root_x=float(x),
        beta=float(beta),
        kl_upper_at_gamma_unit=float(kl_upper_bound(gamma_unit, a_value, l_value)),
        paper_condition_x_lt_4=bool(x < 4.0),
        safety_clamped_to_zero=bool(safety <= 0.0),
        method="paper_cubic_equation_plus_curvature_safety_factor",
    )


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="严格按照 ASC 论文 KL 上界公式求解 gamma。"
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pairs_path", type=str, default=DEFAULT_PAIRS_PATH)
    # 兼容旧命令；纯净版只从 pairs_path 读取 question + long_cot，不再需要 dataset。
    parser.add_argument("--dataset", type=str, default=None, help=argparse.SUPPRESS)
    # 兼容旧命令；严格论文校准固定使用 long_cot 作为自然长推理状态。
    parser.add_argument("--pair_calibration_source", type=str, default="long", help=argparse.SUPPRESS)
    parser.add_argument("--steering_vector", type=str, default=None)
    parser.add_argument("--layer_index", type=int, default=None)
    parser.add_argument(
        "--num_cal",
        type=int,
        default=None,
        help="默认使用 pairs 文件里的全部有效样本；传入后只随机取前 num_cal 个。",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="KL 上界预算。默认使用当前工程配置 1.0；严格论文预算可手动传 1e-3。",
    )
    parser.add_argument("--l_percentile", type=float, default=95.0)
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_impl", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_memory", type=str, default=None, help="例如 0:20GiB,1:20GiB")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--tokenizer_mode", type=str, default="auto", choices=["auto", "fast", "slow"])
    parser.add_argument(
        "--curvature_method",
        type=str,
        default="finite_diff",
        choices=["auto", "jvp_fd", "hvp", "finite_diff"],
        help=(
            "auto 是 jvp_fd 的别名，不再逐样本 fallback；jvp_fd: a 用精确 JVP，"
            "L 用中心差分；hvp: 尝试精确二阶 JVP，显存压力很大；"
            "finite_diff: a 和 L 都用中心差分。"
        ),
    )
    parser.add_argument(
        "--finite_diff_delta",
        type=float,
        default=1.5,
        help=(
            "有限差分步长。混合精度下 1e-2 往往小于 bf16/float16 的有效分辨率，"
            "会把数值噪声放大成异常大的 a/L；默认使用当前工程配置 1.5。"
        ),
    )
    parser.add_argument(
        "--center_logits",
        dest="no_center_logits",
        action="store_false",
        help="显式启用 logits 扰动去均值；这是默认行为，保留该参数是为了命令清单更直观。",
    )
    parser.add_argument(
        "--no_center_logits",
        dest="no_center_logits",
        action="store_true",
        help="默认对 logits 扰动去均值后再取 L2 norm；传入后恢复直接取 full logits L2 norm。",
    )
    parser.set_defaults(no_center_logits=False)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")
    if args.pair_calibration_source != "long":
        raise ValueError("纯净论文版固定使用 question + long_cot；请不要修改 --pair_calibration_source。")
    if args.num_cal is not None and args.num_cal <= 0:
        raise ValueError("--num_cal must be positive when provided.")
    if not (0 < args.l_percentile <= 100):
        raise ValueError("--l_percentile must be in (0, 100].")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = resolve_model_config(args.model_name)
    steering_path = (
        args.steering_vector
        or infer_steering_vector_from_pairs(args.pairs_path)
        or config.get("steering_vector_path")
    )
    layer_index = args.layer_index if args.layer_index is not None else config.get("layer_index")
    if steering_path is None:
        raise ValueError("无法自动推断 steering vector，请传 --steering_vector。")
    if layer_index is None:
        raise ValueError("无法自动推断 layer index，请传 --layer_index。")

    pairs = load_pairs(args.pairs_path)
    calibration_pairs = select_calibration_pairs(pairs, args.num_cal, args.seed)

    print("ASC 论文 KL gamma 求解")
    print(f"  model:              {args.model_name}")
    print(f"  pairs:              {args.pairs_path}")
    print(f"  valid pairs:        {len(pairs)}")
    print(f"  calibration pairs:  {len(calibration_pairs)}")
    print(f"  steering vector:    {steering_path}")
    print(f"  layer index:        {layer_index}")
    print(f"  epsilon:            {args.epsilon}")
    print(f"  L percentile:       {args.l_percentile}")
    print("  text mode:          long_prompt + long_cot")
    print(f"  attention impl:     {args.attn_impl}")

    configure_attention_for_derivatives(args.attn_impl)

    tokenizer = load_tokenizer(args.model_name, args.tokenizer_mode)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_from_arg(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": True,
        "attn_implementation": args.attn_impl,
    }
    max_memory = parse_max_memory(args.max_memory)
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).eval()
    steering_vec_raw = load_steering_vector(steering_path)

    stats, a_samples, l_samples = estimate_a_and_l(
        model=model,
        tokenizer=tokenizer,
        layer_index=int(layer_index),
        steering_vec_raw=steering_vec_raw,
        calibration_pairs=calibration_pairs,
        num_pairs_available=len(pairs),
        num_requested=args.num_cal,
        max_input_tokens=args.max_input_tokens,
        l_percentile=args.l_percentile,
        curvature_method=args.curvature_method,
        finite_diff_delta=args.finite_diff_delta,
        center_logits=not args.no_center_logits,
    )

    selected = solve_paper_gamma(
        a_value=stats.a_median,
        l_value=stats.l_value,
        epsilon=args.epsilon,
        steering_norm=stats.steering_vector_norm,
    )

    report = {
        "selection_type": "strict_paper_kl_upper_bound",
        "selected": asdict(selected),
        "config": {
            "model_name": args.model_name,
            "pairs_path": args.pairs_path,
            "steering_vector": str(steering_path),
            "layer_index_zero_based": int(layer_index),
            "num_cal": args.num_cal,
            "epsilon": args.epsilon,
            "l_percentile": args.l_percentile,
            "text_mode": "long_prompt+long_cot",
            "max_input_tokens": args.max_input_tokens,
            "seed": args.seed,
            "attn_impl": args.attn_impl,
            "device_map": args.device_map,
            "max_memory": args.max_memory,
            "dtype": args.dtype,
            "tokenizer_mode": args.tokenizer_mode,
            "curvature_method_requested": args.curvature_method,
            "curvature_method_effective": stats.curvature_method,
            "finite_diff_delta": args.finite_diff_delta,
            "center_logits": not args.no_center_logits,
            "paper_reference_gamma": config.get("paper_reference_gamma"),
        },
        "calibration_stats": asdict(stats),
        "a_samples": a_samples,
        "l_samples": l_samples,
        "notes": (
            "这是严格论文 KL 公式得到的理论安全 gamma。它控制的是 next-token "
            "logits 分布的局部 KL 上界，不等同于实际正确率/token 压缩率最优值。"
        ),
    }
    write_json(args.output_path, report)

    print("\n" + "=" * 72)
    print("PAPER KL RESULT")
    print(f"  successful samples:       {stats.num_used}/{len(calibration_pairs)}")
    print(f"  raw steering norm:        {stats.steering_vector_norm:.6f}")
    print(f"  a median:                 {stats.a_median:.6g}")
    print(f"  L p{args.l_percentile:g}:                 {stats.l_value:.6g}")
    print(f"  curvature method:         {stats.curvature_method}")
    print(f"  center logits:            {stats.logit_centering}")
    print(f"  gamma_unit_vector:        {selected.gamma_unit_vector:.10g}")
    print(f"  gamma_for_raw_vector:     {selected.gamma_for_raw_vector:.10g}")
    print(f"  KL upper at gamma:        {selected.kl_upper_at_gamma_unit:.6e}")
    print(f"  cubic root x:             {selected.cubic_root_x:.6g}")
    print(f"  safety factor:            {selected.safety_factor:.6g}")
    if selected.safety_clamped_to_zero:
        print("  warning: safety factor clamped gamma to zero; x >= 4 violates the useful paper regime.")
    print(f"\nSaved report to: {args.output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
