# SPDX-License-Identifier: Apache-2.0
"""ReBalance's published online confidence controller."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch


@dataclass(frozen=True)
class ReBalanceParams:
    """Parameters and token markers for one ReBalance request."""

    boundary_token_ids: tuple[int, ...]
    think_start_token_id: int
    think_end_token_id: int
    initial_coef: float = -1.0
    q25c: float = 0.65
    q75c: float = 0.90
    low_val_1: float = -1.0
    q25v: float = 0.0005
    q75v: float = 0.01
    low_val_2: float = -2.0
    high_val_2: float = 0.1
    paper_parameters: tuple[float, ...] | None = None
    curve_tau: float = 0.01

    def __post_init__(self):
        if self.paper_parameters is None:
            return
        p = self.paper_parameters
        if len(p) != 5 or not all(math.isfinite(x) for x in p):
            raise ValueError("Paper parameters require Bm, Bo, Bu, eta_c, eta_v")
        if min(p[:3]) < 0 or min(p[3:]) <= 0:
            raise ValueError("Invalid paper amplitudes or gate widths")
        if not 0 <= self.q25c < self.q75c <= 1:
            raise ValueError("Invalid paper confidence thresholds")
        if not 0 <= self.q25v < self.q75v <= 0.25:
            raise ValueError("Invalid paper variance thresholds")
        if self.initial_coef != 0:
            raise ValueError("Paper reconstruction starts without intervention")

    @classmethod
    def from_request(cls, request) -> "ReBalanceParams":
        """Create parameters from an engine-side steering request."""
        boundary_ids = request.rebalance_boundary_token_ids
        start_id = request.rebalance_think_start_token_id
        end_id = request.rebalance_think_end_token_id
        if not boundary_ids or start_id is None or end_id is None:
            raise ValueError("incomplete ReBalance request parameters")
        return cls(
            boundary_token_ids=tuple(boundary_ids),
            think_start_token_id=start_id,
            think_end_token_id=end_id,
            initial_coef=request.rebalance_initial_coef,
            q25c=request.rebalance_q25c,
            q75c=request.rebalance_q75c,
            low_val_1=request.rebalance_low_val_1,
            q25v=request.rebalance_q25v,
            q75v=request.rebalance_q75v,
            low_val_2=request.rebalance_low_val_2,
            high_val_2=request.rebalance_high_val_2,
            paper_parameters=(
                tuple(request.rebalance_paper_parameters)
                if getattr(request, "rebalance_paper_parameters", None) is not None
                else None
            ),
            curve_tau=getattr(request, "rebalance_curve_tau", 0.01),
        )


def compute_paper_coefficient(confidence, variance, params: ReBalanceParams):
    """Literal Appendix B.3 final surface with explicit reconstruction constants."""
    bm, bo, bu, eta_c, eta_v = params.paper_parameters
    over = torch.sigmoid((params.q25c - confidence) / eta_c)
    over *= torch.sigmoid((variance - params.q75v) / eta_v)
    under = torch.sigmoid((confidence - params.q75c) / eta_c)
    under *= torch.sigmoid((params.q25v - variance) / eta_v)
    amplitude = bm + (bo - bm) * over + (bu - bm) * under
    delta = confidence - params.q75c
    return torch.sign(delta) * amplitude * torch.tanh(delta.abs())


def _solve_k_for_tau(
    q25: float,
    q75: float,
    low_val: float,
    high_val: float,
    tau: float,
    iters: int = 80,
) -> float:
    midpoint = 0.5 * (q25 + q75)
    half_width = max(1e-9, 0.5 * (q75 - q25))
    denominator = max(high_val - low_val, 1e-12)
    target = 2.0 * (tau - 0.5 * (low_val + high_val)) / denominator

    def ratio(k: float) -> float:
        numerator = math.tanh(k * max(0.0, 1.0 - midpoint))
        scaled = k * half_width
        denominator = math.tanh(scaled) if scaled > 1e-12 else scaled
        return numerator / denominator if denominator > 0 else math.inf

    low, high = 1e-6, 1.0
    while ratio(high) > target and high < 1e6:
        high *= 2.0
    for _ in range(iters):
        middle = 0.5 * (low + high)
        if ratio(middle) > target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def validate_curve_targets(q25, q75, low_val, tau=0.01):
    """Reject unattainable three-anchor fits before calibration/evaluation."""
    if not all(math.isfinite(x) for x in (q25, q75, low_val, tau)):
        raise ValueError("Nonfinite curve targets")
    if not 0 <= q25 < q75 < 1 or low_val >= 0 or tau <= 0:
        raise ValueError("Curve needs 0 <= q25 < q75 < 1, low < 0, tau > 0")
    ceiling = -low_val * (1 - q75) / (q75 - q25)
    if not tau < ceiling:
        raise ValueError(
            f"Infeasible curve: F(1) target {tau} must be below {ceiling}"
        )
    return ceiling


def _baseline(values, midpoint, k, intercept, slope):
    values = torch.nan_to_num(
        values, nan=0.5, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    return intercept + slope * torch.tanh(k * (values - midpoint))


@lru_cache(maxsize=128)
def _curve_constants(q25c, q75c, low_val, device, dtype, tau=0.01):
    """Cache only fixed curve values, separately for each device and dtype."""
    high_val_1 = tau
    # Author build_F omits high_val (default 0.0); 0.01 is tau, not high_val.
    curve_high = 0.0

    midpoint = 0.5 * (q25c + q75c)
    half_width = max(1e-9, 0.5 * (q75c - q25c))
    k = _solve_k_for_tau(
        q25c,
        q75c,
        low_val,
        curve_high,
        high_val_1,
    )
    span = math.tanh(k * half_width)
    intercept = 0.5 * (low_val + curve_high)
    slope = (curve_high - low_val) / (2.0 * max(span, 1e-12))

    # Preserve the original scalar tensor operations and their rounding.
    with torch.no_grad():
        at_q25 = _baseline(
            torch.tensor(q25c, device=device, dtype=dtype),
            midpoint, k, intercept, slope,
        )
        at_one = _baseline(
            torch.tensor(1.0, device=device, dtype=dtype),
            midpoint, k, intercept, slope,
        )
    return midpoint, k, intercept, slope, at_q25, at_one


def compute_rebalance_coefficient(
    confidence: torch.Tensor,
    variance: torch.Tensor,
    params: ReBalanceParams,
) -> torch.Tensor:
    """Map step confidence and the two-step variance proxy to a coefficient."""
    q25c, q75c = sorted((params.q25c, params.q75c))
    q25v, q75v = sorted((params.q25v, params.q75v))
    high_val_1 = 0.01
    midpoint, k, intercept, slope, at_q25, at_one = _curve_constants(
        q25c, q75c, params.low_val_1, confidence.device, confidence.dtype,
        params.curve_tau,
    )

    def baseline(values: torch.Tensor) -> torch.Tensor:
        return _baseline(values, midpoint, k, intercept, slope)

    iqrc = max(1e-12, q75c - q25c)
    iqrv = max(1e-12, q75v - q25v)

    def sigmoid(values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(values.clamp(-60.0, 60.0))

    low_gate = sigmoid((q25c - confidence) / iqrc * 1200.0)
    high_gate = sigmoid((confidence - q75c) / iqrc * 1200.0)
    high_variance_gate = sigmoid((variance - q75v) / iqrv * 1200.0)
    low_variance_gate = sigmoid((q25v - variance) / iqrv * 1200.0)

    def scalar_sigmoid(value: float) -> float:
        value = max(min(value, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-value))

    low_normalizer = scalar_sigmoid(0.0) * scalar_sigmoid(0.0)
    high_normalizer = scalar_sigmoid(
        (1.0 - q75c) / iqrc * 12.0
    ) * scalar_sigmoid(0.0)
    low_weight = (low_gate * high_variance_gate / low_normalizer).clamp(max=1.0)
    high_weight = (
        high_gate * low_variance_gate / high_normalizer
    ).clamp(max=1.0)

    updated = baseline(confidence)
    updated += (params.low_val_2 - at_q25) * low_weight
    updated += (params.high_val_2 - at_one) * high_weight
    return updated.clamp(
        min=min(params.low_val_2, params.low_val_1),
        max=max(high_val_1, params.high_val_2),
    )
