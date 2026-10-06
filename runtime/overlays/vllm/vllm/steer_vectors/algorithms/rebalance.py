# SPDX-License-Identifier: Apache-2.0
"""Dynamic ReBalance direction steering."""

import torch

from vllm.forward_context import get_forward_context

from .direct import DirectAlgorithm
from .factory import register_algorithm


@register_algorithm("rebalance")
class ReBalanceAlgorithm(DirectAlgorithm):
    """Add a direction scaled by the request's online ReBalance state."""

    def __init__(self, *args, normalize: bool = False, **kwargs):
        if normalize:
            raise ValueError("rebalance does not support normalize=True")
        super().__init__(*args, normalize=False, **kwargs)

    def _batch_transform_tensor(
        self, hidden_states, positions_tensor, params, residual=None
    ):
        ctx = get_forward_context()
        scales = ctx.steer_token_scales
        if scales is None:
            raise RuntimeError("rebalance steering requires per-token scales")

        selected = hidden_states.index_select(0, positions_tensor)
        selected_scales = scales.index_select(0, positions_tensor).to(selected.dtype)
        transformed = selected + selected_scales.unsqueeze(1) * params
        hidden_states.index_copy_(0, positions_tensor, transformed)
        return hidden_states
