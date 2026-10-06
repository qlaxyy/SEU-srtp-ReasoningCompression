# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steer vector support for the V2 GPU model runner."""

import numpy as np
import torch

from vllm.steer_vectors import trace
from vllm.steer_vectors.rebalance import (
    ReBalanceParams,
    compute_rebalance_coefficient,
    compute_paper_coefficient,
)
from vllm.steer_vectors.request import SteerVectorRequest
from vllm.v1.worker.gpu.input_batch import InputBatch


class SteerVectorState:
    """Per-request steer vector bookkeeping for the V2 model runner.

    Each live request resolves to a config slot at admission time
    (payload loading + layer distribution happen there, never in the
    forward pass).
    """

    def __init__(
        self,
        max_num_reqs: int | None = None,
        device: torch.device | None = None,
        max_model_len: int | None = None,
    ) -> None:
        self._requests: dict[str, SteerVectorRequest] = {}
        self._slots: dict[str, int] = {}
        self._dynamic_params: dict[str, ReBalanceParams] = {}
        self._dynamic_indices: dict[str, int] = {}
        self._boundary_tensors: dict[ReBalanceParams, torch.Tensor] = {}
        self._position_tensors: dict[
            ReBalanceParams, tuple[tuple[int, ...], torch.Tensor]
        ] = {}
        self._coefs: torch.Tensor | None = None
        self._step_prob_sum: torch.Tensor | None = None
        self._step_tok_count: torch.Tensor | None = None
        self._prev_step_mean: torch.Tensor | None = None
        self._in_think: torch.Tensor | None = None
        self._history = None
        self._history_lengths: dict[str, int] = {}
        self._prompt_lengths: dict[str, int] = {}
        self._suspended: dict[str, dict] = {}
        self.replay_counts = {"suspended": 0, "restored": 0}
        if max_num_reqs is not None and device is not None:
            self._allocate_dynamic_state(max_num_reqs, device)
            if max_model_len is not None:
                self._history = torch.ones(
                    (max_num_reqs, max_model_len + 1),
                    dtype=torch.float32, device=device,
                )

    @property
    def supports_kv_replay(self) -> bool:
        return self._history is not None

    def _state_fields(self):
        return (self._coefs, self._step_prob_sum, self._step_tok_count,
                self._prev_step_mean, self._in_think, self._paper_strength,
                self._paper_pending)

    def suspend_request(self, req_id: str) -> None:
        """Retain controller state and historical input scales before KV eviction."""
        idx = self._dynamic_indices.get(req_id)
        if idx is None:
            return
        if not self.supports_kv_replay:
            raise RuntimeError("Dynamic KV eviction requires steering history")
        length = self._history_lengths[req_id]
        self._suspended[req_id] = dict(
            params=self._dynamic_params[req_id],
            prompt_length=self._prompt_lengths[req_id],
            history=self._history[idx, :length].cpu().clone(),
            fields=[field[idx].cpu().clone() for field in self._state_fields()],
        )
        self.replay_counts["suspended"] += 1

    def discard_suspended(self, req_id: str) -> None:
        self._suspended.pop(req_id, None)

    def _allocate_dynamic_state(
        self, max_num_reqs: int, device: torch.device
    ) -> None:
        self._coefs = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        self._step_prob_sum = torch.zeros_like(self._coefs)
        self._step_tok_count = torch.zeros(
            max_num_reqs, dtype=torch.long, device=device
        )
        self._prev_step_mean = torch.full_like(self._coefs, torch.nan)
        self._in_think = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        self._paper_strength = torch.zeros_like(self._coefs)
        self._paper_pending = torch.zeros_like(self._in_think)

    def add_request(
        self,
        req_id: str,
        steer_vector_request: SteerVectorRequest | None,
        manager,
        *,
        req_index: int | None = None,
        prompt_token_ids: list[int] | None = None,
        num_generated_tokens: int = 0,
    ) -> None:
        if steer_vector_request is None:
            return
        if manager is None:
            # Admission should have rejected this; routing the request to
            # the server config (or to nothing) would silently steer with
            # the wrong vector.
            raise RuntimeError(
                f"request {req_id} carries a steering config but this "
                "worker has no steer vector manager (engine launched "
                "without enable_steer_vector=True)"
            )
        self._requests[req_id] = steer_vector_request
        self._slots[req_id] = manager.acquire_config(req_id, steer_vector_request)
        if steer_vector_request.algorithm != "rebalance":
            return
        if req_index is None or self._coefs is None:
            raise RuntimeError("rebalance requires initialized request-index state")
        params = ReBalanceParams.from_request(steer_vector_request)
        self._dynamic_params[req_id] = params
        self._dynamic_indices[req_id] = req_index
        self._coefs[req_index] = params.initial_coef
        self._step_prob_sum[req_index] = 0.0
        self._step_tok_count[req_index] = 0
        self._prev_step_mean[req_index] = torch.nan
        self._paper_strength[req_index] = 0.0
        self._paper_pending[req_index] = True
        prompt_token_ids = prompt_token_ids or []
        self._in_think[req_index] = params.think_start_token_id in prompt_token_ids
        self._prompt_lengths[req_id] = len(prompt_token_ids)
        self._history_lengths[req_id] = len(prompt_token_ids)
        if self._history is not None:
            self._history[req_index].zero_()
        saved = self._suspended.pop(req_id, None)
        if saved is not None:
            length = len(saved["history"])
            if (saved["params"] != params
                    or saved["prompt_length"] != len(prompt_token_ids)
                    or length != len(prompt_token_ids) + num_generated_tokens):
                raise RuntimeError("Resumed ReBalance request/history mismatch")
            self._history[req_index, :length].copy_(saved["history"])
            self._history_lengths[req_id] = length
            for field, value in zip(self._state_fields(), saved["fields"]):
                field[req_index].copy_(value)
            self.replay_counts["restored"] += 1
        elif num_generated_tokens:
            raise RuntimeError("Generated prefix has no saved ReBalance history")

    def remove_request(self, req_id: str, manager) -> None:
        if self._requests.pop(req_id, None) is None:
            return
        self._slots.pop(req_id, None)
        params = self._dynamic_params.pop(req_id, None)
        if params is not None and params not in self._dynamic_params.values():
            self._position_tensors.pop(params, None)
        req_index = self._dynamic_indices.pop(req_id, None)
        self._prompt_lengths.pop(req_id, None)
        self._history_lengths.pop(req_id, None)
        if req_index is not None and self._coefs is not None:
            if self._history is not None:
                self._history[req_index].fill_(1.0)
            self._coefs[req_index] = 0.0
            self._step_prob_sum[req_index] = 0.0
            self._step_tok_count[req_index] = 0
            self._prev_step_mean[req_index] = torch.nan
            self._in_think[req_index] = False
            self._paper_strength[req_index] = 0.0
            self._paper_pending[req_index] = False
        if manager is not None:
            manager.release_config(req_id)

    def slot_of(self, req_id: str) -> int:
        return self._slots.get(req_id, -1)

    def has_routed(self) -> bool:
        return bool(self._slots)

    def has_dynamic(self) -> bool:
        return bool(self._dynamic_params)

    def _group_batch_positions(
        self, req_ids: list[str]
    ) -> dict[ReBalanceParams, list[int]]:
        groups: dict[ReBalanceParams, list[int]] = {}
        for position, req_id in enumerate(req_ids):
            params = self._dynamic_params.get(req_id)
            if params is not None:
                groups.setdefault(params, []).append(position)
        return groups

    def _positions_tensor(
        self, params: ReBalanceParams, positions: list[int], device: torch.device
    ) -> torch.Tensor:
        """Reuse batch positions; state indices still follow the live mapping."""
        key = tuple(positions)
        cached = self._position_tensors.get(params)
        if cached is None or cached[0] != key or cached[1].device != device:
            tensor = torch.tensor(positions, dtype=torch.long, device=device)
            self._position_tensors[params] = (key, tensor)
            return tensor
        return cached[1]

    def batch_scales(self, input_batch: InputBatch) -> torch.Tensor:
        """Return one online coefficient per request in scheduler order."""
        device = input_batch.idx_mapping.device
        scales = torch.ones(input_batch.num_reqs, dtype=torch.float32, device=device)
        if not self.has_dynamic():
            return scales
        assert self._coefs is not None and self._in_think is not None
        for params, positions in self._group_batch_positions(
            input_batch.req_ids
        ).items():
            pos = self._positions_tensor(params, positions, device)
            state_idx = input_batch.idx_mapping.index_select(0, pos).long()
            active_scales = self._coefs.index_select(0, state_idx)
            active_scales *= self._in_think.index_select(0, state_idx)
            scales.index_copy_(0, pos, active_scales)
        return scales

    def token_scales(self, input_batch: InputBatch) -> torch.Tensor:
        """Expand request coefficients to the flattened input-token rows."""
        if self._history is not None:
            indices = torch.repeat_interleave(
                input_batch.idx_mapping.long(),
                torch.diff(input_batch.query_start_loc[:input_batch.num_reqs + 1]),
                output_size=input_batch.num_tokens,
            )
            return self._history[
                indices, input_batch.positions[:input_batch.num_tokens].long()
            ]
        request_scales = self.batch_scales(input_batch)
        repeats = torch.diff(
            input_batch.query_start_loc[: input_batch.num_reqs + 1]
        ).long()
        return torch.repeat_interleave(request_scales, repeats)

    @staticmethod
    def _is_boundary(tokens: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        """Match without isin's data-dependent unique/scalar synchronization."""
        return (tokens[:, None] == boundaries).any(dim=-1)

    def observe_sample(
        self,
        input_batch: InputBatch,
        sampled_token_ids: torch.Tensor,
        max_probabilities: torch.Tensor,
    ) -> None:
        """Update each request after sampling using device-resident state."""
        if not self.has_dynamic():
            return
        if input_batch.num_draft_tokens != 0:
            raise RuntimeError("rebalance does not support speculative decoding")
        if sampled_token_ids.shape != (input_batch.num_reqs, 1):
            raise RuntimeError(
                "rebalance expects exactly one sampled token per request"
            )
        if max_probabilities.shape != (input_batch.num_reqs,):
            raise RuntimeError("rebalance confidence rows do not match the batch")

        assert self._coefs is not None
        assert self._step_prob_sum is not None
        assert self._step_tok_count is not None
        assert self._prev_step_mean is not None
        assert self._in_think is not None
        device = sampled_token_ids.device
        sampled = sampled_token_ids[:, 0]

        for params, positions in self._group_batch_positions(
            input_batch.req_ids
        ).items():
            if hasattr(input_batch, "prefill_len_np"):
                # Partial prefill/replay rows have dummy samples: never count them.
                positions = [p for p in positions if (
                    input_batch.num_computed_tokens_np[p]
                    + input_batch.num_scheduled_tokens[p]
                    >= input_batch.prefill_len_np[p])]
                if not positions:
                    continue
            pos = self._positions_tensor(params, positions, device)
            state_idx = input_batch.idx_mapping.index_select(0, pos).long()
            tokens = sampled.index_select(0, pos)
            probabilities = max_probabilities.index_select(0, pos)
            boundaries = self._boundary_tensors.get(params)
            if boundaries is None:
                boundaries = torch.tensor(
                    params.boundary_token_ids,
                    dtype=tokens.dtype,
                    device=device,
                )
                self._boundary_tensors[params] = boundaries
            is_boundary = self._is_boundary(tokens, boundaries)

            if params.paper_parameters is not None:
                self._observe_paper(
                    state_idx, tokens, probabilities, is_boundary, params
                )
                self._record_scales(input_batch, positions, pos, state_idx)
                continue

            in_think = self._in_think.index_select(0, state_idx)
            in_think |= tokens == params.think_start_token_id
            in_think &= tokens != params.think_end_token_id
            self._in_think.index_copy_(0, state_idx, in_think)

            not_boundary = ~is_boundary
            sums = self._step_prob_sum.index_select(0, state_idx)
            counts = self._step_tok_count.index_select(0, state_idx)
            sums += probabilities * not_boundary
            counts += not_boundary
            self._step_prob_sum.index_copy_(0, state_idx, sums)
            self._step_tok_count.index_copy_(0, state_idx, counts)

            ready = is_boundary & (counts > 0)
            step_means = sums / counts.clamp_min(1).float()
            previous = self._prev_step_mean.index_select(0, state_idx)
            variance = torch.where(
                torch.isfinite(previous),
                (step_means - previous).square() / 4.0,
                torch.zeros_like(step_means),
            )
            updated = compute_rebalance_coefficient(
                step_means, variance, params
            )
            current = self._coefs.index_select(0, state_idx)
            self._coefs.index_copy_(0, state_idx, torch.where(ready, updated, current))
            self._step_prob_sum.index_copy_(
                0, state_idx, torch.where(ready, torch.zeros_like(sums), sums)
            )
            self._step_tok_count.index_copy_(
                0, state_idx, torch.where(ready, torch.zeros_like(counts), counts)
            )
            self._prev_step_mean.index_copy_(
                0,
                state_idx,
                torch.where(ready, step_means.detach(), previous),
            )
            self._record_scales(input_batch, positions, pos, state_idx)

    def _record_scales(self, batch, positions, pos, idx):
        if self._history is None:
            return
        # The sampled token is the next input, at the current sequence length.
        token_pos = batch.seq_lens.index_select(0, pos).long()
        self._history[idx, token_pos] = (
            self._coefs.index_select(0, idx) * self._in_think.index_select(0, idx)
        )
        for p in positions:
            self._history_lengths[batch.req_ids[p]] = int(
                batch.num_computed_tokens_np[p] + batch.num_scheduled_tokens[p] + 1
            )

    def _observe_paper(self, idx, tokens, probabilities, boundary, params):
        """Store a scale only for the next first-content input token."""
        active = self._in_think.index_select(0, idx)
        active &= tokens != params.think_end_token_id
        content = active & ~boundary
        pending = self._paper_pending.index_select(0, idx)
        strength = self._paper_strength.index_select(0, idx)
        self._coefs.index_copy_(
            0, idx, torch.where(content & pending, strength, 0.0)
        )
        pending = torch.where(boundary, True, pending) & ~content & active
        sums = self._step_prob_sum.index_select(0, idx)
        counts = self._step_tok_count.index_select(0, idx)
        logp = probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        sums += torch.where(content, logp, 0.0)
        counts += content
        ready = boundary & active & (counts > 0)
        confidence = torch.exp(sums / counts.clamp_min(1))
        previous = self._prev_step_mean.index_select(0, idx)
        variance = torch.where(
            torch.isfinite(previous), (confidence - previous).square() / 4, 0.0
        )
        updated = compute_paper_coefficient(confidence, variance, params)
        reset = ready | ~active
        self._in_think.index_copy_(0, idx, active)
        self._paper_pending.index_copy_(0, idx, pending)
        self._paper_strength.index_copy_(
            0, idx, torch.where(ready, updated, strength)
        )
        self._step_prob_sum.index_copy_(0, idx, torch.where(reset, 0.0, sums))
        self._step_tok_count.index_copy_(0, idx, torch.where(reset, 0, counts))
        self._prev_step_mean.index_copy_(
            0, idx, torch.where(ready, confidence, previous)
        )


def build_batch_geometry(
    input_batch: InputBatch, state: SteerVectorState | None = None
) -> "BatchGeometry":
    """Build the per-step BatchGeometry from the runner's InputBatch.

    The single producer of batch geometry: steering triggers, capture
    row selection/labels, and the full-graph buffer filler all consume
    this object (directly or via `geometry_samples_info`).
    """
    from vllm.forward_context import BatchGeometry

    num_reqs = input_batch.num_reqs
    num_computed = input_batch.num_computed_tokens_np[:num_reqs]
    prefill_len = input_batch.prefill_len_np[:num_reqs]
    if state is not None and state.has_dynamic():
        # A resumed prefill contains generated tokens. Keep the original boundary.
        prefill_len = np.array([
            state._prompt_lengths.get(req_id, int(prefill_len[i]))
            for i, req_id in enumerate(input_batch.req_ids)
        ], dtype=np.int32)
    is_prefilling = input_batch.is_prefilling_np[:num_reqs]
    # While prefilling, nothing has been generated for this request yet.
    # During decode, the scheduler has computed prefill_len + (k - 1) tokens
    # when the k-th output token is being generated, matching the V1
    # semantics of len(output_token_ids) at execute time.
    num_output = np.where(is_prefilling, 0, num_computed - prefill_len + 1).astype(
        np.int32
    )
    return BatchGeometry(
        query_start_loc=input_batch.query_start_loc[: num_reqs + 1],
        num_computed=torch.from_numpy(np.ascontiguousarray(num_computed)),
        num_prompt=torch.from_numpy(np.ascontiguousarray(prefill_len)),
        num_output=torch.from_numpy(num_output),
        req_ids=list(input_batch.req_ids[:num_reqs]),
        token_ids=input_batch.input_ids[: input_batch.num_tokens],
        query_start_loc_cpu=input_batch.query_start_loc_np[: num_reqs + 1],
    )


def _batch_token_slots(
    input_batch: InputBatch, state: SteerVectorState, default_slot: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-request and per-token config-slot routing for this step."""
    num_reqs = input_batch.num_reqs
    slots_np = np.fromiter(
        (
            slot if (slot := state.slot_of(req_id)) >= 0 else default_slot
            for req_id in input_batch.req_ids
        ),
        dtype=np.int32,
        count=num_reqs,
    )
    token_slots_np = np.repeat(slots_np, input_batch.num_scheduled_tokens[:num_reqs])
    return slots_np, token_slots_np


def _match_positions_np(
    abs_pos: np.ndarray, positions, neg_base: np.ndarray, is_dec: np.ndarray
) -> np.ndarray:
    """Mask of prompt tokens at the given positions (numpy mirror of
    clause._match_positions): negative entries index from each sample's
    prompt length; positive entries past the prompt end clamp to the
    last prompt token; decode tokens never match."""
    mask = np.zeros(abs_pos.shape[0], dtype=bool)
    for p in positions:
        if p < 0:
            mask |= abs_pos == neg_base + p
        else:
            mask |= abs_pos == np.minimum(neg_base - 1, p)
    return mask & ~is_dec


def _match_prompt_window_np(
    abs_pos: np.ndarray, window, neg_base: np.ndarray, is_dec: np.ndarray
) -> np.ndarray:
    """Mask of prompt tokens inside the half-open window (numpy mirror
    of clause._match_prompt_window): negative bounds resolve from each
    sample's prompt length; stop=None means the prompt end."""
    start, stop = window
    lo = neg_base + start if start < 0 else start
    hi = neg_base if stop is None else (neg_base + stop if stop < 0 else stop)
    return ~is_dec & (abs_pos >= lo) & (abs_pos < hi)


def _match_generation_steps_np(
    gen_idx: np.ndarray, is_dec: np.ndarray, steps, window
) -> np.ndarray:
    """Mask of generation tokens at the given 0-based decode steps
    and/or inside the half-open decode-step window (numpy mirror of
    clause._match_generation_steps)."""
    mask = np.zeros(gen_idx.shape[0], dtype=bool)
    if steps is not None:
        mask |= np.isin(gen_idx, np.asarray(list(steps), dtype=np.int64))
    if window is not None:
        start, stop = window
        in_window = gen_idx >= start
        if stop is not None:
            in_window &= gen_idx < stop
        mask |= in_window
    return mask & is_dec


def _clause_mask_np(
    clause: dict,
    is_dec: np.ndarray,
    abs_pos: np.ndarray,
    neg_base: np.ndarray,
    gen_idx: np.ndarray,
    token_ids,
) -> np.ndarray:
    """Evaluate one where-clause over a slot's tokens (numpy mirror of
    clause.collect_positions_apply_spec — same union/veto semantics).

    `token_ids` is a thunk: only token-id filters pay for the host copy.
    """
    n = is_dec.shape[0]

    def _selector_mask(
        prompt_tokens,
        prompt_positions,
        prompt_window,
        generation_tokens,
        generation_positions,
        generation_window,
    ) -> np.ndarray:
        matched = np.zeros(n, dtype=bool)
        if prompt_tokens is not None:
            matched |= (
                np.isin(token_ids(), np.asarray(list(prompt_tokens))) & ~is_dec
            )
        if generation_tokens is not None:
            matched |= (
                np.isin(token_ids(), np.asarray(list(generation_tokens))) & is_dec
            )
        if prompt_positions is not None:
            matched |= _match_positions_np(
                abs_pos, prompt_positions, neg_base, is_dec
            )
        if prompt_window is not None:
            matched |= _match_prompt_window_np(abs_pos, prompt_window, neg_base, is_dec)
        if generation_positions is not None or generation_window is not None:
            matched |= _match_generation_steps_np(
                gen_idx, is_dec, generation_positions, generation_window
            )
        return matched

    mask = np.zeros(n, dtype=bool)
    if clause.get("prompt") == "all":
        mask |= ~is_dec
    if clause.get("generation") == "all":
        mask |= is_dec

    from vllm.steer_vectors.algorithms.clause import _EXCLUDE_KEYS, _INCLUDE_KEYS

    includes = tuple(clause.get(key) for key in _INCLUDE_KEYS)
    if any(value is not None for value in includes):
        mask |= _selector_mask(*includes)

    excludes = tuple(clause.get(key) for key in _EXCLUDE_KEYS)
    if any(value is not None for value in excludes):
        mask &= ~_selector_mask(*excludes)
    return mask


def resolve_slot_positions(
    slot_clauses: dict[int, list[dict | None]],
    active_slots: list[int],
    token_slots_np: np.ndarray,
    device: torch.device,
    geo,
) -> dict[tuple, torch.Tensor | None]:
    """Resolve every active clause's steered positions, once per step.

    Where-clauses are layer-invariant, so this single resolution serves
    every decoder/MoE-gate hook (and the Tier-1 mask filler). Keys are
    (slot, clause_cache_key); a None value means the clause matched no
    token this step.

    Resolution runs host-side in one numpy pass: clauses match phases,
    positions and windows — all host-known geometry — so each slot's
    clauses are evaluated only over that slot's own token rows and the
    matched positions ship to the device in a single copy. Per-step cost
    scales with the batch's tokens, not with the number of distinct live
    configurations. Only token-id filters read the input ids (one cached
    device-to-host copy per step).
    """
    from vllm.steer_vectors.algorithms.clause import (
        clause_cache_key,
        selects_all_tokens,
    )

    resolved: dict[tuple, torch.Tensor | None] = {}
    if not active_slots:
        return resolved

    qsl = geo.query_start_loc_cpu
    assert qsl is not None, "BatchGeometry is missing its host query_start_loc"
    num_computed = geo.num_computed.numpy()
    num_prompt = geo.num_prompt.numpy()
    lens = (qsl[1:] - qsl[:-1]).astype(np.int64)
    starts_all = qsl[:-1].astype(np.int64)

    # Group batch requests by routing slot (a request's slot is its
    # first token's slot; all its tokens share it).
    active = set(active_slots)
    slot_reqs: dict[int, list[int]] = {}
    for r, s in enumerate(token_slots_np[starts_all].tolist()):
        if s in active:
            slot_reqs.setdefault(s, []).append(r)

    keys: list[tuple[int, tuple]] = []
    chunks: list[np.ndarray] = []

    for slot in active_slots:
        reqs = slot_reqs.get(slot)
        clauses = slot_clauses.get(slot, [])
        if not reqs:
            for clause in clauses:
                key = clause_cache_key(clause)
                if key is not None:
                    resolved.setdefault((slot, key), None)
            continue
        reqs_np = np.asarray(reqs, dtype=np.int64)
        seg_lens = lens[reqs_np]
        n = int(seg_lens.sum())
        samp = np.repeat(reqs_np, seg_lens)
        within = np.arange(n, dtype=np.int64) - np.repeat(
            np.cumsum(seg_lens) - seg_lens, seg_lens
        )
        tok_idx = np.repeat(starts_all[reqs_np], seg_lens) + within
        abs_pos = within + num_computed[samp]
        is_dec = abs_pos >= num_prompt[samp]

        for clause in clauses:
            key = clause_cache_key(clause)
            if key is None or (slot, key) in resolved:
                continue
            if selects_all_tokens(clause):
                pos_np = tok_idx
            else:
                mask = _clause_mask_np(
                    clause,
                    is_dec,
                    abs_pos,
                    num_prompt[samp],
                    abs_pos - num_prompt[samp],
                    lambda: geo.token_ids_cpu()[tok_idx],
                )
                pos_np = tok_idx[mask]
            if pos_np.shape[0] == 0:
                resolved[(slot, key)] = None
            else:
                resolved[(slot, key)] = pos_np  # placeholder, replaced below
                keys.append((slot, key))
                chunks.append(pos_np)

    if chunks:
        flat = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        flat_t = torch.from_numpy(flat).to(device, non_blocking=True)
        offset = 0
        for key, chunk in zip(keys, chunks):
            size = chunk.shape[0]
            resolved[key] = flat_t[offset : offset + size]
            offset += size
    return resolved


def make_steer_vector_forward_kwargs(
    input_batch: InputBatch,
    state: SteerVectorState | None = None,
    default_slot: int = -1,
    manager=None,
) -> dict:
    """Build the ForwardContext fields consumed by steering and capture.

    - batch_geometry: the per-step BatchGeometry (see build_batch_geometry)
    - steer_token_slots / steer_active_slots: per-request config routing
      (only when routed configs are live)
    - steer_slot_positions: per-clause steered positions, resolved once
      here and consumed by every layer hook (see resolve_slot_positions)

    `default_slot` is the server-level config's slot (-1 when absent);
    requests without their own steering config are routed to it.
    """
    num_reqs = input_batch.num_reqs
    geo = build_batch_geometry(input_batch, state)
    kwargs = {"batch_geometry": geo}

    if state is not None and (state.has_routed() or default_slot >= 0):
        slots_np, token_slots_np = _batch_token_slots(input_batch, state, default_slot)
        token_slots = torch.from_numpy(token_slots_np).to(
            input_batch.input_ids.device, non_blocking=True
        )
        active_slots = sorted({int(s) for s in slots_np if s >= 0})
        kwargs["steer_token_slots"] = token_slots
        if state.has_dynamic():
            kwargs["steer_token_scales"] = state.token_scales(input_batch)
        kwargs["steer_active_slots"] = active_slots
        if manager is None:
            raise RuntimeError(
                "steering slots are routed but no worker manager was passed "
                "to make_steer_vector_forward_kwargs"
            )
        kwargs["steer_slot_positions"] = resolve_slot_positions(
            manager.slot_clauses(),
            active_slots,
            token_slots_np,
            token_slots.device,
            geo,
        )

        if trace.enabled():
            trace.begin_step(
                req_ids=input_batch.req_ids,
                slots=slots_np.tolist(),
                query_start_loc=input_batch.query_start_loc_np[: num_reqs + 1].tolist(),
                token_ids=geo.token_ids.cpu().tolist(),
                num_computed=geo.num_computed.tolist(),
                num_output=geo.num_output.tolist(),
            )
    return kwargs


def fill_graph_steer_buffers(
    input_batch: InputBatch,
    state: SteerVectorState | None,
    manager,
) -> None:
    """Fill Tier-1 persistent buffers for this step (full-graph mode).

    Writes each token's vector-table row into the shared row buffer and
    sets the per-layer trigger masks to 1 at steered positions. The
    captured kernel `hidden += mask * vectors[row_tok]` then applies the
    right configs without any per-step graph work; row 0 / mask 0 keep
    unsteered and padding tokens untouched. Positions come from the same
    resolver the layer hooks use (resolve_slot_positions).
    """
    from vllm.steer_vectors.algorithms.clause import clause_cache_key

    manager.zero_graph_masks()
    row_buf = manager.token_rows_buf
    row_buf.zero_()
    entries = manager.graph_batch_entries()
    if not entries or state is None:
        return

    slots_np, token_slots_np = _batch_token_slots(
        input_batch, state, manager.server_slot
    )
    token_scales = state.token_scales(input_batch) if state.has_dynamic() else None
    rows_np = np.fromiter(
        (entries[s][0] if s in entries else 0 for s in slots_np),
        dtype=np.int64,
        count=slots_np.shape[0],
    )
    num_scheduled = input_batch.num_scheduled_tokens[: slots_np.shape[0]]
    token_rows_np = np.repeat(rows_np, num_scheduled)
    n = token_rows_np.shape[0]
    if n == 0:
        return
    device = row_buf.device
    row_buf[:n].copy_(torch.from_numpy(token_rows_np).to(device, non_blocking=True))

    # Batch geometry for the trigger collector: the same object the
    # forward context carries, from the single producer.
    geo = build_batch_geometry(input_batch, state)
    batch_slots = set(slots_np.tolist())
    active_slots = sorted(s for s in entries if s in batch_slots)
    resolved = resolve_slot_positions(
        manager.slot_clauses(), active_slots, token_slots_np, device, geo
    )
    from vllm.steer_vectors.algorithms import get_algorithm
    from vllm.steer_vectors.graph_kernels import graph_family_mask_attr

    # One scatter per (module, mask attr), not per slot: slots sharing a
    # layer contribute to the same mask write, so the launch count scales
    # with steered layers, not with live configurations.
    mask_writes: dict[tuple[int, str], tuple] = {}
    for slot in active_slots:
        _, request, controllers = entries[slot]
        positions = resolved[(slot, clause_cache_key(request.apply_spec))]
        if positions is None:
            continue
        # Families whose delta a zero table row cannot neutralize (e.g.
        # replace) carry their own mask; see GRAPH_FAMILY_MASKS.
        mask_attr = graph_family_mask_attr(
            get_algorithm(request.algorithm).graph_family
        )
        for module in controllers:
            mask_writes.setdefault(
                (id(module), mask_attr), (module, mask_attr, [])
            )[2].append(
                (
                    positions,
                    None
                    if request.algorithm != "rebalance"
                    else token_scales.index_select(0, positions),
                )
            )
    for module, mask_attr, position_entries in mask_writes.values():
        position_list = [entry[0] for entry in position_entries]
        positions = (
            position_list[0]
            if len(position_list) == 1
            else torch.cat(position_list)
        )
        mask = getattr(module, mask_attr)
        if any(values is not None for _, values in position_entries):
            value_list = [
                torch.ones(len(pos), dtype=mask.dtype, device=mask.device)
                if values is None
                else values.to(mask.dtype)
                for pos, values in position_entries
            ]
            values = value_list[0] if len(value_list) == 1 else torch.cat(value_list)
            mask.index_copy_(0, positions, values)
        else:
            mask[positions] = 1.0
