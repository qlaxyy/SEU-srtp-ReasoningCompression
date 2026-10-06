"""Instance-only, device-resident sampler adapter for the pinned V2 runner.

No KV forks or per-token CPU output handling. Async readiness remains subject to
the explicit native gate; a CPU reference check is not GPU validation.
"""
import hashlib
import math

from policy import PENALTY, token_flags, trigger_vocabulary, validate_penalty, device_penalty


class Adapter:
    def __init__(self, llm, tokenizer, mode='off', gate_on=False, *, trigger_profile='original14', history_gate='none', penalty_mode='fixed', lower_bound=None, constant_scale=None, lexical_mode='original'):
        self.mode = mode
        self.enabled = mode != 'off'
        if mode not in ('off', 'shadow', 'negative', 'always'):
            raise ValueError(mode)
        if not self.enabled:
            return  # Strict default-off path: install no hooks or device buffers.
        if not gate_on:
            raise ValueError('Explicit experimental gate_on is required')
        if lexical_mode not in ('original','but_wait'):
            raise ValueError('Unknown lexical mode')
        if lexical_mode != 'original' and (mode != 'negative' or trigger_profile != 'original14' or history_gate != 'none' or penalty_mode != 'fixed'):
            raise ValueError('Phrase candidate requires fixed original14 negative mode without other variants')
        self.lexical_mode=lexical_mode
        validate_penalty(penalty_mode, lower_bound, constant_scale)
        if penalty_mode != 'fixed' and (mode not in ('negative','shadow') or trigger_profile != 'original14' or history_gate != 'none'):
            raise ValueError('Penalty candidate requires original14, negative/shadow, no history gate')
        self.penalty_mode, self.lower_bound, self.constant_scale = penalty_mode, lower_bound, constant_scale
        triggers = trigger_vocabulary(trigger_profile)
        self.trigger_profile = trigger_profile
        if history_gate not in ('none','after_first_reflection'):
            raise ValueError('Unknown history gate')
        if history_gate != 'none' and (trigger_profile != 'original14' or mode not in ('negative','shadow')):
            raise ValueError('History candidate requires original14 and negative/shadow mode')
        self.history_gate = history_gate
        import torch
        self.torch = torch
        self.core = llm.llm_engine.engine_core.engine_core
        self.runner = self.core.model_executor.driver_worker.worker.model_runner
        cfg = self.runner.vllm_config
        if (cfg.parallel_config.tensor_parallel_size != 1 or cfg.speculative_config is not None
                or cfg.cache_config.enable_prefix_caching
                or getattr(self.runner, 'hybrid_termination', None) is not None):
            raise ValueError('Requires TP1, no speculation/prefix cache/S64')
        self.original_sampler = self.runner.sampler
        self.original_add = self.runner.add_requests
        self.original_remove = self.runner._remove_request
        self.original_preempt = self.core.scheduler._preempt_request
        self.active, self.completed = {}, {}
        n, device = self.runner.max_num_reqs, self.runner.device
        self.opening = torch.zeros(n, dtype=torch.bool, device=device)
        self.thinking = torch.zeros_like(self.opening)
        self.count = torch.zeros(n, dtype=torch.int64, device=device)
        self.prompt_len = torch.zeros_like(self.count)
        self.eligible_count = torch.zeros_like(self.count)
        self.changed_count = torch.zeros_like(self.count)
        self.first_change = torch.full_like(self.count, -1)
        size = max(tokenizer.get_vocab().values()) + 1
        clean, white = [False]*size, [False]*size
        boundaries = {i for piece,i in tokenizer.get_vocab().items() if 'ĊĊ' in piece}
        for i in range(size):
            clean[i], white[i] = token_flags(tokenizer.decode([i]), i in boundaries)
        boundary = [i in boundaries for i in range(size)]
        self.clean = torch.tensor(clean, device=device)
        self.white = torch.tensor(white, device=device)
        self.boundary = torch.tensor(boundary, device=device)
        self.ids = torch.tensor(list(triggers), device=device)
        if lexical_mode == 'but_wait':
            self.after_but = torch.zeros_like(self.opening)
            self.but_lookup = torch.zeros(size,dtype=torch.bool,device=device)
            self.but_lookup[torch.tensor([3983,1988,8088,714],device=device)] = True
            self.but_columns = torch.tensor([i in (3983,1988,8088,714) for i in triggers],device=device)
            self.wait_columns = torch.tensor([i in (14190,13824,11489,3783) for i in triggers],device=device)
            self.phrase_count = torch.zeros_like(self.count)
        if history_gate != 'none':
            self.first_reflection = torch.full_like(self.count, -1)
            self.reflection_lookup = torch.zeros(size, dtype=torch.bool, device=device)
            self.reflection_lookup[self.ids] = True
        for token, piece in triggers.items():
            if tokenizer.decode([token]) != piece or tokenizer.encode(piece, add_special_tokens=False) != [token]:
                raise ValueError('Tokenizer mismatch')
        self.runner.sampler = Sampler(self)
        self.runner.add_requests = self.add_requests
        self.runner._remove_request = self.remove_request
        self.core.scheduler._preempt_request = self.reject_preempt

    @staticmethod
    def reject_preempt(*args, **kwargs):
        raise RuntimeError('Stop: coordinated adapter has not validated preemption')

    def check_penalty_request(self, rid):
        if getattr(self, 'penalty_mode', 'fixed') != 'fixed':
            params = self.runner.steer_vector_state._dynamic_params[rid]
            if params.paper_parameters is not None or min(params.low_val_1, params.low_val_2) != self.lower_bound:
                raise ValueError('Penalty bound must match actual frozen ReBalance parameters')

    def add_requests(self, output):
        # Reject unsupported requests before mutating the native runner.
        for r in output.scheduled_new_reqs:
            p = r.sampling_params
            if (r.num_computed_tokens or len(r.prefill_token_ids) != len(r.prompt_token_ids)
                    or p is None or p.n != 1 or p.structured_outputs is not None
                    or p.logit_bias or p.bad_words or p.presence_penalty or p.frequency_penalty
                    or p.repetition_penalty != 1 or p.min_tokens):
                raise ValueError('Fresh unmasked single-output requests only')
        self.original_add(output)
        state = self.runner.steer_vector_state
        for r in output.scheduled_new_reqs:
            rid = r.req_id
            if rid in self.active or rid in self.completed:
                raise ValueError('Request ID reused')
            slot = self.runner.req_states.req_id_to_index[rid]
            if self.mode in ('negative','shadow') and rid not in state._dynamic_indices:
                raise ValueError('Negative gate needs the actual ReBalance controller')
            self.check_penalty_request(rid)
            self.active[rid] = slot
            self.opening[slot] = False
            self.thinking[slot] = 151648 in r.prompt_token_ids
            self.count[slot] = self.eligible_count[slot] = self.changed_count[slot] = 0
            self.first_change[slot] = -1
            self.prompt_len[slot] = len(r.prompt_token_ids)
            if getattr(self,'lexical_mode','original') == 'but_wait':
                self.after_but[slot] = False
                self.phrase_count[slot] = 0
            if getattr(self,'history_gate','none') != 'none':
                self.first_reflection[slot] = -1

    def remove_request(self, rid):
        if rid in self.active:
            slot = self.active.pop(rid)
            # One transfer at completion, never in the decoding hot path.
            values = self.torch.stack([self.count[slot], self.eligible_count[slot],
                self.changed_count[slot], self.first_change[slot]]).cpu().tolist()
            state = self.runner.steer_vector_state
            history = None
            if rid in state._dynamic_indices:
                history = hashlib.sha256(state._history[slot,:state._history_lengths[rid]]
                                          .cpu().numpy().tobytes()).hexdigest()
            self.completed[rid] = dict(zip(('tokens','eligible','changed','first_change'),values),
                                       R_history_sha256=history)
            if getattr(self,'lexical_mode','original') == 'but_wait':
                self.completed[rid]['phrase_positions'] = int(self.phrase_count[slot].cpu())
            if getattr(self,'history_gate','none') != 'none':
                self.completed[rid]['first_reflection'] = int(self.first_reflection[slot].cpu())
        return self.original_remove(rid)

    def close(self):
        if not self.enabled:
            return
        self.runner.sampler = self.original_sampler
        self.runner.add_requests = self.original_add
        self.runner._remove_request = self.original_remove
        self.core.scheduler._preempt_request = self.original_preempt


class Sampler:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner.original_sampler, name)

    def __call__(self, logits, batch, **kwargs):
        o, t = self.owner, self.owner.torch
        if batch.num_draft_tokens or logits.shape[0] != batch.num_reqs:
            raise RuntimeError('One prediction per live request required')
        idx = batch.idx_mapping[:batch.num_reqs].long()
        valid = t.as_tensor(batch.num_computed_tokens_np + batch.num_scheduled_tokens
                            >= batch.prefill_len_np, device=logits.device)
        t._assert_async((~valid | (batch.seq_lens[:batch.num_reqs] ==
                        o.prompt_len[idx]+o.count[idx])).all(), 'Accepted-token clock mismatch')
        mask = o.opening[idx] & o.thinking[idx] & valid
        history_on = getattr(o,'history_gate','none') != 'none'
        if o.mode in ('negative','shadow'):
            s = o.runner.steer_vector_state
            coefficient, mean = s._coefs[idx], s._prev_step_mean[idx]
            mask &= t.isfinite(coefficient) & t.isfinite(mean) & (coefficient < 0)
        if history_on:
            mask &= o.first_reflection[idx] >= 0
        phrase_on = getattr(o,'lexical_mode','original') == 'but_wait'
        extended = getattr(o, 'candidate_mask', None)
        if extended is not None:
            lexical_mask = extended(idx, valid)
            mask = lexical_mask.any(dim=1)
        if phrase_on:
            delayed = o.after_but[idx] & o.thinking[idx] & valid & t.isfinite(coefficient) & t.isfinite(mean) & (coefficient<0)
            lexical_mask = (mask[:,None] & ~o.but_columns[None,:]) | (delayed[:,None] & o.wait_columns[None,:])
            o.phrase_count[idx] += delayed.to(t.int64)
            mask = mask | delayed
        o.eligible_count[idx] += mask.to(t.int64)
        if o.mode != 'shadow':
            # Raw R maximum probability was already computed by model_runner.
            # This is before temperature/top-p. Final probability is NOT halved.
            values = logits[:, o.ids]
            penalty_mode = getattr(o, 'penalty_mode', 'fixed')
            if penalty_mode == 'fixed':
                if phrase_on or extended is not None:
                    provider = getattr(o, 'penalty_provider', None)
                    adjusted = values-PENALTY if provider is None else provider(logits, lexical_mask, idx)
                    logits[:, o.ids] = t.where(lexical_mask, adjusted, values)
                else:
                    logits[:, o.ids] = t.where(mask[:,None], values-PENALTY, values)
            else:
                amount = device_penalty(t, coefficient, penalty_mode, o.lower_bound, o.constant_scale)
                adjusted = (values-amount[:,None]).to(values.dtype)
                logits[:, o.ids] = t.where(mask[:,None], adjusted, values)
            o.changed_count[idx] += mask.to(t.int64)
            o.first_change[idx] = t.where(mask & (o.first_change[idx]<0), o.count[idx], o.first_change[idx])
        result = o.original_sampler(logits, batch, **kwargs)
        if result.sampled_token_ids.shape != (batch.num_reqs,1):
            raise RuntimeError('Unsupported sample shape')
        token = result.sampled_token_ids[:,0].long()
        # Invalid partial prefill dummy samples are never used as token IDs.
        safe = t.where(valid, token, t.zeros_like(token))
        t._assert_async(((safe>=0)&(safe<o.clean.numel())).all(), 'Invalid tokenizer ID')
        accepted = getattr(o, 'accept_lexical_sample', None)
        if accepted is not None:
            accepted(idx, safe, valid)
        if phrase_on:
            after = (o.opening[idx] & o.but_lookup[safe]) | (o.after_but[idx] & o.white[safe])
            after &= o.thinking[idx] & (safe!=151648) & (safe!=151649) & ~o.boundary[safe]
            o.after_but[idx] = t.where(valid,after,o.after_but[idx])
        if history_on:
            # Observe the ACTUAL sampled marker at an already clean opening.
            # Interior words, boundary+word mixed tokens and prompt text do not count.
            first = valid & o.opening[idx] & o.thinking[idx] & o.reflection_lookup[safe] & (o.first_reflection[idx]<0)
            o.first_reflection[idx] = t.where(first, o.count[idx], o.first_reflection[idx])
        active = o.thinking[idx]
        active = (active | (safe==151648)) & (safe!=151649)
        opening = t.where(o.boundary[safe], o.clean[safe], o.opening[idx] & o.white[safe])
        opening &= active & (safe!=151648)
        o.opening[idx] = t.where(valid, opening, o.opening[idx])
        o.thinking[idx] = t.where(valid, active, o.thinking[idx])
        o.count[idx] += valid.to(t.int64)
        # The native runner now updates R from the accepted token and its saved
        # pre-penalty confidence; next invocation reads that newly closed step.
        return result
