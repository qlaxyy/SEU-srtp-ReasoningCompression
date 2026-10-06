"""Instance-only dynamic L27 provider. Native controller and gate are unchanged."""
import hashlib
from label_alignment_adapter import AlignmentAdapter
from policy import PENALTY
from strength_policy import validate, adjusted_values


class ResearchAdapter(AlignmentAdapter):
    def __init__(self, llm, tok, *, tables, penalty_multiplier, negative_bound,
                 mode='off', audit_raw=False, penalty_kind='dynamic', curve='linear', shape=1.):
        if mode not in ('off', 'shadow', 'on'):
            raise ValueError('Unknown route mode')
        validate(penalty_multiplier, negative_bound, curve, shape)
        if penalty_kind not in ('dynamic', 'fixed'): raise ValueError('Unknown penalty kind')
        self.penalty_kind = penalty_kind
        self.curve, self.shape = curve, float(shape)
        self.route_mode = mode
        self.penalty_multiplier = float(penalty_multiplier)
        self.negative_bound = float(negative_bound)
        self.audit_raw = audit_raw
        self.raw_hashes, self.route_completed = {}, {}
        super().__init__(llm, tok, tables=tables, large_suppression=True,
                         lexical_control=False, enabled=True)
        t, n, dev = self.torch, self.runner.max_num_reqs, self.runner.device
        self.first_action = t.full((n,), -1, device=dev, dtype=t.long)
        self.counts = t.zeros((n, 5), device=dev, dtype=t.long)
        self.multiplier_sum = t.zeros(n, device=dev, dtype=t.float32)
        self.multiplier_min = t.full((n,), float('inf'), device=dev, dtype=t.float32)
        self.multiplier_max = t.full((n,), -float('inf'), device=dev, dtype=t.float32)
        # Fixed bins [1,2), ... [31,32), {32}; no clipping above the reviewed maximum.
        self.histogram = t.zeros((n, 32), device=dev, dtype=t.long)
        self.hist_bins = t.arange(32, device=dev)
        self.interior = t.zeros(n, device=dev, dtype=t.long)
        self.old_observe = self.rstate.observe_sample
        if audit_raw:
            self.rstate.observe_sample = self.observe
        if mode != 'off':
            self.penalty_provider = self.penalty_values

    def check_penalty_request(self, rid):
        super().check_penalty_request(rid)
        params = self.runner.steer_vector_state._dynamic_params[rid]
        if params.paper_parameters is not None or getattr(params, 'constant_control', False):
            raise ValueError('Requires unchanged native ReBalance controller')
        actual = min(params.low_val_1, params.low_val_2)
        if actual != self.negative_bound:
            raise ValueError('Dynamic penalty bound differs from actual request fit')

    def add_requests(self, output):
        super().add_requests(output)
        for req in output.scheduled_new_reqs:
            i = self.active[req.req_id]
            self.first_action[i] = -1
            self.counts[i] = 0
            self.multiplier_sum[i] = 0
            self.multiplier_min[i] = float('inf')
            self.multiplier_max[i] = -float('inf')
            self.histogram[i] = 0
            self.interior[i] = 0
            if self.audit_raw:
                self.raw_hashes[req.req_id] = hashlib.sha256()

    def observe(self, batch, tokens, probabilities):
        valid = batch.num_computed_tokens_np + batch.num_scheduled_tokens >= batch.prefill_len_np
        vals = probabilities.detach().float().cpu().numpy()
        for j, rid in enumerate(batch.req_ids):
            if valid[j]:
                self.raw_hashes[rid].update(vals[j].tobytes())
        return self.old_observe(batch, tokens, probabilities)

    def candidate_mask(self, idx, valid):
        mask = super().candidate_mask(idx, valid)
        self._penalty_valid = valid
        return mask

    def penalty_values(self, logits, lexical_mask, idx):
        t = self.torch
        values = logits[:, self.ids]
        original = values - PENALTY
        if self.penalty_kind == 'fixed':
            # Exactly the previously tested fixed8 scalar subtraction, including BF16.
            proposal = values - PENALTY * self.penalty_multiplier
            k = t.full((len(idx),), self.penalty_multiplier, dtype=t.float32, device=values.device)
        else:
            proposal, k = adjusted_values(t, values, self.rstate._coefs[idx],
                                          self.penalty_multiplier, self.negative_bound, self.curve, self.shape)
        changed_columns = lexical_mask & (proposal != original)
        changed, eligible = changed_columns.any(dim=1), lexical_mask.any(dim=1)
        self.first_action[idx] = t.where(changed & (self.first_action[idx] < 0),
                                        self.count[idx], self.first_action[idx])
        self.counts[idx] += t.stack([changed.long(), eligible.long(),
            (changed & ~self.thinking[idx]).long(),
            (changed & ~self._penalty_valid).long(), changed_columns.sum(dim=1)], dim=1)
        self.multiplier_sum[idx] += t.where(eligible, k, t.zeros_like(k))
        self.multiplier_min[idx] = t.minimum(self.multiplier_min[idx],
            t.where(eligible, k, t.full_like(k, float('inf'))))
        self.multiplier_max[idx] = t.maximum(self.multiplier_max[idx],
            t.where(eligible, k, t.full_like(k, -float('inf'))))
        bins = t.clamp(t.floor(k).long() - 1, 0, 31)
        self.histogram[idx] += (eligible[:, None] & (bins[:, None] == self.hist_bins[None, :])).long()
        self.interior[idx] += (eligible & (k > 1.) & (k < self.penalty_multiplier)).long()
        return proposal if self.route_mode == 'on' else original

    def remove_request(self, rid):
        if rid in self.active:
            i = self.active[rid]
            v = self.counts[i].cpu().tolist()
            total = float(self.multiplier_sum[i].cpu())
            self.route_completed[rid] = dict(zip(
                ['changed_actions', 'eligible_positions', 'answer_actions', 'prefill_actions', 'changed_columns'], v),
                first_action=int(self.first_action[i].cpu()), repair_actions=0,
                penalty_multiplier=self.penalty_multiplier, negative_bound=self.negative_bound,
                mode=self.route_mode, applied=self.route_mode == 'on',
                penalty_kind=self.penalty_kind, curve=self.curve, shape=self.shape,
                dynamic_rule='1+(maximum-1)*f(clip(alpha/lower_bound,0,1))' if self.penalty_kind == 'dynamic' else 'constant',
                multiplier_sum=total, multiplier_mean=total / v[1] if v[1] else None,
                multiplier_min=float(self.multiplier_min[i].cpu()) if v[1] else None,
                multiplier_max=float(self.multiplier_max[i].cpu()) if v[1] else None,
                multiplier_histogram=self.histogram[i].cpu().tolist(),
                interior_positions=int(self.interior[i].cpu()))
        return super().remove_request(rid)

    def close(self):
        if hasattr(self, 'old_observe'):
            self.rstate.observe_sample = self.old_observe
        super().close()
