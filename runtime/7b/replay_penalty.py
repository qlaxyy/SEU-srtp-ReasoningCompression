"""Preserve lexical and dynamic-penalty state across native synchronous replay.

Lifecycle follows the project's validated ReplayAdapter; the native runner
alone owns ReBalance's coefficient/history replay. No controller math changes.
"""
from penalty_adapter import ResearchAdapter


class ReplayPenalty(ResearchAdapter):
    fields=('opening','thinking','count','prompt_len','eligible_count','changed_count','first_change',
        'lex_state','lex_hit','lex_ready','closed_hit','lex_open','lex_changes',
        'first_action','counts','multiplier_sum','multiplier_min','multiplier_max','histogram','interior')

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.runner.vllm_config.scheduler_config.async_scheduling or self.audit_raw:
            raise ValueError('Replay requires sync and ordinary penalty audit')
        self.suspended={};self.suspending=set();self.replay_events=[]
        self.original_finish=self.runner.finish_requests;self.runner.finish_requests=self.finish_requests
        self.core.scheduler._preempt_request=self.original_preempt

    def finish_requests(self,output):
        self.suspending=set(output.preempted_req_ids or ())-set(output.finished_req_ids)
        try:self.original_finish(output)
        finally:self.suspending=set()
        for rid in output.finished_req_ids:self.suspended.pop(rid,None)

    def remove_request(self,rid):
        if rid in self.suspending and rid in self.active:
            slot=self.active.pop(rid)
            saved={k:getattr(self,k)[slot].clone() for k in self.fields}
            self.suspended[rid]=saved
            self.replay_events.append(dict(request_id=rid,event='suspend',sample_count=int(saved['count'].cpu())))
            return self.original_remove(rid)
        return super().remove_request(rid)

    def add_requests(self,output):
        for r in output.scheduled_new_reqs:
            p=r.sampling_params
            if (r.num_computed_tokens or p is None or p.n!=1 or p.structured_outputs is not None or p.logit_bias or p.bad_words
                    or p.presence_penalty or p.frequency_penalty or p.repetition_penalty!=1 or p.min_tokens):
                raise ValueError('Unsupported replay options')
            generated=len(r.prefill_token_ids)-len(r.prompt_token_ids)
            if r.req_id in self.suspended:
                saved=self.suspended[r.req_id]
                if generated!=int(saved['count'].cpu()) or len(r.prompt_token_ids)!=int(saved['prompt_len'].cpu()):
                    raise ValueError('Penalty replay prefix clock differs')
            elif generated:raise ValueError('Generated prefix lacks penalty state')
        self.original_add(output)
        for r in output.scheduled_new_reqs:
            rid=r.req_id
            if rid in self.active or rid in self.completed:raise ValueError('Request ID reused')
            i=self.runner.req_states.req_id_to_index[rid]
            if rid not in self.runner.steer_vector_state._dynamic_indices:raise ValueError('Native ReBalance missing')
            self.check_penalty_request(rid);self.active[rid]=i
            saved=self.suspended.pop(rid,None)
            if saved is not None:
                for k,v in saved.items():getattr(self,k)[i].copy_(v)
                assert all(self.torch.equal(getattr(self,k)[i],v) for k,v in saved.items())
                self.replay_events.append(dict(request_id=rid,event='restore',sample_count=int(saved['count'].cpu()),
                    replay_prefill_tokens=len(r.prefill_token_ids),all_fields_exact=True))
            else:
                for k in self.fields:getattr(self,k)[i].zero_()
                self.thinking[i]=151648 in r.prompt_token_ids;self.prompt_len[i]=len(r.prompt_token_ids)
                self.first_change[i]=self.first_action[i]=-1
                self.multiplier_min[i]=float('inf');self.multiplier_max[i]=-float('inf')

    def close(self):
        if self.active or self.suspended:raise RuntimeError('Replay penalty not drained')
        self.runner.finish_requests=self.original_finish
        super().close()
