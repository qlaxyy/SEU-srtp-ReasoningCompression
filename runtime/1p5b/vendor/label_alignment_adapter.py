"""Opt-in instance extension of RC14; native R history remains authoritative.

No shared vLLM controller copy or environment mutation. All decoding state and
token lookup tables stay on device; no per-token decode or CPU synchronization.
"""
import numpy as np
from adapter import Adapter


class AlignmentAdapter(Adapter):
    def __init__(self,llm,tokenizer,*,tables=None,large_suppression=False,lexical_control=False,enabled=False):
        if not enabled:
            super().__init__(llm,tokenizer,mode='off')
            return
        if large_suppression and lexical_control:
            raise ValueError('Unplanned simultaneous changes')
        if not large_suppression and not lexical_control:
            raise ValueError('Use unchanged Adapter for T14_T14 and CV_CV')
        if tables is None:
            raise ValueError('Hash-verified compiled lexical tables required')
        self.large_suppression,self.lexical_control=large_suppression,lexical_control
        super().__init__(llm,tokenizer,mode='negative',gate_on=True)
        t=self.torch; device=self.runner.device
        n=self.runner.max_num_reqs
        self.lex_state=t.zeros(n,dtype=t.long,device=device)
        self.lex_hit=t.zeros(n,dtype=t.bool,device=device)
        self.lex_ready=t.zeros_like(self.lex_hit)
        self.closed_hit=t.zeros_like(self.lex_hit)
        self.lex_open=t.zeros_like(self.lex_hit)
        self.lex_changes=t.zeros(n,dtype=t.long,device=device)
        self.lex_transition=t.as_tensor(tables['transition'],device=device)
        self.lex_hits=t.as_tensor(tables['hits'],device=device)
        self.lex_final=t.as_tensor(tables['final'],device=device)
        self.lex_classes=t.as_tensor(tables['token_classes'],device=device)
        if self.lex_classes.numel()!=self.clean.numel():
            self.close(); raise ValueError('Tokenizer/table width mismatch')
        if large_suppression:
            # Only gather columns which can ever complete a phrase.
            ids=tables['candidate_ids']
            if not len(ids): self.close(); raise ValueError('Empty trigger set')
            self.ids=t.as_tensor(ids,device=device)
            self.lex_candidates=self.lex_hits[:,self.lex_classes[self.ids].long()]
        else:
            # No custom mask: retain the original RC14 sampler path exactly.
            self.candidate_mask=None
        self.rstate=self.runner.steer_vector_state
        self.original_record=self.rstate._record_scales
        if lexical_control:
            from vllm.steer_vectors import rebalance as runtime
            self.control_runtime=runtime
            self.rstate._record_scales=self.record_scales

    def add_requests(self,output):
        super().add_requests(output)
        for r in output.scheduled_new_reqs:
            i=self.active[r.req_id]
            self.lex_state[i]=0
            self.lex_hit[i]=self.lex_ready[i]=self.closed_hit[i]=self.lex_open[i]=False
            self.lex_changes[i]=0

    def candidate_mask(self,idx,valid):
        t=self.torch; s=self.runner.steer_vector_state
        gate=valid & self.lex_open[idx] & self.thinking[idx]
        gate &= t.isfinite(s._coefs[idx]) & t.isfinite(s._prev_step_mean[idx]) & (s._coefs[idx]<0)
        return self.lex_candidates[self.lex_state[idx].long()] & gate[:,None]

    def accept_lexical_sample(self,idx,token,valid):
        t=self.torch
        boundary=self.boundary[token]
        start=token==151648; end=token==151649
        state=self.lex_state[idx].long()
        classes=self.lex_classes[token].long()
        nxt=self.lex_transition[state,classes].long()
        hit=self.lex_hits[state,classes]
        active=self.thinking[idx] & valid & ~start & ~end
        if self.lexical_control:
            # Match complete previous step; delimiter tokens are excluded exactly
            # as in calibration. Pending word ends are finalized at boundary.
            ready=active & boundary & (self.runner.steer_vector_state._step_tok_count[idx]>0)
            self.lex_ready[idx]=ready
            self.closed_hit[idx]=self.lex_hit[idx] | self.lex_final[state]
            updated_hit=self.lex_hit[idx] | hit
            self.lex_hit[idx]=t.where(valid & (boundary|start|end),False,
                t.where(active,updated_hit,self.lex_hit[idx]))
        else:
            opening=t.where(boundary,self.clean[token],self.lex_open[idx] & ~hit)
            self.lex_open[idx]=t.where(valid,opening & (self.thinking[idx]|start) & ~start & ~end,self.lex_open[idx])
        self.lex_state[idx]=t.where(valid & (boundary|start|end),0,
            t.where(active,nxt,self.lex_state[idx]))

    def record_scales(self,batch,positions,pos,idx):
        from prepare_label_alignment import lexical_coefficient
        t=self.torch; s=self.rstate
        # Native observe_sample has just updated mean/coefficient; replace only
        # a genuinely completed step, BEFORE its scale is written to KV history.
        params=s._dynamic_params[batch.req_ids[positions[0]]]
        c=s._prev_step_mean[idx]
        result=lexical_coefficient(c,self.closed_hit[idx],params,self.control_runtime)
        ready=self.lex_ready[idx]
        before=s._coefs[idx]
        self.lex_changes[idx]+=(ready & (result!=before)).long()
        s._coefs.index_copy_(0,idx,t.where(ready,result,before))
        self.original_record(batch,positions,pos,idx)

    def remove_request(self,rid):
        extra=None
        if rid in self.active:
            extra=int(self.lex_changes[self.active[rid]].cpu())
        result=super().remove_request(rid)
        if extra is not None: self.completed[rid]['lexical_control_changes']=extra
        return result

    def close(self):
        if hasattr(self,'original_record'):
            self.rstate._record_scales=self.original_record
        super().close()
