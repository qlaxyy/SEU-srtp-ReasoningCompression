"""Offline feature collection only. Inner sampler observes forced accepted tokens."""
import torch


class TeacherForce:
    def __init__(self,llm,max_tokens=16000):
        self.runner=llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        self.original_sampler=self.runner.sampler
        self.original_add=self.runner.add_requests
        self.original_remove=self.runner._remove_request
        n,dev=self.runner.max_num_reqs,self.runner.device
        self.targets=torch.zeros(n,max_tokens,dtype=torch.long,device=dev)
        self.lengths=torch.zeros(n,dtype=torch.long,device=dev)
        self.prompt_len=torch.zeros_like(self.lengths);self.count=torch.zeros_like(self.lengths)
        self.pending,self.active,self.completed={},{},{}
        self.runner.sampler=self
        self.runner.add_requests=self.add_requests
        self.runner._remove_request=self.remove_request

    def __getattr__(self,key):
        return getattr(self.original_sampler,key)

    def register(self,targets):
        if set(targets)&(set(self.pending)|set(self.active)|set(self.completed)):
            raise ValueError('Repeated forced request id')
        for rid,ts in targets.items():
            if not ts or len(ts)>self.targets.shape[1] or ts[-1]!=151643 or 151643 in ts[:-1]:
                raise ValueError('Target must terminate at exactly one EOS within cap')
            self.pending[rid]=list(ts)

    def add_requests(self,output):
        for r in output.scheduled_new_reqs:
            if r.req_id not in self.pending or r.num_computed_tokens:
                raise ValueError('Unregistered or resumed forced sequence')
        self.original_add(output)
        for r in output.scheduled_new_reqs:
            idx=self.runner.req_states.req_id_to_index[r.req_id]
            ts=self.pending.pop(r.req_id)
            self.targets[idx].zero_()
            self.targets[idx,:len(ts)]=torch.tensor(ts,device=self.targets.device)
            self.lengths[idx]=len(ts);self.prompt_len[idx]=len(r.prompt_token_ids);self.count[idx]=0
            self.active[r.req_id]=idx

    def __call__(self,logits,batch,**kwargs):
        result=self.original_sampler(logits,batch,**kwargs)
        if batch.num_draft_tokens or result.sampled_token_ids.shape!=(batch.num_reqs,1):
            raise ValueError('Unsupported forced sampler geometry')
        idx=batch.idx_mapping[:batch.num_reqs].long()
        valid=torch.as_tensor(batch.num_computed_tokens_np+batch.num_scheduled_tokens>=batch.prefill_len_np,device=logits.device)
        torch._assert_async((~valid | (batch.seq_lens[:batch.num_reqs]==self.prompt_len[idx]+self.count[idx])).all(),'Forced clock differs')
        torch._assert_async((~valid | (self.count[idx]<self.lengths[idx])).all(),'Forced sequence exhausted')
        forced=self.targets[idx,self.count[idx].clamp(max=self.targets.shape[1]-1)]
        result.sampled_token_ids[:,0].copy_(torch.where(valid,forced,result.sampled_token_ids[:,0]))
        self.count[idx]+=valid.long()
        # Outer L27 adapter and then native observe_sample see these SAME tokens.
        # Its raw maximum probabilities were computed before either sampler.
        return result

    def remove_request(self,rid):
        if rid in self.active:
            idx=self.active.pop(rid)
            count,length=torch.stack((self.count[idx],self.lengths[idx])).cpu().tolist()
            self.completed[rid]=dict(sampled=int(count),expected=int(length),complete=count==length)
        return self.original_remove(rid)

    def close(self):
        if self.pending or self.active:raise RuntimeError('Forced requests not drained')
        if self.runner.sampler is not self:raise RuntimeError('Remove outer sampler first')
        self.runner.sampler=self.original_sampler
        self.runner.add_requests=self.original_add
        self.runner._remove_request=self.original_remove
