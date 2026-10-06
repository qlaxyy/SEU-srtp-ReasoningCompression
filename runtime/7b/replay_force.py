"""Fixed training-sequence replay check; never enabled for benchmark generation."""
import torch
from probed_teacher_force import ProbedTeacherForce


class ReplayTeacherForce(ProbedTeacherForce):
    fields=('targets','lengths','prompt_len','count','probe_index','probe_at')

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.suspended={};self.suspending=set();self.replay_events=[]
        self.original_finish=self.runner.finish_requests;self.runner.finish_requests=self.finish_requests
        self.tail_logits=torch.zeros_like(self.probe_logits);self.tail_count=torch.zeros_like(self.probe_count)
        self.tail_completed={}

    def finish_requests(self,output):
        self.suspending=set(output.preempted_req_ids or ())-set(output.finished_req_ids)
        try:self.original_finish(output)
        finally:self.suspending=set()

    def add_requests(self,output):
        for r in output.scheduled_new_reqs:
            rid=r.req_id
            if r.num_computed_tokens or (rid not in self.pending and rid not in self.suspended):raise ValueError('Unregistered forced replay')
            if rid in self.suspended:
                saved=self.suspended[rid]
                assert len(r.prefill_token_ids)-len(r.prompt_token_ids)==int(saved['count'].cpu())
        self.original_add(output)
        for r in output.scheduled_new_reqs:
            rid=r.req_id;slot=self.runner.req_states.req_id_to_index[rid];self.active[rid]=slot
            saved=self.suspended.pop(rid,None)
            if saved is not None:
                for k,v in saved.items():getattr(self,k)[slot].copy_(v)
                assert all(torch.equal(getattr(self,k)[slot],v) for k,v in saved.items())
                self.replay_events.append(dict(request_id=rid,event='restore',all_fields_exact=True))
            else:
                ts=self.pending.pop(rid);self.targets[slot].zero_();self.targets[slot,:len(ts)]=torch.tensor(ts,device=self.targets.device)
                self.lengths[slot]=len(ts);self.prompt_len[slot]=len(r.prompt_token_ids);self.count[slot]=0
                i,at=self.probe_pending.pop(rid);self.probe_index[slot]=i;self.probe_at[slot]=at
                self.probe_logits[i].zero_();self.probe_count[i]=0;self.tail_logits[i].zero_();self.tail_count[i]=0

    def __call__(self,logits,batch,**kwargs):
        idx=batch.idx_mapping[:batch.num_reqs].long();probe=self.probe_index[idx]
        valid=torch.as_tensor(batch.num_computed_tokens_np+batch.num_scheduled_tokens>=batch.prefill_len_np,device=logits.device)
        take=valid & (self.count[idx]==self.lengths[idx]-2)
        self.tail_logits.index_copy_(0,probe,torch.where(take[:,None],logits.float(),self.tail_logits.index_select(0,probe)))
        self.tail_count[probe]+=take.long()
        return super().__call__(logits,batch,**kwargs)

    def remove_request(self,rid):
        if rid in self.active:
            slot=self.active[rid]
            if rid in self.suspending:
                self.active.pop(rid);self.suspended[rid]={k:getattr(self,k)[slot].clone() for k in self.fields}
                self.replay_events.append(dict(request_id=rid,event='suspend'))
                return self.original_remove(rid)
            i=int(self.probe_index[slot].cpu())
            self.tail_completed[rid]=dict(count=int(self.tail_count[i].cpu()),logits=self.tail_logits[i].detach().cpu().numpy().copy())
        return super().remove_request(rid)

    def close(self):
        assert not self.suspended
        self.runner.finish_requests=self.original_finish
        super().close()
