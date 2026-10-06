"""Full-logit probe at the answer-entry prediction; engineering sequences only."""
import torch
from teacher_force import TeacherForce


class ProbedTeacherForce(TeacherForce):
    def __init__(self,llm,vocab_size,probe_capacity=2):
        super().__init__(llm)
        dev=self.runner.device;n=self.runner.max_num_reqs
        self.probe_index=torch.zeros(n,dtype=torch.long,device=dev)
        self.probe_at=torch.zeros_like(self.probe_index)
        self.probe_logits=torch.zeros(probe_capacity,vocab_size,dtype=torch.float32,device=dev)
        self.probe_count=torch.zeros(probe_capacity,dtype=torch.long,device=dev)
        self.probe_pending={};self.probe_completed={}

    def register(self,targets):
        super().register(targets)
        if len(targets)>self.probe_logits.shape[0]:raise ValueError('Probe capacity differs')
        for i,(rid,ts) in enumerate(targets.items()):
            if ts.count(151649)!=1:raise ValueError('Exactly one engineering answer entry required')
            self.probe_pending[rid]=(i,ts.index(151649)+1)

    def add_requests(self,output):
        super().add_requests(output)
        for r in output.scheduled_new_reqs:
            i,at=self.probe_pending.pop(r.req_id);slot=self.active[r.req_id]
            self.probe_index[slot]=i;self.probe_at[slot]=at;self.probe_logits[i].zero_();self.probe_count[i]=0

    def __call__(self,logits,batch,**kwargs):
        idx=batch.idx_mapping[:batch.num_reqs].long();probe=self.probe_index[idx]
        valid=torch.as_tensor(batch.num_computed_tokens_np+batch.num_scheduled_tokens>=batch.prefill_len_np,device=logits.device)
        take=valid & (self.count[idx]==self.probe_at[idx])
        self.probe_logits.index_copy_(0,probe,torch.where(take[:,None],logits.float(),self.probe_logits.index_select(0,probe)))
        self.probe_count[probe]+=take.long()
        return super().__call__(logits,batch,**kwargs)

    def remove_request(self,rid):
        if rid in self.active:
            slot=self.active[rid];i=int(self.probe_index[slot].cpu())
            count=int(self.probe_count[i].cpu())
            self.probe_completed[rid]=dict(count=count,logits=self.probe_logits[i].detach().cpu().numpy().copy())
        return super().remove_request(rid)
