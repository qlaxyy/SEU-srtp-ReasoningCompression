"""Natural answer boundary, optional conditional injection, native reasoning intact."""
import math
import torch
from torch import nn

START, END, EOS = 151648, 151649, 151643


class PhaseKernel(nn.Module):
    def __init__(self, hidden_size, bank=None, capacity=256, max_tokens=32768, device='cpu'):
        super().__init__()
        self.capacity, self.max_tokens = capacity, max_tokens
        for key in ('row_indices','input_ids'):
            n = capacity if key=='row_indices' else max_tokens
            self.register_buffer(key, torch.zeros(n,dtype=torch.long,device=device))
        for key in ('row_valid','generated_input','answer_phase'):
            self.register_buffer(key,torch.zeros(capacity,dtype=torch.bool,device=device))
        for key in ('query','answer_sum','direction'):
            self.register_buffer(key,torch.zeros(capacity,hidden_size,dtype=torch.float32,device=device))
        for key in ('entries','body_count','restarts','duplicate_ends','native_answer_conflicts','answer_actions','reason_actions'):
            self.register_buffer(key,torch.zeros(capacity,dtype=torch.long,device=device))
        self.register_buffer('mode',torch.zeros((),dtype=torch.long,device=device))
        self.register_buffer('strength',torch.zeros((),device=device))
        self.register_buffer('slot_numbers',torch.arange(capacity,device=device))
        self.has_bank=bank is not None
        if bank is not None:
            if len(bank.center)!=hidden_size:raise ValueError('Bank dimension differs')
            for key in ('center','basis','scale','mapper','shuffled_mapper','prototypes','global_direction'):
                self.register_buffer(key,torch.as_tensor(getattr(bank,key),dtype=torch.float32,device=device))
            self.register_buffer('reference_norm',torch.tensor(bank.reference_norm,dtype=torch.float32,device=device))

    @torch.no_grad()
    def set_method(self,method='off',strength=.25):
        modes={'off':0,'global':1,'conditional':2,'shuffled':3}
        if method not in modes or (method!='off' and not self.has_bank):raise ValueError('Missing real answer bank')
        if not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Invalid fixed strength')
        self.mode.fill_(modes[method]);self.strength.fill_(strength)

    @torch.no_grad()
    def set_schedule(self,schedule,input_ids):
        for key in ('row_indices','row_valid','generated_input'):
            dst=getattr(self,key);src=torch.as_tensor(schedule[key],dtype=dst.dtype,device=dst.device)
            if src.shape!=dst.shape:raise ValueError('Schedule shape differs')
            dst.copy_(src)
        n=schedule['scheduled_tokens']
        if n>len(input_ids):raise ValueError('Input token count differs')
        self.input_ids.zero_();self.input_ids[:n].copy_(input_ids[:n])

    @torch.no_grad()
    def clear_slot(self,slot):
        for key in ('row_valid','answer_phase','query','answer_sum','direction','entries','body_count',
                    'restarts','duplicate_ends','native_answer_conflicts','answer_actions','reason_actions'):
            getattr(self,key)[slot].zero_()

    def forward(self,hidden,residual,baseline,native_mask):
        q=hidden.index_select(0,self.row_indices)
        if residual is not None:q=q+residual.index_select(0,self.row_indices)
        q=q.float()  # Sum in the model dtype, as in native Qwen2.
        q=torch.where(self.row_valid[:,None],q,torch.zeros_like(q))
        ids=self.input_ids.index_select(0,self.row_indices)
        generated=self.row_valid & self.generated_input
        start=generated & (ids==START);end=generated & (ids==END);eos=generated & (ids==EOS)
        entering=end & ~self.answer_phase
        self.restarts.add_((start & self.answer_phase).long())
        self.duplicate_ends.add_((end & self.answer_phase).long())
        self.entries.add_(entering.long())
        phase=torch.where(start,False,torch.where(end,True,self.answer_phase))
        self.answer_phase.copy_(phase)
        self.query.copy_(torch.where(entering[:,None],q,self.query))
        body=generated & phase & ~start & ~end & ~eos
        self.answer_sum.add_(torch.where(body[:,None],q,torch.zeros_like(q)))
        self.body_count.add_(body.long())
        alpha=native_mask.index_select(0,self.row_indices)
        self.native_answer_conflicts.add_((self.row_valid & phase & (alpha!=0)).long())
        self.reason_actions.add_((self.row_valid & ~phase & (alpha!=0)).long())
        if not self.has_bank:
            return baseline  # Exact object, no numerical operation on the live model.
        z=((q-self.center)@self.basis)/self.scale
        z=torch.cat((torch.ones_like(z[:,:1]),z),dim=1)
        mapper=torch.where(self.mode==3,self.shuffled_mapper,self.mapper)
        weights=torch.softmax(z@mapper,dim=-1)
        mixed=weights@self.prototypes
        mixed=mixed/mixed.norm(dim=-1,keepdim=True).clamp_min(1e-12)*self.reference_norm
        proposed=torch.where(self.mode==1,self.global_direction[None,:],mixed)
        self.direction.copy_(torch.where(entering[:,None],proposed,
            torch.where(start[:,None],torch.zeros_like(proposed),self.direction)))
        on=self.row_valid & phase & ~eos & (self.mode!=0) & (self.strength!=0)
        self.answer_actions.add_(on.long())
        base=baseline.index_select(0,self.row_indices)
        changed=(base.float()+self.strength*self.direction).to(base.dtype)
        changed=torch.where(on[:,None],changed,base)
        n=hidden.shape[0]
        indices=torch.where(self.row_valid,self.row_indices,n+self.slot_numbers)
        extended=torch.cat((baseline,torch.zeros_like(changed)),dim=0)
        return extended.index_copy(0,indices,changed)[:n]
