"""Identical arithmetic for natural decode and all historical answer inputs."""
import torch
from global_kernel import GlobalKernel
from phase_kernel import PhaseKernel,EOS
from stable_add import stable_add,eager_add_reference


class ReplayKernel(GlobalKernel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('history_answer_mask',torch.zeros(self.max_tokens,dtype=torch.bool,device=self.mode.device))
        self.register_buffer('engineering_reference',torch.zeros((),dtype=torch.long,device=self.mode.device))
        self.register_buffer('rounded_delta',torch.zeros_like(self.global_vector,dtype=torch.bfloat16))

    @torch.no_grad()
    def set_method(self,method='off',strength=.25):
        super().set_method(method,strength)
        # Outside the compiled forward: the BF16 cast is an actual stored value.
        self.rounded_delta.copy_((self.global_vector*float(strength)).to(torch.bfloat16))

    @torch.no_grad()
    def set_schedule(self,schedule,input_ids):
        super().set_schedule(schedule,input_ids)
        self.history_answer_mask.copy_(torch.as_tensor(schedule['history_answer_mask'],device=self.mode.device,dtype=torch.bool))

    def forward(self,hidden,residual,baseline,native_mask):
        base=PhaseKernel.forward(self,hidden,residual,baseline,native_mask)
        n=hidden.shape[0];ids=self.input_ids.index_select(0,self.row_indices)
        self.direction.copy_(torch.where(self.answer_phase[:,None],self.global_vector[None,:],torch.zeros_like(self.direction)))
        current=self.row_valid & self.answer_phase & (ids!=EOS) & (self.mode==1) & (self.strength!=0)
        self.answer_actions.add_(current.long())
        historical=self.history_answer_mask[:n] & (self.mode==1) & (self.strength!=0)
        indices=torch.where(self.row_valid,self.row_indices,n+self.slot_numbers)
        scratch=torch.zeros(n+self.capacity,dtype=torch.bool,device=baseline.device)
        mask=scratch.index_copy(0,indices,current)[:n] | historical
        production=stable_add(base,self.rounded_delta,mask)
        reference=eager_add_reference(base,self.rounded_delta,mask)
        return torch.where(self.engineering_reference==1,base,
            torch.where(self.engineering_reference==2,reference,production))
