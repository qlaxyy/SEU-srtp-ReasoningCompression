"""A genuine single mean-difference direction, without invented mixture slots."""
import math
import json
import hashlib
import numpy as np
import torch
from phase_kernel import PhaseKernel,EOS


class GlobalBank:
    def __init__(self,path):
        receipt=json.loads((path.parent/'SRQ_VECTOR_RECEIPT.json').read_text(encoding='utf-8'))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['vector_sha256']:raise ValueError('Frozen SRQ vector changed')
        self.direction=np.load(path,allow_pickle=False);self.fit_ids=tuple(receipt['fit_ids'])
        if self.direction.shape!=(3584,) or not np.isfinite(self.direction).all() or np.linalg.norm(self.direction)<1e-8:
            raise ValueError('Invalid learned global direction')
        if len(self.fit_ids)!=271 or len(set(self.fit_ids))!=271:raise ValueError('Global fit questions differ')


class GlobalKernel(PhaseKernel):
    def __init__(self,hidden_size,bank,capacity=256,max_tokens=32768,device='cpu'):
        super().__init__(hidden_size,None,capacity,max_tokens,device)
        if len(bank.direction)!=hidden_size:raise ValueError('Global bank dimension differs')
        self.register_buffer('global_vector',torch.as_tensor(bank.direction,dtype=torch.float32,device=device))

    @torch.no_grad()
    def set_method(self,method='off',strength=.25):
        if method not in ('off','srq_pca16'):raise ValueError('Conditional bank was rejected, no fake substitute')
        if not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Invalid global strength')
        self.mode.fill_(int(method=='srq_pca16'));self.strength.fill_(strength)

    def forward(self,hidden,residual,baseline,native_mask):
        base=super().forward(hidden,residual,baseline,native_mask)
        ids=self.input_ids.index_select(0,self.row_indices)
        self.direction.copy_(torch.where(self.answer_phase[:,None],self.global_vector[None,:],torch.zeros_like(self.direction)))
        on=self.row_valid & self.answer_phase & (ids!=EOS) & (self.mode==1) & (self.strength!=0)
        self.answer_actions.add_(on.long())
        rows=base.index_select(0,self.row_indices)
        delta=(self.strength*self.direction).to(rows.dtype)
        proposed=rows+delta
        changed=torch.where(on[:,None],proposed,rows)
        n=hidden.shape[0];indices=torch.where(self.row_valid,self.row_indices,n+self.slot_numbers)
        extended=torch.cat((base,torch.zeros_like(changed)),dim=0)
        return extended.index_copy(0,indices,changed)[:n]
