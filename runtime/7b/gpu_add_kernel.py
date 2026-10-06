"""One BF16 output store after adding two already-materialized BF16 operands."""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(H,D,M,O,N:tl.constexpr,W:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=index<N*W
    x=tl.load(H+index,valid,other=0).to(tl.float32)
    delta=tl.load(D+index%W,valid,other=0).to(tl.float32)
    active=tl.load(M+index//W,valid,other=0)
    y=tl.where(active,x+delta,x)
    tl.store(O+index,y,valid)


def launch(hidden,delta,mask):
    assert hidden.is_contiguous() and delta.is_contiguous() and mask.is_contiguous()
    out=torch.empty_like(hidden)
    add_kernel[(triton.cdiv(hidden.numel(),256),)](hidden,delta,mask,out,hidden.shape[0],hidden.shape[1],256)
    return out
