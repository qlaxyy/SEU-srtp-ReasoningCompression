"""Opaque BF16 arithmetic boundary; independent Triton and eager references."""
import torch


@torch.library.custom_op('srq7b::stable_add',mutates_args=())
def stable_add(hidden:torch.Tensor,delta:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    if hidden.dtype!=torch.bfloat16 or delta.dtype!=hidden.dtype:raise ValueError('Explicit BF16 operands required')
    if hidden.ndim!=2 or delta.shape!=(hidden.shape[1],) or mask.shape!=(hidden.shape[0],):raise ValueError('Add geometry differs')
    if hidden.device.type=='cpu':return torch.where(mask[:,None],hidden+delta,hidden)
    from gpu_add_kernel import launch
    return launch(hidden,delta,mask)


@stable_add.register_fake
def stable_add_fake(hidden,delta,mask):return torch.empty_like(hidden)


@torch.library.custom_op('srq7b::eager_add_reference',mutates_args=())
def eager_add_reference(hidden:torch.Tensor,delta:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    # Custom op body is eager, so compilation cannot fuse this addition into
    # the following decoder layer or delete the BF16 output materialization.
    return torch.where(mask[:,None],hidden+delta,hidden)


@eager_add_reference.register_fake
def eager_add_fake(hidden,delta,mask):return torch.empty_like(hidden)
