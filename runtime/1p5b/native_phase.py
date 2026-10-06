"""Instance-scoped natural-phase observer/injector; no edits to native R control."""
import numpy as np
from phase_kernel import PhaseKernel
from positions import LivePositions


def validate_native_params(p,expected):
    """Exact schema of the hash-pinned remote controller, not local HEAD."""
    fields=set(expected)|{'boundary_token_ids','think_start_token_id','think_end_token_id','paper_parameters'}
    if set(vars(p))!=fields:raise ValueError('Pinned native parameter schema differs')
    if any(getattr(p,k)!=v for k,v in expected.items()):raise ValueError('Native fit changed')
    if p.paper_parameters is not None:raise ValueError('Native coefficient mode changed')
    if not p.boundary_token_ids or p.think_start_token_id!=151648 or p.think_end_token_id!=151649:
        raise ValueError('Native boundary markers changed')


class NativePhase:
    def __init__(self,parameters,bank=None,layer=20,hidden_size=1536,capacity=256,max_tokens=32768,kernel_type=PhaseKernel):
        from vllm.steer_vectors import controllers
        from vllm.v1.worker.gpu import model_runner
        self.controllers,self.model_runner=controllers,model_runner
        self.positions=LivePositions(capacity,max_tokens)
        self.parameters=dict(parameters);self.kernel=None;self.runner=None;self.armed=False
        self.completed={};self.checked=set()
        self.old_init=controllers.DecoderSteerController.init_graph_table
        self.old_hook=controllers.DecoderSteerController.process_output_hook
        self.old_fill=model_runner.fill_graph_steer_buffers
        owner=self

        def init(controller,num_rows,dim,dtype,device,max_num_tokens,row_tok,max_rank,families=None):
            owner.old_init(controller,num_rows,dim,dtype,device,max_num_tokens,row_tok,max_rank,families)
            if controller.layer_id!=layer:return
            if owner.kernel is not None or dim!=hidden_size or max_num_tokens!=max_tokens:
                raise ValueError('Duplicate engine or hidden/capacity mismatch')
            if set(controller.graph_tables)!={'additive'}:raise ValueError('Only native additive family supported')
            owner.kernel=kernel_type(dim,bank,capacity,max_tokens,device)
            controller.add_module('_live_phase_kernel',owner.kernel)

        def hook(controller,module,args,output):
            baseline=owner.old_hook(controller,module,args,output)
            if controller.layer_id!=layer:return baseline
            if owner.kernel is None or not controller._graph_mode:raise RuntimeError('in_graph required')
            h,residual,other,form=controllers.split_decoder_output(output)
            base,base_residual,_,_=controllers.split_decoder_output(baseline)
            if base_residual is not residual:raise RuntimeError('Residual ownership changed')
            changed=owner.kernel(h,residual,base,controller.graph_mask)
            return controllers.reconstruct_decoder_output(changed,residual,other,form,output)

        def fill(batch,state,manager):
            owner.old_fill(batch,state,manager)
            if owner.kernel is None or state is None:raise RuntimeError('Missing native phase state')
            if batch.num_draft_tokens:raise ValueError('Speculation unsupported')
            entries=manager.graph_batch_entries()
            # vLLM kernel warmup uses real scheduler requests with no payload.
            # Permit them only during construction, before research admission.
            if not owner.armed:
                if entries or owner.positions.requests:raise ValueError('Unexpected payload before admission')
                owner.kernel.row_valid.zero_();owner.kernel.generated_input.zero_()
                return
            if len(entries)!=1:raise ValueError('Exactly one native payload required')
            _,request,decoders=next(iter(entries.values()))
            if request.algorithm!='rebalance' or request.normalize or request.scale!=1. or {d.layer_id for d in decoders}!={layer}:
                raise ValueError('Original reasoning payload changed')
            n=batch.num_reqs;ids=list(batch.req_ids[:n])
            for j,rid in enumerate(ids):
                if rid not in owner.positions.requests:
                    if int(batch.num_computed_tokens_np[j])!=0:raise ValueError('Missing original KV history')
                    slot=owner.positions.add(rid,int(batch.prefill_len_np[j]));owner.kernel.clear_slot(slot)
                if rid not in owner.checked:
                    p=state._dynamic_params[rid]
                    validate_native_params(p,owner.parameters)
                    owner.checked.add(rid)
            schedule=owner.positions.schedule(ids,batch.num_computed_tokens_np[:n],batch.num_scheduled_tokens[:n],batch.prefill_len_np[:n])
            owner.kernel.set_schedule(schedule,batch.input_ids)

        self.patched_init,self.patched_hook,self.patched_fill=init,hook,fill
        controllers.DecoderSteerController.init_graph_table=init
        controllers.DecoderSteerController.process_output_hook=hook
        model_runner.fill_graph_steer_buffers=fill

    def attach(self,llm):
        runner=llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        self.runner=runner;self.old_remove=runner._remove_request
        def remove(rid):
            if rid in self.positions.requests:
                row=self.positions.finish(rid);slot=row['slot']
                for key in ('entries','body_count','restarts','duplicate_ends','native_answer_conflicts','answer_actions','reason_actions'):
                    row[key]=int(getattr(self.kernel,key)[slot].cpu())
                for key in ('query','answer_sum'):
                    row[key]=getattr(self.kernel,key)[slot].detach().float().cpu().numpy().copy()
                    if not np.isfinite(row[key]).all():raise ValueError('Nonfinite feature')
                row['cached_direction_norm']=float(self.kernel.direction[slot].float().norm().cpu())
                self.completed[rid]=row;self.checked.discard(rid);self.kernel.clear_slot(slot)
            return self.old_remove(rid)
        self.patched_remove=remove;runner._remove_request=remove
        self.armed=True

    def detach(self):
        if self.positions.requests:raise RuntimeError('Requests still live')
        if self.runner._remove_request is not self.patched_remove:raise RuntimeError('Outer wrapper remains')
        self.runner._remove_request=self.old_remove;self.runner=None
