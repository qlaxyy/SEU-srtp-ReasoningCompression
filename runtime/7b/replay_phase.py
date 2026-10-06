"""Save natural answer-phase state and restore it around native KV eviction."""
import numpy as np
import torch
from native_phase import NativePhase
from replay_positions import ReplayPositions
from replay_kernel import ReplayKernel


class ReplayPhase(NativePhase):
    saved_fields=('answer_phase','query','answer_sum','direction','entries','body_count','restarts','duplicate_ends',
        'native_answer_conflicts','answer_actions','reason_actions')

    def __init__(self,*args,**kwargs):
        kwargs['kernel_type']=ReplayKernel
        super().__init__(*args,**kwargs)
        self.positions=ReplayPositions(self.positions.capacity,self.positions.max_tokens)
        self.suspended={};self.suspending=set();self.replay_events=[]

    def attach(self,llm):
        runner=llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        self.runner=runner;self.old_remove=runner._remove_request;self.old_add=runner.add_requests;self.old_finish=runner.finish_requests
        def finish(output):
            self.suspending=set(output.preempted_req_ids or ())-set(output.finished_req_ids)
            try:self.old_finish(output)
            finally:self.suspending=set()
            for rid in output.finished_req_ids:self.suspended.pop(rid,None)
        def remove(rid):
            if rid in self.positions.requests:
                row=self.positions.finish(rid);slot=row['slot']
                if rid in self.suspending:
                    self.suspended[rid]=dict(row=row,buffers={k:getattr(self.kernel,k)[slot].clone() for k in self.saved_fields})
                    self.replay_events.append(dict(request_id=rid,event='suspend',processed=row['next_position']))
                else:
                    for key in ('entries','body_count','restarts','duplicate_ends','native_answer_conflicts','answer_actions','reason_actions'):
                        row[key]=int(getattr(self.kernel,key)[slot].cpu())
                    for key in ('query','answer_sum'):
                        row[key]=getattr(self.kernel,key)[slot].detach().float().cpu().numpy().copy()
                        if not np.isfinite(row[key]).all():raise ValueError('Nonfinite feature')
                    row['cached_direction_norm']=float(self.kernel.direction[slot].float().norm().cpu());self.completed[rid]=row
                self.checked.discard(rid);self.kernel.clear_slot(slot)
            return self.old_remove(rid)
        def add(output):
            self.old_add(output)
            for r in output.scheduled_new_reqs:
                rid=r.req_id;saved=self.suspended.pop(rid,None)
                if saved is None:
                    if len(r.prefill_token_ids)!=len(r.prompt_token_ids):raise ValueError('Missing answer-phase replay state')
                    continue
                row=saved['row'];cutoff=max(row['next_position'],row.get('history_cutoff',0))
                slot=self.positions.resume(rid,len(r.prompt_token_ids),r.prefill_token_ids,cutoff,row.get('replays',0)+1)
                self.kernel.clear_slot(slot)
                for k,v in saved['buffers'].items():getattr(self.kernel,k)[slot].copy_(v)
                assert all(torch.equal(getattr(self.kernel,k)[slot],v) for k,v in saved['buffers'].items())
                hist=int(self.positions.requests[rid]['history_mask'].sum())
                self.replay_events.append(dict(request_id=rid,event='restore',processed=cutoff,
                    replay_prefill_tokens=len(r.prefill_token_ids),history_answer_tokens=hist,all_fields_exact=True))
        self.patched_remove=remove;self.patched_add=add;self.patched_finish=finish
        runner._remove_request=remove;runner.add_requests=add;runner.finish_requests=finish;self.armed=True

    def detach(self):
        if self.positions.requests or self.suspended:raise RuntimeError('Phase replay not drained')
        if self.runner._remove_request is not self.patched_remove:raise RuntimeError('Outer wrapper remains')
        self.runner._remove_request=self.old_remove;self.runner.add_requests=self.old_add;self.runner.finish_requests=self.old_finish
        self.runner=None
