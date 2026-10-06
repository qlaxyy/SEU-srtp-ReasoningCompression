"""CPU causal-phase and inner-sampler checks. No model/accuracy claims."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from types import SimpleNamespace as NS
import unittest
import sys
import types
from unittest.mock import patch
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'runtime/7b'),str(REPO/'runtime/7b/vendor')]
from phase_kernel import PhaseKernel,END,START,EOS
from teacher_force import TeacherForce
from positions import LivePositions
from vendor.mixture import Mixture
from native_phase import NativePhase,validate_native_params
from global_kernel import GlobalKernel,GlobalBank
from probed_teacher_force import ProbedTeacherForce
from pathlib import Path
import json


def synthetic_bank():
    d=8;k=2;r=2
    return Mixture(np.zeros(d),np.eye(d)[:,:r],np.ones(r),np.array([[0,0],[2,-2],[0,0.]]),
        np.array([[0,0],[-2,2],[0,0.]]),np.eye(d)[4:6],np.ones(d),float(np.sqrt(d)),('synthetic',))

class PhaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.pos=LivePositions(3,64);self.pos.add('a',2)
        self.k=PhaseKernel(8,capacity=3,max_tokens=64)

    def step(self,ids,start,hidden=None,base=None):
        n=len(ids)
        sch=self.pos.schedule(['a'],[start],[n],[2]);self.k.set_schedule(sch,torch.tensor(ids))
        h=torch.arange(n*8,dtype=torch.float32).reshape(n,8) if hidden is None else hidden
        residual=torch.ones_like(h)
        b=h+3 if base is None else base
        out=self.k(h,residual,b,torch.zeros(n))
        return h,residual,b,out

    def test_observer_exact_baseline_no_prompt_leak(self):
        _,_,b,out=self.step([END,END],0)
        self.assertIs(b,out);self.assertEqual(int(self.k.entries.sum()),0)

    def test_query_before_answer_and_full_body_pool(self):
        self.step([START,10],0)
        h,r,_,_=self.step([END],2)
        self.assertTrue(torch.equal(self.k.query[0],(h+r)[0]))
        q=self.k.query.clone();self.step([11],3);self.step([12],4)
        self.assertEqual(int(self.k.body_count[0]),2)
        self.assertTrue(torch.equal(q,self.k.query));self.assertEqual(int(self.k.entries[0]),1)
        self.step([EOS],5);self.assertEqual(int(self.k.body_count[0]),2)

    def test_bfloat16_reconstruction(self):
        self.step([START,10],0)
        h=torch.full((1,8),.003,dtype=torch.bfloat16)
        self.step([END],2,hidden=h)
        self.assertTrue(torch.equal(self.k.query[0],(h+torch.ones_like(h))[0].float()))

    def test_natural_restart_pauses_answer(self):
        self.step([START,10],0);self.step([END],2);self.step([11],3);self.step([START],4);self.step([12],5)
        self.assertEqual(int(self.k.body_count[0]),1);self.assertEqual(int(self.k.restarts[0]),1)
        self.step([END],6);self.assertEqual(int(self.k.entries[0]),2)

    def test_injection_only_answer_cached_query_zero_exact(self):
        self.k=PhaseKernel(8,synthetic_bank(),capacity=3,max_tokens=64)
        self.k.set_method('conditional',.25)
        _,_,b,out=self.step([START,10],0);self.assertTrue(torch.equal(b,out))
        _,_,b,out=self.step([END],2);self.assertFalse(torch.equal(b,out))
        direction=self.k.direction.clone()
        self.step([12],3,hidden=torch.ones(1,8)*50)
        self.assertTrue(torch.equal(direction,self.k.direction))
        self.k.set_method('conditional',0.)
        _,_,b,out=self.step([13],4);self.assertTrue(torch.equal(b,out))

    def test_invalid_slots_do_not_overwrite_real_row_zero(self):
        self.k=PhaseKernel(8,synthetic_bank(),capacity=3,max_tokens=64);self.k.set_method('global',.5)
        self.step([START,10],0);_,_,b,out=self.step([END],2)
        self.assertTrue(torch.equal(out,b+.5));self.assertEqual(int(self.k.answer_actions.sum()),1)

    def test_request_reordering_and_reuse(self):
        p=LivePositions(3,64);sa=p.add('a',2);sb=p.add('b',3)
        sch=p.schedule(['b','a'],[0,0],[3,2],[3,2])
        self.assertEqual(sch['row_indices'][sa],4);self.assertEqual(sch['row_indices'][sb],2)
        p.finish('a');self.assertEqual(p.add('c',4),sa)
        with self.assertRaises(ValueError):p.schedule(['b'],[0],[1],[3])

class ForceTests(unittest.TestCase):
    def make(self,teacher_class=TeacherForce):
        state=NS(req_id_to_index={})
        def add(out):
            for r in out.scheduled_new_reqs:state.req_id_to_index[r.req_id]=len(state.req_id_to_index)
        def remove(rid):state.req_id_to_index.pop(rid)
        def sample(logits,batch,**kw):return NS(sampled_token_ids=torch.full((batch.num_reqs,1),999))
        runner=NS(req_states=state,max_num_reqs=3,device='cpu',sampler=sample,add_requests=add,_remove_request=remove)
        llm=NS(llm_engine=NS(engine_core=NS(engine_core=NS(model_executor=NS(driver_worker=NS(worker=NS(model_runner=runner)))))))
        return teacher_class(llm,8),runner

    def test_forced_token_visible_to_outer_and_slot_mapping(self):
        f,r=self.make();f.register({'a':[END,EOS],'b':[17,EOS]})
        r.add_requests(NS(scheduled_new_reqs=[NS(req_id=x,num_computed_tokens=0,prompt_token_ids=[1,2]) for x in ['a','b']]))
        batch=NS(num_reqs=2,num_draft_tokens=0,idx_mapping=torch.tensor([1,0]),num_computed_tokens_np=np.array([0,0]),
            num_scheduled_tokens=np.array([2,2]),prefill_len_np=np.array([2,2]),seq_lens=torch.tensor([2,2]))
        observed=r.sampler(torch.zeros(2,24),batch).sampled_token_ids
        self.assertEqual(observed[:,0].tolist(),[17,END]);self.assertEqual(f.count[:2].tolist(),[1,1])
        batch.num_computed_tokens_np=np.array([2,2]);batch.num_scheduled_tokens=np.array([1,1]);batch.seq_lens+=1
        self.assertEqual(r.sampler(torch.zeros(2,24),batch).sampled_token_ids[:,0].tolist(),[EOS,EOS])
        r._remove_request('a');r._remove_request('b');self.assertTrue(all(v['complete'] for v in f.completed.values()));f.close()

    def test_partial_prefill_not_accepted(self):
        f,r=self.make();f.register({'a':[17,EOS]})
        r.add_requests(NS(scheduled_new_reqs=[NS(req_id='a',num_computed_tokens=0,prompt_token_ids=[1,2,3])]))
        b=NS(num_reqs=1,num_draft_tokens=0,idx_mapping=torch.tensor([0]),num_computed_tokens_np=np.array([0]),
            num_scheduled_tokens=np.array([2]),prefill_len_np=np.array([3]),seq_lens=torch.tensor([2]))
        self.assertEqual(int(r.sampler(torch.zeros(1,24),b).sampled_token_ids[0,0]),999)
        self.assertEqual(int(f.count[0]),0)

    def test_reject_bad_or_unregistered_sequences(self):
        f,r=self.make()
        with self.assertRaises(ValueError):f.register({'a':[EOS,17]})
        with self.assertRaises(ValueError):r.add_requests(NS(scheduled_new_reqs=[NS(req_id='a',num_computed_tokens=0)]))

    def test_probe_full_logits_before_forced_acceptance(self):
        f,r=self.make(ProbedTeacherForce);f.register({'a':[END,17,EOS]})
        r.add_requests(NS(scheduled_new_reqs=[NS(req_id='a',num_computed_tokens=0,prompt_token_ids=[1,2])]))
        for j in range(3):
            b=NS(num_reqs=1,num_draft_tokens=0,idx_mapping=torch.tensor([0]),num_computed_tokens_np=np.array([0 if j==0 else j+1]),
                num_scheduled_tokens=np.array([2 if j==0 else 1]),prefill_len_np=np.array([2]),seq_lens=torch.tensor([2+j]))
            r.sampler(torch.full((1,8),float(10+j)),b)
        r._remove_request('a');probe=f.probe_completed['a']
        self.assertEqual(probe['count'],1);self.assertTrue(np.array_equal(probe['logits'],np.full(8,11.)))
        f.close()

class GlobalTests(unittest.TestCase):
    def test_real_bank_identity(self):
        root=REPO
        bank=GlobalBank(root/'assets/7b/pca_mean16.npy')
        self.assertEqual(len(bank.fit_ids),271);self.assertTrue(np.isfinite(bank.direction).all())

    def test_natural_global_injection_zero_and_restart(self):
        k=GlobalKernel(8,NS(direction=np.ones(8)),capacity=3,max_tokens=64)
        p=LivePositions(3,64);p.add('a',2)
        k.set_method('srq_pca16',.25)
        def step(ids,start):
            sch=p.schedule(['a'],[start],[len(ids)],[2]);k.set_schedule(sch,torch.tensor(ids))
            h=torch.ones(len(ids),8,dtype=torch.bfloat16);return k(h,h,h,torch.zeros(len(ids)))
        self.assertTrue(torch.equal(step([START,9],0),torch.ones(2,8)))
        self.assertTrue(torch.equal(step([END],2),torch.full((1,8),1.25)))
        k.set_method('srq_pca16',0);self.assertTrue(torch.equal(step([10],3),torch.ones(1,8)))
        k.set_method('srq_pca16',.25);self.assertTrue(torch.equal(step([START],4),torch.ones(1,8)))
        with self.assertRaises(ValueError):k.set_method('conditional',.25)


class PenaltyCurveTests(unittest.TestCase):
    def test_monotone_bounded_curve_and_calibration_anchor(self):
        from strength_policy import multiplier
        m=.25;rho=m/(1-m);bound=-1.5
        s=torch.linspace(0,1,101)
        k=multiplier(torch,s*bound,32,bound,'rational',rho)
        self.assertEqual(float(k[0]),1.);self.assertEqual(float(k[-1]),32.)
        self.assertTrue(bool((k[1:]>=k[:-1]).all()))
        anchor=multiplier(torch,torch.tensor([m*bound]),32,bound,'rational',rho)
        self.assertAlmostEqual(float(anchor[0]),16.5,places=5)

    def test_fixed_one_is_exact_historical_bfloat16_subtraction(self):
        from strength_policy import adjusted_values,PENALTY
        values=torch.tensor([[16.2,7.,6.8]],dtype=torch.bfloat16)
        result,_=adjusted_values(torch,values,torch.tensor([-.7]),1,-1.5,'rational',.4)
        self.assertTrue(torch.equal(result,values-PENALTY))

    def test_nonfinite_and_positive_coefficients_do_not_escalate(self):
        from strength_policy import multiplier
        k=multiplier(torch,torch.tensor([float('nan'),float('inf'),.5,0.]),32,-1.5,'rational',.4)
        self.assertTrue(torch.equal(k,torch.ones(4)))

    def test_bad_curve_parameters_rejected(self):
        from strength_policy import validate
        for maximum,bound,curve,shape in [(33,-1,'rational',.4),(32,1,'rational',.4),(32,-1,'unknown',.4),(32,-1,'rational',0)]:
            with self.assertRaises(ValueError):validate(maximum,bound,curve,shape)

class InitializationTests(unittest.TestCase):
    def pinned(self):
        path=REPO/'runtime/7b/vendor/pinned_rebalance.py'
        module=types.ModuleType('pinned_rebalance_test');sys.modules[module.__name__]=module
        code=path.read_text(encoding='utf-8')
        if 'from __future__ import annotations' not in code:code='from __future__ import annotations\n'+code
        exec(compile(code,str(path),'exec'),module.__dict__)
        expected=json.loads((REPO/'assets/7b/original_fit.json').read_text(encoding='utf-8'))['parameters']
        return module.ReBalanceParams(boundary_token_ids=(271,),think_start_token_id=START,think_end_token_id=END,**expected),expected

    def test_actual_pinned_params_with_no_newer_fields(self):
        p,expected=self.pinned();self.assertFalse(hasattr(p,'constant_control'));validate_native_params(p,expected)

    def test_reject_altered_params_and_schema(self):
        p,expected=self.pinned()
        with self.assertRaises(ValueError):validate_native_params(p,dict(expected,high_val_2=.2))
        altered=NS(**vars(p),negative_only=True)
        with self.assertRaises(ValueError):validate_native_params(altered,expected)

    def test_only_unarmed_warmup_may_lack_payload(self):
        controller=type('Controller',(),dict(init_graph_table=lambda *a,**k:None,process_output_hook=lambda *a:None))
        c=types.ModuleType('vllm.steer_vectors.controllers');c.DecoderSteerController=controller
        mr=types.ModuleType('vllm.v1.worker.gpu.model_runner');mr.fill_graph_steer_buffers=lambda *a:None
        mods={n:types.ModuleType(n) for n in ['vllm','vllm.steer_vectors','vllm.v1','vllm.v1.worker','vllm.v1.worker.gpu']}
        mods['vllm.steer_vectors'].controllers=c;mods['vllm.v1.worker.gpu'].model_runner=mr
        with patch.dict(sys.modules,mods):
            phase=NativePhase({});phase.kernel=PhaseKernel(8,capacity=3,max_tokens=64)
            manager=NS(graph_batch_entries=lambda:{})
            mr.fill_graph_steer_buffers(NS(num_draft_tokens=0),NS(),manager)
            self.assertFalse(bool(phase.kernel.row_valid.any()))
            phase.armed=True
            with self.assertRaises(ValueError):mr.fill_graph_steer_buffers(NS(num_draft_tokens=0),NS(),manager)

class ReplayStateTests(unittest.TestCase):
    def test_historical_masks_restart_and_no_future(self):
        from replay_positions import historical_mask
        tokens=[END,START,END,10,START,11,END,12,EOS]
        self.assertEqual(historical_mask(tokens,2,8).tolist(),[False,False,True,True,False,False,True,True,False])
        self.assertFalse(historical_mask(tokens,8,8).any())

    def test_answer_replay_applies_every_old_token_without_double_count(self):
        from replay_positions import ReplayPositions
        from replay_kernel import ReplayKernel
        p=ReplayPositions(2,64)
        tokens=[START,10,END,11,12,13]
        p.resume('a',2,tokens,5,1)
        k=ReplayKernel(8,NS(direction=np.ones(8)),capacity=2,max_tokens=64)
        k.set_method('srq_pca16',.25);k.answer_phase[0]=True;k.entries[0]=1;k.body_count[0]=2;k.answer_actions[0]=3
        q=torch.arange(8).float();k.query[0]=q
        # Recompute old END and two old body tokens, then observe the newest body.
        sch=p.schedule(['a'],[0],[4],[6]);k.set_schedule(sch,torch.tensor(tokens[:4]))
        h=torch.ones(4,8,dtype=torch.bfloat16);out=k(h,h,h,torch.zeros(4))
        self.assertTrue(torch.equal(out[:2],h[:2]));self.assertTrue(torch.equal(out[2:],h[2:]+.25))
        self.assertEqual(int(k.answer_actions[0]),3);self.assertEqual(int(k.body_count[0]),2)
        sch=p.schedule(['a'],[4],[2],[6]);k.set_schedule(sch,torch.tensor(tokens[4:]))
        h=torch.ones(2,8,dtype=torch.bfloat16);out=k(h,h,h,torch.zeros(2))
        self.assertTrue(torch.equal(out,h+.25));self.assertEqual(int(k.answer_actions[0]),4)
        self.assertEqual(int(k.body_count[0]),3);self.assertEqual(int(k.entries[0]),1);self.assertTrue(torch.equal(k.query[0],q))
        k.set_method('off',0.);sch=p.schedule(['a'],[6],[1],[6]);k.set_schedule(sch,torch.tensor([14]))
        h=torch.ones(1,8,dtype=torch.bfloat16);self.assertTrue(torch.equal(k(h,h,h,torch.zeros(1)),h))

    def test_replay_phase_with_chunked_prompt_and_resume_clock(self):
        from replay_positions import ReplayPositions
        p=ReplayPositions(2,64);p.resume('a',5,[START,END,8,END,10],3,1)
        a=p.schedule(['a'],[0],[3],[5]);self.assertFalse(a['row_valid'].any());self.assertFalse(a['history_answer_mask'].any())
        b=p.schedule(['a'],[3],[2],[5]);self.assertTrue(b['row_valid'][0]);self.assertFalse(b['generated_input'][0])
        with self.assertRaises(ValueError):p.resume('b',2,[START,5,6,7],1,1)

    def test_penalty_replay_preserves_float_statistics_and_histogram(self):
        sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
        from replay_penalty import ReplayPenalty
        p=object.__new__(ReplayPenalty);p.torch=torch;p.suspending={'r'};p.active={'r':0};p.suspended={};p.replay_events=[]
        for k in p.fields:setattr(p,k,torch.tensor([3.25,0.]))
        p.count=torch.tensor([9,0]);p.prompt_len=torch.tensor([2,0]);p.histogram=torch.arange(64).reshape(2,32).clone()
        p.counts=torch.arange(10).reshape(2,5).clone();p.original_remove=lambda rid:None
        p.remove_request('r');self.assertNotIn('r',p.active)
        p.completed={};p.original_add=lambda out:None;p.check_penalty_request=lambda rid:None
        p.runner=NS(req_states=NS(req_id_to_index={'r':1}),steer_vector_state=NS(_dynamic_indices={'r':1}))
        opts=NS(n=1,structured_outputs=None,logit_bias=None,bad_words=None,presence_penalty=0,frequency_penalty=0,repetition_penalty=1,min_tokens=0)
        r=NS(req_id='r',num_computed_tokens=0,prompt_token_ids=[START,1],prefill_token_ids=[START,1]+[2]*9,sampling_params=opts)
        p.add_requests(NS(scheduled_new_reqs=[r]));self.assertEqual(p.active['r'],1)
        self.assertEqual(float(p.multiplier_sum[1]),3.25);self.assertTrue(torch.equal(p.histogram[0],p.histogram[1]))
        self.assertTrue(p.replay_events[-1]['all_fields_exact']);self.assertFalse(p.suspended)

class ReplayOracleTests(unittest.TestCase):
    def test_native_and_full_token_references_have_distinct_expected_outputs(self):
        from replay_positions import ReplayPositions
        from replay_kernel import ReplayKernel
        tokens=[START,10,END,11,12]
        baseline=torch.linspace(-3,3,40,dtype=torch.bfloat16).reshape(5,8)
        direction=np.linspace(-.4,.6,8,dtype=np.float32)
        expected=baseline.clone();expected[2:]+= (.25*torch.tensor(direction)).to(torch.bfloat16)
        for mode in [0,1,2]:
            p=ReplayPositions(2,64);p.resume('a',2,tokens,4,1)
            k=ReplayKernel(8,NS(direction=direction),capacity=2,max_tokens=64);k.set_method('srq_pca16',.25)
            k.answer_phase[0]=True;k.entries[0]=1;k.body_count[0]=1;k.answer_actions[0]=2
            k.engineering_reference.fill_(mode);k.set_schedule(p.schedule(['a'],[0],[5],[5]),torch.tensor(tokens))
            actual=k(baseline,baseline,baseline,torch.zeros(5))
            self.assertTrue(torch.equal(actual,baseline if mode==1 else expected))
            self.assertEqual(int(k.answer_actions[0]),3);self.assertEqual(int(k.body_count[0]),2)
