"""Full online evaluation of the frozen old SRQ/PCA16 vector with ReBalance+cal32."""
import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from common import ROOT,read,save,sha,verify,prompt_ids


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--runtime-root',type=Path,required=True)
    a=p.parse_args();plan=read(ROOT/'PILOT_PLAN.json');questions=read(ROOT/'inputs/pilot.json')
    assert len(questions)==plan['n'] and plan['dataset'] in ('math500','gsm8k','amc23','aime2025','custom')
    assert sha(ROOT/'inputs/pilot.json')==plan['inputs_sha256']
    if a.dry_run:
        print(dict(questions=len(questions),arms=plan['methods'],seed=plan['seed'],cap=16000,
            new_reasoning=True,forced_closure=False,answer_bank_ready=(ROOT/'assets/pca_mean16.npy').is_file()));return
    assert a.execute and a.output is not None and os.name=='posix' and os.environ.get('GPU_LOCK_FD');verify()
    assert not os.environ.get('PYTHONOPTIMIZE')
    fit_receipt=read(ROOT/'assets/SRQ_VECTOR_RECEIPT.json')
    assert fit_receipt['new_fit'] is False and fit_receipt['beta']==.25
    assert sha(ROOT/'assets/pca_mean16.npy')==fit_receipt['vector_sha256']
    a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Full dataset deadline')));signal.alarm(plan['gpu_limit_seconds'])
    save(a.output/'JOB.json',dict(command=sys.argv,manifest_sha256=sha(ROOT/'MANIFEST.json'),source_commit=os.environ['SOURCE_COMMIT'],plan=plan))
    try:
        identity=read(ROOT/'assets/source_identity.json');assets=identity['assets']
        for n,h in identity['runtime_hashes'].items():assert hashlib.sha256((a.runtime_root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,n
        for n,item in assets['model_files'].items():assert sha(Path(assets['model_path'])/n)==item['sha256'],n
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_BATCH_INVARIANT='0',
            VLLM_CACHE_ROOT=str(a.output/'compiler_cache'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
        sys.path[1:1]=[str(Path(__file__).resolve().parent/'vendor'),str(a.runtime_root/'sources/EasySteer/vllm-steer'),str(a.runtime_root/'sources/EasySteer')]
        import numpy as np
        import torch
        import vllm
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec
        from easysteer.vectors import from_pt_direction
        from transformers import AutoTokenizer
        from native_phase import NativePhase
        from probed_teacher_force import ProbedTeacherForce
        from penalty_adapter import ResearchAdapter
        from cache_identity import install
        from global_kernel import GlobalBank,GlobalKernel
        save(a.output/'CACHE_IDENTITY.json',install())
        assert Path(vllm.__file__).resolve().is_relative_to(a.runtime_root/'sources/EasySteer/vllm-steer')
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        fit=read(ROOT/'assets/original_fit.json');bank=GlobalBank(ROOT/'assets/pca_mean16.npy')
        assert not set(bank.fit_ids)&{r['problem_sha256'] for r in questions}
        phase=NativePhase(fit['parameters'],bank=bank,kernel_type=GlobalKernel)
        llm=LLM(model=assets['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
            max_num_seqs=256,max_num_batched_tokens=32768,gpu_memory_utilization=.9,enable_steer_vector=True,
            steer_algorithms=['rebalance','direct'],steer_graph_max_rank=32,max_steer_vectors=1,
            steer_graph_mode='in_graph',enforce_eager=False,enable_chunked_prefill=False,
            enable_prefix_caching=False,async_scheduling=True,seed=plan['seed'])
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        boundaries=sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s)
        steer=SteeringSpec(vectors=[VectorSpec(name='native_reasoning_cal32',data=from_pt_direction(str(ROOT/'assets/original_vector.pt'),layers=[20]),
            algorithm='rebalance',scale=1.,layers=[20],normalize=False,apply=ApplySpec(generation_tokens=boundaries),
            params=dict(fit['parameters'],boundary_token_ids=boundaries,think_start_token_id=151648,think_end_token_id=151649))])
        with np.load(ROOT/'assets/opening.npz',allow_pickle=False) as f:tables={k:f[k] for k in f.files}

        def new_penalty():
            return ResearchAdapter(llm,tok,tables=tables,penalty_multiplier=plan['maximum'],negative_bound=min(fit['parameters']['low_val_1'],fit['parameters']['low_val_2']),
                mode='on',penalty_kind='dynamic',curve='rational',shape=plan['calibration_shape'])

        def execute(rows,method,strength,targets=None,filename=None):
            assert not phase.positions.requests
            phase.kernel.set_method(method,strength)
            prompts=[prompt_ids(tok,r) for r in rows]
            ids=llm.enqueue([dict(prompt_token_ids=p) for p in prompts],sampling_params=[SamplingParams(temperature=.7,top_p=.95,seed=plan['seed'],
                max_tokens=len(targets[i]) if targets is not None else 16000,skip_special_tokens=False) for i in range(len(rows))],steering=steer,use_tqdm=False)
            if targets is not None:forced.register(dict(zip(ids,targets)))
            states=llm.llm_engine.output_processor.request_states
            mapping={states[rid].external_req_id:(rid,i) for i,rid in enumerate(ids)}
            found={};last=time.monotonic();started=last
            stream=(a.output/filename).open('x',encoding='utf-8') if filename else None
            try:
                while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                    for out in llm.llm_engine.step():
                        assert out.finished;rid,i=mapping[out.request_id];assert rid not in found
                        ts=list(out.outputs[0].token_ids);assert 0<len(ts)<=16000
                        if targets is not None:assert ts==targets[i] or ts==targets[i][:-1]
                        n=ts.index(151649) if 151649 in ts else len(ts)
                        row=dict(problem_sha256=rows[i]['problem_sha256'],dataset_index=rows[i]['dataset_index'],source=rows[i]['source'],
                            token_ids=ts,text=tok.decode(ts,skip_special_tokens=True),tokens=len(ts),thinking_tokens=n,
                            answer_tokens=max(0,len(ts)-n-1),capped=len(ts)>=16000,finish_reason=out.outputs[0].finish_reason)
                        found[rid]=row
                        if stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    if time.monotonic()-last>30:
                        print(dict(method=method,finished=len(found),n=len(rows),seconds=time.monotonic()-started),flush=True);last=time.monotonic()
                llm.llm_engine.step()
                for rid in ids:
                    if rid in runner.req_states.req_id_to_index:runner._remove_request(rid)
                assert len(found)==len(rows)
                audits=[]
                for rid in ids:
                    audit=phase.completed[rid];pen=penalty.route_completed[rid]
                    assert audit['native_answer_conflicts']==0 and pen['answer_actions']==pen['prefill_actions']==pen['repair_actions']==0
                    if method!='off' and strength and audit['answer_actions']:
                        assert audit['cached_direction_norm']>1e-8
                    audits.append(dict(request_id=rid,phase={k:v for k,v in audit.items() if k not in ('query','answer_sum')},
                        query_sha256=hashlib.sha256(audit['query'].tobytes()).hexdigest(),penalty=pen))
                return [found[rid] for rid in ids],ids,audits,time.monotonic()-started
            finally:
                if stream:stream.close()

        # Real-engine zero identity and nonzero-effect check under the same
        # forced prefix. Synthetic engineering sequences are not scored.
        from engineering import examples
        engineering=examples(tok)
        targets=[r['reasoning_ids']+r['plain_body_ids'][:12]+[151643] for r in engineering]
        forced=ProbedTeacherForce(llm,read(Path(assets['model_path'])/'config.json')['vocab_size']);penalty=new_penalty();phase.attach(llm)
        checks={}
        for label,method,strength in [('off','off',0.),('zero','srq_pca16',0.),('on','srq_pca16',.25),('off_after','off',0.)]:
            rec,ids,audits,seconds=execute(engineering,method,strength,targets)
            checks[label]=dict(ids=ids,audits=audits,seconds=seconds)
            for rid in ids:
                assert forced.completed[rid]['complete'] and phase.completed[rid]['entries']==1
                assert forced.probe_completed[rid]['count']==1
            probe_file=a.output/('engineering_'+label+'_logits.npz')
            with probe_file.open('xb') as f:
                np.savez_compressed(f,logits=np.stack([forced.probe_completed[rid]['logits'] for rid in ids]))
            checks[label].update(probe_file=probe_file.name,probe_sha256=sha(probe_file))
        for other in ('zero','off_after'):
            for a_id,b_id in zip(checks['off']['ids'],checks[other]['ids']):
                for key in ('query','answer_sum'):assert np.array_equal(phase.completed[a_id][key],phase.completed[b_id][key]),'Zero/off restoration changed hidden states'
                assert np.array_equal(forced.probe_completed[a_id]['logits'],forced.probe_completed[b_id]['logits']),'Zero/off answer logits differ'
        changed=[]
        for a_id,b_id in zip(checks['off']['ids'],checks['on']['ids']):
            assert np.array_equal(phase.completed[a_id]['query'],phase.completed[b_id]['query']),'Answer changed pre-entry query'
            # Fixed tokens cannot expose current-layer output edits in that
            # layer's pre-edit states. Check actual downstream logits instead.
            changed.append(not np.array_equal(forced.probe_completed[a_id]['logits'],forced.probe_completed[b_id]['logits']))
            assert phase.completed[b_id]['answer_actions']>0
        assert all(changed),'Nonzero direction had no observed effect'
        save(a.output/'ENGINEERING_COMPLETE.json',dict(passed=True,off_zero_restored_hidden_exact=True,
            off_zero_restored_full_entry_logits_exact=True,entry_queries_unchanged=True,nonzero_entry_logits_changed=True,checks=checks))
        phase.detach();penalty.close();forced.close()
        penalty=new_penalty();phase.attach(llm)
        arms={}
        for method in plan['methods']:
            rec,ids,audits,seconds=execute(questions,method,plan['answer_strength'],filename=method+'.jsonl')
            save(a.output/(method+'_AUDIT.json'),dict(n=len(rec),seconds=seconds,rows=audits))
            arms[method]=dict(n=len(rec),seconds=seconds,total_tokens=sum(r['tokens'] for r in rec),
                thinking_tokens=sum(r['thinking_tokens'] for r in rec),answer_tokens=sum(r['answer_tokens'] for r in rec),capped=sum(r['capped'] for r in rec))
        phase.detach();penalty.close()
        save(a.output/'COMPLETE.json',dict(passed=True,arms=arms,seconds=time.monotonic()-begin,
            manifest_sha256=sha(ROOT/'MANIFEST.json'),free_generation=True,reference_answers_used=False,
            original_reasoning_controller_unchanged=True,forced_closure=False,phase='natural boundary answer injection, original reasoning steering'))
        print(json.dumps(arms),flush=True)
    except BaseException:
        save(a.output/'FAILURE.json',dict(error=traceback.format_exc(),seconds=time.monotonic()-begin));raise


if __name__=='__main__':main()
