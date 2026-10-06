"""Unsteered and single-ReBalance generation under the selected model protocol."""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / os.environ['RC_MODEL_SIZE']))
from common import ROOT,read,save,sha,verify,prompt_ids

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['unsteered','rebalance'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path,required=True)
    a=p.parse_args()
    assert os.environ.get('GPU_LOCK_FD') and a.output.parent.exists()
    verify(); plan=read(ROOT/'PILOT_PLAN.json');rows=read(ROOT/'inputs/pilot.json')
    assert len(rows)==plan['n'] and sha(ROOT/'inputs/pilot.json')==plan['inputs_sha256']
    a.output.mkdir(exist_ok=False)
    start=time.monotonic()
    try:
        identity_record=read(ROOT/'assets/source_identity.json')
        identity=identity_record['assets']
        for n,h in identity_record['runtime_hashes'].items():
            assert hashlib.sha256((a.runtime_root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,n
        for n,item in identity['model_files'].items():
            assert sha(Path(identity['model_path'])/n)==item['sha256'],n
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_BATCH_INVARIANT='0',
            VLLM_CACHE_ROOT=str(a.output/'compiler_cache'),
            PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
        sys.path[1:1]=[str(Path(__file__).resolve().parent/os.environ['RC_MODEL_SIZE']/'vendor'),str(a.runtime_root/'sources/EasySteer/vllm-steer'),str(a.runtime_root/'sources/EasySteer')]
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec
        from easysteer.vectors import from_pt_direction
        from transformers import AutoTokenizer
        import torch
        import vllm
        from cache_identity import install
        save(a.output/'CACHE_IDENTITY.json',install())
        assert Path(vllm.__file__).resolve().is_relative_to(a.runtime_root/'sources/EasySteer/vllm-steer')
        tok=AutoTokenizer.from_pretrained(identity['model_path'],local_files_only=True)
        fit=read(ROOT/'assets/original_fit.json')
        layer=21 if plan['model'].endswith('7B') else 20
        runtime=plan.get('runtime',{})
        llm=LLM(model=identity['model_path'],dtype='bfloat16',tensor_parallel_size=1,
            max_model_len=runtime.get('max_model_len',32768),max_num_seqs=runtime.get('max_num_seqs',256),
            max_num_batched_tokens=runtime.get('max_num_batched_tokens',32768),
            gpu_memory_utilization=runtime.get('gpu_memory_utilization',.9),
            enable_steer_vector=True,steer_algorithms=['rebalance','direct'],
            steer_graph_max_rank=32,max_steer_vectors=1,steer_graph_mode='in_graph',enforce_eager=False,
            enable_chunked_prefill=runtime.get('chunked_prefill',False),enable_prefix_caching=False,
            async_scheduling=runtime.get('async_scheduling',True),seed=plan['seed'])
        boundaries=sorted(i for s,i in tok.get_vocab().items() if '膴膴' in s)
        steer=SteeringSpec(vectors=[VectorSpec(name='frozen_rebalance',
            data=from_pt_direction(str(ROOT/'assets/original_vector.pt'),layers=[layer]),
            algorithm='rebalance',scale=1.,layers=[layer],normalize=False,
            apply=ApplySpec(generation_tokens=boundaries),
            params=dict(fit['parameters'],boundary_token_ids=boundaries,
                think_start_token_id=151648,think_end_token_id=151649))]) if a.method=='rebalance' else None
        prompts=[prompt_ids(tok,r) for r in rows]
        assert all(len(p)+16000<=runtime.get('max_model_len',32768) for p in prompts)
        prompt_hashes=[hashlib.sha256(str(p).encode()).hexdigest() for p in prompts]
        torch.cuda.synchronize();generation_start=time.monotonic()
        ids=llm.enqueue([dict(prompt_token_ids=v) for v in prompts],
            sampling_params=[SamplingParams(temperature=.7,top_p=.95,seed=plan['seed'],max_tokens=16000,
                skip_special_tokens=False) for _ in rows],steering=steer,use_tqdm=False)
        states=llm.llm_engine.output_processor.request_states
        mapping={states[rid].external_req_id:(rid,i) for i,rid in enumerate(ids)}
        core=llm.llm_engine.engine_core.engine_core
        found={};last=time.monotonic()
        with (a.output/(a.method+'.jsonl')).open('x',encoding='utf-8') as stream:
            while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                for out in llm.llm_engine.step():
                    assert out.finished
                    rid,i=mapping[out.request_id];assert rid not in found
                    ts=list(out.outputs[0].token_ids);assert 0<len(ts)<=16000
                    n=ts.index(151649) if 151649 in ts else len(ts)
                    row=dict(problem_sha256=rows[i]['problem_sha256'],dataset_index=rows[i]['dataset_index'],source=rows[i]['source'],
                        prompt_sha256=prompt_hashes[i],
                        token_ids=ts,text=tok.decode(ts,skip_special_tokens=True),tokens=len(ts),
                        thinking_tokens=n,answer_tokens=max(0,len(ts)-n-1),capped=len(ts)>=16000,
                        finish_reason=out.outputs[0].finish_reason)
                    found[rid]=row;stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                if time.monotonic()-last>30:
                    print(dict(method=a.method,finished=len(found),n=plan['n'],seconds=round(time.monotonic()-start,1)),flush=True)
                    last=time.monotonic()
        torch.cuda.synchronize();generation_seconds=time.monotonic()-generation_start
        assert len(found)==plan['n'] and {r['problem_sha256'] for r in found.values()}=={r['problem_sha256'] for r in rows}
        save(a.output/'COMPLETE.json',dict(passed=True,method=a.method,n=plan['n'],seconds=time.monotonic()-start,
            generation_seconds=generation_seconds,prompt_sha256=prompt_hashes,
            total_tokens=sum(r['tokens'] for r in found.values()),capped=sum(r['capped'] for r in found.values()),
            manifest_sha256=sha(ROOT/'MANIFEST.json'),free_generation=True,reference_answers_used=False,
            steering='none' if steer is None else 'frozen single ReBalance'))
    except BaseException:
        save(a.output/'FAILURE.json',dict(error=traceback.format_exc(),seconds=time.monotonic()-start));raise

if __name__=='__main__':main()
