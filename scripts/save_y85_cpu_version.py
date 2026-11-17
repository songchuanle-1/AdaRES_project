"""Retain only the seven actually CPU-tested Y85 files; preserve active checkout."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from prepare_y85_mask_feature_norm import ROOT,CID,files,lease,monitor,ref,tick
from check_ef_execution import validate_frozen
import closure

HEAD='2818322e6049b2541f1590a4fa9ad6854b3c598d'
BRANCH='refs/heads/adares-y85-mask-feature-norm'
PARENT='7bef2cd313b870583c970c56580be5a78097234d'
def git(*args,env=None,data=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,env=env,input=data,timeout=30)

def save(token):
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(c for c in work['candidates'] if c['id']==CID)
    assert c['stage']=='implementation' and not c.get('code_commit')
    result=ROOT/('pipeline/analyses/y85_cpu_'+token+'.json');cpu=tick.load(result)
    flags=('complete','implementation_test_passed','full_model_CPU_constructed','configured_TRAIN_data_collated','released_backbone_loaded','native_YAML_semantically_unchanged',
        'train_eval_train_roundtrip','decoder_initial_RNG_and_parent_weights_exact','norm_bypass_all_stages_W2_anchor_exact','norm_bypass_input_gradients_W2_anchor_close',
        'active_transform_changes_intended_initial_masks','active_initial_scores_unchanged','bypass_restoration_exact','original_mask_raw_norm_batch_prediction_identities_verified',
        'padded_batched_features_zero','original_hard_gate_algorithm_and_consumption_verified','original_shared_head_query_norm_consumption_verified')
    assert all(cpu[k] is True for k in flags)
    assert cpu['id']==CID and cpu['parameters']==164630149 and cpu['additional_parameters']==512
    assert cpu['synthetic_CPU_decoder_optimizer_steps']==2 and len(cpu['CPU_decoder_steps'])==len(cpu['CPU_norm_paths'])==2
    for i,row in enumerate(cpu['CPU_decoder_steps']):
        assert row['step']==i and row['prediction_calls']==row['indicator_calls']==row['query_norm_calls']==7 and row['coarse_gate_calls']==6
        assert row['raw_and_normalized_gradients_finite']
        assert len(row['new_gradient_maxima'])==len(row['parameter_updates'])==2 and all(v>0 for v in row['new_gradient_maxima']+row['parameter_updates'])
    for i,row in enumerate(cpu['CPU_norm_paths']):
        assert row['step']==i and all(row[k] is True for k in ('original_x_mask_output_identity','all_channels_and_original_epsilon','input_undetached','float64_LayerNorm_reference','shared_module_identity'))
    assert {r['dtype'] for r in cpu['component_controls']}=={'torch.float32','torch.float64'}
    assert all(v is True for r in cpu['component_controls'] for k,v in r.items() if k!='dtype')
    assert all(cpu[k] is False for k in ('CUDA_initialized','native_smoke_passed','native_task_backward','native_sparse_forward','validation_used','admission'))
    program=ROOT/('pipeline/checks/y85_cpu_'+token+'_program.py');payload=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert payload['goal_sha256']==cpu['goal_sha256']==ref('pipeline/goal.json')['sha256']
    for p in files():
        value=payload['variant']['config_data'] if p.suffix=='.yaml' else payload['variant']['python_files'][p.name]
        assert ref(p)['sha256']==value['sha256']
    paths=[p.relative_to(ROOT).as_posix() for p in files()]
    assert git('rev-parse','HEAD').decode().strip()==HEAD and not git('ls-files','--',*paths).strip()
    assert subprocess.run(['git','show-ref','--verify','--quiet',BRANCH],cwd=ROOT).returncode==1
    qref,rref=ref('pipeline/queue.json'),ref('pipeline/running.json');queue=monitor.load_queue(str(ROOT))
    for name,cid,pid in [('server-2','AdaRES-Y83',20849),('server-2','AdaRES-Y81',120624),('server-1','AdaRES-Y82',1121817)]:
        task=next(t for t in queue['gpu_queues'][name] if t['id']==cid)
        assert task['status']=='running' and task['launch_verified'] and task['pid']==pid
        closure.admission(ROOT,task)
        keeper=next(c for c in work['candidates'] if c['id']==cid);plan=tick.load(closure.artifact(ROOT,keeper['native_smoke_plan']))
        frozen=ast.literal_eval(ast.parse(closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(keeper,plan,frozen)==7
    paused=next(t for t in queue['gpu_queues']['server-2'] if t['id']=='AdaRES-Y57')
    assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    base=ROOT/('pipeline/checks/y85_cpu_version_'+token);source=base.with_name(base.name+'_source_snapshot.json');index=base.with_suffix('.index');out=base.with_suffix('.json')
    assert not any(p.exists() for p in (source,index,out))
    actual_index=Path(git('rev-parse','--git-path','index').decode().strip())
    if not actual_index.is_absolute():actual_index=ROOT/actual_index
    original=hashlib.sha256(actual_index.read_bytes()).hexdigest()
    tick.save(source,dict(complete=True,id=CID,token=token,CPU_tested_payload=payload,CPU_result=ref(result),CPU_log=ref(result.with_suffix('.log')),CPU_program=ref(program),exact_files=[ref(p) for p in files()],working_HEAD=HEAD))
    assert git('rev-parse','refs/heads/adares-y84-visual-query-init').decode().strip()==PARENT
    assert git('rev-parse',PARENT+'^').decode().strip()==HEAD
    parent_delta=[r.split('\t') for r in git('diff','--no-renames','--name-status',HEAD,PARENT).decode().splitlines()]
    assert len(parent_delta)==7 and all(status=='A' and 'adares_y84_visual_query_init' in path for status,path in parent_delta)
    assert all((ROOT/path).read_bytes()==git('show',PARENT+':'+path) for status,path in parent_delta)
    env=dict(os.environ,GIT_INDEX_FILE=str(index));git('read-tree',PARENT,env=env)
    for p in files():
        blob=git('hash-object','-w','--stdin',data=p.read_bytes()).decode().strip()
        git('update-index','--add','--cacheinfo','100644,'+blob+','+p.relative_to(ROOT).as_posix(),env=env)
    tree=git('write-tree',env=env).decode().strip()
    delta=[r.split('\t') for r in git('diff-tree','-r','--no-commit-id','--name-status',PARENT,tree).decode().splitlines()]
    assert sorted(delta)==sorted([['A',p] for p in paths])
    commit=git('commit-tree',tree,'-p',PARENT,'-m','Add Y85 shared visual mask-feature channel normalization').decode().strip()
    git('update-ref',BRANCH,commit,'0'*40)
    assert git('rev-parse','HEAD').decode().strip()==HEAD and hashlib.sha256(actual_index.read_bytes()).hexdigest()==original
    assert all(p.read_bytes()==git('show',commit+':'+p.relative_to(ROOT).as_posix()) for p in files())
    assert ref('pipeline/queue.json')==qref and ref('pipeline/running.json')==rref
    tick.save(out,dict(complete=True,id=CID,code_commit=commit,parent_commit=PARENT,retained_branch=BRANCH,working_HEAD=HEAD,
        main_index_unchanged=True,source_snapshot=ref(source),CPU_result=ref(result),exact_delta=delta,isolated_index=ref(index),
        original_admissions_and_frozen_sources_verified=True,queue_and_running_preserved=True,Y57_paused=True,
        native_executed=False,admitted=False,production_sync=False,evidence=[ref(__file__),ref('scripts/probe_y85_cpu.py'),ref('scripts/prepare_y85_mask_feature_norm.py')]))
    c.update(code_commit=commit,retained_branch=BRANCH,source_manifest=ref(out),implementation_test_passed=True,
        next_action='Keep active HEAD281832 and original Y81/Y82/Y83 source/observer contracts. Prepare exact Y85 native two-step mask-feature normalization/gradient/consumer checks; exclusive authorized GPU is required. No CPU-based admission or efficacy claim.')
    c['evidence'].extend([ref(source),ref(out),ref(index),ref(result),ref(result.with_suffix('.log')),ref(__file__)])
    work['tick_attempt']=dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],candidate_id=CID,kind='implementation_test',result='passed',
        command='python -B scripts/run_experiment_first_probe.py y85_cpu '+ref(program)['path']+' --token '+token+' --seconds 480',
        observation='CPU native YAML/full model/released backbone/original TRAIN collation; scalar affine LayerNorm/no-RNG/permutation/shift and input-gradient controls, seven-stage W2/output/input-gradient anchors only when the new norm is bypassed, intended active mask changes and exact raw-mask-feature/norm/batching/prediction consumers, original hard gates/shared head/query_norm and two finite synthetic decoder optimizer steps passed. Exact seven tested files retained with HEAD/index and live training preserved. No native GPU task PASS/admission/efficacy.',evidence=[ref(out),ref(result)])
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(version=ref(out),code_commit=commit,HEAD_unchanged=HEAD,native_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,240)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:save(a.token)
    finally:lock.close()
