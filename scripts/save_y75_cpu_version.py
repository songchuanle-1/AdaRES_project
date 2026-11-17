"""Retain only the seven actually CPU-tested Y75 files; preserve active checkout."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from prepare_y75_language_query_init import ROOT,CID,files,lease,monitor,ref,tick
from check_ef_execution import validate_frozen
import closure

HEAD='7da625ab750447eaad76e82309c89d887df93994'
BRANCH='refs/heads/adares-y75-language-query-init'
PARENT='3c91105459f773326910b1eeaf16450e2d6b90ac'
def git(*args,env=None,data=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,env=env,input=data,timeout=30)

def save(token):
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(c for c in work['candidates'] if c['id']==CID)
    assert c['stage']=='implementation' and not c.get('code_commit')
    result=ROOT/('pipeline/analyses/y75_cpu_'+token+'.json');cpu=tick.load(result)
    flags=('complete','implementation_test_passed','full_model_CPU_constructed','configured_TRAIN_data_collated','released_backbone_loaded',
        'native_YAML_semantically_unchanged','train_eval_train_roundtrip','decoder_initial_RNG_and_parent_weights_exact','decoder_all_stages_anchor_exact',
        'decoder_input_gradients_anchor_close','original_generated_query_projected_language_and_padding_identities_verified','decoder_backward_finite',
        'decoder_initialization_gradient_finite_nonzero','nonzero_initialization_changes_initial_and_final_masks','bypassed_initialization_negative_control_exact')
    assert all(cpu[k] is True for k in flags)
    assert cpu['id']==CID and cpu['parameters']==164695173 and cpu['additional_parameters']==65536
    assert cpu['decoder_initialization_calls']==1 and cpu['decoder_indicator_calls']==7 and cpu['initial_prediction_consumption_verified'] is True
    assert {r['dtype'] for r in cpu['component_controls']}=={'torch.float32','torch.float64'}
    assert all(v is True for r in cpu['component_controls'] for k,v in r.items() if k!='dtype')
    assert all(cpu[k] is False for k in ('CUDA_initialized','native_smoke_passed','native_task_backward','native_sparse_forward','validation_used','admission'))
    program=ROOT/('pipeline/checks/y75_cpu_'+token+'_program.py');payload=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert payload['goal_sha256']==cpu['goal_sha256']==ref('pipeline/goal.json')['sha256']
    for p in files():
        value=payload['variant']['config_data'] if p.suffix=='.yaml' else payload['variant']['python_files'][p.name]
        assert ref(p)['sha256']==value['sha256']
    paths=[p.relative_to(ROOT).as_posix() for p in files()]
    assert git('rev-parse','HEAD').decode().strip()==HEAD and not git('ls-files','--',*paths).strip()
    assert subprocess.run(['git','show-ref','--verify','--quiet',BRANCH],cwd=ROOT).returncode==1
    qref,rref=ref('pipeline/queue.json'),ref('pipeline/running.json');queue=monitor.load_queue(str(ROOT))
    for name,cid,pid in [('server-2','AdaRES-Y71',53714),('server-1','AdaRES-Y72',311445),('server-2','AdaRES-Y73',62829)]:
        task=next(t for t in queue['gpu_queues'][name] if t['id']==cid)
        assert task['status']=='running' and task['launch_verified'] and task['pid']==pid
        closure.admission(ROOT,task)
        keeper=next(c for c in work['candidates'] if c['id']==cid);plan=tick.load(closure.artifact(ROOT,keeper['native_smoke_plan']))
        frozen=ast.literal_eval(ast.parse(closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(keeper,plan,frozen)==7
    paused=next(t for t in queue['gpu_queues']['server-2'] if t['id']=='AdaRES-Y57')
    assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    base=ROOT/('pipeline/checks/y75_cpu_version_'+token);source=base.with_name(base.name+'_source_snapshot.json');index=base.with_suffix('.index');out=base.with_suffix('.json')
    assert not any(p.exists() for p in (source,index,out))
    actual_index=Path(git('rev-parse','--git-path','index').decode().strip())
    if not actual_index.is_absolute():actual_index=ROOT/actual_index
    original=hashlib.sha256(actual_index.read_bytes()).hexdigest()
    tick.save(source,dict(complete=True,id=CID,token=token,CPU_tested_payload=payload,CPU_result=ref(result),CPU_log=ref(result.with_suffix('.log')),CPU_program=ref(program),exact_files=[ref(p) for p in files()],working_HEAD=HEAD))
    assert git('rev-parse','refs/heads/adares-y74-language-update-gate').decode().strip()==PARENT
    assert git('rev-parse',PARENT+'^').decode().strip()==HEAD
    parent_delta=[r.split('\t') for r in git('diff','--no-renames','--name-status',HEAD,PARENT).decode().splitlines()]
    assert len(parent_delta)==7 and all(status=='A' and 'adares_y74_language_update_gate' in path for status,path in parent_delta)
    assert all((ROOT/path).read_bytes()==git('show',PARENT+':'+path) for status,path in parent_delta)
    env=dict(os.environ,GIT_INDEX_FILE=str(index));git('read-tree',PARENT,env=env)
    for p in files():
        blob=git('hash-object','-w','--stdin',data=p.read_bytes()).decode().strip()
        git('update-index','--add','--cacheinfo','100644,'+blob+','+p.relative_to(ROOT).as_posix(),env=env)
    tree=git('write-tree',env=env).decode().strip()
    delta=[r.split('\t') for r in git('diff-tree','-r','--no-commit-id','--name-status',PARENT,tree).decode().splitlines()]
    assert sorted(delta)==sorted([['A',p] for p in paths])
    commit=git('commit-tree',tree,'-p',PARENT,'-m','Add Y75 language-conditioned initial query interaction').decode().strip()
    git('update-ref',BRANCH,commit,'0'*40)
    assert git('rev-parse','HEAD').decode().strip()==HEAD and hashlib.sha256(actual_index.read_bytes()).hexdigest()==original
    assert all(p.read_bytes()==git('show',commit+':'+p.relative_to(ROOT).as_posix()) for p in files())
    assert ref('pipeline/queue.json')==qref and ref('pipeline/running.json')==rref
    tick.save(out,dict(complete=True,id=CID,code_commit=commit,parent_commit=PARENT,retained_branch=BRANCH,working_HEAD=HEAD,
        main_index_unchanged=True,source_snapshot=ref(source),CPU_result=ref(result),exact_delta=delta,isolated_index=ref(index),
        original_admissions_and_frozen_sources_verified=True,queue_and_running_preserved=True,Y57_paused=True,
        native_executed=False,admitted=False,production_sync=False,evidence=[ref(__file__),ref('scripts/probe_y75_cpu.py'),ref('scripts/prepare_y75_language_query_init.py')]))
    c.update(code_commit=commit,retained_branch=BRANCH,source_manifest=ref(out),implementation_test_passed=True,
        next_action='Keep active HEAD7da and all three running source/observer contracts. Prepare the exact Y75 native two-step program and its true path/gradient/consumer checks; fresh exclusive authorized GPU is required before execution. No CPU-based admission or efficacy claim.')
    c['evidence'].extend([ref(source),ref(out),ref(index),ref(result),ref(result.with_suffix('.log')),ref(__file__)])
    work['tick_attempt']=dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],candidate_id=CID,kind='implementation_test',result='passed',
        command='python -B scripts/run_experiment_first_probe.py y75_cpu '+ref(program)['path']+' --token '+token+' --seconds 480',
        observation='CPU native YAML/full model/released backbone/actual TRAIN collation and independent query-language product/one exact initial prediction consumer/seven-stage anchors/backward controls passed. Exact seven tested files retained without moving HEAD/index or altering live training. No native GPU pass, admission or efficacy.',evidence=[ref(out),ref(result)])
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(version=ref(out),code_commit=commit,HEAD_unchanged=HEAD,native_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,240)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:save(a.token)
    finally:lock.close()
