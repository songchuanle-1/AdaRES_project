"""Admit measured Y62/Y63 checks and dispatch using the canonical launcher."""
import argparse
import ast
import copy
from datetime import datetime
import json
import math
import shlex

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import experiment_first


def validate_result(raw,candidate,frozen):
    for key in ('id','config','code_commit'):
        assert raw[key]==candidate[key]
    assert raw['goal_sha256']==frozen['goal_sha256']
    assert raw['native_YAML_sha256']==frozen['variant']['config_data']['sha256']
    assert raw['steps']==raw['optimizer_steps']==2 and len(raw['losses'])==2
    assert all(math.isfinite(v) for v in raw['losses'])
    assert raw['native_entry']=='tools/train.py::train' and raw['split']=='train'
    assert raw['native_data_loaded'] and raw['backward_finite'] and raw['intended_path_observed']
    assert raw['validation_used'] is False and raw['model_weights_saved'] is False
    assert raw['shared_head_calls']==14 and raw['query_norm_input_requires_grad']==[True]*14
    assert len(raw['gradient_checks'])==len(raw['batches'])==2
    key='density' if raw['id']=='AdaRES-Y62' else 'bilinear'
    for g in raw['gradient_checks']:
        assert g['all_finite'] and g['indicator_tensors']==11 and g['indicator_calls']==7
        assert g['query_norm_input_requires_grad']==[True]*7
        assert g['query_and_mask_input_gradients_finite'] and g['trainable_query_norm_gradient_nonzero']
        assert math.isfinite(g[key+'_weight_gradient_max']) and g[key+'_weight_gradient_max']>0
    paths=raw[key+'_paths']
    assert len(paths)==14
    assert all(not x[key+'_weight_nonzero'] and x['logit_delta_max']==0 for x in paths[:7])
    assert all(x[key+'_weight_nonzero'] for x in paths[7:])
    assert any(x['logit_delta_max']>0 for x in paths[7:])
    assert 0<raw['peak_reserved_bytes']<24*2**30
    assert all(len(b['scan_ids'])==2 and b['expressions']>0 and b['points']>0 for b in raw['batches'])
    assert raw['data_source']['config_data_root'].rstrip('/')=='/data1/songcl/Project/AdaRES_project/data'


def live(server):
    program="""import json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
print(json.dumps({'gpu':subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.used,power.limit','--format=csv,noheader,nounits']).decode().strip(),
 'compute':subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()}))
"""
    return json.loads(monitor.ssh(server['host'],server['user'],'python3 -B -c '+shlex.quote(program),timeout=30))


def admit(token,cid):
    server=monitor.load_servers(str(ROOT))['server-1']
    closure.validate_adares_server(ROOT,server)
    work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(x for x in work['candidates'] if x['id']==cid)
    assert c['stage']=='validation' and c['required_server']=='server-1' and not c.get('admission_receipt')
    plan=tick.load(closure.artifact(ROOT,c['native_smoke_plan']))
    program=closure.artifact(ROOT,plan['program'])
    frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(c,plan,frozen)==7
    assert c['code_commit']==closure.git(ROOT,'rev-parse','HEAD')
    raw_path=closure.artifact(ROOT,c['native_smoke_result'])
    raw=tick.load(raw_path)
    validate_result(raw,c,frozen)
    bootstrap=tick.load(ROOT/('pipeline/checks/server1_data_verified_'+token+'.json'))
    assert bootstrap['complete'] and bootstrap['required_inputs_equal']
    queue=monitor.load_queue(str(ROOT))
    assert not any(t['id']==cid for group in queue['gpu_queues'].values() for t in group)
    paused=next(t for t in queue['gpu_queues']['server-2'] if t['id']=='AdaRES-Y57')
    assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    q=queue['gpu_queues'].setdefault('server-1',[])
    assert not any(t['status'] in ('running','claiming') for t in q)
    idle=live(server)
    util,used,power=[float(v.strip()) for v in idle['gpu'].split(',')]
    assert util<10 and used<1024 and not idle['compute']
    prefix='pipeline/checks/'+cid+'_server1_'+token
    smoke_path=ROOT/(prefix+'_native_smoke.json')
    freeze_path=ROOT/(prefix+'_post_training_freeze.json')
    receipt_path=ROOT/('pipeline/admissions/'+cid+'.json')
    assert not any(p.exists() for p in (smoke_path,freeze_path,receipt_path))
    smoke=copy.deepcopy(raw)
    command='python -B scripts/run_server1_native_probe.py '+cid+' --token '+token
    smoke.update(command=command,server='server-1',physical_gpu=0,
        admission_resource_check=idle,evidence=[ref(raw_path),ref(raw_path.with_suffix('.log')),
            ref(program),c['native_smoke_plan'],ref(ROOT/('pipeline/checks/server1_data_verified_'+token+'.json'))])
    tick.save(smoke_path,smoke)
    tick.save(freeze_path,dict(complete=True,id=cid,frozen_at=datetime.now().astimezone().isoformat(),
        goal=ref('pipeline/goal.json'),code_commit=c['code_commit'],
        training='Released SPFormer; seed1999; fixed30; native YAML losses/optimizer/scheduler; Q128; single argmax; no validation tuning.',
        evaluation=dict(split='Full frozen ScanRefer val9508 expressions/141 scenes',
            methods=[cid,'AdaRES-Y36','AdaRES-W2','AdaRES-B1'],repeats=2,primary='FIRST',
            metrics=['mIoU','Acc@0.25','Acc@0.5','active_queries','latency','peak_memory'],
            scene_cluster_bootstrap=10000,bootstrap_seed=1999,TRAIN_size_cuts=[1487,3325],
            material_harm='Prespecified target-size subgroup difference CI upper<0 versus Y36/W2; retain negatives.',
            uncertainty='Crossing intervals remain inconclusive; repeats do not measure training-seed uncertainty.'),
        resource_condition=dict(server='server-1',gpu=0,power_limit_watts=power,
            note='New host/runtime compatibility disclosed; run paired trained-cost comparisons on the same device. Two batches are not a whole-dataset memory bound.'),
        resource_authorization=ref('pipeline/authorizations/server1_gpu0_20261025.json'),
        scientific_efficacy='Unknown until training and frozen full evaluation are complete.'))
    receipt=dict(profile='experiment_first',id=cid,candidate_id=cid,config=c['config'],
        code_commit=c['code_commit'],goal_sha256=raw['goal_sha256'],
        authorization=experiment_first.effective(ROOT)['authorization'],native_smoke=ref(smoke_path),
        post_training_plan=ref(freeze_path),resource_authorization=ref('pipeline/authorizations/server1_gpu0_20261025.json'),
        admitted_at=datetime.now().astimezone().isoformat(),
        gates={g:dict(passed=True,evidence=[ref(smoke_path),ref(freeze_path),ref(program)]) for g in experiment_first.GATES})
    experiment_first.validate_smoke(ROOT,receipt)
    decision=monitor.log_decision(str(ROOT),'next_batch',cid+'是否可使用新授权server-1？',
        '原生YAML/TRAIN两步、有限梯度及新分支检查通过，按预算准入并使用既有门控派发；Y57保持暂停。',
        '版本/goal/原数据一致，server-1 GPU0实际独占空闲，固定30轮与seed1999不变；效益仍待测。',ref(smoke_path)['path'])
    receipt['decision']=ref(decision)
    tick.save(receipt_path,receipt)
    task=dict(id=cid,label=cid+' fixed30',config=c['config'],status='waiting',gpu=0,
        server='server-1',depends_on=[],code_commit=c['code_commit'],admission_receipt=ref(receipt_path)['path'],
        scientific_candidate_id=cid,patience=30,min_epochs=30,fixed_epochs=30,requires_exclusive_ownership=True)
    c.update(stage='admission',admission_receipt=task['admission_receipt'],native_smoke=ref(smoke_path),
        post_training_freeze=ref(freeze_path),next_action='Canonical gated server-1 dispatch; confirm increasing native steps. Preserve Y57 pause and foreign jobs.')
    c['evidence'].extend([ref(raw_path),ref(raw_path.with_suffix('.log')),ref(smoke_path),ref(freeze_path),ref(receipt_path),ref(decision),ref(__file__)])
    tick.save(ROOT/'pipeline/research_work.json',work)
    closure.admission(ROOT,task)
    q.append(task)
    monitor.save_queue(str(ROOT),queue)
    print(json.dumps(dict(admitted=cid,server='server-1',receipt=ref(receipt_path),training_verified=False)))


def dispatch(token):
    server=monitor.load_servers(str(ROOT))['server-1']
    queue=monitor.load_queue(str(ROOT))
    tasks=queue['gpu_queues']['server-1']
    assert not any(t['status'] in ('running','claiming') for t in tasks),'Reconcile uncertain launch first'
    before=live(server)
    assert not before['compute']
    events=[]
    monitor._process_queue(queue,'server-1',server['host'],server['user'],str(ROOT),set(),{'0'},events,
        max_tasks_per_gpu=1,allow_external_overlay=False,remote_code_root=server['remote_code_root'],current_tasks=0,max_tasks_total=1)
    queue=monitor.load_queue(str(ROOT))
    work=tick.load(ROOT/'pipeline/research_work.json')
    for task in queue['gpu_queues']['server-1']:
        if task['status']=='waiting':continue
        c=next(c for c in work['candidates'] if c['id']==task['id'])
        c.update(stage='training' if task['status']=='running' and task.get('launch_verified') else 'deployment',
            next_action='Observe exact PID/log and native training step growth; reconcile uncertain launches without blind restart.')
    out=ROOT/('pipeline/checks/server1_dispatch_'+token+'.json')
    assert not out.exists()
    tick.save(out,dict(complete=True,observed_at=datetime.now().astimezone().isoformat(),
        before=before,after=live(server),tasks=queue['gpu_queues']['server-1'],events=events,Y57_still_paused=True))
    for c in work['candidates']:
        if c['id'] in ('AdaRES-Y62','AdaRES-Y63'):c['evidence'].append(ref(out))
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(evidence=ref(out),tasks=queue['gpu_queues']['server-1'],events=events)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('operation',choices=('admit','dispatch'))
    p.add_argument('--id',choices=('AdaRES-Y62','AdaRES-Y63'))
    p.add_argument('--token',required=True)
    a=p.parse_args()
    lease(a.token,300)
    lock=monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        if a.operation=='admit':
            assert a.id
            admit(a.token,a.id)
        else:dispatch(a.token)
    finally:lock.close()

if __name__=='__main__':main()
