"""Admit actual current-HEAD Y69 native checks and use the gated server-1 launcher."""
import argparse
import ast
import copy
from datetime import datetime
import json
import math
import shlex
import subprocess
import sys

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import experiment_first

CID = 'AdaRES-Y69'


def validate_idle(observation):
    util, used, power = [float(v.strip()) for v in observation['gpu'].split(',')]
    assert all(math.isfinite(v) for v in (util, used, power))
    assert 0 <= util < 10 and 0 <= used < 1024 and power == 350.0
    assert not observation['compute'].strip()
    return power


def validate_result(raw, candidate, frozen):
    assert raw['complete'] is True and raw['id'] == CID == candidate['id']
    assert raw['config'] == candidate['config'] == frozen['variant']['config']
    assert raw['code_commit'] == candidate['code_commit'] == frozen['code_commit']
    assert raw['goal_sha256'] == frozen['goal_sha256']
    assert raw['native_YAML_sha256'] == frozen['variant']['config_data']['sha256']
    assert raw['steps'] == raw['optimizer_steps'] == 2
    assert len(raw['losses']) == 2 and all(math.isfinite(v) for v in raw['losses'])
    assert raw['native_entry'] == 'tools/train.py::train' and raw['split'] == 'train'
    assert all(raw[k] is True for k in ('native_data_loaded','backward_finite','intended_path_observed'))
    assert raw['validation_used'] is False and raw['model_weights_saved'] is False
    assert raw['shared_head_calls'] == 14 and raw['geometry_calls'] == 2
    assert raw['coarse_stage_calls'] == raw['attention_stage_calls'] == list(range(6)) * 2
    assert all(type(v) is int for v in raw['coarse_stage_calls'] + raw['attention_stage_calls'])
    assert len(raw['gradient_checks']) == len(raw['batches']) == 2
    from test_y69_native_smoke_plan import N
    for check in raw['gradient_checks']:
        assert check['all_finite'] is True and check['tensors'] > 0 and check['indicator_tensors'] == 10
        N['check_path_calls'](check['coarse_stage_calls'],check['indicator_calls'],check['geometry_calls'],check['attention_calls'])
        assert check['native_hard_masks_preserved'] is True and check['source_and_query_gradients_finite'] is True
        for key in ('geometry_weight_gradient_max','parameter_update_max'):
            assert math.isfinite(check[key]) and check[key] > 0
    N['check_response_rows'](raw['geometry_paths'])
    assert 0 < raw['peak_reserved_bytes'] < 24 * 2**30
    assert all(len(b['scan_ids']) == 2 and b['expressions'] > 0 and b['points'] > 0 for b in raw['batches'])
    assert len({scene for b in raw['batches'] for scene in b['scan_ids']}) == 4
    assert raw['data_source']['dataset_class'] == 'ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'
    assert raw['data_source']['config_data_root'].rstrip('/') == '/data1/songcl/Project/AdaRES_project/data'


def admission_guards(token):
    tests_path=ROOT/('pipeline/checks/y69_server1_admission_guards_'+token+'.log')
    manifest=tests_path.with_suffix('.json')
    sources=[ref(__file__),ref('scripts/test_y69_server1_admission.py'),ref('scripts/test_y69_native_smoke_plan.py'),ref('scripts/probe_y69_native_smoke.py')]
    if manifest.exists():
        cached=tick.load(manifest)
        assert cached['complete'] is True and cached['tests_passed']==14 and cached['sources']==sources
        assert cached['log']==ref(tests_path) and cached['native_result_substitution'] is False
        return tests_path,manifest
    assert not tests_path.exists()
    tested=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','scripts','-p','test_y69_server1_admission.py','-v'],cwd=ROOT,capture_output=True,timeout=60)
    tests_path.write_bytes(tested.stdout+tested.stderr)
    assert tested.returncode==0 and b'Ran 14 tests' in tests_path.read_bytes()
    tick.save(manifest,dict(complete=True,tests_passed=14,sources=sources,log=ref(tests_path),native_result_substitution=False,
        purpose='Unit fixtures verify rejection logic only; actual native result remains mandatory before admission.'))
    return tests_path,manifest


def live(server):
    program="""import json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
print(json.dumps({'gpu':subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.used,power.limit','--format=csv,noheader,nounits']).decode().strip(),
 'compute':subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()}))
"""
    return json.loads(monitor.ssh(server['host'],server['user'],shlex.quote(server['python'])+' -B -c '+shlex.quote(program),timeout=30))


def admit(token,cid):
    assert cid == CID
    server=monitor.load_servers(str(ROOT))['server-1']
    closure.validate_adares_server(ROOT,server)
    work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(x for x in work['candidates'] if x['id']==cid)
    assert c['code_commit']=='9acd73cdc1b96be003c8450a9e22057248a0f531'
    assert c['native_smoke_plan']==c['native_smoke_plan_server1'] and c['required_gpu']==0
    assert c['stage']=='validation' and c['required_server']=='server-1' and not c.get('admission_receipt')
    plan=tick.load(closure.artifact(ROOT,c['native_smoke_plan']))
    program=closure.artifact(ROOT,plan['program'])
    frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(c,plan,frozen)==7
    assert frozen['runtime_server']=='server-1' and frozen['runtime_physical_gpu']==0 and frozen['runtime_power_limit_watts']==350.0
    assert frozen['runtime_path_mappings']==server['runtime_path_mappings']
    closure.artifact(ROOT,frozen['resource_authorization'])
    assert c['code_commit']==closure.git(ROOT,'rev-parse','HEAD')
    raw_path=closure.artifact(ROOT,c['native_smoke_result'])
    raw=tick.load(raw_path)
    validate_result(raw,c,frozen)
    data_binding = dict(path='pipeline/checks/server1_data_verified_63b27dd8efe04a0bab6dc0291cefdbf0.json',
        sha256='8c590fa828fbc6c0df840c21ba88b1802cc0d2323eb0a124a57aa328b0c2361f')
    bootstrap=tick.load(closure.artifact(ROOT,data_binding))
    old_candidate=next(x for x in work['candidates'] if x['id']=='AdaRES-Y64')
    parent_protocol=old_candidate['post_training_freeze']
    closure.artifact(ROOT,parent_protocol)
    tests_path,guard_manifest=admission_guards(token)
    assert bootstrap['complete'] and bootstrap['required_inputs_equal']
    queue=monitor.load_queue(str(ROOT))
    assert tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['queue']['candidate_budget']>=1
    assert monitor.load_experiment(str(ROOT),cid) is None
    assert not any(t['id']==cid for group in queue['gpu_queues'].values() for t in group)
    paused=next(t for t in queue['gpu_queues']['server-2'] if t['id']=='AdaRES-Y57')
    assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    q=queue['gpu_queues'].setdefault('server-1',[])
    assert not any(t['status'] in ('running','claiming') for t in q)
    idle=live(server)
    util,used,power=[float(v.strip()) for v in idle['gpu'].split(',')]
    assert validate_idle(idle) == power
    prefix='pipeline/checks/'+cid+'_server1_'+token
    smoke_path=ROOT/(prefix+'_native_smoke.json')
    freeze_path=ROOT/(prefix+'_post_training_freeze.json')
    receipt_path=ROOT/('pipeline/admissions/'+cid+'.json')
    assert not any(p.exists() for p in (smoke_path,freeze_path,receipt_path))
    smoke=copy.deepcopy(raw)
    command='python -B scripts/run_y69_server1_native_probe.py '+cid+' --token '+token
    smoke.update(command=command,server='server-1',physical_gpu=0,
        admission_resource_check=idle,evidence=[ref(raw_path),ref(raw_path.with_suffix('.log')),
            ref(program),c['native_smoke_plan'],data_binding])
    tick.save(smoke_path,smoke)
    parent=tick.load(closure.artifact(ROOT,parent_protocol))
    freeze=copy.deepcopy(parent)
    freeze.update(id=cid,frozen_at=datetime.now().astimezone().isoformat(),
        goal=ref('pipeline/goal.json'),code_commit=c['code_commit'],parent_protocol=parent_protocol,
        resource_condition=dict(server='server-1',gpu=0,power_limit_watts=power,
            note='Default350W unchanged; paired trained-cost comparisons on the same device. Two batches are not a full-dataset memory guarantee.'),
        resource_authorization=ref('pipeline/authorizations/server1_gpu0_20261025.json'),
        scientific_efficacy='Unknown until training and original frozen full evaluation are complete.')
    freeze['evaluation']['methods'][0]=cid
    restored=copy.deepcopy(freeze['evaluation']);restored['methods'][0]='AdaRES-Y64'
    assert restored==parent['evaluation'] and freeze['training']==parent['training']
    tick.save(freeze_path,freeze)
    receipt=dict(profile='experiment_first',id=cid,candidate_id=cid,config=c['config'],
        code_commit=c['code_commit'],goal_sha256=raw['goal_sha256'],
        authorization=experiment_first.effective(ROOT)['authorization'],native_smoke=ref(smoke_path),
        post_training_plan=ref(freeze_path),resource_authorization=ref('pipeline/authorizations/server1_gpu0_20261025.json'),
        admitted_at=datetime.now().astimezone().isoformat(),
        gates={g:dict(passed=True,evidence=[ref(smoke_path),ref(freeze_path),ref(program)]) for g in experiment_first.GATES})
    experiment_first.validate_smoke(ROOT,receipt)
    decision=monitor.log_decision(str(ROOT),'next_batch',cid+'是否可使用新授权server-1？',
        '原生YAML/TRAIN两步、六次粗层几何偏置注意力/七次共享头、原生位置/原hard masks/独立轴公式/实际消费和有限非零新参数梯度/更新及次步响应检查通过，按预算准入并使用既有门控派发；Y57保持暂停。',
        '版本/goal/原数据一致，server-1 GPU0实际独占空闲，固定30轮与seed1999不变；效益仍待测。',ref(smoke_path)['path'])
    receipt['decision']=ref(decision)
    tick.save(receipt_path,receipt)
    task=dict(id=cid,label=cid+' fixed30',config=c['config'],status='waiting',gpu=0,
        server='server-1',depends_on=[],code_commit=c['code_commit'],admission_receipt=ref(receipt_path)['path'],
        scientific_candidate_id=cid,patience=30,min_epochs=30,fixed_epochs=30)
    c.update(stage='admission',admission_receipt=task['admission_receipt'],native_smoke=ref(smoke_path),
        post_training_freeze=ref(freeze_path),next_action='Canonical gated server-1 dispatch; confirm increasing native steps. Preserve Y57 pause and foreign jobs.')
    c['evidence'].extend([ref(raw_path),ref(raw_path.with_suffix('.log')),ref(smoke_path),ref(freeze_path),ref(receipt_path),ref(decision),ref(__file__),ref(tests_path),ref('scripts/test_y69_server1_admission.py'),ref(guard_manifest)])
    tick.save(ROOT/'pipeline/research_work.json',work)
    closure.admission(ROOT,task)
    work['tick_attempt']=dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=cid,kind='native_smoke',result='passed',command=command,
        observation='Actual current-HEAD native YAML/TRAIN two-step coarse geometry attention path/backward/optimizer passed; admission signed and queued. Training requires gated verified launch.',
        evidence=[ref(raw_path),ref(smoke_path),ref(freeze_path)])
    tick.save(ROOT/'pipeline/research_work.json',work)
    q.append(task)
    monitor.save_queue(str(ROOT),queue)
    print(json.dumps(dict(admitted=cid,server='server-1',receipt=ref(receipt_path),training_verified=False)))


def dispatch(token):
    server=monitor.load_servers(str(ROOT))['server-1']
    queue=monitor.load_queue(str(ROOT))
    tasks=queue['gpu_queues']['server-1']
    other_before=copy.deepcopy(queue['gpu_queues']['server-2'])
    assert [t['id'] for t in tasks if t['status']=='waiting']==[CID]
    closure.admission(ROOT,next(t for t in tasks if t['id']==CID))
    assert not any(t['status'] in ('running','claiming') for t in tasks),'Reconcile uncertain launch first'
    before=live(server)
    validate_idle(before)
    events=[]
    monitor._process_queue(queue,'server-1',server['host'],server['user'],str(ROOT),set(),{'0'},events,
        max_tasks_per_gpu=1,allow_external_overlay=False,remote_code_root=server['remote_code_root'],current_tasks=0,max_tasks_total=1)
    queue=monitor.load_queue(str(ROOT))
    assert queue['gpu_queues']['server-2']==other_before
    work=tick.load(ROOT/'pipeline/research_work.json')
    for task in queue['gpu_queues']['server-1']:
        if task['status']=='waiting':continue
        c=next(c for c in work['candidates'] if c['id']==task['id'])
        c.update(stage='training' if task['status']=='running' and task.get('launch_verified') else 'deployment',
            next_action='Observe exact PID/log and native training step growth; reconcile uncertain launches without blind restart.')
    out=ROOT/('pipeline/checks/y69_server1_dispatch_'+token+'.json')
    assert not out.exists()
    tick.save(out,dict(complete=True,observed_at=datetime.now().astimezone().isoformat(),
        before=before,after=live(server),tasks=queue['gpu_queues']['server-1'],events=events,Y57_still_paused=True))
    for c in work['candidates']:
        if c['id'] == CID:c['evidence'].append(ref(out))
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(evidence=ref(out),tasks=queue['gpu_queues']['server-1'],events=events)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('operation',choices=('precheck','admit','dispatch'))
    p.add_argument('--id',choices=(CID,),default=CID)
    p.add_argument('--token',required=True)
    a=p.parse_args()
    lease(a.token,300)
    lock=monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        if a.operation=='precheck':
            paths=admission_guards(a.token)
            print(json.dumps(dict(guards=ref(paths[1]),native_result_required=True)))
        elif a.operation=='admit':
            assert a.id
            admit(a.token,a.id)
        else:dispatch(a.token)
    finally:lock.close()

if __name__=='__main__':main()
