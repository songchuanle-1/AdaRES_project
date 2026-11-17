"""Check or execute the frozen Y70 GPU1 probe without displacing live work."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from check_ef_execution import validate_frozen
import closure

CID='AdaRES-Y70'
COMMIT='39bdb8d9783512e70cdc2763e9a6097ab0f67a0f'


def slot_blockers(queue,running):
    tasks=queue['gpu_queues']['server-2']
    paused=next(t for t in tasks if t['id']=='AdaRES-Y57')
    assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    active=[t['id'] for t in tasks if str(t.get('gpu'))=='1' and t['status'] in ('waiting','claiming','running')]
    owner=running.get('server-2',{}).get('1')
    return sorted(set(active+([owner] if owner else [])))


def physically_idle(live):
    assert live['hostname']=='dl01-MS-7C60' and live['power_limit_watts']==390.0
    assert live['memory_used_mib']>=0 and 0<=live['utilization_percent']<=100
    return live['memory_used_mib']<1024 and live['utilization_percent']<10 and not live['compute_pids']


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('check','run'));p.add_argument('--token',required=True);a=p.parse_args()
    lease(a.token,800 if a.operation=='run' else 120)
    server=monitor.load_servers(str(ROOT))['server-2'];closure.validate_adares_server(ROOT,server)
    assert server['allowed_gpus']==[0,1] and server['expected_hostname']=='dl01-MS-7C60'
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in work['candidates'] if x['id']==CID)
    assert c['stage'] in ('blocked','validation') and c.get('blocker_kind') in (None,'exclusive_gpu_slot')
    assert c['required_server']=='server-2' and c['required_gpu']==1 and c['code_commit']==COMMIT
    assert c['native_smoke_plan']==c['native_smoke_plan_server2_gpu1']
    assert closure.git(ROOT,'rev-parse','HEAD')==COMMIT
    plan=tick.load(closure.artifact(ROOT,c['native_smoke_plan']))
    program=closure.artifact(ROOT,plan['program']);frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(c,plan,frozen)==7
    assert frozen['runtime_physical_gpu']==1 and frozen['runtime_power_limit_watts']==390.0
    closure.artifact(ROOT,frozen['resource_authorization'])
    queue_before=ref('pipeline/queue.json');running_before=ref('pipeline/running.json')
    queue=monitor.load_queue(str(ROOT));running=monitor.load_running(str(ROOT))
    blockers=slot_blockers(queue,running)
    experiment=monitor.load_experiment(str(ROOT),'AdaRES-Y67')
    dependency_complete=bool(experiment and experiment.get('status')=='done_complete')
    remote="""import json,subprocess
hostname=subprocess.check_output(['hostname']).decode().strip()
assert hostname=='dl01-MS-7C60'
used,util,power=[float(x.strip()) for x in subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu,power.limit','--format=csv,noheader,nounits']).decode().split(',')]
compute=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
print(json.dumps(dict(hostname=hostname,memory_used_mib=used,utilization_percent=util,power_limit_watts=power,compute_pids=[int(p) for p in compute.splitlines()])))
"""
    base=ROOT/('pipeline/checks/y70_gpu1_readiness_'+a.token+'_'+a.operation)
    assert not base.with_suffix('.json').exists() and not base.with_suffix('.log').exists()
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    started=datetime.now().astimezone().isoformat()
    try:
        response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',server['user']+'@'+server['host'],server['python']+' -S -B -'],input=remote.encode(),capture_output=True,timeout=25)
        base.with_suffix('.log').write_bytes(response.stdout+response.stderr);assert response.returncode==0
        live=json.loads(response.stdout);idle=physically_idle(live)
        ready=idle and not blockers and dependency_complete
        result=dict(complete=True,token=a.token,id=CID,operation=a.operation,observed_start_local=started,
            observed_end_local=datetime.now().astimezone().isoformat(),queue_blockers=blockers,live=live,
            Y67_canonical_fixed30_complete=dependency_complete,ready=ready,native_GPU_executed=False,
            native_plan=c['native_smoke_plan'],goal=ref('pipeline/goal.json'),code_commit=COMMIT,
            evidence=[ref(__file__),ref(base.with_suffix('.log')),ref(program)],
            limitation='Read-only queue/source/physical readiness check; no CUDA initialization, native task step or efficacy claim.')
        tick.save(base.with_suffix('.json'),result)
        c['evidence'].extend([ref(base.with_suffix('.json')),ref(base.with_suffix('.log')),ref(__file__)])
        if a.operation=='check':
            work['tick_attempt']=dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],
                candidate_id=CID,kind='resource_check',result='passed' if ready else 'blocked',
                command='python -B scripts/run_y70_gpu1_probe.py check --token '+a.token,
                observation='Exact frozen Y70 plan/source verified. Actual GPU1 '+('idle and unreserved after original Y67 canonical completion.' if ready else 'not ready: preserve original running/claiming/reserved task and wait for Y67 fixed30 canonical archive.'),
                evidence=[ref(base.with_suffix('.json'))])
        tick.save(ROOT/'pipeline/research_work.json',work)
        assert ref('pipeline/queue.json')==queue_before and ref('pipeline/running.json')==running_before
    finally:lock.close()
    print(json.dumps(dict(evidence=ref(base.with_suffix('.json')),ready=ready,live=live,blockers=blockers)))
    if a.operation=='check':return
    assert ready,'Native run refused; preserve busy card and original jobs'
    command=[sys.executable,'-B',str(ROOT/'scripts/run_experiment_first_probe.py'),'y70_gpu1_smoke',str(program),'--token',a.token,'--seconds','600','--gpu','--gpu-id','1']
    subprocess.run(command,cwd=ROOT,check=True,timeout=700)
    lease(a.token,90)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        rawpath=ROOT/('pipeline/analyses/y70_gpu1_smoke_'+a.token+'.json');raw=tick.load(rawpath)
        assert raw['complete'] and raw['id']==CID and raw['code_commit']==COMMIT and raw['steps']==raw['optimizer_steps']==2
        assert raw['native_data_loaded'] and raw['backward_finite'] and raw['intended_path_observed']
        work=tick.load(ROOT/'pipeline/research_work.json');current=next(x for x in work['candidates'] if x['id']==CID)
        current.update(stage='validation',native_smoke_result=ref(rawpath),next_action='Validate actual native result; freeze original full evaluation/admit/gated dispatch in this tick if all gates pass.')
        for key in ('blocker','blocker_kind','blocked_from_stage','retry_after','unblock_condition','occupation_evidence'):current.pop(key,None)
        current['evidence'].extend([ref(rawpath),ref(rawpath.with_suffix('.log'))])
        tick.save(ROOT/'pipeline/research_work.json',work)
        assert ref('pipeline/queue.json')==queue_before and ref('pipeline/running.json')==running_before
    finally:lock.close()


if __name__=='__main__':main()
