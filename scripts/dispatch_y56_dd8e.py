"""Canonical guarded dispatch and exact post-launch observation for Y56."""

from datetime import datetime
import json
from research_y56_dd8e import ROOT, TOKEN, lease, ref, save, bind, tick, monitor
import closure
from dispatch_prepared_candidates import snapshot

lease(400)
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y56')
    queue=monitor.load_queue(str(ROOT))
    task=next(t for t in queue['gpu_queues']['server-2'] if t['id']==candidate['id'])
    assert task['status']=='waiting' and not task.get('pid') and not task.get('dispatch_hold')
    closure.admission(ROOT,task)
    server=monitor.load_servers(str(ROOT))['server-2']
    assert server['host']=='192.168.1.163' and server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
    review=tick.load(ROOT/('pipeline/analyses/y56_gate_review_'+TOKEN+'.json'))
    files={r['path'].removeprefix('AdaRES/'):r['sha256'] for r in review['files']}
    before=snapshot(server,[task],list(files))
    procs=monitor.find_project_processes('AdaRES_project','server-2',server,queue)
    assert not procs, 'Reconcile any existing project run instead of dispatching blindly'
    events=[]
    monitor._process_queue(queue,'server-2',server['host'],server['user'],str(ROOT),set(),{'0','1'},events,
        max_tasks_per_gpu=1,allow_external_overlay=False,remote_code_root=server['remote_code_root'],
        current_tasks=0,max_tasks_total=2)
    queue=monitor.load_queue(str(ROOT))
    task=next(t for t in queue['gpu_queues']['server-2'] if t['id']==candidate['id'])
    after=snapshot(server,[task],list(files))
    head=candidate['code_commit']
    result=dict(complete=True,id=candidate['id'],observed_at=datetime.now().astimezone().isoformat(),
        before=before,after=after,events=events,task=task,code_commit=head,
        exact_deployed_tested_bytes=after['source_hashes']==files,
        native_steps=closure.steps(after['tasks'][task['id']]['tail']),
        limitation='Execution evidence only; no trained benefit or final metric yet.')
    evidence=save('pipeline/analyses/y56_dispatch_'+TOKEN+'.json',result)
    bind(candidate,evidence,ref(__file__))
    if task['status']=='running' and task.get('launch_verified') is True:
        process=after['tasks'][task['id']]
        assert process['process_live'] and process['cwd']==server['remote_code_root']
        assert task['config'] in process['cmdline'] and 'tools/train.py' in process['cmdline']
        assert after['source_marker']==head and after['source_hashes']==files
        assert len(set(result['native_steps']))>=2
        assert not any(s in process['tail'] for s in ('Traceback','CUDA out of memory','RuntimeError'))
        candidate.update(stage='training',experiment_id=task['id'],
            next_action='Monitor SAME Y56 fixed30/1999 single-GPU run and frozen post-training comparison; no epoch/threshold tuning or blind restart.')
    elif task.get('pid') or task.get('dispatch_hold'):
        candidate.update(stage='deployment',next_action='Reconcile exact existing Y56 PID/config/log and native step evidence; never blindly relaunch.')
    else:
        candidate.update(stage='deployment',next_action='Inspect actual canonical dispatch failure in this evidence; do not mislabel it missing user permission.')
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(evidence=evidence,status=task['status'],gpu=task.get('gpu'),pid=task.get('pid'),
        launch_verified=task.get('launch_verified'),native_steps=result['native_steps'],events=events),indent=2))
finally:
    lock.close()
