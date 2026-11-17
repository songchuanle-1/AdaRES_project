"""Apply separate Y57 consent, admit and use canonical dispatch alongside Y56."""

from datetime import datetime
import json,subprocess,hashlib
from research_y56_dd8e import ROOT,TOKEN,lease,ref,save,bind,tick,monitor
import closure
import authorize_prepared_candidates as scoped
from dispatch_prepared_candidates import snapshot

REFERENCE='["request_user_input_async","call_66c334fb17c945d68c7e2cee5ce6063d",0]'
scoped.SPECS={'AdaRES-Y57':('adares_y57_peer_second_moment','pipeline/analyses/y57_anchor_bundle_'+TOKEN+'.json')}
scoped.MESSAGE_ID=REFERENCE
lease(180)
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y57')
    assert c['stage']=='blocked' and c['blocker_kind']=='human_authorization'
    bundle,files=scoped.checked_bundle(c['id'],c)
    head=scoped.git('rev-parse','HEAD').decode().strip()
    assert head=='200486ee12bcf3d7e82393d009d06214dfe83e38'
    path='pipeline/authorizations/AdaRES-Y57_commit.json'
    assert not (ROOT/path).exists()
    now=datetime.now().astimezone().isoformat()
    tick.save(ROOT/path,dict(complete=True,id=c['id'],kind='explicit_human_scoped_commit_authorization',
        recorded_at=now,thread_id='01a07e32-78bf-7670-925b-370911ea48db',human_message_id=REFERENCE,
        human_message='授权 Y57 这 7 个文件的本地提交',files=files,tested_bundle=bundle,starting_HEAD=head,
        context='Separate direct owning-user answer to exact7revisedY57files; not inferred from Y56 permission.',
        permits=['Local Git add/commit only exact seven tested Y57 revision1 files.'],
        excludes=['Push, unrelated files/projects, altered goal/protocol/gates, diagnostic weights or inferred training permission.']))
    prior={k:c[k] for k in ('blocker_kind','blocker','retry_after','unblock_condition') if k in c}
    c.setdefault('blocker_history',[]).append(dict(**prior,resolved_at=now,resolved_by=ref(path)))
    for k in ('blocker_kind','blocker','retry_after','unblock_condition'):c.pop(k,None)
    c.update(stage='validation',next_action='Separate human Y57 consent confirmed; commit tested seven files then canonical admission/dispatch.')
    bind(c,ref(path),ref(__file__))
    work.pop('design_cycle',None)
    tick.save(ROOT/'pipeline/research_work.json',work)
finally:
    lock.close()
scoped.apply('AdaRES-Y57',TOKEN,'commit')

lease(150)
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y57')
    queue=monitor.load_queue(str(ROOT))
    assert len(queue['gpu_queues']['server-2'])==1
    existing=queue['gpu_queues']['server-2'][0]
    assert existing['id']=='AdaRES-Y56' and existing['status']=='running' and existing['launch_verified']
    review_path='pipeline/analyses/y57_anchor_gate_review_'+TOKEN+'.json'
    review=tick.load(ROOT/review_path)
    assert review['goal_sha256']==ref('pipeline/goal.json')['sha256'] and all(g['passed'] for g in review['gates'].values())
    head=c['code_commit']
    for row in review['files']:
        assert ref(row['path'])==row
        assert hashlib.sha256(scoped.git('show',head+':'+row['path'])).hexdigest()==row['sha256']
    plan=tick.load(ROOT/('pipeline/analyses/AdaRES-Y56_post_training_freeze_'+TOKEN+'.json'))
    plan.update(id='AdaRES-Y57',code_commit=head,frozen_at=datetime.now().astimezone().isoformat(),
        preregistered_design=ref('pipeline/designs/exp_design_Y56_Y57_6debf88fed634a13b772f27bdc244bbd.md'),
        connection_revision_prereg=ref('pipeline/analyses/y57_anchor_prereg_'+TOKEN+'.json'),limitations=review['limitations'])
    plan['evaluation']['methods']=['AdaRES-Y57','AdaRES-Y36','AdaRES-W2','AdaRES-B1']
    frozen=save('pipeline/analyses/AdaRES-Y57_post_training_freeze_'+TOKEN+'.json',plan)
    decision=monitor.log_decision(str(ROOT),'manual_adares-y57',
        'Y57修订七文件已获单独授权并提交，是否按现有科学及GPU门禁准入？',
        '复核六项实测及原失败保留，准入同ID固定30；保持Y56原GPU0/PID，规范调度器只可用实时空闲授权GPU。',
        '不推送、不扩展提交范围、不改变原阈值/目标/训练初始化；收益未知。',review_path)
    gates=json.loads(json.dumps(review['gates']))
    binding=ref('pipeline/analyses/AdaRES-Y57_commit_binding_'+TOKEN+'.json')
    gates['independence']['evidence'].append(binding)
    gates['constraints']['evidence'].extend([frozen,ref('pipeline/goal.json')])
    receipt_path='pipeline/admissions/AdaRES-Y57.json'
    receipt=dict(id=c['id'],candidate_id=c['id'],config='configs_song/model_paper5/adares_y57_peer_second_moment_res.yaml',
        code_commit=head,goal_sha256=review['goal_sha256'],decision=ref(decision),gates=gates,
        implementation_revision_sha256=closure.revision_digest(c['revision_history'][-1]),
        admitted_at=datetime.now().astimezone().isoformat(),post_training_plan=frozen,limitations=review['limitations'])
    save(receipt_path,receipt)
    task=dict(id=c['id'],label='AdaRES-Y57 initial moment anchor fixed30',config=receipt['config'],status='waiting',gpu=None,
        depends_on=[],code_commit=head,admission_receipt=receipt_path,patience=30,min_epochs=30,fixed_epochs=30,
        requires_exclusive_ownership=True,scientific_candidate_id=c['id'])
    closure.admission(ROOT,task)
    queue['gpu_queues']['server-2'].append(task)
    c.update(stage='admission',admission_receipt=receipt_path,post_training_freeze=frozen,
        next_action='Canonical dispatch SAME committed Y57 fixed30; preserve existing Y56 PID and verify native steps.')
    bind(c,ref(receipt_path),frozen,ref(decision),binding)
    tick.save(ROOT/'pipeline/research_work.json',work)
    monitor.save_queue(str(ROOT),queue)
    server=monitor.load_servers(str(ROOT))['server-2']
    assert server['host']=='192.168.1.163' and server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
    files={r['path'].removeprefix('AdaRES/'):r['sha256'] for r in review['files']}
    before=snapshot(server,[existing,task],list(files))
    proc=before['tasks']['AdaRES-Y56']
    assert proc['process_live'] and existing['config'] in proc['cmdline'] and proc['cwd']==server['remote_code_root']
    processes=monitor.find_project_processes('AdaRES_project','server-2',server,queue)
    assert set(processes)=={existing['config']}
    events=[]
    monitor._process_queue(queue,'server-2',server['host'],server['user'],str(ROOT),
        {str(p[2]) for p in processes.values()},{'0','1'},events,max_tasks_per_gpu=1,
        allow_external_overlay=False,remote_code_root=server['remote_code_root'],current_tasks=1,max_tasks_total=2)
    queue=monitor.load_queue(str(ROOT))
    task=next(t for t in queue['gpu_queues']['server-2'] if t['id']==c['id'])
    after=snapshot(server,[existing,task],list(files))
    assert after['tasks']['AdaRES-Y56']['process_live'] and after['tasks']['AdaRES-Y56']['pid']==before['tasks']['AdaRES-Y56']['pid']
    steps=closure.steps(after['tasks'][c['id']]['tail'])
    evidence=save('pipeline/analyses/y57_dispatch_'+TOKEN+'.json',dict(complete=True,id=c['id'],
        observed_at=datetime.now().astimezone().isoformat(),before=before,after=after,task=task,events=events,
        code_commit=head,native_steps=steps,Y56_original_PID_preserved=True,
        exact_deployed_tested_bytes=after['source_hashes']==files))
    bind(c,evidence)
    if task['status']=='running' and task.get('launch_verified'):
        assert after['source_hashes']==files and after['source_marker']==head and len(set(steps))>=2
        assert after['tasks'][c['id']]['process_live']
        c.update(stage='training',experiment_id=c['id'],next_action='Monitor SAME fixed30 single-GPU Y57; apply frozen full comparison, no tuning or blind restarts.')
    else:
        c.update(stage='deployment',next_action='Reconcile exact existing Y57 dispatch/PID/log; do not blindly launch again.')
    work.pop('design_cycle',None)
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(evidence=evidence,status=task['status'],gpu=task.get('gpu'),pid=task.get('pid'),
                         launch_verified=task.get('launch_verified'),native_steps=steps),indent=2),flush=True)
finally:
    lock.close()
