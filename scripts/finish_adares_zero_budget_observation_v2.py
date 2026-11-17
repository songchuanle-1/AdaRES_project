"""Close a zero-budget observation event without replaying unchanged blocked tests."""
import argparse,json,subprocess,sys
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import fresh,GOAL,PAUSE
from y102_deployment import Y101,Y102,git
from reconcile_y62_y63_server1 import remote
import closure,gpu_occupation
def evidence(r):return r['path']+' SHA'+r['sha256']


def validate_due_holds(view):
    rows=view['research']['shown']
    assert view['research']['omitted_due_count']==0 and len(rows)==view['research']['due_count']
    assert {r['id'] for r in rows}<={'AdaRES-Y57','AdaRES-Y97'}
    work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};q=monitor.load_queue(str(ROOT));result=[]
    for row in rows:
        candidate=c[row['id']];assert row['stage']==candidate['stage']=='blocked'
        if row['id']=='AdaRES-Y57':
            assert candidate['blocker_kind']=='human_authorization' and candidate['blocked_from_stage']=='training'
            task=next(r for r in q['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57')
            assert task['status']=='paused' and task['dispatch_hold'] and task['resume_requires_user_authorization'] is True and str(task['gpu'])=='1'
            result.append(dict(id=row['id'],stage='blocked',reason='unchanged_separate_user_pause',resume_authorized=False,GPU1_used=False))
        else:
            assert candidate['blocker_kind']=='numerical_control_failure' and candidate['failure_scope']=='implementation'
            assert candidate['CPU_failure']['sha256']=='d9875d4734907fca95bd7ae43f731d5af5feed79af3ffe35bf4e9bea2441ca9a'
            closure.artifact(ROOT,candidate['CPU_failure'])
            last=candidate['revision_history'][-1];assert last['revision']==1 and last['result']=='failed'
            for entry in last['evidence']:closure.artifact(ROOT,entry)
            result.append(dict(id=row['id'],stage='blocked',reason='unchanged_original_and_revision1_numerical_failures',original_failure=candidate['CPU_failure'],revision1_evidence=last['evidence'],scientific_tests_replayed=False,reopened=False))
    return result

def finish(token):
    lease(token,300)
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
    assert view['mode']=='event' and view['preflight']['goal_rc']==4 and not view['preflight']['operational_issues']
    assert view['queue']['candidate_budget']==view['research']['creation_budget']==view['research']['reserve_budget']==0 and not view['changes']['new_terminal_ids']
    due_holds=validate_due_holds(view)
    assert git('rev-parse','HEAD').decode().strip()==Y102
    immutable={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml','pipeline/admissions/AdaRES-Y100.json','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y102.json')}
    assert immutable['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
    index=git('ls-files','--stage','-z')
    work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']}
    y101=c['AdaRES-Y101']['latest_training_observation'];observation=tick.load(closure.artifact(ROOT,y101));fresh(observation,'observed_local')
    assert observation['token']==token and observation['pid']==2263581 and observation['native_progress_confirmed']
    attempted=['scripts/y102_observer_v2.py','observe','--id','AdaRES-Y102','--token',token,'--label','release']
    response=subprocess.run([sys.executable,'-B']+attempted,cwd=ROOT,capture_output=True,timeout=60)
    stderr=ROOT/('pipeline/checks/adares_observation_'+token+'_y102_command.log');assert not stderr.exists();stderr.write_bytes(response.stdout+response.stderr)
    base=ROOT/('pipeline/checks/production_adares-y102_'+token+'_release')
    result=base.with_suffix('.json') if response.returncode==0 else base.with_suffix('.log')
    raw=tick.load(result)
    assert raw['complete'] and raw['process_identity_verified'] and raw['pid']==4747 and raw['source_marker']==Y102
    assert raw['native_YAML_semantics_verified'] and raw['source_files_verified']==6 and raw['no_bounded_tail_error']
    confirmed=raw['native_progress_confirmed']
    if confirmed:assert response.returncode==0
    else:
        assert response.returncode!=0 and raw['before']['latest']==raw['after']['latest']==dict(phase='Train',epoch=1,total=30,step=7,count=4686)
    server=monitor.load_servers(str(ROOT))['server-2']
    program='''import datetime,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
proc=pathlib.Path('/proc/4747');assert proc.is_dir()
assert (proc/'cwd').resolve()==pathlib.Path('/bak_home/songcl/Project/AdaRES_project/AdaRES')
rows=[r.split(None,6) for r in subprocess.check_output(['ps','-eo','pid,ppid,stat,etime,pcpu,wchan:32,comm','--no-headers']).decode().splitlines()]
selected=[r for r in rows if r[0]=='4747' or r[1]=='4747']
print(json.dumps(dict(complete=True,observed_at=datetime.datetime.now().astimezone().isoformat(),original_pid=4747,processes=selected,remote_mutations=False,stack_attachment_attempted=False)))
'''
    worker=ROOT/('pipeline/checks/adares_observation_'+token+'_y102_worker.json');assert not worker.exists();worker_value=remote(server,program,worker);tick.save(worker,worker_value)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y102');assert candidate['stage']=='training'
        diagnostic=ROOT/('pipeline/checks/adares_observation_'+token+'_y102_diagnostic.json');assert not diagnostic.exists()
        tick.save(diagnostic,dict(complete=True,token=token,id=candidate['id'],observed_local=datetime.now().astimezone().isoformat(),pid=4747,current_native_progress_confirmed=confirmed,observation=raw,status='recovered_native_progress' if confirmed else 'unchanged_loader_stall',worker_observation=ref(worker),original_source_runtime_data_and_power_verified=True,original_receipt_preserved=True,no_restart=True,no_checkpoint_loaded=True,no_denied_stack_attachment_retried=True,no_terminal_discovery_replayed=True,evidence=[ref(result),ref(stderr),ref(worker),ref(__file__)]))
        candidate['latest_training_diagnostic']=ref(diagnostic);candidate['evidence'].extend([ref(diagnostic),ref(worker),ref(result),ref(stderr),ref(__file__)])
        if confirmed:candidate['next_action']='Observe original PID4747/native counters and preserve original source/data/runtime/seed/fixed30. Native progress recovered; preserve all prior stall/permission/discovery evidence. Future terminal preparation still needs exact original save announcement. No new admission at zero snapshot budget.'
        tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    for name in ('server-1','server-2'):command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',name,'--label','release'])
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};q=monitor.load_queue(str(ROOT));context=tick.load(ROOT/'pipeline/headless/.tick_context.json');occupations={}
        assert monitor.load_running(str(ROOT))=={'server-1':{'0':'AdaRES-Y101'},'server-2':{'0':'AdaRES-Y102'}}
        for name,cid,version,pid,held in (('server-1','AdaRES-Y101',Y101,2263581,'AdaRES-Y99'),('server-2','AdaRES-Y102',Y102,4747,'AdaRES-Y100')):
            task=next(r for r in q['gpu_queues'][name] if r['id']==cid);assert (task['status'],task['launch_verified'],task['code_commit'],task['pid'],str(task['gpu']))==('running',True,version,pid,'0');closure.admission(ROOT,task)
            entry=ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json');fresh(tick.load(closure.artifact(ROOT,entry)),'observed_at')
            s=next(r for r in context['queue']['servers'] if r['name']==name);state=gpu_occupation.validated_state(ROOT,context,c[held],s,q)
            assert state['busy_gpu_ids']==state['reserved_gpu_ids']==[0] and not state['idle_gpu_ids'];occupations[name]=entry
        fresh(tick.load(closure.artifact(ROOT,y101)),'observed_local');fresh(tick.load(diagnostic),'observed_local')
        assert c['AdaRES-Y100']['scientific_status']=='fixed30_complete_frozen_review_pending' and c['AdaRES-Y100']['terminal_evidence']['sha256']=='4bea26a6e5562995809aed3ae595b6ab93b17c42b78a277c3a4522144f4f0cad'
        assert c['AdaRES-Y103']['additive_version_binding']['sha256']=='7d43d87652f2a50b901f44cadc7fe36bf2758a6b6728eee803be3bc30b181942' and not c['AdaRES-Y103'].get('native_smoke_result') and not c['AdaRES-Y103'].get('admission_receipt')
        assert c['AdaRES-Y101']['terminal_handoff_plan']['sha256']=='410f8494b8ccdef65dd37c8bf7a848e6b912dab23a4fe966c77b191a445a8996'
        paused=next(r for r in q['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57');assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        assert all(ref(p)==r for p,r in immutable.items()) and git('ls-files','--stage','-z')==index
        out=ROOT/('pipeline/checks/adares_observation_'+token+'_closeout.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,observed_at=datetime.now().astimezone().isoformat(),Y101_observation=y101,Y102_diagnostic=ref(diagnostic),Y102_current_progress_confirmed=confirmed,occupations=occupations,snapshot_goal_rc=4,due_hold_verification=due_holds,admission_budget=0,creation_budget=0,new_scientific_candidates=0,scientific_CPU_replays=0,native_GPU_tasks=0,admissions=0,launches=0,archives=0,checkpoint_loads=0,training_restarts=0,failed_terminal_discovery_not_replayed=True,denied_stack_attachment_not_retried=True,old_archives_and_receipts_preserved=True,formal_FIRST_or_cost_executed=False,formal_goal_achievement_claimed=False,evidence=[ref(__file__),y101,ref(diagnostic),ref(worker),ref(result),ref(stderr)]+list(occupations.values())))
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();oldbytes=memory.read_bytes();backup.write_bytes(oldbytes);assert backup.read_bytes()==oldbytes
        old=oldbytes.decode('utf-8').splitlines();phase=observation['after']['latest'];phase_label=phase['phase']+str(phase['epoch'])+'/30 step'+str(phase['step'])+'/'+str(phase['count']);lines=[];changed=set()
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.';changed.add('token')
            elif line.startswith('- Exact previous seven-section memory:'):line='- Exact previous seven-section memory: '+evidence(ref(backup))+'; full original protocols/sources/negative evidence/bans/dedicated recoveries byte-preserved through backup chain; never reopening authority.';changed.add('backup')
            elif line.startswith('- AdaRES-Y101 original PID'):line='- AdaRES-Y101 original PID2263581/server1GPU0/350W, '+phase_label+'; exact original source/YAML/private runtime/power/increasing native counters '+evidence(y101)+'; receipt/freeze unchanged.';changed.add('Y101')
            elif line.startswith('- AdaRES-Y102 original PID'):line='- AdaRES-Y102 original PID4747/server2GPU0/280W; latest '+str(raw['after']['latest'])+'; current_native_progress_confirmed='+str(confirmed)+' '+evidence(ref(diagnostic))+'. Exact original source/YAML/private runtime/power/process verified; worker observations '+evidence(ref(worker))+'. No restart/checkpoint load.';changed.add('Y102')
            elif line.startswith('- Fresh canonical occupation '):line='- Fresh canonical occupation '+'; '.join(evidence(r) for r in occupations.values())+'; both authorizedGPU0 physically busy/reserved by originalY101/Y102. An owned compute PID/memory remains occupied even at0utilization. All GPU1 frozen FIRST/cost remain user-paused.';changed.add('occupation')
            elif line.startswith('- Y102 stalled Train1step7;') or line.startswith('- Y102 current progress ') or line.startswith('- Y102 latest progress '):line='- Y102 latest progress '+('RECOVERED with actual increasing native counters' if confirmed else 'still STALLED Train1step7')+'; exact current diagnosis '+evidence(ref(diagnostic))+'. Prior ptrace denial/missing py-bt retained; no new attachment/root/sudo request or repeated administrator permission question. Failed terminal discovery remains immutable; no save announcement inferred.';changed.add('risk')
            lines.append(line)
        assert changed=={'token','backup','Y101','Y102','occupation','risk'}
        lines+=['## Last Tick','- '+token+': event; due_count'+str(view['research']['due_count'])+' exactly verified parked holds; phase/admission/creation/reserve budgets0. Canonical observation/dispatch first; no ready tasks. Y101 '+phase_label+'; Y102 current_native_progress_confirmed='+str(confirmed)+', original source/runtime/data/power/worker identity checked read-only. Unchanged failed directory preparation/denied stack access/old archives/CPU/native/retention tests not replayed.',
            '- New scientificIDs0/scientificCPUreplays0/nativeGPUtasks0/admissions0/launches0/archives0/FIRST0/cost0/checkpointloads0/restarts0. Raw trainingbest phase gate is not formal FIRST or target achievement. All source/receipts/scientific goal/data/protocols/resources/GPU1+Y57 pauses/scheduler preserved. Closeout '+evidence(ref(out))+'; official finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=['运行：Y101 '+phase_label+'，原PID2263581/源码/原生YAML/精确数据/私有环境/350W与原生递增已核验。','Y102 '+('实测恢复原生计数递增。' if confirmed else '仍Train1step7/4686，当前原生进度未恢复。')+'原PID4747/server2GPU0/280W/源码/原数据加载子进程保留，当前只读诊断已存；没有停止或重启。','本轮event，due_count'+str(view['research']['due_count'])+'项为已核验挂起候选、无可派发待跑任务，准入/创建/预备预算仍0。未重做缺保存声明的目录检查、未重试权限拒绝的调用栈读取、未重做旧归档或已通过科学测试；Y103旧绑定和未执行原生计划保持。','Y100训练best导致自动阶段门控，正式FIRST及完整原设备后评尚未执行，不宣布目标达成。本轮新变体/原生GPU检查/准入/新训练/归档/FIRST/成本均0。'],decisions=['遵守本轮冻结阶段预算；保留两条原运行、Y102既有挂起条件及全部失败证据，不用反复等价测试制造进展。下一轮先查真实进度与预算，后续须满足完整原检查点/归档和独占授权资源。','仅AdaRES，GPU1全用途暂停和Y57单独暂停保持；不跨项目、不用server6、不迁移冻结后评/成本、不改科学goal/原收据/数据/功耗/调度。既有管理员调用栈协助事项没有新增授权，不重复提出相同确认。'],evidence=[ref(out),ref(backup)]))
        assert all(ref(p)==r for p,r in immutable.items()) and git('ls-files','--stage','-z')==index
    finally:writer.close()
    command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)])
    command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists()
    pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status=receipt['status'],postflight=receipt['postflight'],context_committed=True,lease_release='released',memory_lines=len(lines),Y101=phase,Y102=raw['after']['latest'],Y102_current_progress_confirmed=confirmed,admission_budget=0,new_scientific_candidates=0,native_GPU_tasks=0,admissions=0,launches=0,archives=0)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);finish(p.parse_args().token)
