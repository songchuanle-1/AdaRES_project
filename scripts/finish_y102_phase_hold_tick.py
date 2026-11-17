"""Persist this event tick without overriding a zero admission budget or a loader stall."""
import argparse,json,subprocess,sys
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import fresh,GOAL,PAUSE
from y102_deployment import Y101,Y102,git
from reconcile_y62_y63_server1 import remote
import closure,gpu_occupation

def evidence(r):return r['path']+' SHA'+r['sha256']
def close(token):
    lease(token,300)
    command(['scripts/y102_observer_v2.py','observe','--id','AdaRES-Y101','--token',token,'--label','release'])
    attempted=['scripts/y102_observer_v2.py','observe','--id','AdaRES-Y102','--token',token,'--label','release']
    response=subprocess.run([sys.executable,'-B']+attempted,cwd=ROOT,capture_output=True,timeout=60)
    stderr=ROOT/('pipeline/checks/y102_phase_hold_'+token+'_observer.log');assert not stderr.exists();stderr.write_bytes(response.stdout+response.stderr)
    rawpath=ROOT/('pipeline/checks/production_adares-y102_'+token+'_release.log');raw=tick.load(rawpath)
    assert response.returncode!=0 and raw['complete'] and raw['process_identity_verified'] and not raw['native_progress_confirmed']
    assert raw['pid']==4747 and raw['source_marker']==Y102 and raw['before']['latest']==raw['after']['latest']==dict(phase='Train',epoch=1,total=30,step=7,count=4686)
    assert raw['no_bounded_tail_error'] and raw['source_files_verified']==6 and raw['native_YAML_semantics_verified']
    server=monitor.load_servers(str(ROOT))['server-2']
    body='''import datetime,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
proc=pathlib.Path('/proc/5241');assert 'PPid:\\t4747' in (proc/'status').read_text()
rows=subprocess.check_output(['ps','-p','4747,5241','-o','pid,ppid,stat,etime,pcpu,wchan:32,comm','--no-headers']).decode().splitlines()
print(json.dumps(dict(complete=True,observed_at=datetime.datetime.now().astimezone().isoformat(),original_pid=4747,worker_pid=5241,processes=rows,worker_status='\\n'.join(v for v in (proc/'status').read_text().splitlines() if v.startswith(('State:','PPid:','Threads:','VmRSS:'))),remote_mutations=False,stack_attach_attempted=False)))
'''
    worker=ROOT/('pipeline/checks/y102_phase_hold_'+token+'_worker.json');assert not worker.exists();value=remote(server,body,worker);tick.save(worker,value)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['mode']=='event' and view['preflight']['goal_rc']==4 and view['queue']['candidate_budget']==0 and view['research']['creation_budget']==view['research']['reserve_budget']==0
        assert view['generated_at']=='2026-11-12T17:46:51.759889+00:00'
        assert git('rev-parse','HEAD').decode().strip()==Y102 and ref('pipeline/goal.json')['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
        work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};candidate=c['AdaRES-Y102'];assert candidate['stage']=='training' and not candidate.get('terminal_handoff_plan')
        assert c['AdaRES-Y100']['scientific_status']=='fixed30_complete_frozen_review_pending' and c['AdaRES-Y100']['terminal_evidence']['sha256']=='4bea26a6e5562995809aed3ae595b6ab93b17c42b78a277c3a4522144f4f0cad'
        discovery=ref('pipeline/checks/y102_terminal_original_location_'+token+'.log');text=closure.artifact(ROOT,discovery).read_text(encoding='utf-8');assert 'No native checkpoint save announcement yet' in text
        sources=[ref('pipeline/checks/y102_terminal_generation_'+token+'.py'),ref('pipeline/checks/y102_terminal_wrapper_'+token+'.py'),ref('pipeline/checks/y102_terminal_original_location_'+token+'.py'),discovery,ref('scripts/create_y102_terminal_helpers.py')]
        assert all(not (ROOT/'scripts'/name).exists() for name in ('finalize_y102_fixed30.py','test_y102_terminal_handoff.py','prepare_y102_terminal_handoff.py','y102_terminal_contract.py'))
        diagnostic=ROOT/('pipeline/checks/y102_phase_hold_'+token+'_diagnostic.json');assert not diagnostic.exists()
        tick.save(diagnostic,dict(complete=True,token=token,id=candidate['id'],observed_local=datetime.now().astimezone().isoformat(),pid=4747,current_native_progress_confirmed=False,observation=raw,worker=ref(worker),original_native_launch_and_receipt_preserved=True,existing_administrator_stack_request_unchanged=True,no_new_ptrace_or_sudo_attempt=True,terminal_preparation='blocked_no_original_native_checkpoint_save_announcement',terminal_helpers_generated=False,checkpoint_loaded=False,archived=False,training_restarted=False,evidence=[ref(rawpath),ref(stderr),ref(worker)]+sources))
        candidate['latest_training_diagnostic']=ref(diagnostic)
        candidate['next_action']='Original PID4747 remains at Train1step7 and worker5241 CPU-active. Preserve exact source/data/private runtime/seed/fixed30/receipts; no restart or checkpoint claim. Existing administrator stack-help request remains; do not retry denied attachment without changed authority. Future terminal preparation waits for exact original native save announcement; failed discovery immutable. Snapshot admission/creation budget0: no new candidate or launch, no formal target achievement from training-best.'
        candidate['evidence'].extend([ref(diagnostic)]+sources+[ref(worker),ref(rawpath),ref(stderr)])
        work['tick_attempt']=dict(tick_started_at=view['generated_at'],candidate_id=candidate['id'],kind='implementation_test',result='failed',command='python -B scripts/create_y102_terminal_helpers.py --token '+token,observation='Read-only exact native checkpoint-directory discovery failed: original run has no checkpoint save announcement while stalled at Train1step7. Preserved source/log/wrappers; no guessed directory, generated lifecycle helpers, checkpoint loading, GPU task or restart. Current phase budget0 respected.',evidence=[ref(diagnostic)]+sources)
        tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    for name in ('server-1','server-2'):command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',name,'--label','release'])
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml')};index=git('ls-files','--stage','-z')
        work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};context=tick.load(ROOT/'pipeline/headless/.tick_context.json');q=monitor.load_queue(str(ROOT));occupations={}
        assert monitor.load_running(str(ROOT))=={'server-1':{'0':'AdaRES-Y101'},'server-2':{'0':'AdaRES-Y102'}}
        for name,cid,version,pid,held in (('server-1','AdaRES-Y101',Y101,2263581,'AdaRES-Y99'),('server-2','AdaRES-Y102',Y102,4747,'AdaRES-Y100')):
            task=next(r for r in q['gpu_queues'][name] if r['id']==cid);assert (task['status'],task['launch_verified'],task['code_commit'],task['pid'],str(task['gpu']))==('running',True,version,pid,'0');closure.admission(ROOT,task)
            entry=ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json');fresh(tick.load(closure.artifact(ROOT,entry)),'observed_at')
            s=next(r for r in context['queue']['servers'] if r['name']==name);state=gpu_occupation.validated_state(ROOT,context,c[held],s,q);assert state['busy_gpu_ids']==state['reserved_gpu_ids']==[0] and not state['idle_gpu_ids'];occupations[name]=entry
        y101=c['AdaRES-Y101']['latest_training_observation'];observed=tick.load(closure.artifact(ROOT,y101));fresh(observed,'observed_local');assert observed['token']==token and observed['pid']==2263581 and observed['native_progress_confirmed']
        diagnostic_ref=c['AdaRES-Y102']['latest_training_diagnostic'];fresh(tick.load(closure.artifact(ROOT,diagnostic_ref)),'observed_local')
        assert ref('pipeline/admissions/AdaRES-Y100.json')['sha256']=='7091f206c21429c02349c73e68d5e2b06ca3adc9f585465450870ac2a0e019f2' and ref('pipeline/admissions/AdaRES-Y101.json')['sha256']=='379a5b050624141f20345f614877f062f1a61309b55531fe977e023538fecd0c' and ref('pipeline/admissions/AdaRES-Y102.json')['sha256']=='c37ae30c544492ea9e00e7c35ad4290b1062521e8182c13b885a3afb4e96968a'
        assert c['AdaRES-Y103']['additive_version_binding']['sha256']=='7d43d87652f2a50b901f44cadc7fe36bf2758a6b6728eee803be3bc30b181942' and not c['AdaRES-Y103'].get('native_smoke_result')
        assert c['AdaRES-Y101']['terminal_handoff_plan']['sha256']=='410f8494b8ccdef65dd37c8bf7a848e6b912dab23a4fe966c77b191a445a8996'
        paused=next(r for r in q['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57');assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        out=ROOT/('pipeline/checks/y102_phase_hold_'+token+'_closeout.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,observed_at=datetime.now().astimezone().isoformat(),Y101_observation=y101,Y102_diagnostic=diagnostic_ref,occupations=occupations,snapshot_goal_rc=4,admission_budget=0,creation_budget=0,ledger_training_best_triggered_phase_gate=True,formal_FIRST_evaluated=False,formal_goal_achievement_claimed=False,terminal_preparation_passed=False,checkpoint_directory_unknown=True,new_scientific_candidates=0,scientific_CPU_replays=0,native_GPU_tasks=0,admissions=0,launches=0,archives=0,checkpoint_loaded=False,training_restarted=False,original_Y100_archive_not_replayed=True,existing_stack_help_request_not_retried=True,evidence=[ref(__file__),y101,diagnostic_ref]+sources+list(occupations.values())))
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();oldbytes=memory.read_bytes();backup.write_bytes(oldbytes);assert backup.read_bytes()==oldbytes
        old=oldbytes.decode('utf-8').splitlines();phase=observed['after']['latest'];label=phase['phase']+str(phase['epoch'])+'/30 step'+str(phase['step'])+'/'+str(phase['count']);lines=[];changed=set()
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.';changed.add('token')
            elif line.startswith('- Exact previous seven-section memory:'):line='- Exact previous seven-section memory: '+evidence(ref(backup))+'; full original protocols/sources/negative evidence/bans/dedicated recoveries byte-preserved through backup chain; never reopening authority.';changed.add('backup')
            elif line.startswith('- AdaRES-Y101 original PID'):line='- AdaRES-Y101 original PID2263581/server1GPU0/350W, '+label+'; exact source/YAML/private runtime/power/increasing native counters '+evidence(y101)+'; original receipt/freeze preserved.';changed.add('Y101')
            elif line.startswith('- AdaRES-Y102 original PID'):line='- AdaRES-Y102 original PID4747/server2GPU0/280W remains Train1step7/4686. Source/YAML/private runtime/power/process identity preserved; current native progress UNCONFIRMED/STALLED '+evidence(diagnostic_ref)+'. Existing loader5241 stack-help request unchanged. No restart or live checkpoint load.';changed.add('Y102')
            elif line.startswith('- Fresh canonical occupation '):line='- Fresh canonical occupation '+'; '.join(evidence(r) for r in occupations.values())+'; both authorizedGPU0 physically busy/reserved by originalY101/Y102; server2GPU0util0 still has originalPID4747/memory16186MB, never idle. GPU1 full frozen FIRST/cost remain paused.';changed.add('occupation')
            elif line.startswith('- Y102 current progress STALLED'):line='- Y102 stalled Train1step7; worker5241 CPU-active, mainPID4747 waits, no bounded-tail error. Prior ordinary-account gdb ptrace denial/lack of py-bt retained; no new attachment/sudo/root attempt. Existing administrator stack-help request remains unresolved '+evidence(diagnostic_ref)+'.';changed.add('risk')
            elif line.startswith('- No formal FIRST/cost/promotion.'):line='- No formal FIRST/cost/promotion. This snapshot goal_rc4/candidate_budget0/creation_budget0/reserve_budget0 arose from ledger Y100 trainingbest.516448557, NOT formal FIRST. Obey frozen zero budget; no Phase2/seed/ablation/new admission or goal modification. Formalchampion remains Y36=.510725438294; target.5150 pending full original-device review.';changed.add('phase')
            elif line.startswith('- Next tick first observe '):line='- Next tick first observe originalY101/Y102 and phase budget. Y102 original save-directory discovery failed before lifecycle generation: no native checkpoint save announcement; do not guess directory from launch timestamp/filesystem. Preserve wrappers/log '+evidence(discovery)+'. Wait exact original native save announcement before NEW same-run terminal preparation. Y101 existing finalize_y101_fixed30_y102.py valid at HEADY101/Y102. Y103 FIRST native remains unexecuted/current binding preserved; phase budget and exact matching original completion/archive/exclusive GPU0 must all authorize future work. Never replay passed scientific/CPU/native/retention tests.';changed.add('next')
            if line=='## Last Tick':raise AssertionError
            lines.append(line)
        assert changed=={'token','backup','Y101','Y102','occupation','risk','phase','next'}
        lines+=['## Last Tick','- '+token+': canonical observe/dispatch first; event snapshot candidate/creation/reserve budgets0 due ledger-only training-best phase gate. OriginalY101 '+label+'; originalY102 still Train1step7, source/runtime/power/current worker verified read-only. Attempted exact same-run future terminal preparation failed on missing native checkpoint save announcement; immutable wrappers/source/log retained, no terminal helpers generated or registered. Existing admin stack request not repeated, Y100 archive not replayed.',
            '- New scientificIDs0/scientificCPUreplays0/nativeGPUtasks0/admissions0/launches0/archives0/FIRST0/cost0/checkpointloads0/restarts0. Phase gate is not a formal target achievement; all old receipts/scientific goal/source/data/protocols/resources/GPU1+Y57 pauses/scheduler preserved. Closeout '+evidence(ref(out))+'; official receipt/finish/release authoritative.']
        assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=['运行：Y101 '+label+'，原PID2263581/350W/源码/数据/环境及原生递增已核验。Y102仍Train1step7，PID4747/280W和原数据加载子进程5241保留，当前不能确认持续训练进度。','阶段说明：自动台账因Y100训练best.516448557触发goal_rc4/Phase1门控，本轮准入/创建/预备预算均0；正式FIRST和完整冻结后评均未执行，不能宣布达标。保留前置脚本事实表原文，并以本说明限定其含义；不改科学goal或原收据。','准备失败：使用原生保存声明只读定位Y102原检查点目录，实测无checkpoint save announcement；保存失败日志、源码与包装器，不猜目录、不生成无依据终态计划、不加载活跃权重。上一轮Y100完整归档保持，不重做。','两张授权GPU0物理busy/reserved。新科学ID0、科学CPU重跑0、GPU原生检查0、准入0、新训练0、归档0、FIRST0、成本0。'],decisions=['本轮按冻结快照0预算执行，不新增变体、不越过阶段门控或凭训练best宣布收益。保留Y103已有绑定和未执行原生计划，后续须同时满足当轮阶段预算、匹配原owner自然完成/完整精确有限检查点/规范归档和独占空闲GPU0。','Y102原进程和全部负证据保留。已有管理员调用栈协助事项没有变化，不再次尝试权限拒绝、不推断root/sudo、不重启；未来终态准备等待原生保存声明。GPU1全用途与Y57单独暂停、server6禁用、原数据协议/功耗/调度保持。'],evidence=[ref(out),ref(backup)]))
        assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
    finally:writer.close()
    command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)])
    command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists()
    pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status=receipt['status'],postflight=receipt['postflight'],context_committed=True,lease_release='released',memory_lines=len(lines),Y101=phase,Y102=dict(phase='Train',epoch=1,step=7,current_progress_confirmed=False),admission_budget=0,terminal_preparation_passed=False,new_scientific_candidates=0,native_GPU_tasks=0,admissions=0,launches=0,archives=0,formal_target_achievement=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);close(p.parse_args().token)
