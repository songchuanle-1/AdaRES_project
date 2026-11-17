"""Close Y101's original terminal/archive plus verified fixed original-resource review launch."""
import argparse,json,subprocess,sys
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import fresh,GOAL,PAUSE
from finish_adares_zero_budget_observation_v3 import validate_due_holds,evidence
from y102_deployment import git,Y102
import closure


def finish(token):
    lease(token,300);view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
    assert view['mode']=='event' and view['preflight']['goal_rc']==4 and view['queue']['candidate_budget']==view['research']['creation_budget']==view['research']['reserve_budget']==0
    due=validate_due_holds(view);assert git('rev-parse','HEAD').decode().strip()==Y102
    immutable={p:ref(p) for p in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/admissions/AdaRES-Y100.json','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y102.json')};index=git('ls-files','--stage','-z');assert immutable['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
    work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};y101=c['AdaRES-Y101'];terminal_ref=y101['terminal_evidence'];terminal=tick.load(closure.artifact(ROOT,terminal_ref));archive_ref=y101['terminal_archive_binding'];archive=tick.load(closure.artifact(ROOT,archive_ref))
    assert y101['stage']=='validation' and y101['scientific_status']=='fixed30_complete_frozen_review_pending';assert terminal['complete'] and terminal['process_absent'] and terminal['completed_epochs']=={p:list(range(1,31)) for p in ('Train','Val')}
    assert terminal['checkpoints'][0]['epoch']==30 and terminal['checkpoints'][1]['epoch']==24 and all(r['model_finite'] and r['optimizer_finite'] and r['optimizer_state_count']==544 and r['scheduler_last_epoch']==r['epoch'] for r in terminal['checkpoints'])
    assert archive['complete'] and archive['terminal_evidence']==terminal_ref and archive['physical_idle_observed']
    assert monitor.load_experiment(str(ROOT),'AdaRES-Y101')['status']=='done_complete' and not monitor.load_running(str(ROOT)).get('server-1',{}).get('0')
    command(['scripts/launch_y101_frozen_review_job.py','observe','--token',token,'--label','release'])
    work=tick.load(ROOT/'pipeline/research_work.json');y101=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');job=y101['frozen_review_job'];review_ref=job['latest_observation'];review=tick.load(closure.artifact(ROOT,review_ref));assert job['native_progress_confirmed'] and review['original_training_absent']
    assert review['after']['state']['status'] in ('evaluating','method_complete','complete')
    assert review['compute_pids'] or review['after']['state']['status']=='complete','Preserve exact job and wait for actual GPU allocation, not only a PID'
    for entry in (job['launch'],job['program'],y101['frozen_review_worker_plan'],y101['frozen_review_bootstrap_test']):closure.artifact(ROOT,entry)
    response=subprocess.run([sys.executable,'-B','scripts/y102_observer_v2.py','observe','--id','AdaRES-Y102','--token',token,'--label','release'],cwd=ROOT,capture_output=True,timeout=60)
    log=ROOT/('pipeline/checks/y101_review_closeout_'+token+'_y102_command.log');assert not log.exists();log.write_bytes(response.stdout+response.stderr)
    path=ROOT/('pipeline/checks/production_adares-y102_'+token+'_release'+('.json' if response.returncode==0 else '.log'));raw=tick.load(path)
    assert raw['complete'] and raw['process_identity_verified'] and raw['pid']==4747 and raw['source_marker']==Y102 and raw['native_YAML_semantics_verified'] and raw['source_files_verified']==6 and raw['no_bounded_tail_error']
    confirmed=raw['native_progress_confirmed']
    assert (response.returncode==0) if confirmed else (response.returncode!=0 and raw['before']['latest']==raw['after']['latest']==dict(phase='Train',epoch=1,total=30,step=7,count=4686))
    diagnostic=ROOT/('pipeline/checks/y101_review_closeout_'+token+'_y102_diagnostic.json');assert not diagnostic.exists()
    tick.save(diagnostic,dict(complete=True,token=token,id='AdaRES-Y102',observed_local=datetime.now().astimezone().isoformat(),pid=4747,current_native_progress_confirmed=confirmed,observation=raw,status='recovered_native_progress' if confirmed else 'unchanged_loader_stall',original_source_runtime_data_and_power_verified=True,original_receipt_preserved=True,no_restart=True,no_checkpoint_loaded=True,no_denied_stack_attachment_retried=True,no_terminal_discovery_replayed=True,evidence=[ref(path),ref(log),ref(__file__)]))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y102');assert c['stage']=='training';c['latest_training_diagnostic']=ref(diagnostic);c['evidence'].extend([ref(diagnostic),ref(path),ref(log)]);tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    for name in ('server-1','server-2'):command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',name,'--label','release'])
    occupations={name:ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json') for name in ('server-1','server-2')}
    for value in occupations.values():fresh(tick.load(closure.artifact(ROOT,value)),'observed_at')
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};y101=c['AdaRES-Y101'];assert y101['stage']=='validation' and y101['terminal_evidence']==terminal_ref and y101['frozen_review_job']['native_progress_confirmed']
        assert c['AdaRES-Y102']['stage']=='training' and not c['AdaRES-Y103'].get('native_smoke_result') and not c['AdaRES-Y103'].get('admission_receipt')
        assert all(ref(p)==v for p,v in immutable.items()) and git('ls-files','--stage','-z')==index
        failure=ref('pipeline/checks/y101_review_worker_initial_failure_'+token+'.json');closure.artifact(ROOT,failure);binding_failure=y101['frozen_review_source_binding_failure'];closure.artifact(ROOT,binding_failure)
        out=ROOT/('pipeline/checks/y101_terminal_review_closeout_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,id='AdaRES-Y101',recorded_at=datetime.now().astimezone().isoformat(),original_fixed30_verified=True,archives=1,terminal=terminal_ref,archive=archive_ref,review_launch=y101['frozen_review_job']['launch'],review_observation=review_ref,review_native_progress_verified=True,review_state=review['after']['state'],new_training_launches=0,new_admissions=0,new_scientific_candidates=0,scientific_CPU_replays=0,original_terminal_CPU_checkpoints_loaded=2,evaluation_job_starts=1,full_comparison_or_cost_declared_complete=False,automatic_promotion=False,due_holds=due,occupations=occupations,original_goal_source_receipts_private_runtime_data_power_pauses_and_scheduler_preserved=True,evidence=[ref(__file__),terminal_ref,archive_ref,review_ref,ref(diagnostic),failure,binding_failure,y101['frozen_review_worker_plan'],y101['frozen_review_bootstrap_test'],y101['frozen_review_job']['launch'],y101['frozen_review_job']['program']]+list(occupations.values())))
        y101['frozen_review_tick_closeout']=ref(out);y101['evidence'].append(ref(out));tick.save(ROOT/'pipeline/research_work.json',work)
        memory=ROOT/'pipeline/codex_memory.md';oldbytes=memory.read_bytes();backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();backup.write_bytes(oldbytes);assert backup.read_bytes()==oldbytes
        old=oldbytes.decode('utf-8').splitlines();lines=[];changed=set()
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.';changed.add('token')
            elif line.startswith('- Exact previous seven-section memory:'):line='- Exact previous seven-section memory: '+evidence(ref(backup))+'; all original scientific protocols/sources/negative evidence/bans/dedicated recoveries preserved through byte-exact backup chain.';changed.add('backup')
            elif line.startswith('- Current observer='):line='- Y101 original training has naturally completed and was archived ONCE; never live-observe/rearchive/restart it. Current original-resource review observer=launch_y101_frozen_review_job.py observe --token CURRENT --label NEW; original job folder '+job['folder']+' workerPID'+str(job['pid'])+'. Y102 training observer=y102_observer_v2.py observe --id AdaRES-Y102 --token CURRENT --label NEW.';changed.add('observer')
            elif line.startswith('- AdaRES-Y101 original PID'):line='- AdaRES-Y101 original PID2263581 exited; full nativeTrain/Val1..30/stable finite last30-best24/model/optimizer544/scheduler30-24 verified ONCE and canonically archived '+evidence(terminal_ref)+'; archive '+evidence(archive_ref)+'. Trainingbest.5062995553016663 is NOT formal FIRST. Original server1GPU0/350W four-method fixed review workerPID'+str(job['pid'])+' now actual native progress verified '+evidence(review_ref)+'. Full comparison/cost/scientific review pending.';changed.add('Y101')
            elif line.startswith('- AdaRES-Y102 original PID'):line='- AdaRES-Y102 original PID4747/server2GPU0/280W; latest '+str(raw['after']['latest'])+'; native_progress_confirmed='+str(confirmed)+' '+evidence(ref(diagnostic))+'. Original source/YAML/private runtime/data/power/receipt preserved, no restart.';changed.add('Y102')
            elif line.startswith('- Y101 exact original20261112_200238;'):line='- Y101 exact original20261112_200238 full finite terminal proof/canonical archive COMPLETE '+evidence(terminal_ref)+'. Old finalize_y101_fixed30_y102.py and lifecycle evidence preserved; NEVER replay it. Last30/best24 remain original, scientific efficacy pending frozen review.';changed.add('terminal')
            elif line.startswith('- Fresh canonical occupation '):line='- Fresh canonical occupation '+'; '.join(evidence(v) for v in occupations.values())+'; server1GPU0 original Y101 review, server2GPU0 original stalled Y102; preserve all foreign compute and GPU1/Y57 pauses.';changed.add('occupation')
            elif line.startswith('- Y102 latest progress '):line='- Y102 latest progress '+('RECOVERED native counters' if confirmed else 'still STALLED Train1step7')+' '+evidence(ref(diagnostic))+'. Prior ptrace denial and no save announcement retained; no repeated administrator permission request, attachment or discovery.';changed.add('risk')
            elif line.startswith('- No formal FIRST/cost/promotion.'):line='- Complete four-method FIRST/paired subgroup/cost/scientific review pending; no automatic target claim or promotion. Snapshot goal_rc4/admission-creation-reserve0 still derives from Y100 trainingbest.516448557, NOT formal FIRST. Goal bytes and phase gates unchanged, no Phase2/seed/ablation/new admissions.';changed.add('phase')
            elif line.startswith('- Next tick first observe originalY101/Y102'):line='- Next tick first observe exact existing Y101 review job and original Y102, current phase budget and actual resources; never duplicate/retry the review automatically or replay old terminal/CPU/native tests. Reconcile full9508/141 twice EACH Y101/Y36/W2/B1 and paired10000/1999 all/size outcomes; full original cost/scientific disposition pending. Y103 original native unexecuted/phase-gated. Y102 terminal tools still require a genuine original native save announcement; retain all failed directory/permission evidence.';changed.add('next')
            elif line.startswith('- Y101 resource-only evaluator adapter14 '):line='- Preparation resource-only adapter14 CPU boundary tests passed '+evidence(y101['frozen_review_resource_adapter'])+'; first full review now uses the same scientific evaluator with only redundant inner lock removed under its continuously held original GPU0 outer lock. Actual current execution is recorded below; preparation evidence and old negatives remain immutable.'
            elif line.startswith('- Y101/Y36/W2/B1 four isolated original-private-runtime '):line='- Original four isolated private-runtime import/config/class bindings passed '+evidence(y101['frozen_review_runtime_binding'])+'; PointNet project-wrapper/private-egg-binary exact identity and original path-assumption failure preserved. Actual first review launch now follows completed original terminal/archive and fresh original-device checks; no environment change.'
            lines.append(line)
        assert changed=={'token','backup','observer','Y101','Y102','terminal','occupation','risk','phase','next'}
        lines+=['## Last Tick','- '+token+': event; original Y101 naturally fixed30 complete, full exact finite proof/archive1; original-resource fixed review job starts1 with actual increasing evaluation counters '+evidence(review_ref)+'. Existing Y102 remains original '+str(raw['after']['latest'])+'. No new scientific IDs/admissions/training/native-smoke/CPU scientific replay/Phase2 or automatic promotion.','- Prepared11 CPU launch guards plus actual child GPU1-rejection bootstrap; preserve local Windows fcntl preparation failure/correction and evaluator-artifact-versus-Git binding failure/correction '+evidence(failure)+'; '+evidence(binding_failure)+'. Neither first failure created a remote job. Review purpose fixed Y101/Y36/W2/B1 twice each; no scheduler/decision loop, old GPU1 reviews stay held.','- Closeout '+evidence(ref(out))+'; original receipts/goals/data/runtime/350W-280W/Y57-GPU1 pauses/foreign jobs preserved. Official report/finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=['完成：Y101原进程自然退出，原20261112_200238完整Train/Val1..30、稳定last30/best24、有限模型/优化器544和调度器30/24已首次精确核验并规范归档；保留原权重，不重启。训练best.506299555不替代正式FIRST。','运行：原server-1 GPU0/350W固定四组后评已首次启动，Y101/Y36/W2/B1各两遍全9508表达式/141场景，已核验原生计数递增；固定工作程序持续持有原GPU0锁，无自动重试、训练或晋升。全量比较/子组/成本及科学结论尚未完成。','Y102原server-2 GPU0/280W仍'+str(raw['after']['latest'])+'，current_native_progress_confirmed='+str(confirmed)+'；原进程和失败证据保留，不重复附加调试器或索取相同管理员许可。','新准备：11项启动CPU边界检查及实际子入口GPU1拒绝检查通过。Windows预检fcntl导入位置错误、评测工具未纳入训练commit的绑定错误均发生在远端任务创建前，已修正操作入口并保留原失败；评测科学逻辑、原冻结工件SHA和训练来源均未放宽。'],decisions=['本轮新增变体/准入/新训练均0，遵守冻结阶段预算。后评使用Y101原冻结server-1 GPU0/350W，未迁移任何原GPU1挂起评测或成本，Y57仍单独暂停；不改goal/原收据/原数据/功耗/调度。','后续只观察这条精确工作程序及原Y102，完整核验所有原始评测、FIRST、配对置信区间/子组伤害与原成本后再作科学判断；不因PID、训练best或单个完成捕获就宣布达标，不自动重复或重启。'],evidence=[ref(out),ref(backup),terminal_ref,archive_ref,review_ref,ref(diagnostic),failure,binding_failure]))
        assert all(ref(p)==v for p,v in immutable.items()) and git('ls-files','--stage','-z')==index
    finally:writer.close()
    command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)])
    command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists()
    pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status='FINALIZED',postflight='PASS',lease_release='released',memory_lines=len(lines),archives=1,evaluation_job_starts=1,new_training_launches=0,native_evaluation_progress_verified=True,complete_scientific_review=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);finish(p.parse_args().token)
