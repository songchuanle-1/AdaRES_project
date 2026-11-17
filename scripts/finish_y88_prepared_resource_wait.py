"""Close unchanged actual resource waits while preserving the tested remaining-only Y88 continuation."""
import argparse,json
from datetime import datetime,timezone
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import GOAL,PAUSE
from y102_deployment import git,Y102
from y101_frozen_review_job import remote
import closure

IDENTITY=r'''import datetime,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
pids=[int(x) for x in subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().splitlines() if x.strip()];rows=[]
for pid in pids:
    path=pathlib.Path('/proc')/str(pid);row=dict(pid=pid,exists=path.exists())
    if path.exists():row.update(status=[s for s in (path/'status').read_text().splitlines() if s.startswith(('Name:','State:','PPid:','Uid:'))],commandline=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
    rows.append(row)
print(json.dumps(dict(complete=True,observed_at=datetime.datetime.now().astimezone().isoformat(),compute_pids=pids,processes=rows,old_Y88_worker_absent=not pathlib.Path('/proc/3891354').exists(),old_Y88_child_absent=not pathlib.Path('/proc/3891626').exists(),remote_mutations=False,project_files_read=False,foreign_project_files_read=False,GPU_used=False,checkpoint_loaded=False)))
'''

def e(r):return r['path']+' SHA'+r['sha256']
def age(v):return (datetime.now(timezone.utc)-datetime.fromisoformat(v['observed_at'])).total_seconds()

def finish(token):
    lease(token,300);assert __import__('os').environ['CODEX_THREAD_ID']=='01a07e32-78bf-7670-925b-370911ea48db'
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');snapshot=tick.load(ROOT/'pipeline/headless/.tick_context.json');assert view['mode']=='event' and view['experiment_first']['enabled'] and view['preflight']['goal_rc']==4 and view['queue']['candidate_budget']==view['research']['creation_budget']==view['research']['reserve_budget']==0
    protected={p:ref(p) for p in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/experiments.csv','pipeline/admissions/AdaRES-Y102.json','pipeline/admissions/AdaRES-Y90.json','pipeline/admissions/AdaRES-Y88.json')};index=git('ls-files','--stage','-z');assert protected['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE and git('rev-parse','HEAD').decode().strip()==Y102
    initial={s:ref('pipeline/checks/gpu_occupation_'+s+'_start_'+token+'.json') for s in ('server-1','server-2')};label='start'
    if max(age(tick.load(closure.artifact(ROOT,r))) for r in initial.values())>240:
        for s in ('server-1','server-2'):command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',s,'--label','release'])
        label='release'
    occupations={s:ref('pipeline/checks/gpu_occupation_'+s+'_'+label+'_'+token+'.json') for s in ('server-1','server-2')}
    identity_path=ROOT/('pipeline/checks/y88_wait_foreign_identity_'+token+'.json')
    if not identity_path.exists():remote(monitor.load_servers(str(ROOT))['server-1'],IDENTITY,identity_path)
    identity=tick.load(identity_path)
    if age(identity)>240:
        identity_path=ROOT/('pipeline/checks/y88_wait_foreign_identity_'+token+'_release.json');identity=remote(monitor.load_servers(str(ROOT))['server-1'],IDENTITY,identity_path)
    identity_ref=ref(identity_path);assert identity['complete'] and identity['old_Y88_worker_absent'] and identity['old_Y88_child_absent'] and identity['compute_pids'] and 0<=age(identity)<300 and not identity['remote_mutations'] and not identity['foreign_project_files_read']
    assert all(r['exists'] and '/data1/songcl/H100_project/' in r['commandline'] and 'AdaRES_project' not in r['commandline'] for r in identity['processes'])
    for s,r in occupations.items():
        o=tick.load(closure.artifact(ROOT,r));assert o['complete'] and o['token']==token and o['scope']['allowed_gpus']==[0] and 0<=age(o)<300
        pids=[int(row.split(',')[-1].strip()) for row in o['queries']['compute']['stdout'].splitlines() if row.strip()]
        if s=='server-1':assert set(pids)==set(identity['compute_pids']),'Resource identity changed; reconcile fresh occupation rather than closing a stale wait'
        else:assert 4747 in pids
    rawpath=ROOT/('pipeline/checks/production_adares-y102_'+token+'_start.json');rawpath=rawpath if rawpath.exists() else rawpath.with_suffix('.log');raw=tick.load(rawpath)
    assert raw['complete'] and raw['process_identity_verified'] and raw['pid']==4747 and raw['source_marker']==Y102 and raw['native_YAML_semantics_verified'] and raw['source_files_verified']==6 and raw['no_bounded_tail_error'];confirmed=raw['native_progress_confirmed']
    if not confirmed:assert raw['before']['latest']==raw['after']['latest']==dict(phase='Train',epoch=1,total=30,step=7,count=4686)
    diagnostic=ROOT/('pipeline/checks/y88_resource_wait_'+token+'_y102_diagnostic.json');assert not diagnostic.exists();tick.save(diagnostic,dict(complete=True,id='AdaRES-Y102',token=token,pid=4747,current_native_progress_confirmed=confirmed,observation=raw,original_observation_timestamps_preserved=True,no_restart=True,no_checkpoint_loaded=True,no_denied_attachment_retried=True,evidence=[ref(rawpath),ref(__file__)]))
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c={x['id']:x for x in w['candidates']};y=c['AdaRES-Y88'];assert y['stage']=='blocked' and y['blocker_kind']=='exclusive_gpu_slot' and y['frozen_review_job']['failed'] and y['frozen_review_job']['worker_absent'] and len(y['frozen_review_attempts'])==1 and not y.get('frozen_review_continuation_launch_intent') and not y.get('full_frozen_evaluation_result')
        plan_ref=y['frozen_review_continuation_plan'];plan=tick.load(closure.artifact(ROOT,plan_ref));assert plan['completed_capture_identity_coverage_boundary_checks_passed']==8 and plan['native_evaluator_adapter_bootstrap_analysis_payload_byte_identical'] and not plan['GPU_used'] and not plan['checkpoint_loaded'] and not plan['launched']
        partial_ref=y['partial_frozen_review_capture'];partial=tick.load(closure.artifact(ROOT,partial_ref));assert partial['completed_native_passes']==2 and partial['each_expressions']==9508 and partial['each_scenes']==141 and not partial['scientific_rejection'] and not partial['full_four_method_review_complete'];closure.artifact(ROOT,partial['completed_capture_verified']);closure.artifact(ROOT,y['frozen_review_continuation_launch_entry'])
        assert c['AdaRES-Y90']['stage']=='rejected' and c['AdaRES-Y90']['complete_scientific_review']['sha256']=='6aba6007b63c4a0281aa21915e49af7924ad14811627da7776e281b435582f33'
        due=[];ids=snapshot['research_work']['due_ids'];assert len(ids)==view['research']['due_count']
        for cid in ids:
            x=c[cid];assert x['stage']=='blocked'
            if x['blocker_kind']=='exclusive_gpu_slot':
                assert x['required_server']=='server-1' and x['occupation_evidence']==occupations['server-1'];due.append(dict(id=cid,disposition='unchanged_actual_original_GPU0_foreign_compute_occupation',evidence=occupations['server-1']))
            elif x['blocker_kind']=='human_authorization':
                assert cid in ('AdaRES-Y57','AdaRES-Y67','AdaRES-Y70','AdaRES-Y73','AdaRES-Y75','AdaRES-Y78','AdaRES-Y81');due.append(dict(id=cid,disposition='original_user_pause_unchanged_no_reauthorization'))
            else:
                assert cid in ('AdaRES-Y86','AdaRES-Y97') and x['blocker_kind']=='numerical_control_failure' and x['failure_scope']=='implementation';due.append(dict(id=cid,disposition='original_numerical_failure_unchanged_no_replay'))
        for x in c.values():
            if x['stage']=='blocked' and x.get('blocker_kind')=='exclusive_gpu_slot' and x.get('required_server')=='server-1':x['blocker']='Original server1GPU0/350W is physically occupied by other-project compute; use exact occupation_evidence for current PIDs. Preserve all foreign jobs and original-device requirements.'
        for cid in ('AdaRES-Y67','AdaRES-Y70','AdaRES-Y73','AdaRES-Y75','AdaRES-Y78','AdaRES-Y81'):
            x=c[cid];assert x['stage']=='blocked' and x['blocker_kind']=='human_authorization' and x['required_server']=='server-2' and x['required_gpu']==1
        assert c['AdaRES-Y86']['stage']=='blocked' and c['AdaRES-Y97']['stage']=='blocked';paused=next(x for x in monitor.load_queue(str(ROOT))['gpu_queues']['server-2'] if x['id']=='AdaRES-Y57');assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        out=ROOT/('pipeline/checks/y88_prepared_resource_wait_closeout_'+token+'.json');assert not out.exists();tick.save(out,dict(complete=True,token=token,observed_at=datetime.now(timezone.utc).isoformat(),material_change=confirmed,Y102_native_progress_confirmed=confirmed,Y102_counter=raw['after']['latest'],Y88_original_capture_and_preparation_reused_without_replay=True,Y88_remaining_only_plan=plan_ref,Y88_original_partial_capture=partial_ref,Y88_original_worker_absent=True,Y88_continuation_launch_attempted=False,Y88_full_comparison_cost_postaudit_pending=True,current_foreign_compute_identity=identity_ref,occupations=occupations,due_dispositions=due,new_scientific_IDs=0,new_admissions=0,new_training_launches=0,new_GPU_tests=0,new_full_review_jobs=0,new_cost_jobs=0,old_CPU_native_terminal_prediction_cost_replays=0,original_goal_receipts_source_data_mappings_pauses_power_scheduler_limits_and_foreign_jobs_preserved=True,evidence=[ref(__file__),ref(diagnostic),identity_ref,plan_ref,partial_ref,y['frozen_review_continuation_launch_entry']]+list(occupations.values())))
        y['latest_tick_closeout']=ref(out);y['evidence'].extend([ref(out),identity_ref]);c['AdaRES-Y102']['latest_training_diagnostic']=ref(diagnostic);c['AdaRES-Y102']['evidence'].extend([ref(diagnostic),ref(rawpath)]);tick.save(ROOT/'pipeline/research_work.json',w)
        memory=ROOT/'pipeline/codex_memory.md';oldbytes=memory.read_bytes();backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();backup.write_bytes(oldbytes);assert backup.read_bytes()==oldbytes
        old=oldbytes.decode('utf-8').splitlines();lines=[]
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.'
            elif line.startswith('- Exact previous seven-section memory '):line='- Exact previous seven-section memory '+e(ref(backup))+'; original source/CPU/native/receipt/scientific/negative/checkpoint/failed-closeout/dedicated-recovery details retained through exact-byte chain.'
            elif line.startswith('- Original Y102 PID'):line='- Original Y102 PID4747/server2GPU0/280W; native counter='+str(raw['after']['latest'])+'; native_progress_confirmed='+str(confirmed)+' '+e(ref(diagnostic))+'. Original timestamps/source6/nativeYAML/private runtime/data/power/receipt/process preserved; no restart/checkpoint load/repeated denied attachment.'
            elif line.startswith('- Fresh occupation '):line='- Fresh occupation '+'; '.join(e(r) for r in occupations.values())+'. Current originalserver1GPU0 other-project compute '+str(identity['compute_pids'])+' OS identity '+e(identity_ref)+'; no foreign files read or process changed. Originalserver2GPU0 Y102 PID4747 occupied/reserved. Twelve originalserver1GPU0 tasks including prepared Y88 remaining-only review await true exclusive slot; no fake idle/holds or resource substitution.'
            elif line.startswith('- Y102 still stalled this tick;') and confirmed:line='- Y102 current native counters increased; use exact saved current observation. Preserve original process/protocol/receipt; no restart or repeated denied attachment. Original stack-help history remains immutable.'
            lines.append(line)
        lines+=['## Last Tick','- '+token+': event; actual originalGPU0 occupation and other-project OS identity checked. Server1 foreign compute'+str(identity['compute_pids'])+' and server2 originalY102 PID4747 retained; native_progress_confirmed='+str(confirmed)+'. '+str(len(due))+' due original holds have actual resource/pause/numerical-failure dispositions. No new scientific result from a changed foreign PID.','- Existing Y88 two full candidate passes/8 completed-capture boundaries/remaining-only worker and launch entry reused without CPU/native/terminal/prediction/cost replay. No continuation launch attempted while actual originalGPU0 busy. No new IDs/admissions/training/GPUtest/fullreview/cost jobs, no change to Y90 completed rejection, GPU1/Y57 pauses or Y86/Y97 original failures.','- Closeout '+e(ref(out))+'. Exact seven-section backup retained; official report/finish/receipt/release authoritative. User notification only if actual Y102 native progress recovered or another material actionable change occurs; unchanged external occupation is not research progress.']
        assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists();tick.save(notes,dict(actions=['实际资源核验：原server1GPU0被其他项目计算进程'+str(identity['compute_pids'])+'占用，操作系统命令身份已只读确认；原server2GPU0仍为Y102 PID4747占用/保留。没有独占授权空槽，没有调用Y88续评启动器或干预其他项目进程/文件。','Y102原进程、源码6文件/原生YAML/环境已核验；实际计数'+str(raw['after']['latest'])+'，native_progress_confirmed='+str(confirmed)+'。保留原时间和现场，不重启、不加载检查点、不重复此前拒绝的栈附加或管理员请求。','复用原Y88已归集两遍9508/141预测、8项原产物身份/覆盖检查、剩余三种基线续评程序与入口，不重跑已完成CPU/native/终态/预测/成本检查。原失败工作及完整候选输出保持，完整四方法比较/成本/后审计仍待真实原GPU0独占窗口。'],decisions=['本轮'+str(len(due))+'到期项按当前实际占用、原用户暂停或原数值失败保留；资源占用进程变化本身不是新研究结果。Y90有效全科学拒绝不重开，Y57单独暂停和GPU1六项后评暂停不变，Y86/Y97原负证据不重跑。','保持goal_rc4及零新创建/准入/训练额度、原数据/源码/权重/收据/资源/功耗/调度/上限/专用恢复要求。真实独占空槽出现后，继续已准备Y88缺失基线并核验原生计数；不迁移GPU1、不跨项目、不用server6、不驱逐任务。无实质变化且无可行动事项时静默。'],evidence=[ref(out),ref(backup),ref(diagnostic),identity_ref,plan_ref,partial_ref]+list(occupations.values())))
        assert all(ref(p)==r for p,r in protected.items()) and git('ls-files','--stage','-z')==index
    finally:lock.close()
    command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)])
    command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists();pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status='FINALIZED',postflight='PASS',context_committed=True,lease_release='released',material_change=confirmed,user_notification_required=confirmed,new_training_launches=0,new_GPU_jobs=0,memory_lines=len(lines))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
