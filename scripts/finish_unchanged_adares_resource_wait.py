"""Close a fresh unchanged resource observation without replaying held science or reporting new progress."""
import argparse,json
from datetime import datetime,timezone
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import GOAL,PAUSE
from y102_deployment import git,Y102
import closure

def e(r):return r['path']+' SHA'+r['sha256']

def finish(token):
    lease(token,300);view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');snapshot=tick.load(ROOT/'pipeline/headless/.tick_context.json');assert view['mode']=='event' and view['experiment_first']['enabled'] and view['preflight']['goal_rc']==4 and view['queue']['candidate_budget']==view['research']['creation_budget']==view['research']['reserve_budget']==0
    protected={p:ref(p) for p in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/experiments.csv','pipeline/admissions/AdaRES-Y102.json','pipeline/admissions/AdaRES-Y90.json','pipeline/admissions/AdaRES-Y88.json')};index=git('ls-files','--stage','-z');assert protected['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE and git('rev-parse','HEAD').decode().strip()==Y102
    label='start';initial={server:ref('pipeline/checks/gpu_occupation_'+server+'_start_'+token+'.json') for server in ('server-1','server-2')}
    ages=[(datetime.now(timezone.utc)-datetime.fromisoformat(tick.load(closure.artifact(ROOT,r))['observed_at'])).total_seconds() for r in initial.values()]
    if max(ages)>240:
        for server in ('server-1','server-2'):command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',server,'--label','release'])
        label='release'
    occupations={server:ref('pipeline/checks/gpu_occupation_'+server+'_'+label+'_'+token+'.json') for server in ('server-1','server-2')}
    for server,r in occupations.items():
        o=tick.load(closure.artifact(ROOT,r));assert o['complete'] and o['token']==token and o['scope']['allowed_gpus']==[0] and 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(o['observed_at'])).total_seconds()<300
        pids=[int(row.split(',')[-1].strip()) for row in o['queries']['compute']['stdout'].splitlines() if row.strip()];assert (3312380 if server=='server-1' else 4747) in pids
    rawpath=ROOT/('pipeline/checks/production_adares-y102_'+token+'_start.json');rawpath=rawpath if rawpath.exists() else rawpath.with_suffix('.log');raw=tick.load(rawpath)
    assert raw['complete'] and raw['process_identity_verified'] and raw['pid']==4747 and raw['source_marker']==Y102 and raw['native_YAML_semantics_verified'] and raw['source_files_verified']==6 and raw['no_bounded_tail_error'];confirmed=raw['native_progress_confirmed']
    if not confirmed:assert raw['before']['latest']==raw['after']['latest']==dict(phase='Train',epoch=1,total=30,step=7,count=4686)
    diagnostic=ROOT/('pipeline/checks/resource_wait_'+token+'_y102_diagnostic.json');assert not diagnostic.exists();tick.save(diagnostic,dict(complete=True,id='AdaRES-Y102',token=token,pid=4747,current_native_progress_confirmed=confirmed,observation=raw,original_observation_timestamps_preserved=True,no_restart=True,no_checkpoint_loaded=True,no_denied_attachment_retried=True,evidence=[ref(rawpath),ref(__file__)]))
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c={x['id']:x for x in w['candidates']};y=c['AdaRES-Y90'];prepared=c['AdaRES-Y88'];assert y['stage']==prepared['stage']=='blocked' and y['blocker_kind']==prepared['blocker_kind']=='exclusive_gpu_slot'
        assert not y.get('complete_scientific_review') and not y.get('full_trained_cost_result') and y['cost_resource_recovery_status']=='guard_refused_no_remote_job_created' and not prepared.get('frozen_review_job')
        hold=tick.load(closure.artifact(ROOT,y['cost_validation_hold']));assert hold['launch_uncertainty_resolved'] and not hold['GPU_job_started'];plan=tick.load(closure.artifact(ROOT,y['cost_resource_recovery_plan']));assert plan['original_scientific_core_byte_identical'] and plan['safety_guard_boundary_checks_passed']==8
        binding=tick.load(closure.artifact(ROOT,prepared['frozen_review_execution_binding']));assert binding['CPU_identity_and_launch_guards_passed']==11 and binding['source_files_verified']==26 and not binding['GPU_used']
        ids=snapshot['research_work']['due_ids'];assert len(ids)==view['research']['due_count']==15;due=[]
        for cid in ids:
            x=c[cid];assert x['stage']=='blocked'
            if x['blocker_kind']=='exclusive_gpu_slot':
                assert x['required_server']=='server-1' and x['occupation_evidence']==occupations['server-1'];due.append(dict(id=cid,disposition='unchanged_actual_foreign_original_GPU0_occupation',evidence=occupations['server-1']))
            elif x['blocker_kind']=='human_authorization':
                assert cid=='AdaRES-Y57';due.append(dict(id=cid,disposition='original_separate_user_pause_unchanged_no_resume'))
            else:
                assert cid=='AdaRES-Y97' and x['blocker_kind']=='numerical_control_failure' and x['failure_scope']=='implementation';due.append(dict(id=cid,disposition='original_and_revision_CPU_numerical_failure_unchanged_no_replay'))
        for cid in ('AdaRES-Y67','AdaRES-Y70','AdaRES-Y73','AdaRES-Y75','AdaRES-Y78','AdaRES-Y81'):
            x=c[cid];assert x['stage']=='blocked' and x['blocker_kind']=='human_authorization' and x['required_server']=='server-2' and x['required_gpu']==1
        assert c['AdaRES-Y86']['stage']=='blocked';paused=next(x for x in monitor.load_queue(str(ROOT))['gpu_queues']['server-2'] if x['id']=='AdaRES-Y57');assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        c['AdaRES-Y102']['latest_training_diagnostic']=ref(diagnostic);c['AdaRES-Y102']['evidence'].extend([ref(diagnostic),ref(rawpath)])
        out=ROOT/('pipeline/checks/unchanged_resource_wait_closeout_'+token+'.json');assert not out.exists();tick.save(out,dict(complete=True,token=token,observed_at=datetime.now(timezone.utc).isoformat(),material_change=confirmed,original_server1_foreign_compute_unchanged=3312380,original_server2_Y102_pid=4747,Y102_native_progress_confirmed=confirmed,actual_Y102_counter=raw['after']['latest'],Y90_valid_cost_still_pending=True,old_recovery_not_retried=True,Y88_prepared_not_launched=True,occupations=occupations,due_dispositions=due,new_scientific_IDs=0,new_admissions=0,new_training_launches=0,new_GPU_tests=0,new_full_review_jobs=0,new_cost_jobs=0,scientific_CPU_native_terminal_fullprediction_cost_replays=0,goal_receipts_source_data_mappings_pauses_power_scheduler_limits_and_foreign_jobs_preserved=True,evidence=[ref(__file__),ref(diagnostic),y['cost_validation_hold'],y['cost_resource_recovery_plan'],prepared['frozen_review_execution_binding']]+list(occupations.values())))
        for x in (y,prepared):x['latest_tick_closeout']=ref(out);x['evidence'].append(ref(out))
        tick.save(ROOT/'pipeline/research_work.json',w)
        memory=ROOT/'pipeline/codex_memory.md';oldbytes=memory.read_bytes();backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();backup.write_bytes(oldbytes);assert backup.read_bytes()==oldbytes
        old=oldbytes.decode('utf-8').splitlines();lines=[]
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.'
            elif line.startswith('- Exact previous seven-section memory '):line='- Exact previous seven-section memory '+e(ref(backup))+'; original source/CPU/native/receipt/scientific/negative/checkpoint/failed-closeout/dedicated-recovery details retained through exact-byte chain.'
            elif line.startswith('- Original Y102 PID'):line='- Original Y102 PID4747/server2GPU0/280W; current native counter='+str(raw['after']['latest'])+'; native_progress_confirmed='+str(confirmed)+' '+e(ref(diagnostic))+'. Actual timestamps/source6/nativeYAML/private runtime/data/power/receipt and original process preserved; no restart/checkpoint load/repeated attachment.'
            elif line.startswith('- Fresh occupation '):line='- Fresh occupation '+'; '.join(e(r) for r in occupations.values())+'. Same server1GPU0 foreign diagnostic PID3312380, original server2GPU0 Y102 PID4747. No exclusive authorized GPU0 slot; original Y90 valid cost/Y88 prepared review and all original-device waits retained without retry/launch/migration/interference.'
            elif line.startswith('- Y102 still stalled this tick;') and confirmed:line='- Y102 native counters increased this tick; use the exact saved current observation above. Original process/protocol/receipt retained, no restart. Prior denied ptrace/stack-help history remains original; no repeated attachment or permission request.'
            lines.append(line)
        lines+=['## Last Tick','- '+token+': event; fresh official actual GPU0 occupation check retained same foreign server1PID3312380 and original server2Y102PID4747; native_progress_confirmed='+str(confirmed)+'. Fifteen due original holds dispositioned with real resource evidence, Y57 separate pause and Y97 original/revision numerical failures unchanged.','- No new scientific candidate/admission/training/GPUtest/review/cost job, no repeated CPU/native/terminal/prediction/cost check. Y90 invalid first cost and refused recovery remain unaccepted/not retried; Y88 preparation reused without replay. No new scientific conclusion; unchanged resource wait is not research progress.','- Closeout '+e(ref(out))+'. Exact seven-section memory backup retained; official report/finish/receipt/release authoritative. User notification only if real native progress recovered or another material change is observed.']
        assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists();tick.save(notes,dict(actions=['本轮实际资源复查：原server1GPU0仍为其他项目诊断PID3312380占用，原server2GPU0仍为Y102 PID4747占用/保留；没有合法独占空槽，未启动任何GPU任务或重试上轮已拒绝的恢复。','Y102原进程源码6文件/原生YAML/私有环境身份已核验；实际计数'+str(raw['after']['latest'])+'，native_progress_confirmed='+str(confirmed)+'。保留原观察时间和现场，不重启、不加载检查点、不重复此前拒绝的栈附加或索取管理员许可。','复用原证据：Y90完整效能负结果/首轮未接受成本/资源恢复意图拒绝及目录不存在证明均保留；Y88原26文件/原YAML/私有环境及11身份边界准备保留，无重跑。资源条件未变不等于新研究结果。'],decisions=['保持零新创建/准入/训练、所有原资源/数据/功耗/调度/上限/禁用项，13项原server1GPU0工作等待真实空槽；上轮成本恢复需未来先重核原GPU0独占可用和拒绝意图/目录未创建，再以新不可变标签继续，不能盲重试或直接接受首轮未确认独占成本。','本轮15到期项为原GPU0资源等待、Y57单独用户暂停、Y97原和revision1数值失败；GPU1六项暂停/Y86原失败/原收据和专用恢复规则保持。无实质变化且无可行动项时不发用户通知；不跨项目、不用server6、不干预其他项目任务。'],evidence=[ref(out),ref(backup),ref(diagnostic),y['cost_validation_hold'],y['cost_resource_recovery_plan'],prepared['frozen_review_execution_binding']]+list(occupations.values())))
        assert all(ref(p)==r for p,r in protected.items()) and git('ls-files','--stage','-z')==index
    finally:lock.close()
    command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)])
    command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists();pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status='FINALIZED',postflight='PASS',context_committed=True,lease_release='released',material_change=confirmed,Y102_counter=raw['after']['latest'],new_training_launches=0,new_GPU_jobs=0,user_notification_required=confirmed,memory_lines=len(lines))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
