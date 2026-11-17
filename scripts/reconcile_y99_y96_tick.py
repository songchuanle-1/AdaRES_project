"""Resolve the real idle-to-busy transition by advancing Y96, retaining both failed closeout drafts."""
import argparse,json,subprocess,sys
from datetime import datetime,timedelta,timezone
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import GOAL,PAUSE
from y102_deployment import git,Y102
import closure

def e(r):return r['path']+' SHA'+r['sha256']

def finish(token):
    lease(token,600);view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');snapshot=tick.load(ROOT/'pipeline/headless/.tick_context.json')
    assert view['experiment_first']['enabled'] and view['preflight']['goal_rc']==4 and view['queue']['candidate_budget']==view['research']['creation_budget']==view['research']['reserve_budget']==0
    protected={p:ref(p) for p in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/experiments.csv','pipeline/admissions/AdaRES-Y99.json','pipeline/admissions/AdaRES-Y96_provenance_4198029edcde47b7a8686b8e8dcafad4.json','pipeline/admissions/AdaRES-Y102.json')};index=git('ls-files','--stage','-z');assert protected['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE and git('rev-parse','HEAD').decode().strip()==Y102
    w=tick.load(ROOT/'pipeline/research_work.json');c={x['id']:x for x in w['candidates']};job=c['AdaRES-Y96']['frozen_review_job'];obs_ref=job['latest_observation'];obs=tick.load(closure.artifact(ROOT,obs_ref))
    assert job['native_progress_confirmed'] and obs['native_progress_confirmed'] and obs['compute_pids'] and obs['original_training_absent'] and obs['after']['state']['status']=='evaluating'
    assert c['AdaRES-Y99']['stage']=='complete' and tick.load(closure.artifact(ROOT,c['AdaRES-Y99']['complete_scientific_review']))['full_cost_and_postaudit_complete']
    failure=ROOT/('pipeline/checks/y99_idle_backlog_postflight_failure_'+token+'.json');assert not failure.exists();backups=[]
    for label,path in [('memory','pipeline/codex_memory.md'),('notes','pipeline/reports/notes_'+token+'.json'),('report','pipeline/reports/summary_'+token+'.md'),('receipt','pipeline/tick_receipts/'+token+'.json')]:
        src=ROOT/path;dst=ROOT/('pipeline/snapshots/y99_failed_closeout_'+label+'_'+token+src.suffix);assert not dst.exists();dst.write_bytes(src.read_bytes());assert src.read_bytes()==dst.read_bytes();backups.append(ref(dst))
    tick.save(failure,dict(complete=True,token=token,observed_failure='CLOSURE_INCOMPLETE: Pending candidate limit exceeded; advance or resolve existing work. POSTFLIGHT FAILED: lease retained.',cause='Official occupation collector truthfully released16 legacy exclusive-GPU0 waits into validation while idle; this exceeded the unchanged pending_limit4.',resolution='Advance one prepared original-device Y96 frozen review and verify increasing native counters. Rebind only the other15 original GPU0 waits to its actual physical occupation; no fake resource block, no pending-limit/goal/phase/checker changes.',original_idle_closeout=ref('pipeline/checks/y99_complete_closeout_'+token+'.json'),failed_drafts=backups,new_actual_Y96_launch=job['launch'],native_progress=obs_ref,evidence=[ref(__file__),ref('scripts/finish_y99_complete_tick.py'),ref('scripts/finish_y99_complete_tick_v2.py'),job['launch'],obs_ref]+backups))
    original_idle=tick.load(ROOT/('pipeline/checks/gpu_occupation_server-1_release_'+token+'.json'));waiting=[cid for cid in original_idle['candidate_ids'] if cid!='AdaRES-Y96'];assert len(waiting)==15
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c={x['id']:x for x in w['candidates']}
        for cid in waiting:
            x=c[cid];assert x['stage']=='validation' and x['required_server']=='server-1' and not x.get('frozen_review_job')
            if cid=='AdaRES-Y103':assert x['required_gpu']==0 and x['resource_work_kind']=='native_smoke' and not x.get('native_smoke_result') and not x.get('admission_receipt')
            else:
                f=tick.load(closure.artifact(ROOT,x['post_training_freeze']));assert f['complete'] and f['id']==cid and f['resource_condition']['server']=='server-1' and f['resource_condition']['gpu']==0 and f['resource_condition']['power_limit_watts']==350.
            x.update(stage='blocked',blocked_from_stage='validation',blocker_kind='exclusive_gpu_slot',blocker='Actual original-resource Y96 full frozen review now owns server-1 GPU0/350W with verified native counters; preserve original protocol and wait for exclusive idle GPU0.',retry_after=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),unblock_condition='Fresh authorized idle/unreserved original server-1 GPU0 after current Y96 fixed review releases its exclusive lock; no migration, substitution, restart or phase bypass.')
            x['evidence'].extend([job['launch'],obs_ref,ref(failure)])
        c['AdaRES-Y96']['next_action']='Observe exact existing original-resource Y96 four-method full review using scripts/launch_y96_frozen_review.py observe --token CURRENT --label NEW. Never duplicate/retry launch or replay terminal/CPU/native checks. After complete, collect all four methods9508/141 twice and corrected original paired/noise analysis, then original-device frozen cost and source/checkpoint postaudit; no promotion from training best.'
        for cid in ('AdaRES-Y96','AdaRES-Y99'):c[cid]['evidence'].append(ref(failure))
        tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    for server in ('server-1','server-2'):command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',server,'--label','closing'])
    occupations={s:ref('pipeline/checks/gpu_occupation_'+s+'_closing_'+token+'.json') for s in ('server-1','server-2')}
    for r in occupations.values():
        o=tick.load(closure.artifact(ROOT,r));assert o['complete'] and o['token']==token and o['scope']['allowed_gpus']==[0] and 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(o['observed_at'])).total_seconds()<300
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c={x['id']:x for x in w['candidates']};done=c['AdaRES-Y99'];active=c['AdaRES-Y96'];job=active['frozen_review_job'];review_ref=done['complete_scientific_review'];review=tick.load(closure.artifact(ROOT,review_ref));assert active['stage']=='validation' and job['native_progress_confirmed']
        for cid in waiting:assert c[cid]['stage']=='blocked' and c[cid]['blocker_kind']=='exclusive_gpu_slot' and c[cid]['occupation_evidence']==occupations['server-1']
        for cid in ('AdaRES-Y67','AdaRES-Y70','AdaRES-Y73','AdaRES-Y75','AdaRES-Y78','AdaRES-Y81'):
            x=c[cid];assert x['stage']=='blocked' and x['blocker_kind']=='human_authorization' and x['required_server']=='server-2' and x['required_gpu']==1
        paused=next(x for x in monitor.load_queue(str(ROOT))['gpu_queues']['server-2'] if x['id']=='AdaRES-Y57');assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        diagnostic=c['AdaRES-Y102']['latest_training_diagnostic'];d=tick.load(closure.artifact(ROOT,diagnostic));assert d['complete'] and d['pid']==4747 and not d['current_native_progress_confirmed']
        assert all(ref(p)==r for p,r in protected.items()) and git('ls-files','--stage','-z')==index
        out=ROOT/('pipeline/checks/y99_y96_reconciled_closeout_'+token+'.json');assert not out.exists()
        due=[dict(id=cid,disposition=('full_scientific_review_complete_inconclusive' if cid=='AdaRES-Y99' else 'actual_original_GPU0_frozen_review_running' if cid=='AdaRES-Y96' else 'unchanged_original_user_pause' if cid=='AdaRES-Y57' else 'unchanged_original_and_revision_numerical_failure' if cid=='AdaRES-Y97' else 'actual_GPU0_occupied_native_phase0_no_admission' if cid=='AdaRES-Y103' else 'original_GPU0_review_waits_behind_actual_Y96')) for cid in snapshot['research_work']['due_ids']]
        assert len(due)==view['research']['due_count']==19
        tick.save(out,dict(complete=True,token=token,observed_at=datetime.now(timezone.utc).isoformat(),Y99_complete_scientific_review=review_ref,Y99_FIRST=review['FIRST_miou'],Y99_disposition=review['disposition'],Y96_original_review_launch=job['launch'],Y96_native_progress=obs_ref,Y102_diagnostic=diagnostic,due_dispositions=due,original_idle_observation_preserved=True,postflight_failure_preserved=ref(failure),actual_reparked_original_GPU0_ids=waiting,occupations=occupations,new_scientific_IDs=0,new_admissions=0,new_training_launches=0,new_original_Y99_cost_jobs=1,new_original_Y96_full_review_jobs=1,old_CPU_native_terminal_archive_prediction_replays=0,promotion=False,goal_phase_data_receipts_power_pauses_scheduler_and_pending_limit_unchanged=True,evidence=[ref(__file__),review_ref,done['full_frozen_evaluation_result'],done['full_trained_cost_result'],done['post_measurement_audit'],active['frozen_review_execution_binding'],job['program'],job['launch'],obs_ref,diagnostic,ref(failure)]+list(occupations.values())))
        done['final_tick_closeout']=ref(out);active['latest_tick_closeout']=ref(out)
        for x in (done,active):x['evidence'].append(ref(out))
        tick.save(ROOT/'pipeline/research_work.json',w)
        memory=ROOT/'pipeline/codex_memory.md';old=memory.read_text(encoding='utf-8').splitlines();lines=[]
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Y101 and Y99 original training/terminal'):line='- Y101 and Y99 original training/terminal archives/full eight-pass comparisons/trained-cost/postaudit COMPLETE; NEVER replay. Active original Y96 observer=scripts/launch_y96_frozen_review.py observe --token CURRENT --label NEW; folder='+job['folder']+' workerPID'+str(job['pid'])+'. Original Y102 observer unchanged.'
            elif line.startswith('- Y96 original archived20261110'):line='- Y96 original archived20261110_165702 last30-best23 full finite proof reused WITHOUT replay. Prepared26 actual files/private YAML+class import/11 operational guards '+e(active['frozen_review_execution_binding'])+'. FIRST original-resource review launch '+e(job['launch'])+' and native progress '+e(obs_ref)+'. Original modelcommitacb8c1769b7f3d66a92d287da8859a06a138b8b0 separate server1markerY101; continuous exclusive originalGPU0/350W fixed four-method worker; not training/restart.'
            elif line.startswith('- Fresh canonical occupation '):line='- Fresh final canonical occupation '+'; '.join(e(r) for r in occupations.values())+'. Server1GPU0 actual Y96 fullreview; server2GPU0 original stalled Y102 PID4747. Earlier idle evidence/drafts preserved: after idle collector released16 old waits, Y96 was genuinely launched and other15 reblocked only behind actual native compute. No fake busy proof or pending-limit/gate change.'
            elif line.startswith('- Next tick observe original Y102'):line='- Next tick first observe exact existing Y96 original-resource fullreview and Y102; never duplicate/retry review launch. Y99/Y101 science/cost CLOSED. Y96 after completion needs all four9508/141 twice/corrected paired10000-1999/noise, original-device cost/postaudit before disposition. Y103 native still unexecuted/phase-gated; Y102 terminal tools need genuine original save announcement.'
            lines.append(line)
        lines+=['## Last Tick','- '+token+': event; Y99 eight original native captures and corrected original paired/noise analysis/full frozen serial+native stress cost/postaudit COMPLETE, complete_inconclusive/FIRST.5109262481/no target/no promotion. Prepared26-file original Y96 binding+11 CPU operational guards, then launched its FIRST originalGPU0/350W fixed review with actual increasing native counters. Original Y102 still stalled and preserved.','- First authored closeout failed on extra resource note field; first corrected official finish failed pending_limit after truthful idle collector released16 old holds. Both failures/draft memory/notes/report/receipt preserved '+e(ref(failure))+'. Reconciled through real Y96 review progress and physical busy occupation of other15, without changing limits/checker/goal/phase or fabricating a block.','- New scientificIDs0/admissions0/newtraining0/oldCPU-native-terminal-archive-predictionreplays0; first original Y99 costjob1 and Y96 fullreviewjob1. Final closeout '+e(ref(out))+'. Prior seven-section memory exact backup and failed draft backup chain retained; official report/finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'_reconciled.json');assert not notes.exists()
        tick.save(notes,dict(actions=['完成：Y99四组各两遍全9508表达式/141场景和原修正配对10000/1999/重复噪声分析、首次原协议TRAIN8+大场景串行/原生压力成本及源码和检查点后审计均已核验；原训练、预测和终态没有重跑。FIRST0.5109262481，整体配对CI均跨零，无预设尺寸伤害，未达0.515，记complete_inconclusive。','成本：Y99参数+8192，TRAIN8中位84.97ms vsW2 83.87ms，大场景205.03ms vs205.38ms；原生压力保留峰值13.40430GiB vs13.40039GiB。仅冻结样本和顺序描述，不宣称全数据保证/Pareto/等效/增益。','运行：原已归档Y96通过26文件/原YAML/私有环境导入和11项身份边界后，在自己原server1GPU0/350W启动首次固定四组全量后评；已实测原生计数递增，不以PID代签。原模型commit与当前加法marker分别绑定；未新增训练、准入或重启。','Y102仍原PID4747/server2GPU0/280W/Train1step7，原现场和拒绝附加证据保留，无重启、管理员权限假设或重复索权。','收尾修复：原资源note额外字段导致的本地整对象比较失败已保留；官方首次finish因真实idle自动释放16项旧等待超过上限而保留租约。继续真实Y96后评、证实当前GPU占用后，仅将其他15项原GPU0工作重新按实际占用挂起；保留失败草稿/原idle观察，不改上限/goal/官方检查器，不伪造资源忙。'],decisions=['本轮零创建/新准入/新训练预算和Phase2边界不变；只推进既有已归档且原设备匹配的冻结科学后评。Y99/Y101完成不重测；Y96后续收集原四组全量结果及成本/后审计再处置。','原GPU1六项仍用户暂停且不得迁移，Y57单独暂停；Y97原和revision1数值失败不重跑。保留原goal字节、收据、原数据、禁用项、专用恢复要求、私有环境和功耗，不跨项目、不用server6、不改调度或外部进程。'],evidence=[ref(out),ref(failure),review_ref,done['full_frozen_evaluation_result'],done['full_trained_cost_result'],done['post_measurement_audit'],active['frozen_review_execution_binding'],job['launch'],obs_ref,diagnostic]+list(occupations.values())))
        assert all(ref(p)==r for p,r in protected.items()) and git('ls-files','--stage','-z')==index
    finally:writer.close()
    command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)])
    command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists();pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status='FINALIZED',postflight='PASS',context_committed=True,lease_release='released',memory_lines=len(lines),Y99_FIRST=review['FIRST_miou'],Y99_disposition=review['disposition'],Y96_review_native_progress_verified=True,new_training_launches=0,promotion=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
