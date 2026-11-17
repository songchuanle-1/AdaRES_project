"""Persist verified Y99 completion and honest original-resource backlog, then official closeout."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import GOAL,PAUSE
from y102_deployment import git,Y102
import closure

def e(r):return r['path']+' SHA'+r['sha256']

def finish(token):
    lease(token,300)
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');snapshot=tick.load(ROOT/'pipeline/headless/.tick_context.json')
    assert view['mode']=='event' and view['experiment_first']['enabled'] and view['preflight']['goal_rc']==4
    assert view['queue']['candidate_budget']==view['research']['creation_budget']==view['research']['reserve_budget']==0
    assert git('rev-parse','HEAD').decode().strip()==Y102
    protected={p:ref(p) for p in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/experiments.csv','pipeline/admissions/AdaRES-Y96_provenance_4198029edcde47b7a8686b8e8dcafad4.json','pipeline/admissions/AdaRES-Y99.json','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y102.json')};index=git('ls-files','--stage','-z')
    assert protected['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
    w=tick.load(ROOT/'pipeline/research_work.json');c={x['id']:x for x in w['candidates']};done=c['AdaRES-Y99'];prepared=c['AdaRES-Y96']
    review_ref=done['complete_scientific_review'];review=tick.load(closure.artifact(ROOT,review_ref));formal=tick.load(closure.artifact(ROOT,done['full_frozen_evaluation_result']))
    assert done['stage']=='complete' and review['disposition']=='complete_inconclusive' and review['full_cost_and_postaudit_complete'] and not review['promotion'] and not review['target_reached']
    assert review['FIRST_miou']==0.5109262480516882 and formal['full_passes_verified']==8 and not formal['frozen_material_subgroup_harm']
    binding_ref=prepared['frozen_review_execution_binding'];binding=tick.load(closure.artifact(ROOT,binding_ref));assert binding['complete'] and binding['CPU_identity_and_launch_guards_passed']==11 and binding['source_files_verified']==26 and not binding['GPU_used'] and not binding['checkpoint_loaded'] and not prepared.get('frozen_review_job')
    response=subprocess.run([sys.executable,'-B','scripts/y102_observer_v2.py','observe','--id','AdaRES-Y102','--token',token,'--label','release'],cwd=ROOT,capture_output=True,timeout=60)
    log=ROOT/('pipeline/checks/y99_complete_'+token+'_y102_command.log');assert not log.exists();log.write_bytes(response.stdout+response.stderr)
    rawpath=ROOT/('pipeline/checks/production_adares-y102_'+token+'_release'+('.json' if response.returncode==0 else '.log'));raw=tick.load(rawpath)
    assert raw['complete'] and raw['process_identity_verified'] and raw['pid']==4747 and raw['source_marker']==Y102 and raw['native_YAML_semantics_verified'] and raw['source_files_verified']==6 and raw['no_bounded_tail_error']
    confirmed=raw['native_progress_confirmed'];assert (response.returncode==0) if confirmed else (response.returncode!=0 and raw['before']['latest']==raw['after']['latest']==dict(phase='Train',epoch=1,total=30,step=7,count=4686))
    diagnostic=ROOT/('pipeline/checks/y99_complete_'+token+'_y102_diagnostic.json');assert not diagnostic.exists();tick.save(diagnostic,dict(complete=True,token=token,id='AdaRES-Y102',observed_at=datetime.now(timezone.utc).isoformat(),pid=4747,current_native_progress_confirmed=confirmed,observation=raw,no_restart=True,no_checkpoint_loaded=True,no_denied_stack_attachment_retried=True,evidence=[ref(rawpath),ref(log),ref(__file__)]))
    occupations={name:ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json') for name in ('server-1','server-2')}
    for r in occupations.values():
        o=tick.load(closure.artifact(ROOT,r));assert o['complete'] and o['token']==token and o['scope']['allowed_gpus']==[0] and 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(o['observed_at'])).total_seconds()<300
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c={x['id']:x for x in w['candidates']};done=c['AdaRES-Y99'];prepared=c['AdaRES-Y96'];assert prepared['stage']=='validation' and 'blocker_kind' not in prepared and not prepared.get('frozen_review_job')
        prepared['next_action']='Prepared original Y96 modelcommit/source/private-runtime/native-YAML identity and eleven CPU operational guards PASS; do not replay imports/scientific CPU/native/terminal checks. Continue the original server1GPU0/350W four-method full frozen review using the exact registered execution program, a current-stage-aware exclusive launcher, fresh physical idle/unreserved GPU0 and verified native counters. This is archived-model evaluation, not new training/admission or GPU1 substitution.'
        c['AdaRES-Y102']['latest_training_diagnostic']=ref(diagnostic);c['AdaRES-Y102']['evidence'].extend([ref(diagnostic),ref(rawpath),ref(log)])
        due=[]
        assert len(snapshot['research_work']['due_ids'])==view['research']['due_count']==19
        for cid in snapshot['research_work']['due_ids']:
            x=c[cid]
            if cid=='AdaRES-Y99':due.append(dict(id=cid,disposition='full_scientific_review_complete_inconclusive',evidence=review_ref))
            elif cid=='AdaRES-Y57':
                assert x['stage']=='blocked' and x['blocker_kind']=='human_authorization';due.append(dict(id=cid,disposition='original_user_pause_unchanged'))
            elif cid=='AdaRES-Y97':
                assert x['stage']=='blocked' and x['blocker_kind']=='numerical_control_failure';due.append(dict(id=cid,disposition='original_and_revision_numerical_failure_unchanged_no_replay'))
            else:
                assert x['stage']=='validation' and 'blocker_kind' not in x and x['required_server']=='server-1'
                if cid=='AdaRES-Y103':
                    assert x['required_gpu']==0 and not x.get('native_smoke_result') and not x.get('admission_receipt');due.append(dict(id=cid,disposition='CPU_prepared_native_unexecuted_new_admission_phase0',evidence=x['CPU_result']))
                else:
                    f=tick.load(closure.artifact(ROOT,x['post_training_freeze']));t=tick.load(closure.artifact(ROOT,x['terminal_evidence']));assert f['complete'] and t['complete'] and f['id']==t['id']==cid and f['resource_condition']==dict(server='server-1',gpu=0,power_limit_watts=350.)
                    due.append(dict(id=cid,disposition='original_GPU0_physically_idle_backlog_executable_no_fake_resource_block',freeze=x['post_training_freeze'],terminal=x['terminal_evidence'],prepared_this_tick=(cid=='AdaRES-Y96')))
        for cid in ('AdaRES-Y67','AdaRES-Y70','AdaRES-Y73','AdaRES-Y75','AdaRES-Y78','AdaRES-Y81'):
            x=c[cid];assert x['stage']=='blocked' and x['blocker_kind']=='human_authorization' and x['required_server']=='server-2' and x['required_gpu']==1
        paused=next(x for x in monitor.load_queue(str(ROOT))['gpu_queues']['server-2'] if x['id']=='AdaRES-Y57');assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        assert all(ref(p)==r for p,r in protected.items()) and git('ls-files','--stage','-z')==index
        out=ROOT/('pipeline/checks/y99_complete_closeout_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,observed_at=datetime.now(timezone.utc).isoformat(),Y99_full_scientific_review=review_ref,Y99_original_cost_and_postaudit_complete=True,Y96_operational_preparation=binding_ref,Y96_review_not_launched=True,Y102_diagnostic=ref(diagnostic),due_dispositions=due,occupations=occupations,server1_GPU0_idle=True,new_candidates=0,admissions=0,new_training_launches=0,new_original_frozen_cost_jobs=1,new_original_full_prediction_jobs=0,old_CPU_native_terminal_archive_prediction_replays=0,promotion=False,experiment_first_bounded_attempt_complete=True,executable_backlog_preserved=True,evidence=[ref(__file__),review_ref,done['full_frozen_evaluation_result'],done['full_trained_cost_result'],done['post_measurement_audit'],binding_ref,ref(diagnostic)]+list(occupations.values())))
        done['final_tick_closeout']=ref(out);done['evidence'].append(ref(out));prepared['latest_tick_closeout']=ref(out);prepared['evidence'].append(ref(out));tick.save(ROOT/'pipeline/research_work.json',w)
        memory=ROOT/'pipeline/codex_memory.md';oldbytes=memory.read_bytes();backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();backup.write_bytes(oldbytes);assert backup.read_bytes()==oldbytes
        old=oldbytes.decode('utf-8').splitlines();lines=[]
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.'
            elif line.startswith('- Exact previous seven-section memory:'):line='- Exact previous seven-section memory: '+e(ref(backup))+'; original scientific protocols, source/receipts/negative evidence/bans/dedicated recoveries retained through byte-exact backup chain.'
            elif line.startswith('- Y101 original training/archive/full eight-pass'):line='- Y101 and Y99 original training/terminal archives/full eight-pass comparisons/trained cost/postaudit COMPLETE; NEVER replay. Original Y102 observer=scripts/y102_observer_v2.py observe --id AdaRES-Y102 --token CURRENT --label NEW. Y96 original-resource review preparation below is not a launch.'
            elif line.startswith('- AdaRES-Y102 original PID'):line='- AdaRES-Y102 original PID4747/server2GPU0/280W; '+str(raw['after']['latest'])+'; native_progress_confirmed='+str(confirmed)+' '+e(ref(diagnostic))+'. Exact source/YAML/private runtime/data/power/receipt retained; no restart/checkpoint load.'
            elif line.startswith('- Y101 exact original20261112'):line=line.replace('scientific efficacy pending frozen review.','full scientific review/cost complete_inconclusive; no replay or promotion.')
            elif line.startswith('- Fresh canonical occupation '):line='- Fresh canonical occupation '+'; '.join(e(r) for r in occupations.values())+'. Server1GPU0 idle98MiB/no compute/reservation; canonical collector truthfully released16 old GPU0 holds to validation. Server2GPU0 original Y102 PID4747 remains occupied/reserved; GPU1/foreign compute untouched. Do not fake a busy-resource hold or new admission.'
            elif line.startswith('- Y102 latest progress '):line='- Y102 latest progress '+('RECOVERED' if confirmed else 'STILL STALLED Train1step7')+' '+e(ref(diagnostic))+'. Prior denied ptrace/administrator stack-help request retained; no repeated attachment/sudo/permission question or restart.'
            elif line.startswith('- Y101 full scientific review/cost/postaudit complete_inconclusive;'):line='- Y101 and Y99 full original scientific reviews/cost/postaudit complete_inconclusive; neither promoted. Snapshot goal_rc4/new-admission-creation-reserve0 still derives from Y100 trainingbest.516448557, NOT formal FIRST. No goal/gate/ledger/Phase2/seed/ablation/newtraining edits.'
            elif line.startswith('- Next tick first observe exact existing Y99'):line='- Next tick observe original Y102 and current physical resources. Y99 CLOSED: no review/CPU/native/terminal/cost replay. Resume prepared original-device Y96 full frozen review with exact saved execution, current-stage-aware exclusive launcher and verified native progress; do not rerun CPU imports. Y103 native remains unexecuted/phase-gated. Y102 terminal tools still require genuine original native save announcement.'
            elif line.startswith('- Y99 existing archived20261111'):line='- Y99 exact original20261111_180638 last30-best23 retained; original review and cost now complete below. Original source/runtime/11 operational guard evidence and first launch remain immutable; never restart, rearchive or replay predictions.'
            lines.append(line)
        pos=lines.index('## Closed Or Failed Directions')
        lines[pos:pos]=['- Y99 COMPLETE_INCONCLUSIVE '+e(review_ref)+': FIRST.5109262480516882; same-device Y36.5107215774201463/W2.5101072505383117/B1.5022221992542256. All overall paired10000/1999 CIs cross0, frozen subgroup harm absent, target.5150 missed; no confirmed gain/equivalence/falsification/promotion. Eight full9508/141 passes verified '+e(done['full_frozen_evaluation_result'])+'.','- Y99 original cost '+e(done['full_trained_cost_result'])+'; postaudit '+e(done['post_measurement_audit'])+'. TRAIN8 median84.97ms vsW2 83.87ms; largestpair205.03ms vs205.38ms; native stress peakReserved13.40430GiB vs13.40039GiB, +8192parameters. Twenty fresh-optimizer memory-only stress updates including warmups; original weights/raw source preserved; limited scenes/order/descriptivep95, no full-data/Pareto bound.','- Y96 original archived20261110_165702 last30-best23 exact full finite proof reused WITHOUT replay. Prepared26 actual source files/private YAML+class import/11 operational guards '+e(binding_ref)+'. Original modelcommitacb8c1769b7f3d66a92d287da8859a06a138b8b0 separate current server1markerY101; execution='+e(binding['program'])+'. No CUDA/checkpoint loading/review launch; original server1GPU0/350W only.']
        lines+=['## Last Tick','- '+token+': event; completed original Y99 eight-capture reconciliation, corrected original paired/repeat-noise analysis, frozen TRAIN8+largestpair serial/native stress cost and source/checkpoint postaudit. complete_inconclusive; FIRST.5109262481; targetmiss/no promotion. Prepared original Y96 source/private runtime/YAML and11 boundary guards; no review launch.','- Real Y102 process still stalled and preserved; original negative observer logs retained '+e(ref(diagnostic))+'. Official occupation released idle server1GPU0 holds truthfully; executable archived-model review backlog remains validation, not external blockage. Experimental-first bounded attempt completed; no legacy exhaustion/handoff fabrication.','- New scientificIDs0/admissions0/newtraining0/predictionjobs0/oldCPU-native-terminal-archive-predictionreplays0; original Y99 costjob1 and Y96 operational preparation1. All original goal/receipts/data/mappings/pauses/power/scheduler protected. Closeout '+e(ref(out))+'; canonical report/finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=['完成：原Y99四组各两遍完整9508表达式/141场景、原修正配对10000/1999/重复噪声结果已归集核验；首次原设备冻结串行成本、原生内存压力成本和源码/原检查点后审计完成，原训练/终态/预测没有重跑。','科学结果：FIRST mIoU0.5109262481；同设备Y36 0.5107215774、W2 0.5101072505、B1 0.5022221993。整体配对区间均跨零，预设尺寸子组无伤害判据触发，未达0.515；记录complete_inconclusive，不宣称增益、等效或晋升。','成本：参数+8192；TRAIN8串行中位84.97ms vs83.87ms，大场景205.03ms vs205.38ms，原生压力峰值保留13.40430GiB vs13.40039GiB；仅冻结样本/顺序的描述成本，不构成全数据或Pareto结论。','新准备：原已归档Y96完成26文件身份、原YAML/私有环境类导入及11项操作边界；无GPU或权重加载，未启动后评或新训练。','运行：Y102原PID4747/server2GPU0/280W仍Train1step7；保存真实停滞日志和现场，不重启、不重复附加此前拒绝的栈或索取管理员许可。官方占用显示server1GPU0空闲，已真实释放16项旧GPU0资源挂起到validation；可执行后评保留待续，不伪称资源忙。'],decisions=['按本轮experiment_first真实有结果的有界尝试完成；本轮零新训练/准入/创建预算保持。已完成Y99/Y101不重测，下一原设备后评从准备好的Y96继续；不以训练best代签正式FIRST或绕过Phase2。','GPU1六项原冻结后评仍用户暂停且不迁移，Y57单独暂停；Y97原数值失败不重跑；保留原goal字节、运行收据、原数据、专用恢复规则及独立环境，不跨项目、不用server6、不改功耗/调度/外部进程。'],evidence=[ref(out),ref(backup),review_ref,done['full_frozen_evaluation_result'],done['full_trained_cost_result'],done['post_measurement_audit'],binding_ref,ref(diagnostic)]+list(occupations.values())))
        assert all(ref(p)==r for p,r in protected.items()) and git('ls-files','--stage','-z')==index
    finally:writer.close()
    command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)])
    command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists();pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status='FINALIZED',postflight='PASS',context_committed=True,lease_release='released',memory_lines=len(lines),Y99_disposition=review['disposition'],Y99_FIRST=review['FIRST_miou'],Y96_CPU_preparation_passed=True,new_training_launches=0,server1_GPU0_idle=True,promotion=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
