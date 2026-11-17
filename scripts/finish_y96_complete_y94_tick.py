"""Close verified Y96 scientific completion and the first original-resource Y94 frozen review."""
import argparse,json
from datetime import datetime,timezone
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import GOAL,PAUSE
from y102_deployment import git,Y102
import closure

def e(r):return r['path']+' SHA'+r['sha256']

def finish(token):
    lease(token,600);view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');snapshot=tick.load(ROOT/'pipeline/headless/.tick_context.json')
    assert view['mode']=='event' and view['experiment_first']['enabled'] and view['preflight']['goal_rc']==4
    assert view['queue']['candidate_budget']==view['research']['creation_budget']==view['research']['reserve_budget']==0
    protected={p:ref(p) for p in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/experiments.csv','pipeline/admissions/AdaRES-Y94.json','pipeline/admissions/AdaRES-Y96_provenance_4198029edcde47b7a8686b8e8dcafad4.json','pipeline/admissions/AdaRES-Y99.json','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y102.json')};index=git('ls-files','--stage','-z')
    assert protected['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE and git('rev-parse','HEAD').decode().strip()==Y102
    w=tick.load(ROOT/'pipeline/research_work.json');c={x['id']:x for x in w['candidates']};done=c['AdaRES-Y96'];active=c['AdaRES-Y94'];job=active['frozen_review_job'];obs_ref=job['latest_observation'];obs=tick.load(closure.artifact(ROOT,obs_ref))
    assert job['native_progress_confirmed'] and obs['native_progress_confirmed'] and obs['compute_pids'] and obs['original_training_absent'] and obs['after']['state']['status']=='evaluating'
    review_ref=done['complete_scientific_review'];review=tick.load(closure.artifact(ROOT,review_ref));formal=tick.load(closure.artifact(ROOT,done['full_frozen_evaluation_result']));assert done['stage']=='complete' and review['full_cost_and_postaudit_complete'] and review['disposition']=='complete_inconclusive' and review['FIRST_miou']==0.5082609879270078 and not review['promotion'] and not formal['frozen_material_subgroup_harm']
    rawpath=ROOT/('pipeline/checks/production_adares-y102_'+token+'_start.log');raw=tick.load(rawpath)
    assert raw['complete'] and raw['process_identity_verified'] and raw['pid']==4747 and raw['source_marker']==Y102 and raw['native_YAML_semantics_verified'] and raw['source_files_verified']==6 and raw['no_bounded_tail_error'] and not raw['native_progress_confirmed']
    assert raw['before']['latest']==raw['after']['latest']==dict(phase='Train',epoch=1,total=30,step=7,count=4686)
    diagnostic=ROOT/('pipeline/checks/y96_complete_'+token+'_y102_diagnostic.json');assert not diagnostic.exists();tick.save(diagnostic,dict(complete=True,id='AdaRES-Y102',token=token,recorded_at=datetime.now(timezone.utc).isoformat(),pid=4747,current_native_progress_confirmed=False,observation=raw,original_observation_timestamp_preserved=True,repeated_probe=False,no_restart=True,no_checkpoint_loaded=True,no_denied_stack_attachment_retried=True,evidence=[ref(rawpath),ref(__file__)]))
    failure=ref('pipeline/checks/y96_cost_shared_import_failure_'+token+'.json');closure.artifact(ROOT,failure)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c={x['id']:x for x in w['candidates']};waiting=[x for x in w['candidates'] if x['stage']=='blocked' and x.get('blocker_kind')=='exclusive_gpu_slot' and x.get('required_server')=='server-1'];assert len(waiting)==14
        for x in waiting:
            x['blocker']='Actual original-resource Y94 four-method full frozen review owns server-1 GPU0/350W with verified increasing native counters; original protocol and products retained.'
            x['unblock_condition']='Fresh exclusive authorized idle/unreserved original server-1 GPU0 after current Y94 fixed review releases its lease; no migration, substitution, restart or phase bypass.'
            x['evidence'].extend([job['launch'],obs_ref])
        c['AdaRES-Y102']['latest_training_diagnostic']=ref(diagnostic);c['AdaRES-Y102']['evidence'].extend([ref(diagnostic),ref(rawpath)])
        tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    for server in ('server-1','server-2'):command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',server,'--label','release'])
    occupations={server:ref('pipeline/checks/gpu_occupation_'+server+'_release_'+token+'.json') for server in ('server-1','server-2')}
    for r in occupations.values():
        o=tick.load(closure.artifact(ROOT,r));assert o['complete'] and o['token']==token and o['scope']['allowed_gpus']==[0] and 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(o['observed_at'])).total_seconds()<300
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c={x['id']:x for x in w['candidates']};done=c['AdaRES-Y96'];active=c['AdaRES-Y94'];assert active['stage']=='validation' and done['stage']=='complete'
        due=[];ids=snapshot['research_work']['due_ids'];assert len(ids)==view['research']['due_count']==23
        for cid in ids:
            x=c[cid]
            if cid=='AdaRES-Y96':due.append(dict(id=cid,disposition='full_scientific_review_complete_inconclusive',evidence=review_ref))
            elif cid=='AdaRES-Y94':due.append(dict(id=cid,disposition='original_device_full_frozen_review_running',evidence=obs_ref))
            elif x.get('blocker_kind')=='human_authorization':
                assert cid in ('AdaRES-Y67','AdaRES-Y70','AdaRES-Y73','AdaRES-Y75','AdaRES-Y78','AdaRES-Y81') and x['stage']=='blocked' and x['required_server']=='server-2' and x['required_gpu']==1
                f=tick.load(closure.artifact(ROOT,x['post_training_freeze']));assert f['id']==cid and f['resource_condition']['gpu']==1 and f['resource_condition']['server']=='server-2' and f['resource_condition']['power_limit_watts']==390.
                due.append(dict(id=cid,disposition='original_GPU1_review_still_user_paused',freeze=x['post_training_freeze'],no_migration=True))
            elif x.get('blocker_kind')=='numerical_control_failure':
                assert cid=='AdaRES-Y86' and x['stage']=='blocked' and x['failure_scope']=='implementation' and not x.get('native_smoke_result') and x['native_smoke_failure']['sha256']=='8382c924aa1297e5836d53af7c46517917336e269d8ff3656fbaeab547a84b6c';closure.artifact(ROOT,x['native_smoke_failure']);due.append(dict(id=cid,disposition='unchanged_original_native_numerical_failure_no_replay',evidence=x['native_smoke_failure']))
            else:
                assert x['stage']=='blocked' and x['blocker_kind']=='exclusive_gpu_slot' and x['required_server']=='server-1' and x['occupation_evidence']==occupations['server-1']
                if cid=='AdaRES-Y103':
                    assert x['required_gpu']==0 and x['resource_work_kind']=='native_smoke' and not x.get('native_smoke_result') and not x.get('admission_receipt');due.append(dict(id=cid,disposition='native_unexecuted_phase0_and_actual_original_GPU0_busy',evidence=x['CPU_result']))
                else:
                    f=tick.load(closure.artifact(ROOT,x['post_training_freeze']));t=tick.load(closure.artifact(ROOT,x['terminal_evidence']));assert f['complete'] and t['complete'] and f['id']==t['id']==cid and f['resource_condition']['server']=='server-1' and f['resource_condition']['gpu']==0 and f['resource_condition']['power_limit_watts']==350.
                    due.append(dict(id=cid,disposition='original_GPU0_frozen_review_waits_behind_actual_Y94',freeze=x['post_training_freeze'],terminal=x['terminal_evidence'],no_terminal_or_checkpoint_replay=True))
        paused=next(x for x in monitor.load_queue(str(ROOT))['gpu_queues']['server-2'] if x['id']=='AdaRES-Y57');assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        assert all(ref(p)==r for p,r in protected.items()) and git('ls-files','--stage','-z')==index
        out=ROOT/('pipeline/checks/y96_complete_y94_closeout_'+token+'.json');assert not out.exists();tick.save(out,dict(complete=True,token=token,observed_at=datetime.now(timezone.utc).isoformat(),Y96_full_scientific_review=review_ref,Y96_FIRST=review['FIRST_miou'],Y96_disposition=review['disposition'],Y94_original_review_launch=job['launch'],Y94_native_progress=obs_ref,Y102_diagnostic=ref(diagnostic),due_dispositions=due,occupations=occupations,new_scientific_IDs=0,new_admissions=0,new_training_launches=0,new_original_Y96_cost_jobs=1,new_original_Y94_full_review_jobs=1,old_CPU_native_terminal_archive_prediction_replays=0,promotion=False,goal_phase_data_receipts_power_pauses_scheduler_pending_limit_and_source_preserved=True,local_shared_import_failure_preserved=failure,evidence=[ref(__file__),review_ref,done['full_frozen_evaluation_result'],done['full_trained_cost_result'],done['post_measurement_audit'],active['frozen_review_execution_binding'],job['program'],job['launch'],obs_ref,ref(diagnostic),failure]+list(occupations.values())))
        done['final_tick_closeout']=ref(out);active['latest_tick_closeout']=ref(out)
        for x in (done,active):x['evidence'].append(ref(out))
        tick.save(ROOT/'pipeline/research_work.json',w)
        memory=ROOT/'pipeline/codex_memory.md';oldbytes=memory.read_bytes();backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();backup.write_bytes(oldbytes);assert backup.read_bytes()==oldbytes
        old=oldbytes.decode('utf-8').splitlines();lines=[]
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.'
            elif line.startswith('- Exact previous seven-section memory:'):line='- Exact previous seven-section memory: '+e(ref(backup))+'; original source/receipt/science/negative-evidence/bans/dedicated recoveries and failed old-closeout drafts retained through exact-byte chain.'
            elif line.startswith('- Y96 original archived20261110'):line='- Y96 exact original20261110_165702 last30-best23/full finite terminal and original native/CPU/binding/launch evidence retained; FULL SCIENTIFIC REVIEW COMPLETE below. Never replay training/terminal/CPU/native/predictions/cost or promote its training best.'
            elif line.startswith('- Y101 and Y99 original training/terminal'):line='- Y101/Y99/Y96 original fixed30 archives/full eight-pass comparisons/trained-cost/postaudit COMPLETE; NEVER replay. Active original Y94 observer=scripts/y94_frozen_review_job.py observe --token CURRENT --label NEW; folder='+job['folder']+' workerPID'+str(job['pid'])+'. Original Y102 observer remains scripts/y102_observer_v2.py observe --id AdaRES-Y102 --token CURRENT --label NEW.'
            elif line.startswith('- AdaRES-Y102 original PID'):line='- AdaRES-Y102 original PID4747/server2GPU0/280W; '+str(raw['after']['latest'])+'; native_progress_confirmed=False '+e(ref(diagnostic))+'. Actual observation timestamps preserved, no repeated equivalent probe; exact source/YAML/private runtime/data/power/receipt and original process retained.'
            elif line.startswith('- Fresh final canonical occupation '):line='- Fresh final canonical occupation '+'; '.join(e(r) for r in occupations.values())+'. Server1GPU0 original Y94 fullreview; server2GPU0 original stalled Y102 PID4747. Fourteen other originalGPU0 tasks wait on actual Y94 compute; no transient idle collector/candidate-limit change/fake busy evidence. Older idle/failed-closeout evidence stays in byte-exact backup chain.'
            elif line.startswith('- Y102 latest progress '):line='- Y102 latest progress STILL STALLED Train1step7 '+e(ref(diagnostic))+'. Prior denied ptrace/administrator stack-help request retained; no repeated attachment/sudo/permission question or restart.'
            elif line.startswith('- Y101 and Y99 full original scientific reviews/'):line='- Y101/Y99/Y96 full original science/cost/postaudit complete_inconclusive; no promotion. Snapshot goal_rc4/new-admission-creation-reserve0 still derives from Y100 trainingbest.516448557, NOT formal FIRST. No goal/gate/ledger/Phase2/seed/ablation/newtraining edits.'
            elif line.startswith('- Next tick first observe exact existing Y96'):line='- Next tick first observe exact existing Y94 original-resource fullreview and original Y102; never duplicate/retry launch. Y101/Y99/Y96 science/cost CLOSED. After Y94 completes collect four9508/141 twice/corrected paired10000-1999/repeatnoise, original-device cost/postaudit before disposition. Y103 native still unexecuted/phase-gated; Y102 terminal tools need a genuine original save announcement.'
            lines.append(line)
        pos=lines.index('## Closed Or Failed Directions');lines[pos:pos]=['- Y96 COMPLETE_INCONCLUSIVE '+e(review_ref)+': FIRST.5082609879270078 versus original-device Y36.5107204851656577/W2.5100091052719461/B1.5021153087615295; target.515 missed. All overall scene-paired CIs cross0, no frozen subgroup harm or demonstrated gain/equivalence/falsification/promotion; eight full9508/141 verified '+e(done['full_frozen_evaluation_result'])+'.','- Y96 original cost '+e(done['full_trained_cost_result'])+'; source/checkpoint postaudit '+e(done['post_measurement_audit'])+'. Parameters+65536; TRAIN8 median87.00ms vsW2 83.70ms, largestpair205.59ms vs208.56ms; native stress peakReserved13.40234GiB vs13.40039GiB. Frozen serial warm2/read3 +20 memory-only fresh-optimizer native updates; limited scenes/order/descriptivep95, no full-data/Pareto proof; raw source/original weights preserved.','- Y94 archived original20261109_111838 last30-best21/oldPID218513; independent W2 zero head-axis geometry bias inside same six query-self allowed SCG neighborhoods (+24,total164629661); complete original masks/heads/gates/data/seed1999/fixed30 intact. Prepared26 source files/private original YAML+class import/11 guards '+e(active['frozen_review_execution_binding'])+'. Original modelcommit3f1c351f8a04e12745526e9abb3dbac46af806f1 separate current markerY101; FIRST original server1GPU0/350W fixed review launch '+e(job['launch'])+' with native progress '+e(obs_ref)+'. Not training or GPU1 substitution.']
        lines+=['## Last Tick','- '+token+': event; Y96 original eight captures/corrected original paired-noise analysis/frozen serial+native stress cost and source/checkpoint postaudit COMPLETE, complete_inconclusive/FIRST.5082609879/no target/no promotion. Prepared Y94 exact26-file runtime/YAML+11 operational guards and launched FIRST original-device full frozen review with actual native counters. Y102 original stalled process preserved.','- Local Y96 cost identity wrapper first shared import failed before plan/GPU/remote; old helper and failure '+e(failure)+' retained. V2 restores only original shared remote import, original frozen AST and all identities/guards preserved; first actual cost then completed. No scientific/CPU/native/prediction/terminal replay or checker/limit changes.','- New scientificIDs0/admissions0/newtraining0/oldCPU-native-terminal-archive-predictionreplays0; first original Y96 costjob1 and Y94 frozenreviewjob1. Final closeout '+e(ref(out))+'. Exact seven-section backup retained; canonical report/finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists();tick.save(notes,dict(actions=['完成：Y96原四组八遍全9508表达式/141场景和修正的原配对10000/1999/重复噪声结果已归集校验；首次原冻结串行/原生压力成本与源码和原检查点后审计均完成，原训练、预测、终态检查无重跑。','科学处置：正式FIRST mIoU0.5082609879；原同设备Y36 0.5107204852/W2 0.5100091053/B1 0.5021153088。整体配对区间均跨零，无预设尺寸伤害，未达0.515，complete_inconclusive，不宣称增益、等效、机制证伪或晋升。','成本：参数+65536，TRAIN8串行中位87.00ms vs83.70ms，大场景205.59ms vs208.56ms，原生压力峰值保留13.40234GiB vs13.40039GiB。冻结样本/顺序描述，不代表全数据上界或Pareto；原权重/源码字节不变。','运行：已归档Y94原26文件/原YAML/私有环境导入及11身份边界通过，在自己原server1GPU0/350W首次启动四组全量后评，实测原生计数递增；原模型commit与当前加法marker独立绑定。无新训练、准入、资源迁移或重启。','Y102本轮原PID4747/server2GPU0/280W仍Train1step7，原现场保留；不重复此前拒绝的栈附加、管理员索权或重启。Y96成本包装器一次共享导入名误替换发生在计划/GPU/远端前，原失败保留；仅恢复共享运输导入后首次实际成本通过。'],decisions=['保持本轮零新候选/准入/新训练及Phase2限制；仅推进既有已训练归档且原设备匹配的冻结后评。Y101/Y99/Y96科学结果关闭不重测；Y94完整原配对/成本/后审计后再处置。','23项到期工作已按原证据核验：Y96完成、Y94实际后评、14项原GPU0等待真实占用、GPU1六项仍用户暂停、Y86原数值失败不重跑。Y57单独暂停和Y97失败继续保留；不跨项目、不用server6、不改goal/数据/收据/功耗/调度/上限或外部任务。'],evidence=[ref(out),ref(backup),review_ref,done['full_frozen_evaluation_result'],done['full_trained_cost_result'],done['post_measurement_audit'],active['frozen_review_execution_binding'],job['launch'],obs_ref,ref(diagnostic),failure]+list(occupations.values())))
        assert all(ref(p)==r for p,r in protected.items()) and git('ls-files','--stage','-z')==index
    finally:writer.close()
    command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)])
    command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists();pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status='FINALIZED',postflight='PASS',context_committed=True,lease_release='released',memory_lines=len(lines),Y96_FIRST=review['FIRST_miou'],Y96_disposition=review['disposition'],Y94_native_progress_verified=True,new_training_launches=0,promotion=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
