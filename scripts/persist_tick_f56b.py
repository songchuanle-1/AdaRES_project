"""Persist actual final dispositions and report notes under the owned lease."""
import argparse,hashlib,json,os,statistics,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
parser=argparse.ArgumentParser()
parser.add_argument('--token',required=True)
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
now=datetime.now(timezone.utc)
remaining=(datetime.fromisoformat(owner['deadline_at'])-now).total_seconds()
assert 30<remaining<=180,'Only an actual near-deadline disposition may defer executable work'
def path(stem):return 'pipeline/analyses/'+stem+'_'+args.token+'.json'
def ref(relative):return dict(path=relative,sha256=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest())
def result(stem):
    relative=path(stem)
    value=tick.load(ROOT/relative)
    assert value['complete']
    return value,ref(relative)
cpu46,rCPU=result('y46_revision2_compat_cpu_f56b')
native46,rN=result('y46_revision2_compat_native_f56b')
controls46,rC=result('y46_revision2_compat_controls_f56b')
stress46,rS=result('y46_revision2_stress_f56b')
cpu49,r49C=result('y49_production_cpu_f56b')
native49,r49N=result('y49_production_native_f56b')
inference49,r49I=result('y49_production_inference_f56b')
stress49,r49S=result('y49_production_stress_f56b')
resources,rR=result('tick_resources_f56b')
assert len(controls46['native_rows'])==8 and len(controls46['inference_rows'])==10
assert all(c['repeat_exact'] for row in controls46['native_rows'] for c in row['cached_controls'])
assert native46['model_optimizer_scheduler_restore_exact'] and native49['model_optimizer_scheduler_restore_exact']
def inference_summary(rows):
    return {name:dict(median_seconds=statistics.median(v['seconds'] for row in rows[:8] for v in row[name]),largest_pair_median_seconds=statistics.median(v['seconds'] for row in rows[-2:] for v in row[name])) for name in ('parent','candidate')}
def stress_summary(rows):
    return {name:dict(median_seconds=statistics.median(v['seconds'] for v in rows[-1][name]),peak_reserved_gib=max(v['peak_reserved_gib'] for v in rows[-1][name])) for name in ('parent','candidate')}
measured46=dict(inference=inference_summary(controls46['inference_rows']),stress=stress_summary(stress46['rows']))
measured49=dict(inference=inference_summary(inference49['inference_rows']),stress=stress_summary(stress49['rows']))
review=dict(complete=True,created_at_local=datetime.now().astimezone().isoformat(),token=args.token,seconds_remaining=remaining,Y46=dict(measured=measured46,CPU_controls=True,native_cached_controls=32,native_exact_repeat=True,actual_tools_train_steps=2,exact_model_optimizer_scheduler_restore=True,scope='Current realization only; global-component mechanism not universally falsified.',resource_decision='Reject this two-revision realization for the measured severe time cost against the project lightweight Pareto objective. No new numerical cutoff or efficacy threshold fitted; memory feasibility alone is not Pareto approval.',evidence=[rCPU,rN,rC,rS]),Y49=dict(measured=measured49,production_CPU_controls=True,actual_tools_train_steps=2,exact_model_optimizer_scheduler_restore=True,diagnosis='INCONCLUSIVE: zero observed equal-component-partition/different-global-value witnesses in the original TRAIN8/32 expressions; constructed degree-confounded and abstract controls do not establish native task deficiency.',next_test='Prediction-blind extension of the SAME W2 native capture to the first24 previously unseen TRAIN scene IDs beyond the original first8, first4 expressions each, seed1999; preserve original zero count and frozen8eps*nodes definition. Check actual source component-mass and global-descriptor witnesses without validation selection. Preregister exact cohort/program before capture.',evidence=[r49C,r49N,r49I,r49S]),resources=resources,resource_evidence=rR,no_admissions=True,new_training=0,new_candidate_IDs=0,deadline_unchanged=True,limitations=['All native updates/stress were temporary diagnostics, not full training.','No trained efficacy for Y46/Y49; no novelty, necessity, no-harm or Phase2 approval.','Y48 realized subgroup harm rejected; no validation retuning or restarts.'])
review_path=path('final_review_f56b')
assert not (ROOT/review_path).exists()
work_path=ROOT/'pipeline/research_work.json'
work=tick.load(work_path)
c46=next(c for c in work['candidates'] if c['id']=='AdaRES-Y46')
c49=next(c for c in work['candidates'] if c['id']=='AdaRES-Y49')
assert len(c46['revision_history'])==2
refs46=[rCPU,rN,rC,rS,ref('pipeline/analyses/y46_revision2_api_compat_prereg_f56b.md'),ref('pipeline/analyses/y46_revision2_cpu_f56b_'+args.token+'.log'),ref('pipeline/analyses/y46_revision1_dec_f56b_'+args.token+'.py'),ref('pipeline/analyses/y46_revision2_api_failed_dec_f56b_'+args.token+'.py'),ref(path('y46_revision2_test_bindings_f56b_compat').replace('_'+args.token+'.json',args.token+'.json'))]
c46['revision_history'][1].update(result='passed',evidence=[rCPU,rN,rC,rS])
c46['evidence'].extend(e for e in refs46 if e not in c46['evidence'])
c46['evidence'].append(ref('pipeline/analyses/Y46_revision2_gate_review_f56b.md'))
c46.update(stage='rejected',failure_scope='implementation',scientific_status='realization_rejected_measured_excess_runtime_after_two_revisions',rejection_reason=review['Y46']['resource_decision'],next_action='Do not admit this costly realization or make a third local revision. Preserve measured resource failures, repaired exact-repeat evidence and all earlier negatives; broader mechanism is not falsified, historical rejected ID requires explicit reopen authorization.')
c46['diagnostic_gates'].update(controls='Revision2 CPU32 and actual native32 cached exact repeats/wrong correspondence passed.',native_backward='Revision2 actual released-init production tools.train2steps; all8 updates and full exact model/AdamW/PolyLR restore.',resources=review['Y46']['resource_decision'])
c46['pending_gates']=[]
refs49=[r49C,r49N,r49I,r49S,ref(path('y49_production_bundle_f56b_nativeparent').replace('_'+args.token+'.json','_'+args.token+'.json')),ref('pipeline/analyses/y49_native_parent_binding_f56b.md')]
c49['evidence'].extend(e for e in refs49 if e not in c49['evidence'])
c49['evidence'].extend([ref('pipeline/analyses/Y49_production_gate_review_f56b.md'),ref(path('y49_train_extension_freeze_f56b'))])
c49.update(stage='validation',failure_scope='inconclusive',scientific_status='production_execution_verified_native_task_information_diagnosis_inconclusive',next_action=review['Y49']['next_test'])
c49['diagnostic_gates'].update(diagnosis=review['Y49']['diagnosis'],controls='Actual independent physical-class CPU32 hooks and prior same-helper native TRAIN8 cached controls passed.',native_backward='Production released-init tools.train2steps/all4 parameter updates and exact model/AdamW/PolyLR restore passed.',independence='Seven additive independent files; physical parent AST/constructor/RNG/source-byte checks passed.',resources='Fixed serial TRAIN8/largest-pair inference and native stress measured; feasibility does not establish trained Pareto or efficacy.')
c49['pending_gates']=['native_task_diagnosis','resource_and_final_six_gate_review']
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    tick.save(ROOT/review_path,review)
    review_ref=ref(review_path)
    for c in (c46,c49):c['evidence'].append(review_ref)
    work['design_cycle']=dict(stop_reason='tick_deadline_near',attempted_ids=['AdaRES-Y48','AdaRES-Y46','AdaRES-Y49'],evidence=[review_ref,rR],resume_condition='Resume SAME Y49 pending native TRAIN diagnosis with a preregistered larger TRAIN-only capture; retain all negative/resource evidence and all admission gates. GPU idleness still requires live checks/exclusive lease; no automatic launch.')
    tick.save(work_path,work)
    decision=monitor.log_decision(str(ROOT),'other','Y48终态和Y46/Y49修订后的实际准入审查','拒绝Y48当前监督实现和Y46高成本实现；Y49继续原生信息缺口诊断，不准入、不启动正式训练。','Y48冻结小目标伤害；Y46修订2重复/更新/恢复通过但实测耗时严重；Y49生产执行通过不能替代尚缺的原生区分见证。',review_path)
    memory=['# AdaRES_project Research Memory','## Goal And Phase',f'- Current token {args.token}; owner {owner["thread_id"]}; snapshot {owner["generated_at"]}; immutable deadline {owner["deadline_at"]}; design. Finish still pending, never infer completion from this memory.','- ScanRefer target .515 unmet; historical rounded CSV champion Y36 .5110, not promotion. W2 retained scientific comparator; no Phase2 approval.','- Only AdaRES/server2 GPUs0,1, one experiment/card; closure enabled; scheduler remains Disabled. No server6/other project/new task/nested Codex/scheduler change.','- Released SPFormer/seed1999/fixed30/native128/singleargmax unchanged; no validation tuning.','## Champion And Baseline','- Current frozen FIRST Y48 .5034034117 / W2 .5099093531 / B1 .5021230823; each9508expressions/141scenes/twice EACH, FIRST primary.','- Y48-B1 delta+.0012803294 CI[-.0075489146,.0102441221]; Y48-W2 delta-.0065059414 CI[-.0150799728,.0021267774]; overall inconclusive.','- Historical Y36 frozen .510716624292 versus SAME historical B1 .502112417447; not a fresh current comparison or superiority proof.','## Active Hypotheses','- SAME Y49 global-mask-resolvent-coherence revision1: forest Schur elimination preserves SAME unit resolvent; cyclic graphs keep original sparse-LU. No graph/mask/mass/head/target changes.','- CPU32 descriptor error2.22e-16/partitions exact; original/revised19.979/6.692s; prior native witness count0 retained.','- Revised native TRAIN8 all4 gradients finite/nonzero; minimum.0001537401. Prototype costs improved but are not trained efficacy.',f'- Physical Y49 production CPU/import/parent AST/initialization/RNG and released-init tools.train2steps/full exact model+AdamW+PolyLR restore passed. Costs {json.dumps(measured49)}.','- Y49 remains validation, NOT admitted/training: task-relevant same-component/global-information diagnosis INCONCLUSIVE. Next: preregister first24 unseen TRAIN scenes beyond old8, first4 expressions each, no validation selection.','## Closed Or Failed Directions','- Y48 original fixed30/best14 verified and canonically finalized; no restart. Frozen small group2816expressions/124scenes vsW2 delta-.0233349321 CI[-.0392752156,-.0079237523] triggers original harm rule. Current OR-supervision realization REJECTED/implementation scope.','- Y48 trained TRAIN8 inference median.076851/.075527s; native stress13.400391GiB both; no Pareto dominance/equivalence/no-harm claim.','- Y46 revision1 CUDA atomic pooling failed original exact cached-repeat rule. Forensics all8 repeats each4 expressions unequal/max5.96e-8; inherited forensic repeat_exact=true fields are NOT checked PASS. Raw retained.','- Y46 revision2 prereg12:46:40+08 only stable grouped sum. Unsupported stable argsort API failure preserved; unique integer keys implement SAME stable order, not a third model revision.','- Y46 revision2 CPU32 max old/new error1.67e-16; native32 cached repeats exact, all8 native gradients/updates and exact full restore passed.',f'- Y46 final resource measurements {json.dumps(measured46)}. Current two-revision realization REJECTED for severe runtime cost under lightweight objective, not universal hypothesis falsification. No third revision or silent reopening.','- Y49 first local package/native-parent byte mismatch preserved in v1 tar/manifest; final package bound to actual native parent bytes, no shared entry edits. Builder pre-execution syntax/encoding failures did not alter scientific evidence.','- Y43 subgroup harm rejected; Y44/Y45/Y42/Y41/Y39 complete_inconclusive; Y47 existing pairwise path disconfirmed; closed ranking/readout families remain closed.','## Running And Queue','- Preflight12:22:37+08 active0/running0/claim0/wait0/target2, admissionbudget2, reserve3/preparation0. Snapshot GPU utilization unknown; no idle inference from whitelist.',f'- Post-tests authorized GPU occupation {json.dumps(resources["rows"])}. Foreign GPU0 job preserved; no new official training or admissions.','- Y48 original checkpoints remain remote, bestSHAc9c2ecbc7a010ecbff58c37f461c3bca4250d5bd1fd2f31beeeb2b1882687973; lastSHA3b095efc1b02a5863ebdb8cee1a801f1e76c41ab609d3201d736af0d0c8afea5.','## Open Questions And Risks','- Y49 original TRAIN8/32 captured expressions had zero native equal-component-mass/different-global-value witnesses; do not erase/upgrade to diagnosis PASS.','- All production optimizer/stress updates were diagnostic and memory-only or temporary checkpoints, never deployed training weights.','- Official unique/multiple metadata unknown; one trained seed/method; repeats are not seed uncertainty and W2/B1 per-expression instability retained.','- New source packages are uncommitted/unadmitted; no queue receipt or dispatch until exact tested commit and every substantive gate passes.','## Last Tick','- Old12be turn interrupted before postflight, original deadline elapsed; no helper writers remained; canonical abort returned intentional1/released, independent lock absence verified. Old tick INCOMPLETE, not completed.','- This fresh tick used one begin/preflight; six full frozen passes plus actual local revisions/native/resource tests. No new IDs, admissions or full training.','- Actual near-deadline disposition only, no fabricated resource block/cooldown or deadline change.',f'- Current review {review_path}; decision {Path(decision).relative_to(ROOT).as_posix()}. Finish PENDING; require explicit closure PASS/COMPLETE/FINALIZED and independent absent lease.','- Receipt usage is partial begin-to-record window; cache inside input, reasoning inside output; missing unknown, never a bill; no post-release project writes; next automation time unknown.']
    assert len(memory)<120
    (ROOT/'pipeline/codex_memory.md').write_text('\n'.join(memory)+'\n',encoding='utf-8')
    actions=[f'旧12be轮因中断以INCOMPLETE保存收据并规范abort，释放已独立核实；本轮新token={args.token}，仅一次begin/preflight，原截止时间未改。','Y48原训练固定30轮终态已复核；本轮六次全量冻结评测和训练后成对成本完成。FIRST mIoU0.503403，较B1+0.1280个百分点、较W2-0.6506个百分点，总体区间跨0；小目标相对W2下降2.3335个百分点且CI上界<0，当前实现拒绝，不晋升。',f'Y46修订2已实际重测CPU32、原生32缓存精确重复、released-init两步更新/完整恢复和压力测试；最大对峰值{measured46["stress"]["candidate"]["peak_reserved_gib"]:.3f}GiB，但训练步中位数{measured46["stress"]["candidate"]["median_seconds"]:.3f}s/父{measured46["stress"]["parent"]["median_seconds"]:.3f}s，当前高成本实现不准入并拒绝；原子重复失败/API失败和旧代码全部保留。',f'Y49修订1保持描述符等价，CPU计算19.98s降至6.69s，实际原生/独立生产/两步更新/完整恢复及资源测量已做；最大对峰值{measured49["stress"]["candidate"]["peak_reserved_gib"]:.3f}GiB。原生等组件质量分区区分见证仍0，科学状态validation/inconclusive，不把可运行当准入。',f'后续占用仅只读核验：{json.dumps(resources["rows"],ensure_ascii=False)}；GPU0外部作业保护，GPU1无正式训练。新增ID0、准入0、正式训练0；下轮调度时间未知。','报告与七段记忆已保存；此时finish/postflight仍待执行，最终执行状态以随后实际命令和收据及锁释放核验为准。收据用量窗口只覆盖begin至记录，缺失未知，缓存属于输入、推理属于输出，不重复相加，不是账单。']
    notes='pipeline/analyses/tick_notes_f56b_'+args.token+'.json'
    tick.save(ROOT/notes,dict(actions=actions,decisions=['拒绝Y48已训练当前监督实现，依据原冻结小目标伤害规则，不外推整个一致性机制被证伪。','拒绝Y46两次修订后的高成本实现；不以小误差为由放宽精确重复门槛，不添加第三次修订。','保留Y49同ID续做原生TRAIN信息缺口诊断；不降低准入、不凑变体、不重启或改变调度。','仅因真实截止时间已近而保留明确待续测试，不造假外部阻塞、不设置无证据冷却。']))
    print(json.dumps(dict(review=review_path,notes=notes,decision=decision,remaining_seconds=remaining,measured46=measured46,measured49=measured49),ensure_ascii=False))
finally:lock.close()
