"""Record a real resource recheck without replaying completed diagnostics."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = 'eaa6e00eba9249fa86a1f63da10440cb'
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
import closure


def ref(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 180
view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
assert view['research']['shown'][0]['id'] == 'AdaRES-Y52' and view['research']['due_count'] == 1
resource_path = 'pipeline/analyses/y52_resume_status_eaa6_' + TOKEN + '.json'
resource = tick.load(ROOT / resource_path)
assert resource['complete'] and all(resource['resources']['all_authorized_slots_occupied'].values())
prior_path = 'pipeline/analyses/postflight_failure_4567_4567f5a09ecb4aa89df8f0c2932652df.json'
prior = tick.load(ROOT / prior_path)
checker = Path('C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/closure.py')
assert hashlib.sha256(checker.read_bytes()).hexdigest() == prior['checker']['sha256']
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
    assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
    for evidence in candidate['evidence']:
        assert ref(evidence['path']) == evidence
    bundle_path = 'pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json'
    bundle = tick.load(ROOT / bundle_path)
    for row in bundle['files']:
        assert ref(row['path']) == row
    # Full snapshot is used only by this program to evaluate the exact checker predicate.
    context_path = ROOT / 'pipeline/headless/.tick_context.json'
    assert hashlib.sha256(context_path.read_bytes()).hexdigest() == owner['context_sha256']
    context = tick.load(context_path)
    queue = context['queue']
    rejected_predicate = closure.exclusive_gpu_slot_available(candidate, queue['servers'], queue['active_tasks'])
    assert rejected_predicate is True, 'Known closure discrepancy changed: reassess rather than abort'
    failure_path = 'pipeline/analyses/reconciliation_eaa6_' + TOKEN + '.json'
    assert not (ROOT / failure_path).exists()
    now = datetime.now().astimezone().isoformat()
    tick.save(ROOT / failure_path, dict(complete=True, token=TOKEN, observed_at_local=now,
        scientific_status='Y50 training; SAME Y52 CPU-prepared and genuinely exclusive-GPU blocked',
        execution_status='INCOMPLETE: abort pending', current_resource=ref(resource_path), previous_failure=ref(prior_path),
        unchanged_checker_sha256=prior['checker']['sha256'], current_queue_opening_predicate=True,
        scope='Read-only evaluation of the exact known rejecting predicate, not a new finish invocation or native scientific test.',
        reason='The checker still sees an opening in own-project queue, while GPU0 is occupied by foreign PID19723 and GPU1 by Y50 PID31539. No authorized idle exclusive slot exists.',
        preserved_retry_after=candidate['retry_after'], prototype=ref(bundle_path),
        actions_not_taken=['No repeated CPU experiment, equivalent route review or invented reserve candidate.', 'No full GPU diagnostic, code change, admission, Git commit, dispatch, scheduler change or other-project operation.', 'No repeated known-failing finish or blocker recategorization to bypass checks.'],
        new_candidates=0, new_scientific_tests=0, model_changes=0, admissions=0, launches=0))
    candidate['evidence'].append(ref(resource_path))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    memory = f'''# AdaRES_project Research Memory
## Goal And Phase
- Currenttoken{TOKEN}; snapshot{view['generated_at']}; immutabledeadline{view['deadline_at']}; design/dueY52. ExecutionINCOMPLETE: knownclosurepredicateverifiedunchanged; canonicalabortpending, trainingNOTstopped.
- OnlyAdaRES/server2GPU0,1/oneexperiment-card/owner01a07e32-78bf-7670-925b-370911ea48db/schedulerDisabled. ScanRefertarget.515unmet; CSVY36.5110/B1.5020.
## Champion And Baseline
- HistoricalexactY36.510716624292/B1.502112417447 differsfromfrozenY48.5034034117/W2.5099093531/B1.5021230823. Do notmix orclaimnewpairedbenefit.
- CurrentcanonicalmonitorY50best.3657 fromsequence.3115,.3657; stdoutlatestmIoU.366/Acc50.406/Acc25.470/spmIoU.337. Early2validationresults, training3/30; notprojectbest/fullprotocolresult.
## Active Hypotheses
- Y50samehigher-orderjointmaskincidenceTRAINING; scopedhumancommitresolved/commit26ee1cec57c53ee2f94d6969d0bb637565b6ed59/realadmissionpreserved; no diagnosticweights.
- SAMEY52stableID: directindicatorquality-regressionpremisefalse/actualmatchedCE. RemainingnarrowhypothesisauxiliaryscoreMSEusesunweightedSP-IoUwhilepointmetric; scoreNOTdirectlyconsumedbyRESselection; sharedqueryefficacyunknown.
- PriorY52CPUpreparationCOMPLETEnotrepeated:35sourcefixtures, frozen8TRAINscenes/32expressions/all4096masks/47existinghalf-filterdifferences, NOTmatchedqueryfrequency. Independent7fileuncommittedprototype/newparams0/inferenceunchanged.
- Physical32countcontrols/5raggedexpressions/fullCriterionfinal+auxCPUcontrolscompleted. Fullnativebackbone/FPSmatching/backward/resources/restoreUNRUN; noadmissionPASS. Manifest{bundle_path}.
- Y52blockedexclusive_gpu_slot/server2. Originalretry{candidate['retry_after']} preserved, notpushedback. NexttestactualnativeTRAINmatchedloss/backwardplusresource/restoreonceauthorizedGPUtrulyidle+exclusivelease. NoautomaticY52commitauthorization.
## Closed Or Failed Directions
- Y51gaugemismatchrejected/selectoractualsoftmax; Y49currentrev1resourceimplementationrejected/nativeoriginal+extendedwitness0; nounsupportedrevision2. Y46two-revisionresourceimplementationrejected;Y48OR-targetsmallharmrejected/fixed30terminal.
- Y47/Y29/Y40/Y19/Y32closed; no renamedreopen. Y52doesnotchangeindicatorlabels/rankingloss. Preservev1strictgradientfailure; v2maskgrad2.273736754e-13equalsidenticalrepeat/scoregradexact; nobitwisefailureerasure.
## Running And Queue
- Begin16:17:47+08 active1/running1/claim0/wait0/target2/admissionbudget1/reserve1of2/preparationbudget1; noextraexperimentthisturn.
- Client16:20:31.296955to16:20:31.793127+08 identityverifiedserver2/GPU0foreignPID19723/100percent/23326MiB;GPU1ownY50PID31539/85percent/22694MiB. Whitelist2, ownedtraining1, genuinely_idle_GPU_count=0.
- Y50sametrain_adares_y50_joint_incidence_res_20261016_145628.log/Train3/30step402to403/4686/no boundedtailcrash. ReleasedSPFormer/seed1999/fixed30/128query/singleargmax/weightsunchanged.
## Open Questions And Risks
- Y52cachedallmaskdiagnosis/componentgradientsdonotprovenativematchedfrequency/helpfulsharedquerygradient/trainedbenefit; novelty/accuracy-resourcePareto unknown. Preservebothoccupiedjobs; noserver6/otherproject/background/newtasks/nestedcodexexec.
- Y50 frozenfull9508/141 Y50/W2/B1twiceEACH/FIRSTprimary;scene-bootstrap10000/1999/TRAINcuts1487/3325/subgroupCIupper<0harmreview; novalidationselection/threshold/weight/epochscans.
- KnowncheckerSHA{prior['checker']['sha256']} unchanged: ownqueueopeningpredicateTRUEwhileliveauthorizedGPUsoccupied. Do notchangeblockerkind/goal/context/checker orfabricatequeueentry; reportmismatchandabort.
## Last Tick
- Prior4567finishfailedBlockedGPUcandidatehasaqueueopening; canonicalabort16:13:52savedINCOMPLETE/contextuncommitted;16:14:30ownedleaseabsent/Y50alive. Oldmemoryabortpendingwaspre-releasewording, notanunreleasedlock. Neverresumeoldtoken.
- Currentonecanonicalbegin --auto-steady, compactfirst/stableY52only, freshreadonlyresourceproof+boundhashaudit. No repeatedCPUscientifictest/newvariant/newcommit/admission/launch ornewno-opdecisionindex.
- Exactknownrejectingpredicatewasre-evaluatedread-onlyonprogram-onlyfullsnapshot, no fullfinishretry. Evidence{failure_path}. Currentabortnext; verifyreceiptandabsentlease, thennopostreleasewrites.
- Receiptusagebegin-to-receiptlocalwindowonly; cachedinputandreasoningoutputalreadyincluded, notbill; missingunknown. Nextactualautomationtimeunknown.
'''
    assert len(memory.splitlines()) < 120
    (ROOT / 'pipeline/codex_memory.md').write_text(memory, encoding='utf-8')
    notes_path = 'pipeline/analyses/notes_eaa6_' + TOKEN + '.json'
    tick.save(ROOT / notes_path, dict(actions=[
        '执行状态INCOMPLETE，准备按契约abort；科学状态Y50 training、同一Y52 blocked/exclusive_gpu_slot，未作新的科学PASS。项目/当前任务身份与调度Disabled已核实。',
        '先读新紧凑快照，再按稳定ID续做Y52。原型七文件和既有21条证据哈希复核未变；CPU准备已完成，不重复相同控制，不为储备预算造候选。Y52原retry17:10:31+08保持不变。',
        '客户端2026-10-16 16:20:31.30至16:20:31.79+08只读核验：GPU0外来PID19723/100%/23326MiB；GPU1本项目Y50 PID31539/85%/22694MiB；没有真实空闲独占卡，不运行GPU诊断、不触碰外来作业。',
        'Y50同一进程Train[3/30]402→403/4686，无本次有限尾部崩溃标记。begin监控序列mIoU .3115→.3657；stdout最新约.366、Acc@.25约.470/Acc@.5约.406、spmIoU约.337；是早期验证，不是项目最佳或已证明提升。延迟与峰值显存未知。',
        '上一轮abort已释放，本轮不是恢复旧租约。已知检查器SHA未变；程序只读执行本轮同一queue-opening拒绝谓词仍为真，虽然实时两卡分别被外来与本项目占用。本轮不重复必失败finish，也不改阻塞类型、目标、调度或检查器；保留证据并abort。',
        '实际产出：新候选0、新科学测试0、模型改动0、准入0、启动0；新增占用/训练进度证据1，Y52科学阶段不变。收据用量及窗口待abort后只读核实，当前未知；缓存和推理不重复相加，不作账单解释。'],
        decisions=['无新科学路线或准入决策，保持同一Y52冻结实现与原恢复条件；只有真实空闲授权卡且成功取得独占租约，才续做原生匹配/反向与资源验收。',
            '不反复推送已知阻塞故障或常规训练进度提醒；保存完整六段事实报告。Y50训练继续，等待实际结果或资源变化；下一真实调度时间未知。']))
    print(json.dumps(dict(token=TOKEN, notes=str(ROOT / notes_path), evidence=ref(failure_path),
        known_rejecting_predicate=rejected_predicate, candidate_stage_unchanged=candidate['stage'],
        retry_after_unchanged=candidate['retry_after'], memory_lines=len(memory.splitlines()), next_action='report then canonical abort; no finish repeated'), ensure_ascii=False, indent=2))
finally:
    lock.close()
