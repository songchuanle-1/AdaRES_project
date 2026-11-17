"""Persist real CPU progress and an evidenced exclusive-GPU continuation."""
import ast
from datetime import datetime, timedelta
import json
from research_tick_4567 import ROOT, TOKEN, lease, ref, tick, monitor

owner = lease()
now = datetime.now().astimezone()
paths = {
    'diagnosis': 'pipeline/analyses/y52_mass_target_4567_' + TOKEN + '.json',
    'train': 'pipeline/analyses/y52_train_cpu_4567_' + TOKEN + '.json',
    'bundle': 'pipeline/analyses/y52_prototype_bundle_4567_' + TOKEN + '.json',
    'physical': 'pipeline/analyses/y52_physical_cpu_v2_4567_' + TOKEN + '.json',
    'resource': 'pipeline/analyses/y52_final_status_v2_4567_' + TOKEN + '.json',
}
facts = {name: tick.load(ROOT / path) for name, path in paths.items()}
assert all(row['complete'] for row in facts.values())
assert all(facts['resource']['resources']['all_authorized_slots_occupied'].values())
assert facts['physical']['native_full_model_backward'].startswith('not executed')
for row in facts['bundle']['files']:
    assert ref(row['path']) == row, 'Prototype bytes changed after tests'
for name, expected in facts['physical']['source_files'].items():
    path = ROOT / 'AdaRES/ipdn/model_paper5/adares_y52_point_target' / name
    assert ref(path.relative_to(ROOT).as_posix())['sha256'] == expected
package = ROOT / 'AdaRES/ipdn/model_paper5/adares_y52_point_target'
for path in package.glob('*.py'):
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module and any(s in node.module for s in ('adares_w2', 'adares_y50', 'adares_b1')))
        if isinstance(node, ast.Import):
            assert not any(any(s in alias.name for s in ('adares_w2', 'adares_y50', 'adares_b1')) for alias in node.names)
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    lease()
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
    assert candidate['stage'] == 'implementation'
    for evidence in candidate['evidence']:
        assert ref(evidence['path']) == evidence
    supporting = [
        'scripts/build_y52_point_target_4567.py', 'scripts/probe_y52_physical_cpu_4567.py',
        'scripts/prepare_y52_physical_cpu_4567.py', 'scripts/probe_y52_final_status_4567.py',
        'pipeline/analyses/y52_physical_cpu_program_4567_' + TOKEN + '.py',
        'pipeline/analyses/y52_physical_cpu_4567_' + TOKEN + '.log',
        'pipeline/analyses/y52_physical_cpu_v2_program_4567_' + TOKEN + '.py',
        'pipeline/analyses/y52_physical_cpu_v2_prereg_4567_' + TOKEN + '.json',
        'pipeline/analyses/y52_final_status_4567_' + TOKEN + '.log',
    ] + list(paths.values())
    seen = {e['path'] for e in candidate['evidence']}
    candidate['evidence'].extend(ref(path) for path in supporting if path not in seen)
    goal = tick.load(ROOT / 'pipeline/goal.json')
    review_path = 'pipeline/reviews/motivation_review_Y52_4567.md'
    design_path = 'pipeline/designs/exp_design_Y52_4567.md'
    assert not (ROOT / review_path).exists() and not (ROOT / design_path).exists()
    review = f'''# Motivation Review - AdaRES-Y52
实际整理时间：{now.isoformat()}。前置预注册与测试各有独立时间；本整理不伪装成事前文件。
## 1. Goal
```json
{json.dumps(goal, ensure_ascii=False, indent=2)}
```
历史CSV Y36 .5110/B1 .5020；目标.515未达标。Y50首次原生验证约.312，仅是早期值。
## 2. Story Status
| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| 最终indicator直接回归质量 | 不成立 | 实际loss_indi为匹配CE；score扰动不改变固定推理 | 不能宣称直接修复选择器 |
| 辅助质量目标与点级IoU口径不同 | CPU支持 | 8TRAIN场景/32表达/每表达全部128mask；47/4096筛选差异 | 原生匹配后频率未知 |
| 修正口径改善共享query并提升选择 | 未验证 | 独立CPU原型及梯度对照 | 全模型原生测试、资源与训练结果 |
## 3. Experiment Assessment
| Experiment | mIoU | 与动机的关系 | 对故事线贡献 |
|---|---|---|---|
| W2 | 冻结FIRST .5099093531 | 科学参照 | 不被未训练原型取代 |
| Y36 | 历史CSV .5110 | 目标检查数值champion | 不等于显著优于W2 |
| Y50 | 首次约.312 | 高阶共同覆盖训练中 | 非最终效果或项目最佳 |
| Y52 | 未训练 | 已有辅助目标的测度一致性 | 新候选仅CPU组件准备 |
## 4. Gap Analysis
等权超点IoU、点质量加权硬标签IoU、真实点目标IoU已分别计算。平均绝对质量目标差.00865423，最大.48834160；这些不是mIoU收益。超点内部混合贡献均值.000017484，不能与占用加权混为一谈。
## 5. Design Rationale
保持最终CE、选择器、mask、权重和固定协议；只将既有辅助MSE的标签测度改为精确点交并计数。不是Y19/Y32全query软indicator目标、Y29排序损失或Y48所有权目标重开；不宣称文献新颖性。
## 6. Next Batch
同一Y52继续：原型冻结清单见{paths['bundle']}。全模型匹配/梯度和资源未运行，不准入，不自动提交。
'''
    design = f'''# Experiment Design - AdaRES-Y52
整理时间：{now.isoformat()}；状态：CPU_PREPARED / NOT_ADMITTED。
## 1. Motivation
{goal['description']}
本轮容量2、运行1、准入预算1；遵循先登记候选再测试，非为填卡而准入。
## 2. Current Best Results
| ID | State | Best mIoU | Acc@50 | spmIOU | Epochs |
|---|---|---|---|---|---|
| Y36 | 历史完成 | CSV .5110 | 未知 | 未知 | 固定30 |
| Y50 | 原始进程训练中 | 早期约.312 | 约.335 | 约.284 | 当前2/30 |
| Y52 | CPU准备 | 未知 | 未知 | 未知 | 未训练 |
## 3. Analysis & Direction
已确认的是辅助目标差异，不是训练收益。直接置信度质量回归的前提被否定；较窄的共享query监督假设仍需测试。
## 4. Experiment Plan
| ID | Config Path | Purpose | Next Test | Depends On |
|---|---|---|---|---|
| AdaRES-Y52 | configs_song/model_paper5/adares_y52_point_target_res.yaml | 仅修正辅助point-IoU目标 | 原生TRAIN匹配/反向、资源/恢复 | 真实空闲独占授权卡；六门禁及后续提交许可 |
最终及所有既有辅助层均用detached精确点计数；原生>.5筛选不变。单目标RES，显式拒绝不支持的多目标训练。推理、采样、decoder和参数命名保持不变。
## 5. GPU Allocation Plan
| Server | GPU | Assignment | Justification |
|---|---|---|---|
| server-2 | 0 | 外来PID19723，保留 | 非AdaRES占用 |
| server-2 | 1 | Y50 PID31539，保留 | 不移动、不重启 |
| server-2 | 首个真实空闲授权卡 | 仅下一项诊断，尚无预留 | 重新实时检查并取得独占租约 |
## 6. Implementation Checklist
- [x] 独立六文件model包及唯一basename YAML；无跨变体import，原型七文件hash冻结。
- [x] 精确计数32随机控制、空目标/空mask控制、跨2scene/5expression映射；独立模块导入。
- [x] 最终+辅助Criterion组件损失和score梯度与展开点参考精确一致；其余loss不变。
- [ ] 实际稀疏backbone/FPS及原生匹配后的全模型梯度、资源、真实两步/完整恢复。
- [ ] 提交许可、真实commit绑定、完整六门禁receipt、规范dispatch；本轮均未执行。
## 7. Risk Assessment
新增参数0；推理代码未变不等于已测训练资源PASS。CPU v1逐位梯度断言失败保留；v2最大2.273736754e-13，与原样重复误差相同，score梯度精确；非降低科学阈值。最终选择不直接消费score，辅助任务影响可能无效或有害，须训练后判定。
## 8. Smell Check
这是既有监督标签从等超点测度到真实点测度的因果纠正，不是阈值/温度/权重/epoch扫描，也不包装为新架构。文献新颖性未知；原生匹配影响缺证，不作为已准入贡献。
## 9. Prospective Evaluation
若以后六门禁通过，released SPFormer/seed1999/固定30/singleGPU/128queries/singleargmax不变。主指标ScanRefer mIoU，W2/B1冻结比较，全9508表达/141scene各两次且FIRST为主；scene-bootstrap10000/1999；TRAIN尺寸切点1487/3325，子组相对任一参照delta区间上界<0触发伤害审查。报告Acc@.25/.5、查询数、延迟、峰值显存；无验证选择。当前此计划不是准入凭证。
'''
    (ROOT / review_path).write_text(review, encoding='utf-8')
    (ROOT / design_path).write_text(design, encoding='utf-8')
    candidate['evidence'].extend([ref(review_path), ref(design_path)])
    candidate.update(stage='blocked', blocker_kind='exclusive_gpu_slot', required_server='server-2',
        blocker='CPU/source/prototype/component preparation finished; actual sparse-backbone/FPS native matched TRAIN forward/backward and resource gates require an exclusive GPU. GPU0 has foreign PID19723; GPU1 has existing Y50 PID31539.',
        retry_after=(now + timedelta(hours=1)).isoformat(),
        unblock_condition='An authorized server-2 GPU0 or GPU1 is genuinely idle with no compute PID, memory<1024MiB/util<10%, and its exclusive diagnostic lease can be acquired; preserve both existing jobs and recheck live state.',
        next_action='Run SAME frozen Y52 independent package through actual native matched TRAIN loss/backward; quantify matched-target changes, shared-query gradients, exact disabled/reference controls and native resources/restore before any admission. No automatic Git commit permission for Y52.',
        attempted_alternatives=['Native source/selector audit and35 CPU fixtures completed.', 'Prediction-blind replay of8 frozen TRAIN scenes/32 expressions/all4096 masks completed without GPU.', 'Independent seven-file implementation,32 exact-count controls, ragged5-expression mapping and full Criterion final/auxiliary CPU component controls completed.', 'Fresh authorizedGPU0/1 occupation and live Y50 command/progress verified; no idle exclusive slot.'],
        failure_scope='inconclusive',
        gate_summary=dict(diagnosis='supported_on_frozen_masks; native_matched_frequency_unknown', controls='CPU_component_only; v1_failure_retained',
                          native_backward='pending_exclusive_GPU_not_PASS', resources='pending_native_full_model_not_PASS',
                          independence='physical_package_and_static_checks_passed', constraints='preparation_only_no_admission'),
        wake_paths=[paths['bundle']])
    work['design_cycle'] = dict(stop_reason='external_resource_blocked', attempted_ids=['AdaRES-Y52'],
        evidence=[ref(paths['resource']), ref(paths['train']), ref(paths['physical']), ref(design_path)],
        resume_condition=candidate['unblock_condition'], retry_after=candidate['retry_after'])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    record = 'pipeline/analyses/final_review_4567_' + TOKEN + '.json'
    tick.save(ROOT / record, dict(complete=True, token=TOKEN, observed_at_local=now.isoformat(),
        execution_status='POSTFLIGHT_PENDING', science='Y50 training; Y52 CPU-prepared/exclusive-GPU blocked',
        sources={k: ref(v) for k,v in paths.items()}, review=ref(review_path), design=ref(design_path),
        local_prototype_files=7, admitted=0, launched=0, no_other_project_or_scheduler_change=True,
        failures_retained=['Builder initial SyntaxError before any model writes; corrected.', 'Physical CPU v1 strict-gradient assertion; v2 quantified repeat-level2.27e-13, no model change.', 'Narrow ps command matching failed; -ww full command verification passed.'],
        previous_tick='75486f9263d34214b3bd2e1412c69639 was INCOMPLETE and already aborted/released; do not resume its token.'))
    memory = f'''# AdaRES_project Research Memory
## Goal And Phase
- Current token{TOKEN}; snapshot{owner['generated_at']}; deadline{owner['deadline_at']} unchanged; design. PostflightPENDING, not completed yet.
- ScanRefer target.515 unmet; historicalCSVY36.5110/B1.5020; onlyAdaRES/server2GPU0,1/oneexperiment-card/owner01a07e32-78bf-7670-925b-370911ea48db/schedulerDisabled.
## Champion And Baseline
- Historical exactY36.510716624292/B1.502112417447 is separate from frozenY48.5034034117/W2.5099093531/B1.5021230823. Do not mix.
- Y50 firstnative validation stdoutrounded mIoU.312/Acc50.335/Acc25.408/spmIoU.284; early, not projectbest; unrounded metricunknown.
## Active Hypotheses
- Y50 SAME higher-orderjoint maskincidence TRAINING; scopedhumancommit authorizationresolved, sevenfilescommit26ee1cec57c53ee2f94d6969d0bb637565b6ed59/admissionpreserved. No diagnosticweightsdeployed.
- NewY52 SAMEstableID: initialdirectindicator-regressionpremisefalse; actualindicator matchedCE. Narrow auxiliaryscore target usesunweightedSP-IoU whilemetricpoint-IoU; no directscore consumptioninRESselector. Indirectsharedqueryeffectunknown.
- Y52 prereg/source35CPUcontrols; frozen8TRAINscenes/32expressions/4096masks replayidentityPASS;47existing>.5filter differences/allquery, NOTnative matchedfrequency. Meanabsqualitytargetdifference.00865423/max.48834160, notaccuracygain.
- Y52 independent7filelocalUNCOMMITTEDprototype; onlyauxqualitypointcounts/targetplumbing; no newparameters/inferencechanged.32exactcountcontrols/5raggedexpressions/fullCriterionfinal+auxCPUcontrols; nativefullmodelbackward/resources/restoreUNRUN. Manifest{paths['bundle']}.
- Y52 BLOCKEDexclusive_gpu_slot/server2, retry{candidate['retry_after']}; CPUpreparationcomplete. NextactualnativeTRAINmatchedloss/backward andresourcecheck on genuinely idleauthorizedexclusiveGPU. No automaticY52commitauthorization.
## Closed Or Failed Directions
- Y51commongaugemismatchrejected/actualselectorsoftmax; Y49currentrev1resourceimplementationrejected/nativeoriginal+extendedwitness0; no unsupportedrevision2/reopen.
- Y46two-revisionresourceimplementationrejected;Y48OR-targetsmallharmrejected/fixed30terminal. Y47/Y29/Y40/Y19/Y32closed; Y52 does not alterindicator labels orrankingloss.
- PreserveY52v1strictgradfailure; v2samebytesmaskgradmax2.273736754e-13 equalsidenticalrepeat; scoregradexact/unchangednon-scorelosses. Builderparsefailureandpsnarrowcommandfailurealsoinfinalreview; no historicalevidenceerased.
## Running And Queue
- Preflight15:18:07+08 running1/claim0/wait0/active1/target2/budget1/reserve0of2. No newadmission/queueentry/launch this tick.
- Freshclient16:06:33.709708to16:06:34.227651+08 server2GPU0foreignPID19723/100percent/22322MiB;GPU1Y50PID31539/45percent/22694MiB. Whitelistcapacity2, ownedtraining1; neitheravailableforGPUdiagnostics.
- SAMEY50logtrain_adares_y50_joint_incidence_res_20261016_145628.log; Train2/30step3615to3616/4686, no boundedtailcrash; releasedSPFormer/seed1999/fixed30/128query/singleargmax/nativeweights unchanged.
## Open Questions And Risks
- FulltrainedY50 efficacy and fullY52 nativevalidation unknown. Y52 auxiliaryheadnotdirectlyselected; diagnosis does not ensurehelpful gradient or trainedbenefit. Noveltyunknown/noParetoproof.
- Y50frozen full9508/141 Y50/W2/B1twiceEACH/FIRSTprimary;scene-bootstrap10000/1999/TRAINcuts1487/3325/subgroupCIupper<0harmreview. No validationselection/thresholds/weights/epochs tuning.
- Preserveforeignjobs/server6ban/nootherproject/scheduler/background/newtasks/nestedcodexexec. LatestY50humancommitpermission doesnotextendtoY52.
## Last Tick
- Onecanonicalbegin --auto-steady; designbootstrapstableY52registeredbeforebroadreview. ActualCPUtests andprototypehashesevidencepreserved, realresourceblock not invented to passclosure.
- Previous7548tickfinishfailedExternal_resource_stop_lacks_blocked_candidate; canonicalabortalreadyreleasedlease andsavedINCOMPLETEreceipt. TrainingwasNOTaborted. Oldmemorypendingclaimsupersededbycommand/receipt.
- Current report/notes andseven-sectionmemoryprepared; callfinishonce, trustexplicitPASS/COMPLETE plusabsentownedlease. Ifunfixable, preserveandabort; nopostreleasewrites.
- Receiptusageonlybegin-to-receiptlocalwindow; cacheincludedininput/reasoninginoutput, notbill; unavailableunknown. Nextautomationtimeunknown.
'''
    assert len(memory.splitlines()) < 120
    (ROOT / 'pipeline/codex_memory.md').write_text(memory, encoding='utf-8')
    decision = monitor.log_decision(str(ROOT), 'other', 'Y52辅助质量目标差异和独立CPU原型如何继续？',
        '保持Y50原训练；Y52完成CPU准备后真实独占GPU阻塞，不准入、不自动提交。',
        '8TRAIN场景32表达47/4096筛选差异；完整Criterion最终及辅助score梯度精确；全模型匹配/资源尚未验收。GPU0外来、GPU1Y50均占用。', str(ROOT / record))
    notes_path = 'pipeline/analyses/notes_4567_' + TOKEN + '.json'
    tick.save(ROOT / notes_path, dict(actions=[
        '科学状态：Y50 training；新稳定候选Y52完成CPU准备，blocked/exclusive_gpu_slot。执行状态：POSTFLIGHT_PENDING，尚未称完成。',
        'Y52先登记diagnosis再做源审计与35CPU控制；实际indicator是匹配CE，不是质量回归；仅辅助score存在等超点与点级IoU口径差异。',
        '冻结8TRAIN场景/32表达/每表达128掩码重放，47/4096现有0.5筛选边界不同；匹配后频率未知，非性能提升。独立七文件未提交原型/新增参数0，32精确计数与5表达映射、完整Criterion最终及辅助层组件对照已执行。',
        '保留v1逐位梯度断言失败；v2最大2.27e-13与原样重复一致，辅助score梯度精确；其余损失不变。不能替代全模型原生反向、资源及恢复门禁。',
        '客户端16:06:33.71至16:06:34.23+08复核：GPU0外来PID19723/100%/22322MiB，GPU1本项目Y50 PID31539/45%/22694MiB。Y50 Train[2/30]3615→3616/4686；早期验证mIoU约.312、Acc@.25约.408/Acc@.5约.335、spmIoU约.284，精确未舍入值未知。',
        '本轮新增研究候选1、独立本地原型1、准入0、启动0；Y50提交授权已解决不再索要。收据用量窗口及计数须finish后读取；当前未知，不将缓存重复相加或将用量视为账单。'],
        decisions=['真实下一步是同一Y52的原生匹配/反向与资源测试；CPU准备完成后两张授权卡均被占用，保留明确retry/unblock，不造假填卡或重开旧路线。',
                   'GPU0外来占用不算本项目第二个实验；不动现有Y50、外来进程、目标、门禁、调度或其他项目。Y52提交许可未获，当前尚未达到需提交阶段。',
                   '上一轮7548已INCOMPLETE并abort释放；本轮按真实blocked候选验证收尾，finish成功才称完成，否则保留证据并abort核实释放。下一调度时间未知。']))
    print(json.dumps(dict(candidate='AdaRES-Y52', stage='blocked', retry_after=candidate['retry_after'],
        notes=str(ROOT / notes_path), evidence=ref(record), decision=decision, memory_lines=len(memory.splitlines())), ensure_ascii=False, indent=2))
finally:
    lock.close()
