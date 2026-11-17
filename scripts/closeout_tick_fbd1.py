"""Persist the completed CPU-only audit and unchanged external authorization hold."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
TOKEN='fbd16ee677ad4d3aa2bbc403acdc2a67'
parser=argparse.ArgumentParser()
parser.add_argument('--decision',required=True)
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
now=datetime.now().astimezone()
assert now<datetime.fromisoformat(owner['deadline_at'])
def ref(relative):
    path=(ROOT/relative).resolve()
    assert ROOT in path.parents and path.is_file()
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
source='pipeline/analyses/y51_source_fbd1_'+TOKEN+'.json'
result_path='pipeline/analyses/y51_gauge_controls_v2_fbd1_'+TOKEN+'.json'
result=tick.load(ROOT/result_path)
assert result['complete'] and result['id']=='AdaRES-Y51'
assert result['exact_native_predict_AST'] and result['literal_native_float32_loss_and_gradient_exact']
assert result['fixture_count']==32 and result['native_TRAIN_expressions']==32
assert result['native_selected_masks_unchanged']==65 and result['raw_foreground_changed_cases']>0
assert all(r['native_top_equal'] and r['selected_pointmask_exact'] for r in result['rows']+result['native_rows'])
previous_hold='pipeline/analyses/y50_commit_authorization_block_1cb1_1cb15474bfd346dcbc34837f3e9b79d3.json'
goal_text=(ROOT/'pipeline/goal.json').read_text(encoding='utf-8')
goal=json.loads(goal_text)
review_path='pipeline/reviews/motivation_review_Y51_fbd1.md'
disposition_path='pipeline/analyses/Y51_gauge_disposition_fbd1.md'
summary_path='pipeline/analyses/final_review_fbd1_'+TOKEN+'.json'
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y51')
    held=next(c for c in work['candidates'] if c['id']=='AdaRES-Y50')
    assert candidate['stage']=='diagnosis' and held['stage']=='blocked' and held['blocker_kind']=='human_authorization'
    held_before=json.dumps(held,sort_keys=True)
    assert datetime.fromisoformat(held['retry_after'])>now
    for relative in (review_path,disposition_path,summary_path):assert not (ROOT/relative).exists()
    review=f'''# Motivation Review - AdaRES-Y51
整理时间：{now.isoformat()}。这是已完成诊断的整理，不伪造事前审查时间；原始判据见本轮预注册和数值程序冻结。

## 1. Goal
```json
{goal_text.strip()}
```
目标.515未达标；台账Y36 .5110、B1 .5020为原有记录，不是本轮复测。

## 2. Story Status
| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| Y50高阶共同覆盖 | 既有提交授权阻塞，未到期 | 上轮独立包/原生/恢复/资源证据保留 | 不重复测试或重设重试时间 |
| Y51公共logit偏移造成监督与部署错配 | 本诊断前提已否定 | native RES已softmax前景概率；65组完整选择掩码不变 | 不由此创建数值校准变体 |

## 3. Experiment Assessment
| Experiment | mIoU | 与动机的关系 | 对故事线贡献 |
|---|---|---|---|
| 原有Y36/W2/B1 | 本轮未评测 | 保留既有科学参照 | 没有新指标提升证据 |
| Y51 | 未训练/不创建模型 | 审计训练置信度与部署取值的一致性 | 排除本实际RES路径的公共模态错配前提 |

## 4. Gap Analysis
实际父模型不是原始前景logit排序，而是两类softmax前景概率singleargmax。原训练CE与该选择路径没有被本判据发现的公共平移错配。
该结论不证明全部数值鲁棒性、全模型泛化或任务收益。32条原生TRAIN输入的float64重建不是完整GPU推理等价声明。

## 5. Design Rationale
固定32随机fixture、精确半整数阳性对照及8场景32条既有TRAIN输入；原始前景logit规则有29组排名变化，而真实native RES始终选择相同掩码，说明控制并非无效扰动。
float64仅显式适配原生固定float32 one-hot缓冲区的dtype叶子，标签/公式/权重不变；字面原生float32损失及梯度另在精确阳性fixture上逐位相等。未放宽判据。

## 6. Next Batch
本诊断不进入模型设计/准入，不扫描排名、温度、权重或阈值。Y50继续原有授权阻塞；没有新增人工请求或训练启动。
'''
    (ROOT/review_path).write_text(review,encoding='utf-8')
    disposition=f'''# Y51 Gauge Audit Disposition

审查时间：{now.isoformat()}；stage=rejected；failure_scope=hypothesis，仅指本公共模态错配前提。
精确native IPDN.predict_by_feat的RES分支已经F.softmax(pred_indis,dim=-1)[:,1]后singleargmax。
32fixture+1半整数阳性对照+32条原TRAIN输入，65/65完整选择掩码不变；原始前景logit排序29组变化。
最大原生选择器分值差{result['maximum_native_score_error']:.17g}；float64对照CE差{result['maximum_native_CE_error']:.17g}，梯度差{result['maximum_native_gradient_error']:.17g}，均在事前8eps*nodes*Q包络内。
字面原生float32损失/梯度阳性控制精确相等。float64损失one-hot缓冲dtype适配显式记录，不能称未改动的完整原生训练。
第一次探针因get_iou误从ipdn.utils导入而在任何数值测试前失败；原始脚本/日志保留，v2只提取实际loss.py的原生get_iou AST，判据/模型不变。
不重开已拒绝ranking/readout家族，不创建缺乏此诊断前提的校准模型，不把CPU审计当mIoU收益。无GPU使用、模型修改、正式准入或训练。
Y50授权阻塞未到期，不重复它的测试、不增加证据或改变retry；其原始提交授权条件仍为唯一已登记未完成事项。
'''
    (ROOT/disposition_path).write_text(disposition,encoding='utf-8')
    tick.save(ROOT/summary_path,dict(complete=True,reviewed_at_local=now.isoformat(),
        y51=dict(stage='rejected',failure_scope='hypothesis',fixture_count=32,native_TRAIN_expressions=32,
                 unchanged_selected_masks=65,raw_foreground_changed_cases=result['raw_foreground_changed_cases'],
                 maximum_score_error=result['maximum_native_score_error'],literal_native_float32_loss_gradient_exact=True),
        new_model_variants=0,GPU_tests=0,training_launches=0,new_admissions=0,
        tests=ref(result_path),disposition=ref(disposition_path),review=ref(review_path),decision=ref(args.decision),
        retained_external_block=ref(previous_hold),blocked_candidate='AdaRES-Y50',blocked_retry_after_unchanged=held['retry_after'],
        reason='Bounded new diagnosis tested and rejected. Remaining registered work retains original external human commit-authorization hold, not a GPU block/no-information-gain/exhaustion/deadline claim.'))
    candidate.update(stage='rejected',failure_scope='hypothesis',
        next_action='Do not create or admit a common-mode calibration/ranking realization from this disconfirmed premise. Actual native RES already uses two-class softmax foreground probability. This rejects this mismatch diagnosis only, not every confidence mechanism.')
    for entry in map(ref,(result_path,review_path,disposition_path,summary_path,args.decision,'scripts/closeout_tick_fbd1.py')):
        if entry not in candidate['evidence']:candidate['evidence'].append(entry)
    work['design_cycle']=dict(stop_reason='external_resource_blocked',attempted_ids=['AdaRES-Y51'],
        blocker_kind='human_authorization',evidence=[ref(summary_path),ref(previous_hold)],
        resume_condition=held['unblock_condition'],retry_after=held['retry_after'],
        detail='New preparation budget1 used for the completed independent Y51 audit, no model manufactured. Only registered unfinished candidate Y50 remains on its unchanged not-yet-due human commit-authorization block. No unchanged-block recheck/probe, no GPU block, no exhaustion/no-gain assertion or deadline padding.')
    assert json.dumps(held,sort_keys=True)==held_before
    tick.save(ROOT/'pipeline/research_work.json',work)
    memory=f'''# AdaRES_project Research Memory
## Goal And Phase
- Current token {TOKEN}; owner{owner['thread_id']}; snapshot2026-10-16T06:16:42.533745+00:00; immutable deadline{owner['deadline_at']}; design. Finish PENDING, not completed by memory alone.
- ScanRefer target.515 unmet; historicalCSV Y36 .5110/B1 .5020. No new trained metric/Phase2. ReleasedSPFormer/seed1999/fixed30/128/singleargmax/no validation tuning retained.
- OnlyAdaRES/server2GPU0,1/oneexperiment-card; schedulerDisabled verifiedthisbegin; no other project/newtask/nestedCodex/background/schedulerchange.
## Champion And Baseline
- Prior frozenFIRST Y48 .5034034117/W2 .5099093531/B1 .5021230823; Y48-small harm rule already triggered. No new validation run.
- Historical exactY36 .510716624292 versus historicalB1 .502112417447 is a separate frozen comparison, not fresh significance evidence.
## Active Hypotheses
- Y50 SAME higher-order joint candidate-mask incidence, unchanged BLOCKED/human_authorization; retry{held['retry_after']}. No new human authorization, evidence rewrite, repeated test or retry reset this tick.
- PriorY50 full128matched area/Gram32seeds parentindicatorerror0/newdescriptor gap7.7500155e-6; native128expressions nonconstant128; whole upstream blindness/novelty/utility unknown.
- PriorY50 independent bytes/AST/init/RNG/final-only CPU and nativeTRAIN8/32 all4gradients/cachedwrongcontrols passed; released-inittools.train2steps/updates/fullmodel+AdamW+PolyLR restore passed.
- PriorY50 inference medianparent/candidate .090737872/.090891707s; stress .697397007/.703661315s/reserved14.080078/14.351563GiB. Measured feasibility, not Pareto superiority. Codeuntracked, inheritedd18c228fd3796b5a87dbe313075b6ee748b2d51b cannot bind source; canonicalcontract prohibitsautomaticgitadd/commit.
## Closed Or Failed Directions
- NewY51 confidence-gauge mismatch diagnosis REJECTED/hypothesis scope: actual nativeRES uses two-classsoftmax foregroundprobability already, not rawforegroundlogit.65matchedfullnative selections unchanged; rawforeground positivecontrol29changes.
- Y51 maximum scoreerror2.220446e-16/float64CEerror1.776357e-15/gradienterror5.551115e-17 within original8eps*nodes*Q. Literal nativefloat32 CE/gradient exact on fixedhalf-integercontrol; float64onlyonehot-bufferdtype adaptation explicitlyfrozen, not fullnativeGPUtrain.
- Y49 currentrev1 high-runtime realization rejected/implementation; TRAIN8/32original0 plusTRAIN24/96new0 witnesses; all6314newsubgraphsforests. No unsupportedcyclicrevision2 or universal mechanism falsification.
- Y46 two-revision realization remainsresource-rejected; Y48 currentOR-target realization remainssubgroupharm-rejected/fixed30terminal. Y47pairwisemissing/Y29ranking/Y40word-support remainclosed, no silentreopen.
## Running And Queue
- Begin14:16:42+08 active0/running0/claim0/wait0/target2/admissionbudget2/reserve1/preparation1; due0/blockednotdue1.
- CurrentGPUutilization/memory/PIDs UNKNOWN; thisCPU-onlytick did not query them or allocate GPU. Prior14:05:14+08 cachedGPU0foreignPID19723/100%/24102MiB andGPU1noPID/0%/1MiB are not fresh availability evidence.
- No new model variant, queue/admission/marker/source change or formaltraining. One new diagnosis stableIDY51 only.
## Open Questions And Risks
- AwaitY50explicit scoped-tested-byte commit authorization or supplied matching commit, then recheckgoal/source/evidence before receipt/canonicaldispatch/nativeprogress. No blanketstage, oldHEADbinding or fabricatedGPUblock.
- FrozenfutureY50/W2/B1full9508/141twiceEACH/FIRSTprimary; scene-bootstrap10000/1999/TRAINsizecuts1487,3325/subgroupCIupper<0harmreview. No validation selection.
- Y51 audit neither full-loader/GPUfloat32 inference bitwise equivalence nor global numerical robustness/efficacy proof. ZeroGTmask and fixedpositiveindices were fixture bookkeeping only.
- No route exhaustion/no-information-gain assertion. Boundedpreparation completed; unchangedexternalauthorizationhold is retained, no new human request or timer reset.
## Last Tick
- Onecanonicalbegin/preflight; actualdesignbootstrap required one diagnosis before broadreview. RegisteredY51 first, exactsource capture thenpreregistered32fixtures+positive+oldTRAIN8/32controls, allCPUonly.
- Originalget_iouimportfailure before numeric execution preserved; v2exactnativeget_iouAST extraction andoriginalprecisionrule, no model/tolerancechange.
- Prior1cb receiptFINALIZED/PASS and releasedverified14:12:47; its oldFinishPENDINGmemory is not an abandonedlease. Old12be separatelyaborted, neverresumed.
- Evidence{summary_path}; decision{args.decision}; sixsectionreport notesprepared; finishstillPENDING. Afterrelease no projectwrites.
- Receiptusage is partialbegin-to-receipt localwindow, excludesearlierreads/lateusage/finalreply; cachedinput/reasoningoutputalreadyincluded, not a bill. Nextautomationtimeunknown.
'''
    assert len(memory.splitlines())<120
    (ROOT/'pipeline/codex_memory.md').write_text(memory,encoding='utf-8')
    notes='pipeline/analyses/tick_notes_fbd1_'+TOKEN+'.json'
    tick.save(ROOT/notes,dict(actions=[
        '本轮仅AdaRES_project，14:16:42+08快照为design；due0、未到期阻塞1、储备准备预算1。先登记稳定Y51诊断，再做定向原生源码和CPU控制；未重做Y50或重置授权重试时间。',
        'Y51审计公共logit平移是否造成训练CE与部署选择错配。实际native RES已使用两类softmax前景概率singleargmax；32固定fixture、1精确阳性对照、既有TRAIN8/32共65组选中掩码不变，原始前景logit规则29组排名变化。',
        '最大原生选择器分值差2.220446e-16、float64对照CE差1.776357e-15、梯度差5.551115e-17，均在预注册包络内；字面原生float32损失/梯度精确相等。float64仅显式适配one-hot缓冲dtype，不冒称完整原生GPU训练。',
        '首个程序get_iou误导入在数值测试前失败，原程序/日志保留；v2仅提取实际loss.py原生函数，模型、fixture与判据不变。未创建模型变体、未用GPU、未做验证集评估、未修改队列或启动训练。',
        '执行状态待finish核验，科学状态Y51 rejected/hypothesis、Y50原有blocked/human_authorization未变。GPU实时占用本轮未知，不将上轮缓存当空闲证据。',
        '本轮收据用量在finish前未知；统计仅begin至收据的本地响应窗口，早先读档/晚到用量/最终回复不含；缓存属于输入、推理属于输出，不重复相加、不当账单。实际本轮新诊断1、模型变体0、GPU测试0、准入0、训练启动0。'],
        decisions=[
        '只否定Y51所声称的公共模态错配前提，保留原生选择器，不创建无此诊断依据的数值校准或排序变体，不外推所有置信度机制无效。',
        'Y50尚未到18:08:33+08授权重查时间且无新授权；其既有提交阻塞原样保留，不重复审查、重测、改写证据或延后计时。',
        '本轮一个新诊断预算已执行；其余已登记未完成事项仍为真实人工提交授权阻塞。按外部授权阻塞收尾，不声称路线穷尽、无信息增益或GPU不足，不硬撑deadline，不降低准入。',
        '没有新的训练后指标、准入或启动，不需要重复发出上轮已经提出的人工授权通知；六段报告和原始证据完整保存。']))
    print(json.dumps(dict(review=summary_path,notes=notes,y51='rejected',y50='unchanged_blocked',retry_after=held['retry_after'],finish='PENDING'),indent=2))
finally:lock.close()
