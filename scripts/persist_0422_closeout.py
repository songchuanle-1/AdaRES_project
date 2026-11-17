"""Seven-section memory and Chinese report notes, before actual postflight."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--decision", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor


def load(name):
    return json.loads((ROOT / ("pipeline/analyses/" + name + "_" + args.token + ".json")).read_text())


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


analysis = load("y41_clustered_comparison")
cost = load("y41_trained_cost_0422_v2")
native = load("y43_native_controls_0422_v2")
physical = load("y43_physical_native_0422")
resources = load("y43_resources_0422")
initial = load("y43_initial_loss_0422")
launch = load("y43_launch_0422")
commit = load("y43_commit_0422")
assert all(row["complete"] for row in (analysis, cost, native, physical, resources, initial, launch, commit))
assert launch["native_steps_increased"] and launch["actual_gpu"] == 0
assert launch["code_commit"] == commit["code_commit"] and not commit["origin_push"]
assert launch["preserved_y42_pid"] == 91215 and launch["preserved_y42_gpu"] == 1
summaries = analysis["summary"]
pair, baseline = analysis["versus_w2"]["all"], analysis["versus_b1"]["all"]
stress = cost["summary"]
steps = "->".join(str(v["step"]) for v in launch["captures"])
observed = datetime.now().astimezone().isoformat()
notes_path = "pipeline/headless/.foreground_notes_" + args.token + ".json"
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue = monitor.load_queue(str(ROOT))
    active = [t for entries in queue["gpu_queues"].values() for t in entries
              if t["status"] in ("running", "waiting", "claiming")]
    assert len(active) == 2 and {t["id"] for t in active} == {"AdaRES-Y42", "AdaRES-Y43"}
    assert all(t["status"] == "running" and t["launch_verified"] for t in active)
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text())
    y41 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y41")
    y43 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y43")
    assert y41["stage"] == "complete" and y43["stage"] == "training"
    script = "scripts/persist_0422_closeout.py"
    y43["evidence"].append(evidence(script))
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")

    memory_path = ROOT / "pipeline/codex_memory.md"
    old = memory_path.read_bytes()
    backup = "pipeline/analyses/codex_memory_before_" + args.token + ".md"
    assert not (ROOT / backup).exists()
    (ROOT / backup).write_bytes(old)
    old_text = old.decode("utf-8")
    historical = old_text.split("## Champion And Baseline", 1)[1].split("## Active Hypotheses", 1)[0].strip()
    closed = old_text.split("## Closed Or Failed Directions", 1)[1].split("## Running And Queue", 1)[0].strip()
    memory = f"""# AdaRES_project Research Memory
## Goal And Phase
- Current{args.token} DESIGN; env thread01a07e32-78bf-7670-925b-370911ea48db verified. Snapshot2026-10-13 01:49:06.538573+08/deadline02:49:06.538573+08; beginONCE/compactFIRST, no repeatedpreflight.
- ScanRefer target.515/Phase1unmet/Phase2unapproved/W2retained. RoundedCSVY36.5110 is not promotion. GoalSHA53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966 unchanged.
- Explicit GPU0/design human grant; server2 allowed[0,1]/one singleGPU job/card; keySSH/hostname dl01-MS-7C60. WindowsPipeline disabled/closureenabled unchanged. No otherproject/server6/scheduler/newthread/nestedCodex/background/password/sudo/server4.
- ReleasedSPFormer/native128/argmax1/seed1999/fixed30/nativeoptimizer/loss/scheduler unchanged. No validation thresholds/temperatures/feature subsets/neighbors/weights/epochs/gate relaxation.

## Champion And Baseline
{historical}
- CurrentY41 canonicaldone_complete/originalPID116906absent/last30/best21. BestSHA490a0a1c50d69293836302f92b5c9b6ebfbbc15655186d2bf9bc8e2195a99c88/lastSHAbdd11e943b56b23fcd4f9d600a6b50226268bcfd5be788e26837378d9571e66a, original checkpoints read-only.
- Full9508/141 Y41/W2/B1 twiceEACH/FIRSTprimary/window01:59:27.319894->02:11:20.608763+08. FIRSTmIoU{json.dumps({k:v['miou'] for k,v in summaries.items()})}; Y41Acc25{summaries['y41']['acc25']}/Acc50{summaries['y41']['acc50']}/spmIoU{summaries['y41']['spmiou']}.
- Y41-W2delta{pair['paired_miou_difference']}/scene95CI{pair['scene_clustered_ci95']}; B1delta{baseline['paired_miou_difference']}/CI{baseline['scene_clustered_ci95']}. Bootstrap10000seed1999/TRAINcuts1487,3325/observedsize noCIupper<0 trigger, not no-harm/equivalence.
- SmallversusB1CIpositive butnot overall superiority; mediumvsW2 adverse-.0104083/CIcross0. Objectcountsingle9508/141/multi0/0 insufficient, NOTofficialunique/multiple; officialunknown.
- Individualrepeatmax{json.dumps({k:v['repeat_max_iou_difference'] for k,v in analysis['cost_first_chunk_only'].items()})}; single checkpoint/seed/split limitations retained.
- TrainedY41/W2pressuremedian{stress['candidate']['native_train_stress']['median_seconds']}/{stress['parent']['native_train_stress']['median_seconds']}s/reserved{stress['candidate']['native_train_stress']['peak_reserved_gib']}/{stress['parent']['native_train_stress']['peak_reserved_gib']}GiB; inferenceTRAIN8median{stress['candidate']['train8']['median_seconds']}/{stress['parent']['train8']['median_seconds']}s. Params164696711/164629637; no Pareto superiority.
- SAMEY41COMPLETE/scientificcomplete_inconclusive; reviewpipeline/analyses/AdaRES-Y41_frozen_review_0422d0d8.md. W2 retained/targetunmet/no promotion/restart/retune.

## Active Hypotheses
- OriginalY42 full528covariance remainsGPU1/PID91215/configadares_y42_point_covariance_res.yaml/commit243e0b6827073831e0d68090857755d575efbf16. Historical00:47raw27/30/roundedbest.5050; currentstructuredmetricsUNKNOWN. Originalconfig/source/PID preserved through this tick, no late efficacy inference.
- SAMEY43 all-channel-point-distribution-asymmetry registeredDIAGNOSIS first, testedexisting32-seedreflection gap0/thirdgap>=4. Initial negative-feature caveat retained; actualindependentnonnegative32-case witness passed, minreflection.02874938/controlfloor1.53e-16/all8gradients/zerostatfinite.
- IndependentW2 copy: all32 withinSPthirdmoments/stablesignedcuberoot/nativepointmass FG/BG/query finalconfidenceONLY/detachedmask. Completephysicalcode sixfiles+uniqueYAML, sevenlocal/gitblob/remotehashes equal. Wholedec/ipdn samePythonASTparentidentity afterexactnewpath removal/configdiffs[].
- NativeTRAIN8/init4 all8 finitegrads/min{native['minimum_added_gradient']}; wrongmaskmin{native['wrong_masks_min']}/rolledthirdmin{native['wrong_third_min']}/reflectionmin{native['reflected_third_min']}; repeat/nonindicatorloss/poison0/permutation{native['permutation_max']}. Narrowstatisticpath gap, not upstreamblindness/necessity/taskutility.
- Initialnativedeltas{json.dumps(native['loss_deltas'])}; fixed2newonlyAdamWchanges{json.dumps({phase:{m:v['mean_change'] for m,v in outcomes.items()} for phase,outcomes in initial['summary'].items()})}. Alladverseindividualchanges/wrongcontrolslearning retained; no diagnosticweightdeployment.
- Native tools.train2steps/all8updated; exactmodel/AdamW/PolyLRrestore beforecontinuation, lossdelta{physical['continuation']['loss_difference']}. Fourcachedscene disabledphysicalW2masks/indicators/selectionequal0; native128/argmax1.
- Y43new271618params/resources{json.dumps(resources['summary'])}; frozenTRAIN8/pressurepair728,6780/2warm3read/params<1%/reserved<24GiB/noOOMorNonfinite. Prospectivefeasibility only, not trainedPareto/efficacy.
- Sixgate reviewpipeline/analyses/Y43_six_gate_review_0422d0d8.md; sevensectiondesign+smell2/3defense/motivation/actualevidence. Codeexact7commit{commit['code_commit']}/originpushFALSE; receiptpipeline/admissions/AdaRES-Y43.json/decision{args.decision}; canonicalGPU0dispatch/noextra preflight.
- Launchwindow{launch['observed_start_local']}->{launch['observed_end_local']}/PID{launch['pid']}/GPU0/Train{launch['captures'][0]['epoch']}/30 steps{steps}/{launch['captures'][0]['total_steps']}; actualcwd/config/source/codeverified, launch_verifiedtrue. Scientifictraining_not_efficacy.
- TerminalY43 requiresfrozen9508/141twiceEACH/FIRST/scenebootstrap10000seed1999/TRAINcuts1487,3325/objectgroups/harmvsEither/noise/trainedpairedcost. PreserveoriginalY42/Y43/checkpoints/noearlystop/blindrestart/retune.

## Closed Or Failed Directions
{closed}
- CurrentY41 complete_inconclusive. FirstY41costadapter failed protectedPID91215 bytesexpectedY41/GPU0, v2correctY42/GPU1 only; originalfailedlog/hash retained.
- Y43two preexecutionpathbinding mistakes corrected beforetests; firstnativeadapter failed resultkeywordKeyError(wrong_third_max). Newv2changesresultkeywordfieldsONLY; originalhelper/log/hash retained. No candidate model/gate/tolerance change aftertests.

## Running And Queue
- BEGINactive1/running1/claim0/wait0/target2/budget1/due0/newterminalY41. Sections0-3 deterministicfacts bound to01:49:06 snapshot, unknownGPUutilization/memory/process/structuredmetrics notidle/zero.
- Postdispatchverified{observed}: active2/running2/claim0/wait0 originalY42GPU1+newY43GPU0; oneadmissiontargetmet. No fakeblock/exhaustion/cooldown or wholetrainingwait.
- GPU0atlaunch{launch['gpu0_device_memory_mib']}MiB/{launch['gpu0_device_utilization']}% deviceoccupationonly; prediagnosticidle snapshotseparate. NextscheduledtimeUNKNOWN; no automation/scheduler edits.

## Open Questions And Risks
- No Y43efficacy/novelty/necessity/upstreamblindness/noharm/humanPhase2approval. Momentpath may be redundant/taskirrelevant; signedrootmachinefloor bounds derivative notsearchedsetting.
- Singlecheckpoint/seed/split/unknownofficialuniqueness and comparatorperexpressionrepeat instability limitconclusions. Sceneintervalcross0 is inconclusive, notsafety/equivalence.
- Runnerlocalwindows used; remoteclockrelationunknown, originaltrainingstart/exit/durationunknown. Boundedlogabsence notwholelogproof; PIDalone notlaunchsuccess.
- Oldmemoryverbatim{backup}/SHA{hashlib.sha256(old).hexdigest()}; all closed original evidence preserved.

## Last Tick
- CurrentDESIGN closedSAMEY41 scientificallyinconclusive and progressedSAMEY43diagnosis->actualsixgatevalidation->oneGPU0training; originalY42preserved. Targetactive2met.
- Memory/Chineseactions+decisions written UNDER ownedlease at{observed}; executionfinishPENDING. RequireactualCLOSURE_POSTFLIGHT_PASS/COMPLETE+independentownedlockabsence. No postreleasewrites.
- Receiptusagebegin->recordlocalwindowONLY/excludesearlierreads/laterecords/finalreply, notbill; cachedpartofinput/reasoningpartofoutput/notaddedagain. MissingUNKNOWN; priorreceiptsnotsummed. Actualcountsreportedafterfinishinchat.
"""
    assert len(memory.splitlines()) < 120
    memory_path.write_text(memory, encoding="utf-8")
    notes = {
        "actions": [
            "本轮 DESIGN；快照2026-10-13 01:49:06+08，0–3段仍绑定该时点。归属、冻结、资源授权已核实；begin一次、紧凑视图先读，无重复preflight。",
            f"Y41原训练终态已核实，last30/best21、模型/优化器/调度器只读保留。冻结复评窗口01:59:27–02:11:20+08：Y41/W2/B1各9508表达、141场景、两遍，FIRST主分析。mIoU分别{summaries['y41']['miou']:.8f}/{summaries['w2']['miou']:.8f}/{summaries['b1']['miou']:.8f}；Y41 Acc25 {summaries['y41']['acc25']:.8f}/Acc50 {summaries['y41']['acc50']:.8f}/spmIoU {summaries['y41']['spmiou']:.8f}，128查询/单argmax。",
            f"Y41对W2差{pair['paired_miou_difference']:.8f}，场景95%CI{pair['scene_clustered_ci95']}；对B1差{baseline['paired_miou_difference']:.8f}，CI{baseline['scene_clustered_ci95']}。总体跨0、目标未达，科学状态complete_inconclusive。小组对B1为正但不证明总体优势；medium对W2点估计不利，区间跨0。multi-object0/0无数据，official unique/multiple未知，不宣称无害/等价。",
            f"Y41/W2训练后配对压力成本中位{stress['candidate']['native_train_stress']['median_seconds']:.5f}/{stress['parent']['native_train_stress']['median_seconds']:.5f}秒，保留{stress['candidate']['native_train_stress']['peak_reserved_gib']:.4f}/{stress['parent']['native_train_stress']['peak_reserved_gib']:.4f}GiB；固定TRAIN8及最大样本对、2预热3测量，临时更新不落原检查点。无Pareto优势证据。",
            f"先登记稳定Y43，再执行32种子原路径反射和非负实际新读出对照、完整父AST及配置一致性、TRAIN8/init4错掩码/滚动统计/反射/重复/零路径、原生两步保存恢复、配对资源。六项可行性门通过；新增271618参数，压力训练Y43/W2 {resources['summary']['candidate']['stress_train_median']:.5f}/{resources['summary']['parent']['stress_train_median']:.5f}秒、{resources['summary']['candidate']['stress_train_reserved']:.4f}/{resources['summary']['parent']['stress_train_reserved']:.4f}GiB，时间回退明确保留。所有初始不利损失、错误对照同样学习、续步数值差仍保留，不是效果或必要性证明。",
            "保留首次反射样例允许负特征的限制，另做两臂均非负确认；保留Y41成本绑定故障和Y43原生记录字段故障的原日志/哈希。仅修测试绑定/记录字段，未改模型、门槛、样本或容差。",
            f"仅七个已测试新文件提交{commit['code_commit']}，未push或改无关变更；哈希绑定六门收据和决策{args.decision}后走canonical GPU0派发。Y43独立启动窗口{launch['observed_start_local']}至{launch['observed_end_local']}：PID{launch['pid']}，Train{launch['captures'][0]['epoch']}/30，步数{steps}/{launch['captures'][0]['total_steps']}，cwd/config/提交/源哈希及GPU0核实。Y42 GPU1原PID91215及源文件保持不动。",
            f"后续核实活跃2/目标2，Y43科学状态training_not_efficacy；GPU0启动窗口设备占用{launch['gpu0_device_memory_mib']}MiB/{launch['gpu0_device_utilization']}%，不是孤立峰值。本轮执行状态待finish；完成后在消息中补收据用量窗口和实际产出，缺失记未知、缓存不重复相加、不当账单。未操作其他项目或调度。",
        ],
        "decisions": [
            "Y41终态证据尚不能证明优于W2，目标0.515未达；继续保留W2，不重启、微调或晋升Y41，不把跨0区间当等价/无害。",
            "Y43只获可证伪的固定30轮Phase1训练准入；第三阶统计机制待训练结果检验，不宣称效果、必要性、新颖性或人类审批，Phase2未授权。",
            "容量已达2，保留原Y42 GPU1和新Y43 GPU0。终态后按冻结9508/141各两遍、FIRST、场景bootstrap、规模/对象组、对任一参照伤害、重复噪声及训练后配对成本复核。下一调度时刻未知。",
        ],
    }
    assert not (ROOT / notes_path).exists()
    (ROOT / notes_path).write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"memory_lines": len(memory.splitlines()), "backup": evidence(backup),
                      "notes": notes_path, "active": len(active), "execution": "finish_pending"}))
finally:
    lock.close()
