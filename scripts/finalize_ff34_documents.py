"""Persist bounded Chinese event/design memory and report notes before finish."""
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
    return json.loads((ROOT / ("pipeline/analyses/" + name + "_" + args.token + ".json")).read_text(encoding="utf-8"))


full = load("y42_frozen_full_ff34")
analysis = load("y42_clustered_comparison")
cost = load("y42_trained_cost_ff34")
native = load("y44_native_controls_ff34_v3")
physical = load("y44_physical_native_ff34")
resources = load("y44_resources_ff34")
launch = load("y44_launch_ff34")
commit = load("y44_commit_ff34")["code_commit"]
assert all(v["complete"] for v in (full, analysis, cost, native, physical, resources, launch))
assert launch["native_steps_increased"] and launch["actual_gpu"] == 1
queue = monitor.load_queue(str(ROOT))
active = [t for entries in queue["gpu_queues"].values() for t in entries
          if t["status"] in ("running", "claiming", "waiting")]
assert len(active) == 2 and {t["id"] for t in active} == {"AdaRES-Y43", "AdaRES-Y44"}
assert all(t["status"] == "running" and t["launch_verified"] is True for t in active)
steps = launch["captures"]
candidate_resource = resources["summary"]["candidate"]
notes_path = ROOT / ("pipeline/analyses/foreground_notes_" + args.token + ".json")
assert not notes_path.exists()
snapshot_local = datetime.fromisoformat(owner["acquired_at"]).astimezone().isoformat() if "acquired_at" in owner else "2026-10-13T05:50:06.621319+08:00"
notes = {
    "actions": [
        "本轮仅 AdaRES_project；快照 2026-10-13 05:50:06.621319+08，DESIGN。0-3段为该快照事实，以下记录本轮后续变化；不重复 preflight，不改调度或其他项目。",
        f"Y42 原始固定30轮完成、last30/best21；Y42/W2/B1各9508表达式/141场景两遍，FIRST mIoU分别{full['summaries']['y42']['miou']:.9f}/{full['summaries']['w2']['miou']:.9f}/{full['summaries']['b1']['miou']:.9f}。场景10000bootstrap/seed1999、TRAINcuts1487/3325及对象分组复评已落证据。",
        f"Y42相对W2差值{analysis['versus_w2']['all']['paired_miou_difference']:.9f}，95%区间{analysis['versus_w2']['all']['scene_clustered_ci95']}跨零；目标0.515未达。训练后配对压力成本中位{cost['summary']['candidate']['native_train_stress']['median_seconds']:.3f}秒，保留显存{cost['summary']['candidate']['native_train_stress']['peak_reserved_gib']:.3f}GiB。SAME Y42科学收尾complete_inconclusive，不晋升、不重启、不调参。",
        "先登记稳定Y44诊断再评审。32seed实际旧读出保持不变，而完整坐标-特征配对统计分离；独立新类32例及整模块父模型结构/配置一致性通过。随后原生TRAIN8/init4真实捕获、错配/零输入/禁用/重复、全部8张量梯度与两步更新、模型/优化器/调度器恢复、禁用物理W2一致性及资源均已执行通过。",
        f"保留本地GBK上传失败、旧PyTorch排序参数失败、独立归约逐位相等断言失败与本机Git路径故障。只修正工具适配；精度审计在观测前冻结推导边界，所有分量通过，缓存重复及禁用W2仍严格为0。原生初始不利场景、错配也能学习及恢复后微小数值差异保留；不称必要性、泛化、创新或收益。",
        f"Y44新增288002参数；压力保留显存{candidate_resource['stress_train_reserved']:.3f}GiB，推理中位{candidate_resource['inference_median']:.3f}秒，W2为{resources['summary']['parent']['inference_median']:.3f}秒，开销如实报告。恰好七个测试文件/git原始字节提交{commit}，六门槛收据和决策绑定，经规范调度仅发往server2 GPU1；无额外preflight/后台/Codex子任务。",
        f"Y44启动核验窗口{launch['observed_start_local']}至{launch['observed_end_local']}：原始PID{launch['pid']}，GPU1，Train{steps[0]['epoch']}/30步数{steps[0]['step']}→{steps[-1]['step']}，cwd/config/源码/提交一致。Y43 GPU0/PID13155保持不变；最新活跃2/目标2。Y44科学状态training_not_efficacy。",
        "本轮报告与七段记忆已在租约内落盘。执行状态待finish，不预称完成；finish失败将保留证据并abort核实释放。用量与实际产出以结束收据为准，缺失未知；窗口仅begin至收据，不含早读、晚记录及最终回复，缓存属于输入、推理属于输出，不重复相加或作账单。",
    ],
    "decisions": [
        f"2026-10-13 06:29:21 next_batch：同一Y42 complete_inconclusive，保留W2/目标0.515/Phase1；仅准入同一Y44至GPU1固定30轮，保留Y43 GPU0。见{args.decision}。",
        "容量2/2已满足，停止本轮新增设计；不为补位放宽门槛。Y43/Y44终态再做各模型全量两遍FIRST、场景区间、规模/对象分组、伤害对照、重复噪声与训练后配对成本。无盲重启、改名重入或验证集调参。",
        "创新性、上游是否已编码配对、方法必要性、训练收益与官方unique/multiple标签仍未知；下一次计划时间未知，不改写调度。",
    ],
}
memory = f"""# AdaRES_project Research Memory
## Goal And Phase
- Current{args.token} DESIGN; actualenvthread01a07e32-78bf-7670-925b-370911ea48db; snapshot2026-10-13 05:50:06.621319+08/deadline06:50:06.621319+08. beginONCE/compactFIRST/no repeatedpreflight.
- ScanRefertarget.515/Phase1unmet/Phase2unapproved/W2retained. RoundedCSVY36.5110 is not promotion. GoalSHA53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966 unchanged.
- ExplicitGPU0/design human authorization; onlyAdaRES/server2allowed[0,1]/onejobcard/keySSH/hostnameverified/PipelineWindowsdisabled/closureenabled. No otherproject/server6/scheduler/newchat/nestedCodex/background/password/sudo/server4.
- ReleasedSPFormer/native128/argmax1/seed1999/fixed30/nativeoptimizer/loss/scheduler unchanged; no validation thresholds/temperatures/components/neighbors/weights/epochs/gate relaxation.

## Champion And Baseline
- W2scientificcomparator; Y39FIRST.507705983/W2.510009850/CI[-.011360390,.006338310]; SAMEY39complete_inconclusive. Reviewpipeline/analyses/AdaRES-Y39_frozen_review_062ae665.md. No repeatedclosedtests.
- Y41FIRST.509047119/W2.510105160/CI[-.009716838,.007546363]; SAMEY41complete_inconclusive/reviewpipeline/analyses/AdaRES-Y41_frozen_review_0422d0d8.md/no promotion/retune/restart.
- SAMEY42canonicaldone_complete/last30/best21/originalPID91215absent. BestSHA6d8caaf4bb31e78404b4297e75171258e87505dad938af4f92f8d34cdb8a8172/lastSHA99d4abc4cd997cd9f8b1f0ddd698f4da1a47fcd9928770246753718ebdf55257; checkpointsreadonly.
- Y42/W2/B1 EACHfull9508/141/twice/FIRST; localwindow2026-10-13 05:59:43.541918->06:11:03.674806+08; originalevaluatorSHA7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13.
- FIRSTmIoU{json.dumps(analysis['summary'])}; repeatmeans{json.dumps(analysis['repeated_miou'])}.
- Y42-W2delta{analysis['versus_w2']['all']['paired_miou_difference']}/scene95CI{analysis['versus_w2']['all']['scene_clustered_ci95']}; B1delta{analysis['versus_b1']['all']['paired_miou_difference']}/CI{analysis['versus_b1']['all']['scene_clustered_ci95']}. Scene10000seed1999/TRAINcuts1487,3325/allsizeCIcross0/noharmtrigger NOT no-harm/equivalence.
- Singleobject9508/141/multi0/0 insufficient; NOTofficialunique/multiple latterUNKNOWN. Individualrepeatmax{json.dumps({k:v['repeat_max_iou_difference'] for k,v in analysis['cost_first_chunk_only'].items()})}; singlecheckp/seed/split limitsretained.
- TrainedY42/W2pressure{json.dumps(cost['summary'])}; params{json.dumps(cost['parameters'])}. Temporarynativeupdates memoryonly/checkpimmutable/2warm3read/descriptivep95/candidateorder notPareto superiority.
- SAMEY42COMPLETE/scientificcomplete_inconclusive/unfavorableW2point/targetunmet; reviewpipeline/analyses/AdaRES-Y42_frozen_review_ff347697.md. No promotion/restart/retune/renamed equivalent.

## Active Hypotheses
- OriginalY43 all32thirdmoments, independentW2readout; priornative6gates passed/commit6fc7d4e46b6c26a11fede7f12d4e79b3471cc1cc. PreserveGPU0/PID13155/configadares_y43_point_asymmetry_res.yaml/fixed30/scientifictraining_not_efficacy.
- SAMEY44hypothesisjoint-coordinate-feature-association registeredDIAGNOSIS FIRST. Frozengeometry/fullfeaturemultiset perSP but pairingdiffers; oldactualW2/Y39/Y42/Y43CPU32 marginal/readout0/crossgap>=9.
- Y44 independentall3x32rawcenteredcross96/nativecoords_float/point-orderSP/expressionmapping; query+pointmassFG/BG finalconfidenceonly/detachedmask. No selectedcomponents/normalization/sweep/newloss.
- ActualnewclassCPU32minpairresponse.0010000116/controlfloor1.11e-16/all8finitegrads; wholedec/ipdnASTparentidentity afterexactnewpath removal/configdiffs[]/uniqueconfig/sevenlocalremotehashes equal.
- NativeTRAIN8/init4actualcapturedall3x32SPcross, withinSPfeaturebijection/wrongpair/rolledcross/zero/wrongmasks. Eightgradientsmin{native['minimum_added_gradient']}; repeat/nonindicatorloss/padding0/permutation{native['permutation_max']}.
- V3capturednative mean reductions/CPU64reference; allcomponenterror inside prospectivelyderived machineeps/pointcount/magnitude envelope; maxCPU64error{max(r['incidence']['native_cpu64_error_max'] for r in native['rows'])}. Wrongpairgapmin{min(r['incidence']['wrong_pairing_cross_max'] for r in native['rows'])}. No bitwisecrossclaim or changedmodel/gates.
- Native tools.train2steps/all8actualupdates; exact model/AdamW/PolyLRrestore BEFOREcontinuation, nextstepdifferences{json.dumps(physical['continuation'])} retained. Fourcachedscene disabledphysicalW2mask/indicator/selectionequal0/native128/argmax1.
- Nativeinitiallossdeltas{json.dumps(native['loss_deltas'])}; fixed2newonlylearner correct/wrongcontrols alllearn, adverse individualscenesretained; legacywrong_thirdalias=rolled96cross NOTthirdmoment. No diagnosticweightdeployment/benefit/necessityclaim.
- Added288002params/resources{json.dumps(resources['summary'])}; <1%W2/reserved<24GiB/noOOM/nonfinite, actualtimeoverheadretained/nottrainedPareto.
- Y44sixgatereviewpipeline/analyses/Y44_six_gate_review_ff347697.md/designpipeline/designs/exp_design_Y44_ff347697.md/motivationfullgoal/smell2,3defense/terminalfreeze. EXACT7rawbytecommit{commit}; perinvocationautocrlffalse preservingtestedbytes, noreposetting/sourcechange/originpush.
- Canonicaldecision{args.decision}/receiptpipeline/admissions/AdaRES-Y44.json; scopedserver2GPU1dispatch/no repeatedpreflight.
- OriginalY44PID{launch['pid']}/GPU1/nativeTrain{steps[0]['epoch']}/30 {steps[0]['step']}->{steps[-1]['step']}, verifiedwindow{launch['observed_start_local']}->{launch['observed_end_local']}, originalcwd/config/source/commitverified/launch_verifiedtrue. Scientifictraining_not_efficacy.
- Y43/Y44terminal EACH9508/141twice/FIRST/scenebootstrap10000seed1999/TRAINcuts1487,3325/objects/harmvsEither/repeat/trainedpairedcost; no blindrestart/retune.

## Closed Or Failed Directions
- Y39/Y41/Y42complete_inconclusive; historicalY40readoutoverlap/Y38/Y34overlap/Y32ASTduplicate/Y33/Y30/Y28subgroupharm retained, not reopened or repeated.
- CurrentY44 localGBKreaderfault beforeSSH fixedUTF8; nativev1unsupportedstable-sort beforecontrols correctedbyuniquelabel+pointindex keys, same withinSPreverse.
- Nativev2assert exactindependentscatter failedbeforedecodedcontrols; originalsource/logpreserved. V3quantifies actualerror/derivedbound and independentrepeat withoutfittedtolerance. Candidatepackage unchanged.
- DecisionwriterGitpathmissing preventedprocessstart/noindexmutation; verifiedinstalledProgramFiles(x86) path then canonicalwrite. Othermodel/gates/originalweights unchanged.
- All original failedprobe sources/logs and hashed evidence retained; no failed observation overwritten.

## Running And Queue
- BEGIN05:50:06 snapshotactive1/running1/claim0/wait0/target2/budget1/newterminalY42/due0; deterministicreport0-3boundONLYthissnapshot. StructuredGPUutil/memory/process/metricsUNKNOWN notzero/idle.
- LATER verifiedoriginalY43GPU0/PID13155 andY44GPU1/PID{launch['pid']}: active2/running2/claim0/wait0/target2, bothnativeprogressverified. CurrentHEAD{commit}.
- DeviceY44launchoccupation{launch['gpu1_device_memory_mib']}MiB/{launch['gpu1_device_utilization']}% is its timedwindow, notisolatedpeak. No source/queue mutation outsideAdaRES.
- Capacitytargetmet => stopnewdesign, no fakeexhaustion/resourceblock/cooldown/deadline. NextscheduledtimeUNKNOWN; no schedulingchange.

## Open Questions And Risks
- NoY43/Y44efficacy/novelty/necessity/upstreamblindness/no-harm/humanPhase2approval. Coordinateunits/rawmoments/backbonegeometryredundancy may limitY44utility.
- Wrongcontrolslearnsimilarly, initializedloss/tasksubsetcannotestablish populationbenefit. Latency/memoryoverheads andperexpression repeatinstability retained.
- Originaltrainingstart/exit/duration/remoteclockrelationunknown; localobserverwindows explicit. Boundedlogs no crashmarkers notwholelogproof; PIDalone notlaunchsuccess.
- PreserveoriginalY42checkpoints/Y43training and frozenposttrainingharmrules; don'ttune/restart toreach.515.

## Last Tick
- Currentff34DESIGNscientificallyclosedSAMEY42inconclusive andadvancedSAMEY44fromdiagnosis throughactualsixgates toverifiedGPU1training. W2/Phase1targetunchanged; capacity2/2.
- Memory/Chineseactions+decisions underownedlease; executionfinishPENDING, requireactualPASS/COMPLETE andindependentownedlockabsence; no postreleasewrites.
- Receiptusagewindowbegin->recordONLY/excludesearlyreads/laterecords/finalreply/cacheinsideinput/reasoninginsideoutput/notbill; unavailableUNKNOWN. Actualcountsreportedafterfinishinchat, notsummedacrossreceipts.
"""
assert len(memory.splitlines()) < 120
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/codex_memory.md"
    backup = ROOT / ("pipeline/analyses/codex_memory_before_" + args.token + ".md")
    assert not backup.exists()
    old = path.read_bytes()
    backup.write_bytes(old)
    path.write_text(memory, encoding="utf-8")
    notes_path.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"memory_lines": len(memory.splitlines()), "notes": str(notes_path),
                      "prior_memory_sha256": hashlib.sha256(old).hexdigest(),
                      "current_active": len(active), "execution_status": "finish_pending"}, ensure_ascii=False))
finally:
    lock.close()
