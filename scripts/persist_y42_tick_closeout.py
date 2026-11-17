"""Persist seven-section memory and Chinese notes only under this owned lease."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--decision", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text(encoding="utf-8"))
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


base = "pipeline/analyses/"
launch_path = base + "y42_launch_" + args.token + ".json"
launch = load(launch_path)
cost = load(base + "y42_resources_" + args.token + ".json")
commit_path = base + "y42_commit_" + args.token + ".json"
commit = load(commit_path)
review_path = base + "Y42_admission_review_062ae665.md"
terminal_review = base + "AdaRES-Y39_frozen_review_062ae665.md"
assert launch["complete"] and launch["native_steps_increased"] and launch["actual_gpu"] == 1
assert launch["code_commit"] == commit["code_commit"] and not commit["origin_push"]
assert launch["preserved_y41_pid"] == 116906 and launch["preserved_y41_gpu"] == 0
observed = datetime.now().astimezone().isoformat()
captured = launch["captures"]
steps = str(captured[0]["step"]) + "->" + str(captured[1]["step"])
notes_path = "pipeline/headless/.foreground_notes_" + args.token + ".json"
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue = monitor.load_queue(str(ROOT))
    active = [t for entries in queue["gpu_queues"].values() for t in entries if t["status"] in ("running", "waiting", "claiming")]
    assert len(active) == 2 and {t["id"] for t in active} == {"AdaRES-Y41", "AdaRES-Y42"}
    assert all(t["status"] == "running" and t["launch_verified"] for t in active)
    work_path = ROOT / "pipeline/research_work.json"
    work = load("pipeline/research_work.json")
    candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y42")
    y39 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y39")
    assert candidate["stage"] == "training" and y39["stage"] == "complete"
    for ref in (commit_path, review_path, "scripts/persist_y42_tick_closeout.py"):
        entry = evidence(ref)
        prior = next((e for e in candidate["evidence"] if e["path"] == ref), None)
        assert prior is None or prior == entry
        if prior is None:
            candidate["evidence"].append(entry)
    work_path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    memory_path = ROOT / "pipeline/codex_memory.md"
    old = memory_path.read_bytes()
    backup_path = base + "codex_memory_before_" + args.token + ".md"
    assert not (ROOT / backup_path).exists()
    (ROOT / backup_path).write_bytes(old)
    old_sha = hashlib.sha256(old).hexdigest()
    memory = f"""# AdaRES_project Research Memory
## Goal And Phase
- Current{args.token} DESIGN; actual environment thread01a07e32-78bf-7670-925b-370911ea48db verified. Snapshot2026-10-11 21:49:58.987108+08/deadline22:49:58.987108+08, begin ONCE/compact FIRST/monitorRC0/goalRC0.
- ScanRefer target.515; rounded CSVchampY36.5110/gap.004 NOT promotion. W2 retained; Phase1 unmet/Phase2 unapproved. GoalSHA53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966 unchanged.
- Explicit GPU0/design grant plus current server2 allowed[0,1]/one singleGPU job per card; hostname dl01-MS-7C60/key SSH. Closure enabled/WindowsPipeline disabled unchanged. No other project/server6/scheduler/new task/nestedCodex/background decision/password/sudo/server4.
- ReleasedSPFormer/native128/one argmax/seed1999/fixed30/native optimizer/loss/scheduler unchanged. No validation thresholds/temperatures/feature subsets/neighbors/epoch search/gate relaxation.

## Champion And Baseline
- W2 is scientific comparator; historicalY36 frozen FIRST.5107166243/CIvsW2[-.007218155,.008880425] inconclusive. HistoricalY37 FIRST.505061023/CIvsW2[-.012977527,.002909512] inconclusive; no repeated closed tests.
- CurrentY39 canonical done_complete/originalPID102267 absent/last30/best23. BestSHA4a50225433aa755bf9b0f5643b9bcee8b025c828dcd5aaede2bc092b95fb68e1/lastSHAfe5444138102d2045a2746e3c61b60507466d374763f805f06beb8709c205f76; original checkpoints unchanged.
- Y39/W2/B1 EACH9508expressions/141scenes/twice/FIRSTprimary. Localwindow21:59:12.860011->22:10:35.313285+08/fullSHAcdb0188acbae92545e501549ce5102bcd78309f8dde0fc4126e6a1039cad44b7; original evaluatorSHA7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13.
- FIRSTmIoU Y39.5077059831652888/W2.5100098499225277/B1.5022241613397069. Y39 Acc25.6069625578460244/Acc50.5612116112747161/spmIoU.47172301744895867; target unmet.
- Scene10000seed1999/TRAINcuts1487/3325: Y39-W2delta-.0023038667572389342/CI[-.011360390287967127,.006338310071024288]; versusB1+.005481821825581824/CI[-.0032655849737820318,.014512773945359362].
- All/size observed CI cross0/no upper<0 trigger; NOT equivalent/no-harm. Single-object9508/141 sameall; multi-object0/0 insufficient data/noCI. Objectcount NOT officialunique/multiple metadata; latter unknown.
- Per-expression repeatmaxY39 1.0/W2.02990561724/B1.97389560938 despite similar aggregate means. Single seed/checkpoint/split/repeat instability limit conclusions.
- Trained pairedY39/W2 stressmedian.9877400119/.9408668727s/reserved13.513671875/13.400390625GiB; TRAIN8+maxpair728/6780/2warm3read. CostSHAb4d5145529cdc2354a96c15767b490f5e3fd59aea05dc868177be03487f88ddd; no Pareto superiority.
- SAMEY39 stageCOMPLETE/scientificcomplete_inconclusive; six-sectionreview{terminal_review}/SHA{evidence(terminal_review)['sha256']}. W2 retained/no promotion/retune/restart.

## Active Hypotheses
- SAMEY41 final-mask-boundary-probability remains TRAINING originalGPU0/PID116906/logtrain_adares_y41_boundary_probability_res_20261011_162911.log. Six priorgates/receiptAdaRES-Y41 unchanged, code79ce2de9582c369422facb27c3cbe85559bf227c.
- PreflightY41 raw7/30/roundedbest.4697/no_improve0/30; structured metrics UNKNOWN. Current launch verification preserves originalY41PID/config/GPU0/sourcebytes; no late structured validation poll or restart.
- SAMEY42 full-channel-point-covariance registered DIAGNOSIS as first durable design action, not broad-route rejection. Complete32-channel centered528 covariance entries/signedlog1p/528->256->256, detachedfinalmaskpointmass FG/BG/query->2, finalindicatoronly.
- Y42 CPU32 equalmeans/diagvariance/counts/masks/query/geometry: actualW2/Y39delta0/covgap2. Actualextract/readout matchedjointoutputgap.00465405/gradientmin.00323320203/invariance5.96e-8. Narrow readout gap, not upstream blindness/semantic benefit/novelty.
- NativeTRAIN8/init4: new8 finitegradients/min1.54949885e-5; wrongmaskmin1.97440386e-5/offdiagmin8.69780779e-5/PSDreflectionmin.0001042485. Repeat/nonindicatorloss/poison0, permutation8.94e-8/maskgradientdetached.
- Init enabled lossadverse+.0183002949 retained. Fixed2 new-only AdamW correct traineddelta-.0010590702/init-.0422233343; wrongcontrols similarlylearn and someindividualcorrectlossesrise. No necessity or diagnosticweightdeployment.
- Native tools.train2steps/all8changed/408decodergradienttensors; exactmodel/AdamW/PolyLR restore beforecontinuation. Nextloss0/model1.5258789e-5/optimizer2.3625034e-8, not bitwise trajectory. Fourcachedscene disabledphysicalW2equal0/128/oneargmax.
- Y42/W2 params165028231/164629637/add398594. Pressuremedian1.123867346/.912553675s (+23.16%)/reserved14.046875/13.400390625GiB; infermedian.100903064/.095824677s. Prospective+params<1%/reserved<24GiB/nonfinite/OOMpassed, not trained Pareto efficacy.
- ResourceSHA35b4444fac2f9f7c8c6c72e130cd60a97d4c376453645b5e00dd73073e9e610a; samefrozenTRAIN8/maxpair728/6780/2warm3read/serialprediction. No new cost/metric admission threshold.
- Wholedec/ipdn samePython3.8 AST parentidentity afterexactnewpathstripping/configdiffs[], SHA2f00e85079fc0e51691e5651c4889e9b2b72b80ae5cf583f6b800d66ddeac545. Independentsixfiles/uniqueYAML/7localgitblobremotehashes equal; no rawparentbyteidentity claim.
- Six-sectionmotivation/seven-sectiondesign+Smell/admissionreview{review_path}/SHA{evidence(review_path)['sha256']}; receiptpipeline/admissions/AdaRES-Y42.json/decision{args.decision}. Sixactualfeasibilitygates only, not humanapproval/efficacy.
- Exactly7 testedfiles commit{commit['code_commit']}; originpushFALSE/unrelateddirtyfilesuntouched. CanonicalGPU1additive-markerdispatch/no repeatedpreflight.
- Independentlaunchwindow{launch['observed_start_local']}->{launch['observed_end_local']}/PID{launch['pid']}/GPU1/Train{captured[0]['epoch']}/30steps{steps}/{captured[0]['total_steps']}; cwd/config/commit/source hashes/launch_verifiedtrue. SHA{evidence(launch_path)['sha256']}.
- SAMEY42stageTRAINING/training_not_efficacy. Terminal->frozenfull9508/141 eachY42/W2/B1twice/FIRST/scene10000seed1999/TRAIN1487/3325/harmversuseither/repeat/trainedpairedcost. Noearlystop/originalcheckpointoverwrite/retune.

## Closed Or Failed Directions
- Y39 nowcomplete_inconclusive. HistoricalY40readoutoverlap/Y38/Y34overlap/Y32ASTduplicate/Y33/Y30/Y28subgroupharm and allclosed evidence retained without reopened renamedvariants or fullhistory review.
- CurrentGBKlocalbuildfault before remote writes retained; explicitUTF8 fix recovered only6copiednativefiles thenLFformatted beforetests. Original failedparentAST tuple-return binding, guessedcallname and double-substitutedconfig logs/programs retained.
- V2/V3 test-adapter repairs only; exactactualextract_return/forward_iter_pred/uniqueconfig binding. No candidate source/gate/tolerance edits aftertests; no failed outcome overwritten.
- PSDreflection preregistered before native data; offdiagroll mayleavePSDcone. Wrongcontrolslearning and adverseinitloss retained as scientificlimitations, not gatebenefit.

## Running And Queue
- BEGIN21:49:58+08 active1/running1/claim0/wait0/target2/budget1/due0/newterminalY39. Sections0-3 facttables bound to this snapshot, GPU utilization/memory/process andstructuredmetrics UNKNOWN.
- Verifiedpostdispatch at{observed}: active2/running2/claim0/wait0, Y41GPU0original andY42GPU1new; admissiontargetmet. No invented resourceblock/exhaustion/cooldown or wholetrainingwait.
- ActualGPU1device atlaunch{launch['gpu1_device_memory_mib']}MiB/{launch['gpu1_device_utilization']}% is timestamped deviceoccupation, not isolatedpeak; prediagnostic1MiB/util0 was separate evidence. No freeGPUinference fromunknownsnapshot.
- NextscheduledtimeUNKNOWN; heartbeat/Windowsdisabledschedulerunchanged. Futureaction canonicalterminal/crash/frozenevidence only, preserveoriginaljobs.

## Open Questions And Risks
- Y42 realizes jointcovariance readout and feasibletraining, NOT efficacy/necessity/generalization/noharm/Pareto. Correct/wrong pathslearnsimilarly; nativebackbone mayindirectly encode jointstatistics; no humanPhase2approval.
- Frozen bootstrapcrossing0 inconclusive; absentmulticountpopulation/unknownofficialuniqueness/instableexpressionrepeats/singleseedcheckpointsplit limitclaims. No fabricated missing subgroup/latency evidence.
- Localrunnerwindowsused/remoteclockrelationunknown. Originaltrainingstart/exit/durationunknown. Boundedlogcrashabsence not wholelogproof; PIDalone notlaunchsuccess.
- Oldmemoryverbatim preserved{backup_path}/SHA{old_sha}; allhistoricalevidenceuntouched. RunnerSHAb1f4e6b818cacbae148abef2d5ac0d605f6ace5e13e80dba48003539581dcd50 unchanged.

## Last Tick
- ThisDESIGN closesY39scientificallyinconclusive and advancesstableY42diagnosis->validation->training, exactly1newadmission/independentlaunch, originalY41preserved. Tests/failures/hashes saved beforeclosure.
- Seven-sectionmemory/Chineseactions+decisions persisted UNDER ownedlease at{observed}. Finish/receiptPENDING atwriting; require actualCLOSURE_POSTFLIGHT_PASS/COMPLETE and independentmatchinglockabsence. No postreleasewrites.
- Receiptusageonlybegin->record localwindow/excludesearlierreads/laterecords/finalreply, NOT bill. Cachedtokenspartofinput/reasoningpartofoutput, not addedagain; missingUNKNOWN. Currentusage/outcome counts reportedafterfinishinchat, notsumofhistory.
"""
    assert len(memory.splitlines()) < 120
    memory_path.write_text(memory, encoding="utf-8")
    notes = {
        "actions": [
            "本轮 design；begin 一次、紧凑视图先读。快照为2026-10-11 21:49:58+08；0–3段仍绑定该时点，不代表后续训练状态；当前线程、AdaRES归属、closure授权、冻结协议和资源白名单已核实。",
            "Y39终态真实核实：原PID102267已消失，last30/best23和模型/优化器/调度器保留；本轮全9508表达/141场景，Y39/W2/B1各两遍、FIRST主分析。窗口21:59:12–22:10:35+08；Y39 mIoU0.50770598/Acc25 0.60696256/Acc50 0.56121161/spmIoU0.47172302。对W2差-0.00230387，场景CI[-0.01136039,0.00633831]；对B1差+0.00548182，CI[-0.00326558,0.01451277]。",
            "Y39科学状态complete_inconclusive，W2保留；观测规模组区间跨0不等于无害，multi-object0/0缺数据，official unique/multiple未知；每表达重复最大差Y39=1.0。训练后配对压力成本Y39/W2为0.98774/0.94087秒、13.5137/13.4004GiB，无Pareto提升证据；原检查点未改写。",
            "先注册稳定Y42诊断，再完成32例等边际/实际提取对照、TRAIN8/init4原生错误掩码/协方差/保PSD反射、两次原生更新和精确保存恢复、配对资源、两份完整模块同解释器父AST/配置/七文件独立哈希。六项可行性门通过；初始损失+0.01830、错误对照同样学习及下一步数值差均保留，不冒充效果或必要性。",
            "Y42资源窗口22:27:55–22:29:53+08：新增398594参数；压力训练Y42/W2中位1.123867/0.912554秒、保留14.046875/13.400391GiB，时间回退约23.16%；低于冻结24GiB上限，不宣称训练后Pareto。一个GBK构建故障和三个测试适配器绑定失败原件保留，修复未改变模型或准入标准。",
            f"仅七个已测试新文件提交{commit['code_commit']}，未push/未改无关变更；哈希绑定六门收据和决策{args.decision}后走canonical GPU1准入派发，无重复preflight。",
            f"Y42启动独立核验窗口{launch['observed_start_local']}至{launch['observed_end_local']}：PID{launch['pid']}/GPU1/Train{captured[0]['epoch']}/30，步数{steps}/{captured[0]['total_steps']}，cwd/config/提交和源哈希匹配。Y41 GPU0原PID116906及源字节保持；启动时GPU1设备占用{launch['gpu1_device_memory_mib']}MiB/利用率{launch['gpu1_device_utilization']}%，仅该窗口观测。",
            "科学状态Y42 training_not_efficacy；活跃2/目标2，固定30轮原生训练继续，不等待全训练。第0–3段未知值不补零；未操作其他项目/server6/调度/新任务/后台。当前执行状态待finish；实际收据用量窗口和产出在完成后消息中补充，缺失记未知，缓存不重复相加，不当账单。",
        ],
        "decisions": [
            "Y39证据不充分、目标未达，继续保留W2；不把四舍五入CSV、相似均值或跨0区间当科学提升/等价/无害。",
            "Y42完整点协方差仅获可证伪固定30轮训练准入；保留不利初始损失、错误对照学习和资源回退，无效果/必要性/新颖性/人类审批声明，Phase2未授权。",
            "容量已达2，只保留原Y41 GPU0与新Y42 GPU1；终态后依冻结9508/141各两遍/FIRST/场景bootstrap/规模及对象分组/对任一基线伤害/重复噪声/训练后成本复核，不重启或微调。下一调度时刻未知。",
        ],
    }
    assert not (ROOT / notes_path).exists()
    (ROOT / notes_path).write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"memory_lines": len(memory.splitlines()), "backup": evidence(backup_path),
                      "notes": notes_path, "active": len(active), "scientific_status": candidate["scientific_status"],
                      "execution": "finish_pending"}))
finally:
    lock.close()
