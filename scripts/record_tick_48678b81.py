"""Save the compact seven-section memory and real research notes under lease."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "48678b818b134bea9d9c3c3e1f3d3305"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

memory = """# AdaRES_project Research Memory
## Goal And Phase
- Tick48678b818b134bea9d9c3c3e1f3d3305 DESIGN; actualenv owner01a07e32-78bf-7670-925b-370911ea48db verified. Snapshot2026-10-10 19:48:21.288800+08/deadline20:48:21.288800+08 unchanged. BeginONCE/compactFIRST/monitorRC0/goalRC0/no repeatedpreflight.
- ScanRefertarget.515; automatic CSVroundedchampionY36.5110/gap.004, not scientificpromotion. Frozen scientificcomparatorW2 retained. Phase1unmet/Phase2unapproved; goalSHA53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966 unchanged.
- HumanGPU0/designgrant; server2allowed[0,1]/onejobperGPU/singleGPU/server6[]. Closureenabled/WindowsPipelineDisabled; no otherproject/newtask/schedulerchange/background/nestedcodex/password/sudo/server4formerIP.
- ReleasedSPFormer/native128/oneargmax/seed1999/fixed30/optimizer/lossweights unchanged; no validationthreshold/temperature/feature-subset/neighborhood/epochsearch or gate relaxation.

## Champion And Baseline
- Prior1fb792ef frozen FIRSTfullY36/W2/B1=.5107166242917184/.5100098499225277/.5021124174472842, each9508expressions/141scenes/twice/FIRSTprimary. Targetunmet; no repeated GPU evaluation this tick.
- Y36-W2+.000706774369/CI[-.007218154973,.008880424700]; Y36-B1+.008604206844/CI[.000486970108,.016646870070]. W2superiorityunresolved; B1positiveonlyonefixedcheckpoint/seed/split, notcausalnecessity.
- SmallvsW2-.008495133019/CI[-.024612470092,.007851595067]. All6subgroupCIs cross0, no CIupper<0 harmtrigger, not equivalence/noharm. Scene10000bootstrapseed1999/TRAINcuts1487/3325 unchanged.
- PriorY36trainedcostparams165089671/W2164629637; stressreserved14.0390625/13.400390625GiB. Repeatnoise/checkpoint/sourcepostaudit retained in pipeline/analyses/AdaRES-Y36_frozen_review_1fb792ef.md SHA6fc28819d50441d649526dff199102554b2012f079538409addf43cc94686571.
- SAME Y36 canonicaldone_complete/30of30 confirmed bybegin, queueprunednormally. ScientificstageCOMPLETE/complete_inconclusive closed underlease; queue/CSV hashesunchanged byclosurehelper. No promotion/restart/retune.

## Active Hypotheses
- SAME Y37 TRAINING final-mask-global-pair-distance-distribution; originalPID20348/GPU0/configadares_y37_pair_distribution_res/code67f4a1b2. Snapshot10/30/roundedbest.4926. IndependentY39launchcapture20:22:02->20:22:07 verifiedoriginalPID/cwd/config/GPU0/sevenownsourcehashesunchanged. No wholetrainedefficacy yet.
- SAME Y39 within-superpoint-feature-distribution-confidence registereddiagnosis BEFOREbroadreview; prereg5ea039ce/realizationfreeze_48678b81 unchanged. Native all32 prepoolcenteredpopulationfeaturevariance/log1p/shared32->256->256/maskpointmassFG-BGandmidpoint/finalquery/768->256->2 finalindicatoronly;8tensors/271618params.
- No mask/auxloss/querycount/sampling/schedule/oneargmax changes. New masksdetached inaddedreadout; originalnativegradpathsretained. Native pointvariance replicated across expressions beforeexpand_and_fps, no GT/inferencesubset.
- CPU32 equalmeans/counts/geometry/W2state heldfixed: variancecontrast.25/W2delta0/repeat0/rawpointalgebra max1.192092896e-7. Actualreadout32 repeat/padding0/perm5.960464478e-8/wrongvariance>=.001359522343/wrongmask>=.001102492213/all8nonzerofinitegrad.
- NativeTRAIN64/256 exactoptimalwhole-SPIoUceiling:29nonzerogaps/mean.999771170370/min.989761092150/median1. NEGATIVE majorquantization-bottleneck diagnosis; heterogeneityassociation and learnedmeansindirectencoding unproved. CPUexhaustive256solverchecks maxdiff0. No arbitrarypretrain-effectsize veto.
- NativeTRAIN8/init4 controls ad61cc51:8gradients/min1.403968872e-5/wrongmask>=9.284168482e-5/wrongvariance>=.000209212303/perm8.940696716e-8; repeat/nonindicatorloss/maskdelta/padding0. Native all32variance nonzero; detachedmaskgradient and variancegradient verified.
- InitialmeanlossW2+.001077540219/releasedinit-.015290141106. Fixednew8-only2AdamWsteps1c0b0927: correct/rolled/wrongvariance trainedchanges-.001247614622/-.001245014369/-.001243680716; releasedinit-.038121938705/-.038123607635/-.038033723831.5of8correcttrainindividualsrise; allcontrolslearn, notnecessity/generalization. No diagnosticweightsdeployed.
- Physicalnative1581c333 tools.train2steps/all8updated/408finitedecodergradtensors; exactmodel/AdamW/PolyLRrestore beforecontinuation. Nextlossdiff0/modelmax1.52587890625e-5/optimizer9.332779882e-9, notbitwisewholetrajectory. DisabledY39/physicalW2 masks/indicators/sampling0 on4cachedscenes.
- Resources2716efaa params164901255/164629637(+.1649873%); stressmedian.9439128898/.9181046952s/peakreserved13.513671875/13.40234375GiB; infermedian.0930267386/.1022411818s/p95.2231962607/.2228620847s/reserved1.8203125/1.7578125GiB. Serial2warmups3reads, descriptive/initializedfeasibility nottrainedPareto/speedup.
- Motive6sections/design7+smell/review persisted beforeadmission. Reviewconsolidates prospectivefreeze, NOT backdatedprereg. Nativeadverse/near1ceiling/correct-and-wronglearnability allretained. No novelty/humanapproval/efficacy/noharm claim.
- ONLYsevenY39files committed79a23936d5bd5980fa337388131762078f52fa07; originpushFALSE. Candidate3CRLF->LFonly/ASTsame/git-local-remoteexact0b184b91; freshCPUreadout/extract20:19:40->20:19:48 identicala9012134/d54538d5. Otherdirtyfilesuntouched.
- Sixrealgates/decision20261010-202011_next_batch/receipt0ee8ada0/canonicalGPU1dispatch20:21:10/PID102267/unique20261010_202034log. Independent846b6174 at20:22:02->20:22:07 Train1/30steps102->110/4686, codehash/cwd/config/GPU1 verified, originalY37preserved.
- SAME Y39stageTRAINING/scientifictraining_not_efficacy;39hashrefs. Frozenposttraining full9508/141 EACHY39/W2/B1twice/FIRSTprimary/scene10000bootstrap1999/TRAINcuts1487/3325/subgroupCIupper<0harmvseither/repeatnoise/trainedpairedcost. Promoteonly.515+frozenharm/Pareto review.

## Closed Or Failed Directions
- SAME Y38 registeredstablefirst thenREJECTED: actualexistingY30QueryMaskTrajectoryReadout consumes proposedpoint-masshistory/GRU. CPU32 equalfinalquery/mask changedhistory: W2delta0/Y30min.005445152521/repeat0/permutation0. Evidence5a545e0c. Tested missing-path realizationcovered, notbanallhistory or renamedY30admission.
- Y36nowcomplete_inconclusive, Y35priorcomplete_inconclusive retained. Y33smallharm/Y30mediumharm/Y28smallharm/Y34contextoverlap/Y32ASTduplicate/Y31inconclusive and earlierclosedrecords unchanged. No full-historyreview/repeatedfullmeasurement.
- FirstY39CPUWelford.var strictpermutationassert failed/rawlogretained. V2usescenteredsquaremean, samefixtures/strictzero/no tolerance change, b7e62db3. Actualextract_featCPU pathpermutation latermax1.192092896e-7 underpredeclarednative1e-5 bound.
- Additiveupload firstlocalGBKdecodeerror BEFOREremoteoperations; explicitUTF8fixed. Resourceoutput stalewordtrajectory is documentedwordingdefect; actualresolvedpackagepointvariance, originalresultnotrewritten.

## Running And Queue
- BEGINsnapshot19:48:21active1/running1/claiming0/waiting0/target2/budget1/due0/newterminalY36. DeterministicGPUutil/memory/process/tablemetricsUNKNOWN. RawmonitorY37GPU0 10/30/.4926/noimprove0of30.
- Laterindependent20:22:07 active2/running2/claiming0/waiting0/Y37GPU0PID20348/Y39GPU1PID102267, bothoriginalfixed30 preserved. GPU1device16312MiB/31% atcaptureONLY, notisolatedpeak/futureidle; GPU0utilmemoryUNKNOWN.
- Admissiontargetmet; no extraadmission/test to filltime. Futureonlycanonicalterminal/crashevents/frozencomparisons, no blindrestart. NextscheduledtimeUNKNOWN; heartbeatconfigurationunchanged.

## Open Questions And Risks
- Within-SP featurevariance notprovennecessary/useful; meanfeature/backbone canindirectlyencodeheterogeneity. Nearlyperfectwhole-SPceiling isnegativeevidence, nottargetdeficit explanation. Initializedgrad/control/Pareto screens don'testablishtrainedbenefit.
- Masksdetachedonlynewbranch; addedvariance gradients reachrawbackbonefeatures. CUDA scatter/sparse continuation and onefixedseed remain. Correct/wronginputscanalllearn and5of8lossesrise; preservealloutcomes.
- Frozenprotocol/goal unchanged; no emptyliteraturesearch novelty claim. RemoteclockrelationtoLocalUNKNOWN: use runnerlocalwindows. Authorizedrunnerb1f4e6b818cacbae148abef2d5ac0d605f6ace5e13e80dba48003539581dcd50 unchanged.
- Previousmemory archivedverbatim at pipeline/analyses/codex_memory_before_48678b81.md; alloldhistoricalevidence preserved, no no-op decision log.

## Last Tick
- Previouse367eb89 EVENT finishPASS/COMPLETE/contextcommit18:50:12.365887, independent18:50:31.1818775ownedlockabsent. ReceiptFINALIZED/local18:47:59.336296->18:50:12.365887/6calls/input898946(cache889216included)/output3531(reasoning914included)/newcandidates0/changedrefs0/newtraining0/complete0.
- Current48678b81 DESIGN: canonicalY36complete/scientificinconclusive reusedoldreview; stableY38 testedrejected; stableY39 independentdiagnosis/nativecontrols/backward/resume/resources/adverseinvestigation/review/only7filecommit/exactbytes/sixgates/admitted/verifiedGPU1training. Targetcapacity2met, preserveY37.
- Sevensectionmemory/reportnotes savedUNDERownedlease. Currentfinish/receiptPENDING atwriting; requireactualCLOSURE_POSTFLIGHT_PASS/COMPLETE plusindependentownedlockabsence before tickCOMPLETED. No postreleasewrites.
- Receiptusagebegin-to-recordedwindowonly/excludesearlierreads/lateusage/finalreply/notproviderbill; cacheincludedinput/reasoningincludedoutput/notaccumulated. MissingfieldsUNKNOWN; actualreceiptcountsreportedafterfinishinchat.
"""
notes = dict(
    actions=[
        "执行状态：本报告保存时 finish 尚待验证，不能预称完成。19:48:21+08 快照为 design；begin/compact 各一次，监控与目标检查返回码均为0，无重复 preflight。",
        "同一 Y36 经本轮正常终态确认，复用原冻结9508/141、三模型各两遍 FIRST 主结果及区间/成本/审计，科学状态 complete_inconclusive；没有重跑评估或晋升。完整 mIoU=0.510716624292，对W2差+0.000706774369但95%区间跨0，目标0.515仍未达。",
        "先登记稳定 Y38，再以实际 Y30 类执行32组等最终状态历史对照：W2差0、已有Y30最小响应0.005445152521，重复/置换差0。拒绝重复缺失路径实现，未重命名旧失败路线。",
        "稳定 Y39 的超点内全32通道特征方差路径完成CPU等均值/计数/几何对照及原生TRAIN64/256诊断。29条有量化误差但平均最优整超点上限0.999771，不能宣称量化是主要瓶颈。首个CPU严格置换失败日志保留，中心矩实现修复未放宽容差。",
        "实际 Y39 完成TRAIN8/init4错误掩码/方差关联、重复、padding、梯度及两步原生AdamW/恢复/物理W2关闭对照。8新张量均有限非零并更新；关闭后掩码、指标、采样差0；恢复后下一步参数数值差约1.53e-5，不称全轨迹逐位复现。",
        "负面结果保留：W2初始化平均损失+0.00107754；固定新路径两步修正平均-0.00124761但5/8场景损失上升，错误对照也能学习。参数+271618约0.165%，压力训练预留13.514/13.402GiB、耗时0.944/0.918秒；仅可执行性，不是必要性、训练效果或Pareto。",
        "只提交7个Y39文件79a23936d5bd5980fa337388131762078f52fa07，3个自身文件仅CRLF转LF，Git/本地/远端字节一致及新CPU复验通过。六项真实门槛、冻结终态评估计划及自主审查存证，没有声称用户逐项审批或origin push。",
        "20:21:10+08 canonical GPU1 调度 Y39，PID102267；20:22:02至20:22:07独立证明 Train[1/30]步数102增至110/4686。原Y37 PID20348/GPU0/源文件完全保留。此时active=2达到目标，Y39科学状态training，训练效果未知；0至3段仍是19:48快照，非收尾实时表。",
        "收据用量和实际产出在 finish 后读真实收据再展示；当前未知。仅begin到收据记录窗口，缓存包含在输入、推理包含在输出，不重复加总或当账单。下一次计划时间未读取，未知；未操作调度/其他项目/后台或新任务。",
    ],
    decisions=[
        "Y36完成科学收尾但不宣称优于W2或无伤害；自动目标检查的rounded champion Y36=0.5110不是科学晋升。Phase1未达，Phase2未授权。",
        "Y38真实对照拒绝重复实现；Y39仅准入可证伪训练，唯一新准入额度已用完。近1量化上限、初始损失上升、错误对照可学及数值噪声均作为风险保留，未放宽门槛。",
        "保持 Y37/GPU0 与 Y39/GPU1 原始固定30轮、seed1999、发布SPFormer、原生128查询/单argmax/损失/优化器；禁止盲目重启、调参、变体凑数或验证集搜索。",
        "Y39终态后按既定9508/141候选/W2/B1各两遍 FIRST、整场景bootstrap10000/1999、TRAIN切分1487/3325、子组CI上界<0伤害规则、重复噪声和训练后配对成本评审。区间跨0为未决，只有目标0.515及冻结伤害/Pareto审查通过才可晋升。",
    ],
)
assert len(memory.splitlines()) < 120
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = json.loads((ROOT / "pipeline/research_work.json").read_text(encoding="utf-8"))
    assert next(c for c in work["candidates"] if c["id"] == "AdaRES-Y36")["stage"] == "complete"
    assert next(c for c in work["candidates"] if c["id"] == "AdaRES-Y38")["stage"] == "rejected"
    assert next(c for c in work["candidates"] if c["id"] == "AdaRES-Y39")["stage"] == "training"
    previous = ROOT / "pipeline/codex_memory.md"
    archive = ROOT / "pipeline/analyses/codex_memory_before_48678b81.md"
    assert not archive.exists()
    shutil.copyfile(previous, archive)
    previous.write_text(memory, encoding="utf-8", newline="\n")
    notes_path = ROOT / "pipeline/analyses/tick_notes_48678b81.json"
    assert not notes_path.exists()
    notes_path.write_text(json.dumps(notes, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(dict(memory_lines=len(memory.splitlines()), notes=str(notes_path),
                          previous_memory_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                          observed_at_local=datetime.now().astimezone().isoformat())))
finally:
    lock.close()
