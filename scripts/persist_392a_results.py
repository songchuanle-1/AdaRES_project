"""Persist real frozen outcomes and SAME Y45 evidence under the owned lease."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--stop", action="store_true")
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
now = datetime.now().astimezone()
remaining = (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds()
assert remaining > 0
assert not args.stop or remaining <= 180, "No fabricated deadline-near stop"


def read(stem):
    path = ROOT / ("pipeline/analyses/" + stem + "_" + args.token + ".json")
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["complete"]
    return path, value


def evidence(path):
    if not isinstance(path, Path):
        path = ROOT / path
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    terminal_reviews = []
    for name in ("y43", "y44"):
        comparison_path, comparison = read(name + "_clustered_392a")
        cost_path, cost = read(name + "_trained_cost_392a")
        terminal_path, terminal = read("y43_terminal_7a40" if name == "y43" else "y44_terminal_392a")
        cid = "AdaRES-" + name.upper()
        candidate = next(c for c in work["candidates"] if c["id"] == cid)
        adverse = comparison["frozen_harm_triggered_groups"]
        rejected = any(adverse.values())
        status = "rejected_realization_frozen_subgroup_harm" if rejected else "complete_inconclusive"
        review_path = ROOT / ("pipeline/analyses/" + cid + "_frozen_review_392a.md")
        if not review_path.exists():
            summary = comparison["summary"][name]
            best = terminal["checkpoints"][1]
            review = f"""# Experiment Result Analysis - {cid}
Observed locally {now.isoformat()}; original SAME training done_complete30.

## 1. Basic Info
Server2/singleGPU; resumed original run, last/scheduler30, best epoch{best["epoch"]}.
Original checkpoints immutable. Training start/end/duration clock relation unknown.
Terminal proof: {evidence(terminal_path)}.

## 2. Best Metrics
| Metric | Frozen FIRST | Population |
|---|---|---|
| mIoU | {summary["miou"]} | 9508 expressions/141 scenes |
| Acc@0.25 | {summary["acc25"]} | same |
| Acc@0.5 | {summary["acc50"]} | same |
| spmIoU | {summary["spmiou"]} | same |
Checkpoint metadata {best["best_miou"]} is not the frozen efficacy endpoint.

## 3. Convergence Summary
Fixed30 unchanged; best saved epoch{best["epoch"]}. FIRST is primary, not best repeat.
Second mIoU {comparison["repeated_miou"][name]}.
Whole-scene bootstrap10000seed1999; TRAIN cuts1487/3325 unchanged.
Full comparison/noise/subgroup evidence: {evidence(comparison_path)}.

## 4. Comparison
W2: {json.dumps(comparison["versus_w2"], indent=2)}
B1: {json.dumps(comparison["versus_b1"], indent=2)}
Frozen harm triggers versus either: {json.dumps(adverse)}.
Trained paired costs: {json.dumps(cost["summary"], indent=2)}.
Parameters {json.dumps(cost["parameters"])};128 queries/singleargmax.
Costs use2warmups/3reads, TRAIN8 and fixed largest pair; stress updates memory-only.
Order/hardware effects and descriptive p95 remain limitations, no Pareto superiority.

## 5. Conclusion
Scientific status {status}; target0.515 not reached. W2 retained, no promotion.
{("Small-target subgroup harm versus W2 disconfirms this admitted realization under its original freeze; no restart/retune/renamed successor." if rejected else "Intervals cross zero; no benefit, equivalence or no-harm conclusion; do not retune or restart.")}
Official unique/multiple metadata unknown; single-object9508/141 is not a substitute.
One seed/checkpoint; full-input repeat instability is retained, not explained away.
Y44 reused the current-tick W2/B1 twice-EACH captures; no extra reference passes counted.
"""
            review_path.write_text(review, encoding="utf-8")
        items = [evidence(comparison_path), evidence(cost_path), evidence(terminal_path), evidence(review_path)]
        candidate.update(stage="rejected" if rejected else "complete",
                         scientific_status=status,
                         next_action="Preserve this frozen terminal outcome and immutable checkpoints; no promotion, restart, retuning or renamed realization.")
        if rejected:
            candidate["rejection_reason"] = "Frozen small-target scene CI upper<0 versus W2, actual trained harm; original material-harm rule retained."
        for item in items:
            if item not in candidate["evidence"]:
                candidate["evidence"].append(item)
        candidate.setdefault("history", []).append(dict(
            observed_at=now.isoformat(), action="frozen_terminal_outcome_persisted", evidence=items))
        terminal_reviews.append(evidence(review_path))
    y45 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y45")
    stems = [
        "y45_native_targets_392a", "y45_decomposition_controls_392a",
        "y45_local_bundle_392a", "y45_source_bundle_392a", "y45_audit_repair_392a",
        "y45_package_cpu_392a_v2", "y45_native_path_392a_v2",
        "y45_physical_native_392a", "y45_resources_392a", "y45_resources_binding_392a",
        "y45_initial_learning_392a"]
    actual = []
    pending = []
    for stem in stems:
        path = ROOT / ("pipeline/analyses/" + stem + "_" + args.token + ".json")
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            assert result["complete"]
            actual.append(evidence(path))
        else:
            pending.append(stem)
    for path in [
            "pipeline/designs/exp_design_Y45_392a.md",
            "pipeline/reviews/motivation_review_Y45_392a.md",
            "pipeline/analyses/Y45_native_capture_prereg_392a.md",
            "pipeline/analyses/Y45_gate_review_392a.md",
            "scripts/probe_y45_package_cpu_392a.py",
            "scripts/probe_y45_native_path_392a.py",
            "pipeline/analyses/y45_package_cpu_392a_" + args.token + ".log",
            "pipeline/analyses/y45_native_path_392a_" + args.token + ".log"]:
        actual.append(evidence(path))
    for item in actual:
        if item not in y45["evidence"]:
            y45["evidence"].append(item)
    y45.update(stage="validation", scientific_status="independent_native_validation_not_admitted",
               next_action="Measure separate legacy matched-indicator and new categorical-likelihood gradients on the same cached prediction-blind native TRAIN cohort; compare signed inner products under correct/cyclic-wrong/collapsed targets and repeats to explain trained legacy loss regression. No balancing-weight tuning; gate review/commit/admission only after interpretation.",
               pending_validation_outputs=pending, admission_receipt_issued=False)
    y45.setdefault("history", []).append(dict(
        observed_at=now.isoformat(), action="native_and_independent_package_progress",
        evidence=actual, limitations="Full-input exact-repeat assertion failed; cached native-input controls pass. No global bitwise, efficacy or novelty claim."))
    if args.stop:
        work["design_cycle"] = dict(
            stop_reason="tick_deadline_near", attempted_ids=["AdaRES-Y43", "AdaRES-Y44", "AdaRES-Y45"],
            evidence=terminal_reviews + actual,
            resume_condition="Continue SAME Y45 validation/admission review from hashed real outputs on live-idle authorized GPU1; preserve GPU0 PID19723 and frozen Y43/Y44 outcomes.",
            observed_at=now.isoformat(), remaining_seconds=remaining)
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    memory = f"""# AdaRES_project Research Memory
## Goal And Phase
- Current token {args.token}; actual owning thread {owner["thread_id"]}; design snapshot2026-10-14 12:50:49.868453+08, immutable deadline13:50:49.868453+08.
- ScanRefer target.515 unmet; W2 retained scientific comparator; roundedCSVY36.5110 is not promotion.
- Only AdaRES/server2GPUs0,1; one job/card; enabled closure; background scheduler Disabled. No server6/other project/new chat/nested Codex/scheduler changes.
- Released SPFormer/seed1999/fixed30/native128/singleargmax/optimizer/scheduler unchanged; no validation thresholds/temperatures/features/lossweight/epoch search.
## Champion And Baseline
- SAME Y43 and Y44 shutdown resumes verified on2026-10-13 20:06+08; model/optimizer/scheduler preserved, not fresh restarts.
- Y43 canonicaldone_complete30/best17; frozen FIRST.5083255213/W2.5101079952/B1.5022218180. Overall W2 CI[-.01003174,.00638104].
- Y43 small2816expressions/124scenes delta-.01677876/CI[-.03184727,-.00175832]; frozen harm triggered; realized candidate rejected, no promotion/retune/restart.
- Y44 canonicaldone_complete30/best18; FIRST.5102523338/W2.5101079952; delta.00014434/CI[-.00743152,.00761913]. Complete_inconclusive, no promotion/no-harm claim.
- Each method full9508/141/twice, FIRST primary; W2/B1 captures reused for Y44 with explicit source hash, not new passes.
- Reviews: pipeline/analyses/AdaRES-Y43_frozen_review_392a.md; AdaRES-Y44_frozen_review_392a.md. Trained costs/empirical repeats retained; official unique/multiple unknown.
## Active Hypotheses
- SAME Y45 metric-confusion-decomposition; initial scalar-confidence premise disconfirmed/preserved. Actual parent final selector matched CE.
- Native prediction-blind TRAIN8/32expressions/128queries: point TP/FP/FN exact;29nonidentical wrong truths separate; same labels contain multiple partitions.
- Actual historical B2/G1/C1 CPU32 equal-IoU loss/gradient gaps0; categorical FP/FN gradient gap>=.002604167. Information distinction, not utility/novelty.
- Independent six-file W2 package/config adares_y45_metric_decomposition; query/FG/BG point mass predicts3categories, final metric-odds residual and TRAIN-only categorical likelihood.
- Source/config byte identity remote; whole-parent AST stripping exact additions; inherited dimensions/no new numeric knobs/no GT inference.
- Native cached controls TRAIN8:4new tensor gradients finite/nonzero min.14382629; wrong-mask indicator response>=.32885742; literal target error0; disabled physical W2 equal.
- Released native two optimizer steps reach/update all4 tensors; model/AdamW/PolyLR restoration exactly equal before continuation. Nextstepmodelmax1.5258789e-5/optmax1.6596914e-8 retained, not bitwise trajectory.
- Resource/initial-learning results, if present, are hashed in research_work; pending outputs: {json.dumps(pending)}. No admission receipt, commit or training launch yet.
## Closed Or Failed Directions
- Y43 realized subgroup harm rejected; Y44/Y42/Y41/Y39 complete_inconclusive; prior Y29 renamed ranking/Y40/readout overlaps remain closed.
- Y45 first AST audit failed Python3.8 ast.Index handling; raw local/native byte differences but all six source ASTs equivalent. V1 source/log retained; v2 audit only, model unchanged.
- Y45 full-input repeat assertion failed; raw v1 source/log retained. V2 fixes matched-control scope to actual cached decoder inputs/RNG, not a widened tolerance or global determinism claim.
- No historical checkpoint/model/protocol/gate/failed evidence overwritten.
## Running And Queue
- Begin snapshot active1/runningY44/deathwatch1/claim0/wait0/budget1; do not relabel those preflight facts as final.
- Y44 verified last/scheduler30, PID15520absent; canonical finalized13:05:50+08, full resumed log archived with SHAa236e35d1f5ef130007f417aad404eea9c5edfe803cde02aa026cbdeae5a77b6.
- After terminal reconciliation queueactive0; no new training/admission. GPU0 external PID19723 preserved; GPU1 exclusively leased for bounded diagnostics only, current free status needs live check.
- Training checkpoints remain remote: Y43bestSHA5738e63c1180d3cedb7c467d6be901e3becd95cabb07815293580d8c81dd7c28; Y44bestSHA369d761e24ebc593799247b28d43cc12e6652bf4bdda7d5c4ae33e80812275fe.
## Open Questions And Risks
- No trained Y45 efficacy, population learnability, necessity, novelty, upstream blindness or human Phase2 approval.
- Raw legacy native loss increased in7/8 initial scenes. Fixed2 new-only TRAIN8/released-init4 correct and wrong controls alllearn newloss; trained legacy correct mean+.00027923 remains OPEN, not a PASS.
- Single seed/checkpoint and full-input repeat instability limit conclusions; wrong controls learning similarly does not establish necessity.
- Current full native loader/RNG trajectory is not bitwise reproducible; exact restored checkpoint state is a distinct observation.
## Last Tick
- This tick continued due candidates; original resume task fully handled. Y43 harm/Y44 inconclusive persisted; SAME Y45 independent/native diagnosis advanced.
- Execution finish not yet verified; claim COMPLETED only after explicit PASS/COMPLETE and independently absent owned lease.
- Stop disposition {("tick_deadline_near backed by actual immutable deadline" if args.stop else "work ongoing; no early-exhaustion or resource-block claim")}; next action SAME Y45 validation/admission review.
- Usage only begin-to-receipt local log window, excludes initial reads/final reply/late records; cache inside input, reasoning inside output; unavailable unknown, not a bill.
- Next saved automation time unknown; no schedule operations.
"""
    assert len(memory.splitlines()) < 120
    (ROOT / "pipeline/codex_memory.md").write_text(memory, encoding="utf-8")
    notes = dict(actions=[
        "本轮为 design；0-3 段仍是北京时间12:50:49的 preflight 快照，不代表以下后续状态。",
        "已核实并规范归档同一Y44固定30轮终态，13:05:50队列归档完成；关机前Y43/Y44恢复请求已完成，未从头训练。现项目队列活跃0，未新增训练。",
        "Y43/Y44各两次9508表达/141场景全量评估；FIRST主结果分别0.50832552/0.51025233。Y44复用本轮W2/B1各两次参照，不重复计算为新推理。",
        "Y43小目标相对W2差异-0.01677876，场景95%区间[-0.03184727,-0.00175832]，按原冻结规则拒绝该实现；Y44整体区间跨零，科学状态complete_inconclusive。保留W2与目标0.515。",
        "真实训练权重的成对时间/显存压力测试均已完成，代价与重复波动全部保留，不宣称Pareto优势。",
        "同一Y45完成真实TRAIN8/32表达点级目标取证、旧目标CPU对照、独立实现、字节一致性、CPU/原生匹配对照、四张量梯度和原生两步/优化器/调度器恢复；新增参数197635。",
        "Y45首版AST解析与整输入逐位重复失败均保留。修复仅涉及检查器Python3.8下标解析及固定原生输入对照；不宣称整流水线逐位重复，不降低误差阈值。",
        "Y45已完成输出："+", ".join(stem for stem in stems if stem not in pending)+"；缺失输出："+(", ".join(pending) if pending else "无")+"。尚未签发六门槛准入收据、提交测试包或启动训练。",
        f"北京时间{now.isoformat()}持有本轮租约；剩余{remaining:.1f}秒。执行状态postflight待验证，科学状态Y43 rejected/Y44 complete_inconclusive/Y45 validation。收据用量窗口和实际计数待finish后补充，缺失值未知。"],
        decisions=[
            "Y43不晋升、不调参重训，按原小目标伤害证据拒绝该实现；Y44保留证据不充分结论，W2仍为科学对照。",
            "仅推进同一Y45；保留原生负面结果、错配控制、失败脚本/日志及下一步准入审查，不以空闲GPU放宽门槛。",
            ("依据实际deadline在180秒内，以tick_deadline_near保留未完成候选，下一轮从真实哈希证据续做；不是路线耗尽或资源阻塞。" if args.stop else "尚有可执行工作，继续本轮真实测试；不声称完成或提前耗尽。")])
    notes_path = ROOT / ("pipeline/reports/notes_392a_" + args.token + ".json")
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(notes=str(notes_path), stop=args.stop, remaining_seconds=remaining,
                         scientific_status={"Y43": "rejected", "Y44": "complete_inconclusive", "Y45": "validation"},
                         pending=pending), ensure_ascii=False))
finally:
    writer.close()
