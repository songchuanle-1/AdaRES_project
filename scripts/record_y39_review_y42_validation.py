"""Close measured Y39 scientifically and record tested same-ID Y42 progress."""

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
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor


def ref(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def result(name):
    path = "pipeline/analyses/" + name + "_" + args.token + ".json"
    value = json.loads((ROOT / path).read_text())
    assert value["complete"]
    return path, value


terminal_path, terminal = result("y39_terminal_current")
analysis_path, analysis = result("y39_clustered_comparison")
cost_path, cost = result("y39_trained_cost_current")
objects_path, objects = result("y39_object_groups")
full_path, full = result("y39_frozen_full_current")
assert not analysis["target_reached"]
assert all(not v["frozen_material_harm_ci_upper_below_zero"]
           for values in (analysis["versus_w2"], analysis["versus_b1"])
           for v in values.values())
review_path = "pipeline/analyses/AdaRES-Y39_frozen_review_062ae665.md"
review = ROOT / review_path
assert not review.exists()
summary = analysis["summary"]
pair = analysis["versus_w2"]["all"]
best = next(v for v in terminal["checkpoints"] if v["path"].endswith("/best.pth"))
text = (
    "# Experiment Result Analysis: AdaRES-Y39\n\n## 1. Basic Info\n"
    "Canonical done_complete; original PID102267 absent before/after read; "
    "GPU1. Config adares_y39_point_distribution_res.yaml. Original start/exit/"
    "duration unknown; local frozen evaluation21:59:12.860011->22:10:35.313285+08 "
    "on2026-10-11. No original checkpoint overwrite/resume.\n"
    f"Last epoch30, best epoch{best['epoch']}, bestSHA{best['sha256']}.\n\n"
    "## 2. Best Metrics\n"
    "| Metric | FIRST frozen full | Source epoch |\n|---|---|---|\n"
    f"| mIoU | {summary['y39']['miou']} | {best['epoch']} |\n"
    f"| Acc25 | {summary['y39']['acc25']} | {best['epoch']} |\n"
    f"| Acc50 | {summary['y39']['acc50']} | {best['epoch']} |\n"
    f"| spmIoU | {summary['y39']['spmiou']} | {best['epoch']} |\n"
    "9508 expressions/141 scenes/twice EACH Y39/W2/B1/FIRST primary/"
    "128 native queries/one argmax; do not substitute original rounded best.\n\n"
    "## 3. Convergence Summary\n"
    f"Original best metadata{best['best_miou']} at23/30; original last30 and "
    "model/optimizer/scheduler verified. This does not prove optimal epoch or "
    "convergence; no validation epoch selection or retune.\n\n"
    "## 4. Comparison\n"
    f"FIRST W2{summary['w2']['miou']}/B1{summary['b1']['miou']}; "
    f"Y39-W2{pair['paired_miou_difference']}, CI{pair['scene_clustered_ci95']}. "
    f"Y39-B1{analysis['versus_b1']['all']['paired_miou_difference']}, "
    f"CI{analysis['versus_b1']['all']['scene_clustered_ci95']}.\n"
    "Whole-scene10000 bootstrap/seed1999/TRAIN cuts1487/3325. Every observed "
    "all/size interval crosses0; no predeclared upper<0 harm trigger, not no-harm/"
    "equivalence. Small-versus-W2 point estimate is adverse(-.0147890).\n"
    "Single-object population9508/141; multi-object0, no CI or safety inference. "
    "Official unique/multiple metadata unknown, not substituted with object count.\n"
    "Repeat maximum expression difference Y39/W2/B1=1/.0299056/.9738956; "
    "nearly equal aggregate means do not imply reliable expression-level repeatability.\n"
    f"Trained stress median Y39/W2={cost['summary']['candidate']['native_train_stress']['median_seconds']}/"
    f"{cost['summary']['parent']['native_train_stress']['median_seconds']}s; reserved="
    f"{cost['summary']['candidate']['native_train_stress']['peak_reserved_gib']}/"
    f"{cost['summary']['parent']['native_train_stress']['peak_reserved_gib']}GiB; "
    "params164901255/164629637. Serial frozen TRAIN8+maxpair728/6780, "
    "two warmups/three readings; temporary updates in memory only. No Pareto superiority.\n\n"
    "## 5. Conclusion\n"
    "SAME stableY39 complete_inconclusive. Target.515 unmet; retain W2, no "
    "promotion/resume/retune/renamed successor. All initial negative ceiling/"
    "adverse losses/incorrect-control learning and repeat limitations remain.\n\n"
    "## 6. Crash Info\n"
    "Not a canonical crash. Bounded terminal tail had no traceback/OOM; "
    "not whole-log proof. Y41 PID116906/GPU0 and source retained through probes.\n"
    f"Hashed sources: {json.dumps([ref(p) for p in (terminal_path, full_path, analysis_path, objects_path, cost_path)])}\n"
)
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    review.write_text(text, encoding="utf-8", newline="\n")
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    y39 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y39")
    assert y39["stage"] == "training"
    y39.update(stage="complete", scientific_status="complete_inconclusive",
               result="Frozen FIRST .5077059832 versus W2 .5100098499; "
                      "paired CI crosses0, target unmet, no superiority/Pareto.",
               next_action="Retain W2 and original immutable Y39 evidence; "
                           "no resume/retune. Revisit only new evidence, not rounded CSV.")
    for p in (terminal_path, full_path, analysis_path, objects_path, cost_path, review_path):
        y39["evidence"].append(ref(p))
    y42 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y42")
    assert y42["stage"] == "diagnosis"
    y42.update(stage="validation", scientific_status="native_validation_not_efficacy",
               next_action="Execute original native two-step/resume, adverse "
                           "initial-loss investigation and paired resources "
                           "before any receipt/commit/dispatch.")
    for name in ("y42_covariance_cpu", "y42_extract_readout_cpu", "y42_source_bundle",
                 "y42_native_controls_v2", "y42_parent_semantics_v3",
                 "y42_probe_binding_repair", "y42_parent_ast_binding_repair"):
        p, _ = result(name)
        y42["evidence"].append(ref(p))
    for p in (
        "pipeline/analyses/Y42_build_encoding_failure_062ae665.md",
        "pipeline/analyses/Y42_resource_freeze_062ae665.md",
        "pipeline/analyses/Y42_native_control_amendment_062ae665.md",
    ):
        y42["evidence"].append(ref(p))
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"y39": y39["scientific_status"], "y42": y42["stage"],
                      "review": ref(review_path)}))
finally:
    lock.close()
