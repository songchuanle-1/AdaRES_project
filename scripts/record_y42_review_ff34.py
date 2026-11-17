"""Same Y42 terminal scientific disposition from actual frozen evidence."""
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
BASE = "pipeline/analyses/"


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


names = ("y42_terminal_ff34", "y42_frozen_full_ff34", "y42_clustered_comparison",
         "y42_object_groups", "y42_trained_cost_ff34")
refs = [BASE + n + "_" + args.token + ".json" for n in names]
values = [json.loads((ROOT / p).read_text(encoding="utf-8")) for p in refs]
assert all(v["complete"] for v in values)
terminal, full, analysis, objects, cost = values
assert not analysis["target_reached"]
assert all(not row["frozen_material_harm_ci_upper_below_zero"]
           for ref in ("versus_w2", "versus_b1") for row in analysis[ref].values())
assert analysis["versus_w2"]["all"]["scene_clustered_ci95"][0] < 0 < analysis["versus_w2"]["all"]["scene_clustered_ci95"][1]
assert full["summaries"]["y42"]["miou"] == analysis["summary"]["y42"]["miou"]
best = next(c for c in terminal["checkpoints"] if c["path"].endswith("/best.pth"))
metrics = analysis["summary"]["y42"]
review = BASE + "AdaRES-Y42_frozen_review_ff347697.md"
assert not (ROOT / review).exists()
text = f"""# Experiment Result Analysis: AdaRES-Y42
Observed {datetime.now().astimezone().isoformat()}; same stableID.

## 1. Basic Info
Configconfigs_song/model_paper5/adares_y42_point_covariance_res.yaml.
Server2/GPU1/originalPID91215 absent/no matching original config process.
Canonical done_complete; actual training start/exit/duration unknown.
Last30/best{best['epoch']}; bestSHA{best['sha256']}; original last/best
read-only, model/optimizer/scheduler intact. Canonical finished_at is monitor
observation, not verified remote process exit time.
Full9508expressions/141scenes, Y42/W2/B1 twiceEACH/FIRSTprimary/native128/
argmax1. Capturewindow05:59:43.541918->06:11:03.674806+08 on2026-10-13.

## 2. Best Metrics
| Metric | FIRST Frozen Full | Checkpoint Epoch |
|---|---|---|
| mIoU | {metrics['miou']} | {best['epoch']} |
| Acc25 | {metrics['acc25']} | {best['epoch']} |
| Acc50 | {metrics['acc50']} | {best['epoch']} |
| spmIoU | {metrics['spmiou']} | {best['epoch']} |
Original best metadata{best['best_miou']} is not a replacement for frozen full.

## 3. Convergence Summary
Fixed30 completed; best first saved21. Bounded final30log rounded mIoU.500.
No optimal-epoch, convergence or retuning claim. No restart or resumed training.

## 4. Comparison
FIRSTW2{analysis['summary']['w2']['miou']}/B1{analysis['summary']['b1']['miou']}.
Y42-W2{analysis['versus_w2']['all']['paired_miou_difference']}/scene95CI
{analysis['versus_w2']['all']['scene_clustered_ci95']}; versusB1
{analysis['versus_b1']['all']['paired_miou_difference']}/CI
{analysis['versus_b1']['all']['scene_clustered_ci95']}.
Whole-scene10000bootstrapseed1999/TRAINcuts1487/3325. All/size intervals
cross0/no upper<0 materialharm trigger, NOT equivalence/no-harm. Largeversus
B1 adversepoint{analysis['versus_b1']['large']['paired_miou_difference']}/CI
{analysis['versus_b1']['large']['scene_clustered_ci95']} retained.
Objectcountsingle9508/141; multi0/0 insufficient. Officialunique/multiple
unknown; object count is not a substitute.
RepeatsmIoU{json.dumps(analysis['repeated_miou'])}; perexprrepeatmax
{json.dumps({k:v['repeat_max_iou_difference'] for k,v in analysis['cost_first_chunk_only'].items()})}.
Trained pairedcost summary{json.dumps(cost['summary'])}; parameters
{json.dumps(cost['parameters'])}. TRAIN8+pressurepair728/6780,2warm3read,
serialpredictbatch1/nativebatch2backward/AdamW. Temporary updates memory-only;
immutable original checkp hashes. Candidate-then-parent order,descriptivep95/
singlecheckpoint/seed/split limitinterpretation; no trainedPareto superiority.

## 5. Conclusion
SAMEY42 complete_inconclusive, unfavorable W2 pointestimate; target.515 unmet.
RetainW2. No promotion/retune/restart/renamed equivalent. Full covariance
benefit unproven. New Y44 joint-coordinate-feature pairing is a distinct
preregistered mechanism, not a rescue or tunedY42.

## 6. Crash Info
No canonical training crash. Bounded tail traceback/OOM absence is not whole
log proof. Full/cost probes completed and protected originalY43GPU0/PID13155/
sources unchanged. Y44 localuploadGBK/native-sort test-adapter faults are
separate; neither is a Y42 scientific or training crash.
Hashed evidence:
{json.dumps([evidence(p) for p in refs])}
"""
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    (ROOT / review).write_text(text, encoding="utf-8")
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y42")
    assert candidate["stage"] == "training"
    candidate.update(stage="complete", scientific_status="complete_inconclusive",
                     result=f"FrozenFIRST{metrics['miou']}, pairedW2CIcross0/unfavorablepoint, targetunmet.",
                     next_action="Retain W2 and immutable original evidence; no promotion/restart/retune.")
    for ref in refs + [review]:
        entry = evidence(ref)
        assert not any(e["path"] == ref for e in candidate["evidence"])
        candidate["evidence"].append(entry)
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"id": candidate["id"], "stage": candidate["stage"],
                      "scientific_status": candidate["scientific_status"], "review": evidence(review)}))
finally:
    lock.close()
