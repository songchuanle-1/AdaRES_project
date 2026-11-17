"""Persist real terminal/CPU evidence and a live-occupation resource hold."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor


def evidence(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text(encoding="utf-8"))
    assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        prefix = "pipeline/analyses/"
        terminal = prefix + "y43_terminal_7a40_" + args.token + ".json"
        cpu = prefix + "y45_confusion_cpu_7a40_v2_" + args.token + ".json"
        mapping = prefix + "y45_point_targets_cpu_7a40_" + args.token + ".json"
        terminal_data = json.loads((ROOT / terminal).read_text(encoding="utf-8"))
        cpu_data = json.loads((ROOT / cpu).read_text(encoding="utf-8"))
        mapping_data = json.loads((ROOT / mapping).read_text(encoding="utf-8"))
        assert all(v["complete"] for v in (terminal_data, cpu_data, mapping_data))
        assert terminal_data["checkpoints"][0]["epoch"] == 30
        assert cpu_data["case_count"] == mapping_data["case_count"] == 32
        assert "0, 22940" in mapping_data["gpu_occupation_csv"]
        assert "1, 20772" in mapping_data["gpu_occupation_csv"]
        assert "19723" in mapping_data["compute_processes_csv"]
        assert "15520" in mapping_data["compute_processes_csv"]
        now = datetime.now().astimezone()
        retry = (now + timedelta(minutes=30)).isoformat()
        review_path = "pipeline/reviews/motivation_review_Y45_7a40_" + args.token + ".md"
        review = f"""# Motivation Review - AdaRES_project
Observed locally: {now.isoformat()}; owned token {args.token}.

## 1. Goal
Complete unchanged project goal:
```json
{(ROOT / "pipeline/goal.json").read_text(encoding="utf-8").strip()}
```
Target0.515 remains unmet. Rounded ledgerY36 .5110 is not scientific promotion;
W2 remains the scientific comparator.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Y43 all-channel asymmetry | fixed30 training complete | {terminal} | full frozen efficacy/harm/cost pending |
| Y45 metric-confusion decomposition | CPU target mapping only | {cpu}; {mapping} | native TRAIN incidence/controls pending |

## 3. Experiment Assessment
| Experiment | mIoU | Relation to motivation | Contribution |
|---|---|---|---|
| Y43 | .508529961 checkpoint metadata, not frozen evaluation | confidence from point asymmetry | no efficacy conclusion before frozen comparison |
| Y44 | snapshot training30/30, terminal unconfirmed | coordinate-feature association | preserve originalGPU1 run |
| Y45 | unknown; not trained | explicit FP/FN supervision at fixed masks | generated CPU witnesses distinguish error partitions only |

## 4. Gap Analysis
Actual W2 final selection uses matched indicator classification, not scalar
quality regression. The initial scalar-confidence premise is disconfirmed and
retained in its original preregistration. Thirty-two float32 actual-function
cases have identical classification losses/gradients and identical scalar IoU
but different [TP,FP,FN] values [4,2,0] and [6,0,3].
Thirty-two separate point-mapping cases agree with literal point counts;
joint-point/query permutation, padding and empty-mask checks pass; wrong
expression targets differ by at least3 points. These are generated CPU
fixtures with actual parent method bodies, not a captured native TRAIN batch,
not evidence of useful learnability or an admission gate PASS.

## 5. Design Rationale
The corrected stable Y45 asks whether a learned decomposition gives confidence
useful TRAIN supervision beyond matched binary labels. It does not reopen
scalar-IoU/listwise/pairwise ranking implementations, use oracle inference
inputs, tune weights/thresholds/temperatures, or add a higher-moment successor.
Native alignment and empirical task relevance must be checked before model
implementation/admission. Prospective independent-package and frozen
post-training requirements remain mandatory.

## 6. Next Batch
No admission or new training. GPU0 has unregistered PID19723, memory22940MiB;
GPU1 has originalY44 PID15520, memory20772MiB, in the12:02 local observation.
Preserve both processes. The actual GPU0 evaluation attempt was refused by
the outer idleness guard; retained failure log:
{prefix}y43_frozen_full_7a40_{args.token}.log.
The initial float64 CPU fixture failed native float32 label assignment before
results; its original source/log are retained. Corrected v2 did not alter any
model/source/config/checkpoint.
Resume SAME Y43 frozen9508/141 twice EACH/FIRST vs W2/B1, then trained paired
cost, and SAME Y45 native TRAIN correct/wrong target capture when an authorized
GPU is actually idle and its exclusive lease can be acquired. Retry after
{retry}, or sooner on resource/evidence change. No claimed route exhaustion.
"""
        destination = ROOT / review_path
        assert not destination.exists()
        destination.write_text(review, encoding="utf-8")
        path = ROOT / "pipeline/research_work.json"
        work = json.loads(path.read_text(encoding="utf-8"))
        shared = [evidence(terminal), evidence(cpu), evidence(mapping), evidence(review_path)]
        blocker = ("Live authorized GPU occupation: GPU0 PID19723 unregistered to this project's queue; "
                   "GPU1 original Y44 PID15520. No free exclusive GPU for frozen inference or native TRAIN capture.")
        for cid in ("AdaRES-Y43", "AdaRES-Y45"):
            candidate = next(c for c in work["candidates"] if c["id"] == cid)
            candidate.setdefault("history", []).append(dict(
                observed_at=now.isoformat(), prior_stage=candidate["stage"],
                action="terminal_or_CPU_evidence_persisted_live_GPU_hold",
                prior_hypothesis=candidate.get("hypothesis"), evidence=shared))
            candidate["evidence"].extend(shared)
            candidate.update(stage="blocked", blocker=blocker,
                             blocker_kind="live_external_gpu_occupation",
                             required_server="server-2", retry_after=retry,
                             unblock_condition="Either authorized GPU becomes live idle with no compute process; acquire its exclusive lease and preserve all foreign/original jobs.")
            if cid == "AdaRES-Y43":
                candidate.update(scientific_status="fixed30_complete_frozen_validation_blocked",
                                 next_action="Resume SAME Y43 frozen full9508/141 twice EACH/FIRST vs W2/B1 and trained paired resource/harm review on an actually idle authorized GPU; do not restart training.")
                candidate["evidence"].extend([
                    evidence("scripts/probe_y43_terminal_7a40.py"),
                    evidence("scripts/probe_y43_frozen_full_7a40.py"),
                    evidence(prefix + "y43_frozen_full_7a40_" + args.token + ".log")])
            else:
                candidate.update(
                    hypothesis="Matched binary indicator labels may omit distinct FP/FN point-mass supervision; learned error decomposition is untested in native TRAIN.",
                    scientific_status="CPU_target_mapping_only_native_diagnosis_blocked",
                    next_action="Capture actual prediction-blind native TRAIN masks and point targets; test correct/wrong expression TP/FP/FN targets and inspect matched-loss gradients before selecting an independent implementation. CPU witnesses do not admit training.",
                    pending_gates=["diagnosis", "controls", "native_backward", "resources", "independence", "constraints"])
                candidate["evidence"].extend([
                    evidence("scripts/probe_y45_confusion_cpu_7a40.py"),
                    evidence(prefix + "y45_confusion_cpu_7a40_" + args.token + ".log"),
                    evidence("scripts/probe_y45_confusion_cpu_7a40_v2.py"),
                    evidence("scripts/probe_y45_point_targets_cpu_7a40.py")])
        work["design_cycle"] = dict(
            stop_reason="external_resource_blocked",
            attempted_ids=["AdaRES-Y43", "AdaRES-Y45"], evidence=shared,
            resume_condition="Live GPU0 external job release or natural Y44 GPU1 completion with verified idleness; resume the same frozen evaluation and native target diagnosis.",
            retry_after=retry, observed_at=now.isoformat())
        path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(blocked_ids=["AdaRES-Y43", "AdaRES-Y45"],
                              retry_after=retry, review=evidence(review_path),
                              stop_reason="external_resource_blocked", new_admissions=0)))
    finally:
        writer.close()


if __name__ == "__main__":
    main()
