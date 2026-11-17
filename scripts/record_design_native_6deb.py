"""Bind this tick's native diagnosis evidence without manufacturing gate passes."""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import authorize_prepared_candidates as scoped

ROOT = scoped.ROOT
TOKEN = "6debf88fed634a13b772f27bdc244bbd"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    assert args.token == TOKEN
    scoped.lease(args.token)
    lock = scoped.monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work_path = ROOT / "pipeline/research_work.json"
        work = scoped.tick.load(work_path)
        for cid, name in (("AdaRES-Y56", "y56_native_batch2"),
                          ("AdaRES-Y57", "y57_native_task")):
            candidate = next(item for item in work["candidates"] if item["id"] == cid)
            assert candidate["stage"] == "blocked" and candidate[
                "blocker_kind"] == "exclusive_gpu_slot"
            relative = "pipeline/analyses/" + name + "_" + TOKEN + ".json"
            result = scoped.tick.load(ROOT / relative)
            assert result["complete"] and result["id"] == cid
            assert result["expression_layer_records"] == 224
            assert result["fixed_scenes"] == 8 and result["expressions"] == 32
            assert result["parameters_unchanged"] and result["buffers_restored"]
            assert result["optimizer_steps"] == 0 and not result["weights_saved"]
            reference = scoped.ref(relative)
            history = {key: deepcopy(candidate[key]) for key in
                       ("blocker_kind", "blocker", "retry_after", "unblock_condition",
                        "required_server") if key in candidate}
            history.update(resolved_at=datetime.now().astimezone().isoformat(),
                           resolved_by=reference)
            candidate.setdefault("blocker_history", []).append(history)
            for key in ("blocker_kind", "blocker", "retry_after",
                        "unblock_condition", "required_server"):
                candidate.pop(key, None)
            candidate["evidence"].extend([
                reference, scoped.ref(relative[:-len(".json")] + ".log")])
            candidate.update(stage="diagnosis", failure_scope="inconclusive")
            candidate["gate_summary"].update(
                native_backward="Not executed; the completed native capture used no_grad.",
                resources="Source/checkpoint-bound native capture completed on an idle "
                          "authorized exclusive GPU; this is not a production resource gate.",
                independence="No formal candidate model/config implemented.",
                constraints="No optimizer, saved weights, new queue entry or training.")
            if cid == "AdaRES-Y56":
                assert result["exact_joint_duplicate_excess"] == 0
                candidate.update(
                    scientific_status="No exact joint copies in fixed native8scene/32expression/"
                                      "all7layer capture; synthetic nuisance not empirical prevalence.",
                    next_action="Review fixed native incidence with the matched multiplicity "
                                "control; preregister any broader prediction-blind incidence "
                                "test before a correction. Do not equate binary-mask equality "
                                "with identical features/soft masks or claim population falsification.")
                candidate["gate_summary"]["diagnosis"] = (
                    "Actual224 correlated expression-layer records: exact joint copies0; "
                    "no empirical support here for the exact-copy correction.")
            else:
                candidate.update(
                    scientific_status="Fresh source/checkpoint/current-GT-bound capture "
                                      "complete; task-causal/incremental relevance unestablished.",
                    next_action="Execute frozen y57_native_signal_" + TOKEN +
                                "_program.py on CPU, final layer6 primary, with held-out-scene "
                                "confidence-conditional and same-capacity shifted controls; "
                                "do not treat observational gain as causal or admission PASS.")
                candidate["evidence"].extend([
                    scoped.ref("pipeline/analyses/y57_native_signal_" + TOKEN + "_program.py"),
                    scoped.ref("pipeline/analyses/y57_native_signal_" + TOKEN + "_prereg.json")])
                candidate["gate_summary"]["diagnosis"] = (
                    "Native source/checkpoint/point-GT bindings passed; task-specific "
                    "incremental evidence and reachable counterfactuals remain untested.")
        scoped.lease(args.token)
        scoped.tick.save(work_path, work)
        print("Bound native evidence; cleared only the two resolved GPU holds; no admission.")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
