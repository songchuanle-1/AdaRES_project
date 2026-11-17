"""Persist actual feasibility review and hashes, without claiming launch or efficacy."""
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
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text(encoding="utf-8"))
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


names = ("covariance_cpu", "extract_readout_cpu", "native_controls_v2", "physical_native",
         "initial_loss_v2", "resources", "source_bundle", "parent_semantics_v3")
paths = ["pipeline/analyses/y42_" + n + "_" + args.token + ".json" for n in names]
results = {n: json.loads((ROOT / p).read_text(encoding="utf-8")) for n, p in zip(names, paths)}
assert all(r["complete"] for r in results.values())
native, physical, cost, initial = (results[n] for n in ("native_controls_v2", "physical_native", "resources", "initial_loss_v2"))
assert native["added_trainable_tensors"] == 8 and native["minimum_added_gradient"] > 0
assert min(native["wrong_masks_min"], native["wrong_covariance_min"], native["reflected_covariance_min"]) > 0
assert native["cached_repeat_max"] == native["nonindicator_loss_delta_max"] == 0
assert all(c["padding_poison"] == 0 and c["detached_mask_gradient"] for c in native["cached_controls"])
assert results["parent_semantics_v3"]["whole_modules_parent_identity"]
assert physical["native_optimizer_steps"] == 2
assert all(physical["resume_before_continuation"][k]["equal"] for k in ("model", "optimizer", "scheduler"))
assert cost["parameters_added"] == 398594 < .01 * cost["summary"]["parent"]["parameters"]
assert cost["summary"]["candidate"]["stress_train_reserved"] < 24
assert not initial["source_weights_or_production_checkpoint_changed"]
additional = [
    "pipeline/reviews/motivation_review_Y42_062ae665.md",
    "pipeline/designs/exp_design_Y42_062ae665.md",
    "scripts/admit_dispatch_y42.py", "scripts/capture_finalize_y42_launch.py",
    "scripts/commit_y42_code.py",
]
review_path = "pipeline/analyses/Y42_admission_review_062ae665.md"
assert not (ROOT / review_path).exists()
parent, candidate = cost["summary"]["parent"], cost["summary"]["candidate"]
text = f"""# Y42 Six-Gate Feasibility Review
Observed {datetime.now().astimezone().isoformat()}; token {args.token}.
This operational admission review is not human approval or efficacy.

## 1. Diagnosis
Actual W2/Y39 CPU32 equal-marginal readouts both delta0/covariance gap2.
Actual extract/readout CPU32 uses complete528 covariance entries; matched
joint-only readout gap>=.00465405, finite gradients, invariance<=5.96e-8.
Native TRAIN8/init4 joint covariance is nonzero. Narrow readout information
gap supported; upstream blindness, novelty and semantic utility unknown.

## 2. Controls
Wrong mask/off-diagonal/PSD reflection responses are nonzero; all8 new
tensors have finite nonzero gradients, repeated and nonindicator loss delta0,
padding poison0, mask inputs detached. PSD reflection preserves all diagonal
entries. Rolled off-diagonal matrices need not remain PSD; retained limitation.
Released-init enabled loss delta+.0183002949 is adverse and not erased.
Fixed two new-only steps: correct trained delta{initial['summary']['trained']['enabled']['mean_change']},
released-init delta{initial['summary']['released_init']['enabled']['mean_change']};
wrong mask/covariance also learn similarly. Individual correct losses rise.
This does not establish covariance necessity or generalization.

## 3. Native Backward
Actual tools.train two steps/all8 tensors update, finite native gradients.
Exact model/AdamW/PolyLR restoration before continuation. Next-step loss
delta{physical['continuation']['loss_difference']}, modelmax{physical['continuation']['model']['max']},
optimizermax{physical['continuation']['optimizer']['max']}; no bitwise trajectory claim.
Disabled-path physical W2 cached four-scan masks/indicators/selection equal0.
128 queries/one native argmax unchanged; not a whole-loader determinism claim.

## 4. Resources
New params398594 (<1% parent); candidate/parent native pressure-batch median
{candidate['stress_train_median']}/{parent['stress_train_median']} seconds,
reserved{candidate['stress_train_reserved']}/{parent['stress_train_reserved']} GiB.
Observed training time regression{100*(candidate['stress_train_median']/parent['stress_train_median']-1):.3f}%.
Inference median{candidate['inference_median']}/{parent['inference_median']} seconds.
Prospective reserved<24GiB/no OOM/nonfinite passed, not trained Pareto benefit.
TRAIN8 plus maxpair728/6780,2warm/3reads; descriptive, not significance/tails.

## 5. Independence
Isolated six native Python files and unique YAML; seven local/remote hashes
equal. Same-Python3.8 whole dec/ipdn AST equals W2 after removing exactly new
path; structured config differences[]. No raw parent-byte identity claimed.
Git rawblob identity still required after exact-seven-file commit. Original
Y41 GPU0/PID116906 source and Y39 completed sources preserved.
GBK build failure and three original test-adapter binding failures retained;
only adapters repaired, no model/source/gate/tolerance change after tests.

## 6. Constraints And Disposition
Six feasibility gates support falsifiable phase-1 fixed30 training ONLY.
W2 retained/target.515 unmet/Phase2 unapproved. ReleasedSPFormer, seed1999,
native optimizer/loss/scheduler/128 queries/one argmax unchanged. No numeric
sweep, diagnostic weight deployment, scalar/radius/temperature/feature subset.
Full9508/141 Y42/W2/B1 twice EACH/FIRST/scene10000seed1999/TRAIN1487/3325
and versus-either subgroup harm/repeat noise/trained paired costs remain
required on terminal. Unknown official uniqueness; empty multi-object data
not no-harm. Queue admission, original PID/config/cwd/GPU/increasing steps
still REQUIRED, not already complete at review writing.

## Hashed Evidence
{json.dumps([evidence(p) for p in paths + additional], indent=2)}
"""
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y42")
    assert candidate["stage"] == "validation"
    (ROOT / review_path).write_text(text, encoding="utf-8")
    for ref in paths + additional + [review_path]:
        entry = evidence(ref)
        prior = next((e for e in candidate["evidence"] if e["path"] == ref), None)
        assert prior is None or prior == entry
        if prior is None:
            candidate["evidence"].append(entry)
    candidate.update(scientific_status="six_gate_feasibility_not_efficacy",
                     next_action="Commit exactly seven tested files; canonical GPU1 admission/dispatch and independent increasing native steps. Preserve original Y41 GPU0, all adverse controls, fixed30/seed1999 and terminal frozen protocol.")
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"review": evidence(review_path), "stage": candidate["stage"],
                      "scientific_status": candidate["scientific_status"]}))
finally:
    lock.close()
