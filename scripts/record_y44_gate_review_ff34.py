"""Review actual six-gate evidence; preserve every failed test adapter."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import yaml

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


names = ("y44_pairing_cpu_ff34", "y44_parent_cpu_ff34", "y44_source_bundle",
         "y44_native_controls_ff34_v3", "y44_physical_native_ff34",
         "y44_initial_loss_ff34", "y44_resources_ff34")
refs = [BASE + n + "_" + args.token + ".json" for n in names]
loaded = [json.loads((ROOT / ref).read_text(encoding="utf-8")) for ref in refs]
assert all(v["complete"] for v in loaded)
old_cpu, cpu, bundle, native, physical, initial, resources = loaded
assert old_cpu["seeds"] == 32 and old_cpu["both_feature_arms_nonnegative"]
assert old_cpu["existing_marginal_max"] == old_cpu["existing_readout_max"] == 0
assert old_cpu["minimum_pairing_gap"] > old_cpu["control_floor"]
assert cpu["whole_modules_parent_identity"] and cpu["config_differences"] == []
assert cpu["cpu_cases"] == 32 and cpu["all_feature_arms_nonnegative"]
assert cpu["minimum_pairing_response"] > cpu["control_floor"] and cpu["minimum_new_gradient"] > 0
assert native["added_trainable_tensors"] == 8 and native["minimum_added_gradient"] > 0
assert native["cached_repeat_max"] == native["nonindicator_loss_delta_max"] == 0
assert native["permutation_max"] < 1e-5
assert len(native["rows"]) == 12 and len(native["cached_controls"]) == 12
for row in native["rows"]:
    assert all(row[n + "_max"] > 0 for n in ("wrong_masks", "wrong_pairing", "rolled_cross", "zero_cross"))
    incidence = row["incidence"]
    assert incidence["all_components_within_derived_envelope"]
    assert incidence["feature_multiset_preserved_by_within_sp_bijection"]
    assert incidence["wrong_pairing_cross_max"] > max(incidence["native_reconstruction_max"],
                                                      incidence["independent_reconstruction_repeat_max"])
assert all(c["padding_poison"] == 0 and c["detached_mask_gradient"] for c in native["cached_controls"])
assert physical["native_optimizer_steps"] == 2
assert len(physical["peer_parameter_changes"]) == 8 and all(v > 0 for v in physical["peer_parameter_changes"].values())
assert all(physical["resume_before_continuation"][k]["equal"] for k in ("model", "optimizer", "scheduler"))
assert all(r["disabled_parent_mask_max"] == r["disabled_parent_indicator_max"]
           == r["disabled_parent_sample_mismatches"] == r["enabled_repeat_mask_max"] == 0
           for r in physical["cached_inference_rows"])
assert not initial["source_weights_or_production_checkpoint_changed"] and initial["temporary_new_path_steps"] == 2
assert all(v["mean_change"] < 0 for outcomes in initial["summary"].values() for v in outcomes.values())
assert resources["parameters_added"] == native["added_parameters"] == physical["added_parameters"] == 288002
assert resources["parameters_added"] < .01 * resources["summary"]["parent"]["parameters"]
assert resources["summary"]["candidate"]["stress_train_reserved"] < 24
assert resources["inference_queries"] == physical["inference_queries"] == 128
assert resources["inference_selected"] == physical["inference_selected"] == 1
assert bundle["physical_independence"] and bundle["remote_byte_identity"]
paths = [entry["path"] for entry in bundle["files"]]
assert len(paths) == 7
for entry in bundle["files"]:
    assert evidence(entry["path"]) == entry
    path = ROOT / entry["path"]
    if path.suffix == ".py":
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert "model_paper5" not in (node.module or "")
            if isinstance(node, ast.Import):
                assert not any("model_paper5" in alias.name for alias in node.names)
config = ROOT / "AdaRES/configs_song/model_paper5/adares_y44_coordinate_feature_res.yaml"
matches = list((ROOT / "AdaRES/configs_song").rglob(config.name))
assert matches == [config]
cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
assert cfg == yaml.safe_load((config.parent / "adares_w2_pointmass_competition_res.yaml").read_text(encoding="utf-8"))
assert cfg["train"]["epochs"] == 30 and cfg["train"]["seed"] == 1999
assert cfg["train"]["pretrain"].endswith("sp_unet_backbone.pth")
assert cfg["optimizer"]["type"] == "AdamW" and cfg["lr_scheduler"]["type"] == "PolyLR"
assert evidence("pipeline/goal.json")["sha256"] == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
queue = monitor.load_queue(str(ROOT))
active = [t for entries in queue["gpu_queues"].values() for t in entries
          if t["status"] in ("running", "claiming", "waiting")]
assert len(active) == 1 and active[0]["id"] == "AdaRES-Y43"
assert active[0]["pid"] == 13155 and str(active[0]["gpu"]) == "0"
view = json.loads((ROOT / "pipeline/headless/.tick_model_view.json").read_text())
assert view["queue"]["candidate_budget"] == 1 and view["mode"] == "design"
constraints = [
    "Only server2 GPUs0/1, testsGPU1 preserve originalY43GPU0, onejobcard.",
    "B3 not active; current design budget1 admits only sameY44, no extra candidate.",
    "B2/B4 fixed30 resume protocols untouched; Y44 released initializer/fixed30/native optimizer and scheduler unchanged.",
    "ReleasedSPFormer sparseUNet nativepretrain retained; authored independent decoder.",
    "mIoU/Acc25/Acc50/queries/latency/peakmemory terminal and resource protocols frozen.",
    "No validation threshold/temperature/loss/feature/axis/epoch sweep; no server6/b005/CFNet-asB0/rankingmargin/occupancy-weight scan.",
    "Goal.515 unchanged, Phase1unmet/Phase2unapproved/W2retained.",
]
gates = {name: True for name in ("diagnosis", "controls", "native_backward", "resources", "independence", "constraints")}
checks_path = BASE + "y44_gate_checks_" + args.token + ".json"
review = BASE + "Y44_six_gate_review_ff347697.md"
assert not (ROOT / checks_path).exists() and not (ROOT / review).exists()
extra = [
    BASE + "Y44_native_precision_audit_freeze_ff347697.md",
    BASE + "y44_native_controls_ff34_v2_" + args.token + ".log",
    "scripts/probe_y44_native_controls_ff34_v3.py",
]
checks = {"complete": True, "id": "AdaRES-Y44", "gates": gates, "constraints_checked": constraints,
          "evidence": [evidence(p) for p in refs + extra],
          "observed_at_local": datetime.now().astimezone().isoformat(),
          "scientific_status": "six_gate_feasibility_not_efficacy"}
text = f"""# Y44 Six-Gate Feasibility Review
Observed {checks['observed_at_local']}; not efficacy/human approval.

## 1. Diagnosis
Actual oldW2/Y39/Y42/Y43 CPU32seed feature-marginal/readout invariance0,
crossgap>=9. New actualclass32seed minpairresponse{cpu['minimum_pairing_response']}
versuscontrolfloor{cpu['control_floor']}; featurearmsnonnegative.
Actual TRAIN8/init4 captures coords3/features32/96crossall/SPincidence.
WithinSP feature bijection preserves entire feature multiset and fixed geometry.
Narrow explicit information gap, not upstream blindness/proven task bottleneck.

## 2. Controls And Precision
12native matchedwrongpair/rolledcross/zerocross/wrongmasks separated. Repeat/
nonindicatorloss/paddingpoison0; detachedmasks; permutationmax
{native['permutation_max']}. Minwrongpairconfidence
{min(r['wrong_pairing_max'] for r in native['rows'])}.
V1 unsupported old-PyTorch stable-sort failed before controls; V2 bitwise
independent scatter-reconstruction assertion failed before decoded controls.
Both sources/logs preserved. V3 uses exact same label/point-order bijection,
captures actual native means and independentCPUfloat64 cross. EVERYcomponent
within prospectively frozen float32 forward-error envelope derived from
machineeps/counts/inputmagnitudes, not a fitted or raised fixed tolerance.
MaximumnativeCPU64difference
{max(r['incidence']['native_cpu64_error_max'] for r in native['rows'])};
GPUindependentreconstructrepeatmax
{max(r['incidence']['independent_reconstruction_repeat_max'] for r in native['rows'])};
minimumwrongpaircrossgap{min(r['incidence']['wrong_pairing_cross_max'] for r in native['rows'])}.
No candidate source, scientific control, gradient/resource gate changed.
Cached repeats/disabledphysicalW2 still exact0. No bitwise cross claim.
Nativeinitialdeltas{json.dumps(native['loss_deltas'])}; allper-sceneadversities
retained. Fixed2newonlynativeAdamWlearningchanges
{json.dumps({phase:{mode:v['mean_change'] for mode,v in outcomes.items()} for phase,outcomes in initial['summary'].items()})}.
Legacy resultaliaswrong_third in reused learner denotes rolled96cross input,
NOT a third-moment statistic. Wrong controls learn similarly; no necessity,
benefit,generalization,no-harm claim or diagnosticweightdeployment.

## 3. Native Backward And Restoration
All8newtensors finite/nonzero nativeTRAIN8/init4, minimum
{native['minimum_added_gradient']}; actual tools.train2steps/all8changed.
Exact model/AdamW/PolyLR restore beforecontinuation; continuation
{json.dumps(physical['continuation'])} retained, not bitwise trajectory claim.
Fourcachedscenes disabledphysicalW2masks/indicators/selectionequal0.
Native128/oneargmax unchanged.

## 4. Resources
Added288002params(<1%W2); nativepressure reserved<24GiB/noOOM/nonfinite.
FrozenTRAIN8/pressurepair728,6780/2warm3read actual pairedsummary:
{json.dumps(resources['summary'])}.
Shared initial tensors exactparent; inferenceW2+initializednewbranch.
Observed time/memory regression retained; no trainedPareto superiority.

## 5. Independence
Exactly seven physically independent local/remote source/YAML hashes equal;
unique basename and native resolver, no crossvariant imports. Wholedec/ipdn
samePythonAST parentidentity afterexactnewpath removal/configdiffs[].
Native disabledphysicalW2 equality corroborates. Y42 terminal/Y43GPU0 original
sources/protectedprocess unchanged. FirstlocalGBKupload fault beforeSSH repaired
only explicitUTF8 reading. Commit commands disable automatic newline conversion
PERINVOCATION only to preserve alreadytestedrawbytes; no repository setting or
tested source modification. Immutablegitblob/commit/receipt still pending.

## 6. Constraints And Disposition
{json.dumps(constraints)}.
All six executed gates support a falsifiable Phase1 training admission, not
efficacy/novelty/necessity/no-harm/humanPhase2approval. Prospective sevensection
design/motivation preserve fullgoal/terminal plan: EACHfull9508/141twice/FIRST/
scenebootstrap10000seed1999/TRAINcuts1487,3325/object/harmvsEither/repeat/
trainedpairedcost. W2retained/target.515unmet. No extra admission beyondbudget1.
Hashed evidence{json.dumps([evidence(p) for p in refs + extra])}.
"""
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    (ROOT / review).write_text(text, encoding="utf-8")
    (ROOT / checks_path).write_text(json.dumps(checks, indent=2), encoding="utf-8")
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y44")
    assert candidate["stage"] == "validation"
    for ref in refs + extra + [checks_path, review]:
        if not any(e["path"] == ref for e in candidate["evidence"]):
            candidate["evidence"].append(evidence(ref))
    candidate.update(scientific_status="six_gate_feasibility_not_efficacy",
                     next_action="Commit exactly tested7rawbytefiles, bind genuine receipt/decision, canonicalGPU1dispatch and increasing original native steps.")
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "review": evidence(review), "checks": evidence(checks_path)}))
finally:
    lock.close()
