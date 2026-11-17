"""Admit only hash-bound Y41 gates and dispatch GPU0 through the canonical path."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import closure
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--commit", required=True)
parser.add_argument("--decision", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 360
assert closure.git(ROOT, "rev-parse", "HEAD") == args.commit and len(args.commit) == 40
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["allowed_gpus"] == [0, 1]
base = "pipeline/analyses/"


def evidence(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


goal_sha = evidence("pipeline/goal.json")["sha256"]
assert goal_sha == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
names = ("boundary_cpu_nonempty", "native_controls", "physical_native",
         "initial_loss", "resources", "source_bundle", "parent_semantics")
paths = {name: base + "y41_" + name + "_" + args.token + ".json" for name in names}
results = {name: json.loads((ROOT / path).read_text(encoding="utf-8"))
           for name, path in paths.items()}
assert all(row["complete"] for row in results.values())
cpu, native, physical, resources = (
    results["boundary_cpu_nonempty"], results["native_controls"],
    results["physical_native"], results["resources"])
assert cpu["cpu_cases"] == 32 and cpu["exact_binary_w2_max"] == 0
assert cpu["minimum_gradient"] > 0 and cpu["invariance_max"] < 1e-5
assert native["added_trainable_tensors"] == 4 and native["minimum_added_gradient"] > 0
assert native["cached_repeat_max"] == native["nonindicator_loss_delta_max"] == 0
assert min(native["wrong_masks_min"], native["wrong_pairs_min"]) > 0
assert native["permutation_max"] < 1e-5
assert physical["native_optimizer_steps"] == 2
assert len(physical["peer_parameter_changes"]) == 4
assert all(v > 0 for v in physical["peer_parameter_changes"].values())
assert all(physical["resume_before_continuation"][k]["equal"]
           for k in ("model", "optimizer", "scheduler"))
assert all(r["disabled_parent_mask_max"] == r["disabled_parent_indicator_max"] ==
           r["disabled_parent_sample_mismatches"] == 0
           for r in physical["cached_inference_rows"])
assert resources["parameters_added"] == 67074
assert resources["parameters_added"] < .01 * resources["summary"]["parent"]["parameters"]
assert resources["summary"]["candidate"]["stress_train_reserved"] < 24
assert not results["initial_loss"]["source_weights_or_production_checkpoint_changed"]
bundle = results["source_bundle"]
assert len(bundle["files"]) == 7 and bundle["physical_independence"] and bundle["remote_byte_identity"]
assert results["parent_semantics"]["whole_decoder_module_parent_identity"]
assert results["parent_semantics"]["config_differences"] == []
for item in bundle["files"]:
    blob = subprocess.check_output([
        "git", "-C", str(ROOT), "show", args.commit + ":" + item["path"],
    ])
    assert hashlib.sha256(blob).hexdigest() == item["sha256"]
    assert evidence(item["path"]) == item
review = base + "Y41_admission_review_5e172bd4.md"
motivation = "pipeline/reviews/motivation_review_Y41_5e172bd4.md"
design = "pipeline/designs/exp_design_Y41_5e172bd4.md"
assert args.decision.startswith("pipeline/decisions/")
gates = {
    "diagnosis": [base + "Y41_diagnosis_prereg_5e172bd4.md",
                  paths["boundary_cpu_nonempty"], paths["native_controls"]],
    "controls": [paths["boundary_cpu_nonempty"], paths["native_controls"],
                 paths["initial_loss"], base + "Y41_nonempty_control_repair_5e172bd4.md"],
    "native_backward": [paths["physical_native"], paths["native_controls"]],
    "resources": [paths["resources"], base + "Y41_resource_freeze_5e172bd4.md"],
    "independence": [paths["source_bundle"], paths["physical_native"], paths["parent_semantics"]],
    "constraints": [motivation, design, review],
}
receipt = {
    "id": "AdaRES-Y41",
    "config": "configs_song/model_paper5/adares_y41_boundary_probability_res.yaml",
    "code_commit": args.commit, "goal_sha256": goal_sha, "decision": evidence(args.decision),
    "gates": {key: {"passed": True, "evidence": [evidence(p) for p in refs]}
              for key, refs in gates.items()},
    "scientific_status": "admitted_for_falsifiable_training_not_efficacy",
    "frozen_protocol": {"epochs": 30, "seed": 1999, "server": "server-2",
                        "allowed_gpus": [0], "post_training_design": evidence(design)},
    "limitations": "All failed controls, initial adverse losses, wrong-control "
                   "learning and numerical continuation differences retained. "
                   "No trained benefit, necessity, novelty, no-harm, Pareto or human approval claim.",
}
task = {
    "id": "AdaRES-Y41", "label": "Y41 final boundary-probability confidence",
    "config": receipt["config"], "status": "waiting", "depends_on": [],
    "code_commit": args.commit, "admission_receipt": "pipeline/admissions/AdaRES-Y41.json",
    "requires_exclusive_ownership": True, "min_epochs": 30, "patience": 30,
    "total_epochs": 30,
}
old_receipt = json.loads((ROOT / "pipeline/admissions/AdaRES-Y39.json").read_text())
old_bundle = json.loads((ROOT / old_receipt["gates"]["independence"]["evidence"][0]["path"]).read_text())
original_ssh = monitor.ssh


def scoped_ssh(host, user, command, *positional, **keywords):
    assert host == server["host"] and user == server["user"]
    inventory = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total"
    if command.startswith(inventory):
        command = command.replace("nvidia-smi ", "nvidia-smi -i 0 ", 1)
    if "nvidia-smi" in command:
        assert "nvidia-smi -i 0 " in command
    return original_ssh(host, user, command, *positional, **keywords)


monitor.ssh = scoped_ssh
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue = monitor.load_queue(str(ROOT))
    active = [t for entries in queue["gpu_queues"].values() for t in entries
              if t["status"] in ("running", "waiting", "claiming")]
    assert len(active) == 1 and active[0]["id"] == "AdaRES-Y39"
    assert active[0]["pid"] == 102267 and str(active[0]["gpu"]) == "1"
    assert not any(t["id"] == task["id"] for entries in queue["gpu_queues"].values() for t in entries)
    preserved = copy.deepcopy(active[0])
    program = r'''
import hashlib,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
cmd=pathlib.Path('/proc/102267/cmdline').read_bytes().split(b'\0')
assert b'configs_song/model_paper5/adares_y39_point_distribution_res.yaml' in cmd
assert cmd[cmd.index(b'--gpu_ids')+1]==b'1'
for row in P['old']+P['new']:
    assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
print('SOURCE_AND_Y39_IDENTITY_VERIFIED')
'''
    payload = {"old": old_bundle["files"], "new": bundle["files"]}
    command = server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)
    assert scoped_ssh(server["host"], server["user"], command).strip() == "SOURCE_AND_Y39_IDENTITY_VERIFIED"
    receipt_path = ROOT / task["admission_receipt"]
    assert not receipt_path.exists(), "Do not replay admission"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    closure.admission(ROOT, task)
    queue["gpu_queues"]["server-2"].append(task)
    monitor.save_queue(str(ROOT), queue)
    events = []
    monitor._process_queue(
        queue, "server-2", server["host"], server["user"], str(ROOT), {"1"}, {"0"},
        events, max_tasks_per_gpu=1, remote_code_root=server["remote_code_root"],
        current_tasks=1, max_tasks_total=2)
    latest = monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
    assert next(t for t in latest if t["id"] == "AdaRES-Y39") == preserved
    selected = next(t for t in latest if t["id"] == task["id"])
    output = ROOT / (base + "y41_dispatch_" + args.token + ".json")
    assert not output.exists()
    output.write_text(json.dumps({
        "observed_at_local": datetime.now().astimezone().isoformat(),
        "task": selected, "events": events, "receipt": evidence(task["admission_receipt"]),
        "preserved_y39_exact": True, "scope": "Canonical GPU0-only dispatch, not preflight.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"task": selected, "events": events,
                      "preserved_y39_exact": True}, indent=2))
finally:
    monitor.ssh = original_ssh
    lock.close()
