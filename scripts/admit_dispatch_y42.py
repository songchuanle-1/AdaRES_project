"""Bind six actual Y42 gates, then canonical GPU1-only dispatch preserving Y41."""
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
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text(encoding="utf-8"))
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
names = ("covariance_cpu", "extract_readout_cpu", "native_controls_v2", "physical_native",
         "initial_loss_v2", "resources", "source_bundle", "parent_semantics_v3")
paths = {name: base + "y42_" + name + "_" + args.token + ".json" for name in names}
results = {name: json.loads((ROOT / path).read_text(encoding="utf-8")) for name, path in paths.items()}
assert all(row["complete"] for row in results.values())
cpu, extraction = results["covariance_cpu"], results["extract_readout_cpu"]
assert cpu["cpu_cases"] == 32 and cpu["w2_max"] == cpu["y39_max"] == 0
assert cpu["covariance_min"] > 0 and cpu["repeat_max"] == cpu["permutation_max"] == 0
assert extraction["cpu_cases"] == 32 and extraction["covariance_width"] == 528
assert extraction["added_tensors"] == 8 and extraction["minimum_added_gradient"] > 0
assert extraction["algebra_and_permutation_max"] < 1e-5
assert min(extraction["wrong_offdiagonal_min"], extraction["matched_covariance_only_min"]) > 0
native, physical, resources = (results["native_controls_v2"], results["physical_native"], results["resources"])
assert native["added_trainable_tensors"] == 8 and native["minimum_added_gradient"] > 0
assert native["cached_repeat_max"] == native["nonindicator_loss_delta_max"] == 0
assert min(native["wrong_masks_min"], native["wrong_covariance_min"], native["reflected_covariance_min"]) > 0
assert native["permutation_max"] < 1e-5
assert all(row["padding_poison"] == 0 and row["detached_mask_gradient"] for row in native["cached_controls"])
assert physical["native_optimizer_steps"] == 2 and physical["added_parameters"] == 398594
assert len(physical["peer_parameter_changes"]) == 8
assert all(v > 0 for v in physical["peer_parameter_changes"].values())
assert all(physical["resume_before_continuation"][k]["equal"] for k in ("model", "optimizer", "scheduler"))
assert all(r["disabled_parent_mask_max"] == r["disabled_parent_indicator_max"] ==
           r["disabled_parent_sample_mismatches"] == 0 for r in physical["cached_inference_rows"])
assert resources["parameters_added"] == 398594
assert resources["parameters_added"] < .01 * resources["summary"]["parent"]["parameters"]
assert resources["summary"]["candidate"]["stress_train_reserved"] < 24
assert not results["initial_loss_v2"]["source_weights_or_production_checkpoint_changed"]
assert results["initial_loss_v2"]["temporary_new_path_steps"] == 2
bundle = results["source_bundle"]
assert len(bundle["files"]) == 7 and bundle["physical_independence"] and bundle["remote_byte_identity"]
assert results["parent_semantics_v3"]["whole_modules_parent_identity"]
assert results["parent_semantics_v3"]["config_differences"] == []
for item in bundle["files"]:
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", args.commit + ":" + item["path"]])
    assert hashlib.sha256(blob).hexdigest() == item["sha256"] and evidence(item["path"]) == item
review = base + "Y42_admission_review_062ae665.md"
motivation = "pipeline/reviews/motivation_review_Y42_062ae665.md"
design = "pipeline/designs/exp_design_Y42_062ae665.md"
assert args.decision.startswith("pipeline/decisions/")
gates = {
    "diagnosis": [base + "Y42_diagnosis_prereg_062ae665.md", paths["covariance_cpu"], paths["extract_readout_cpu"]],
    "controls": [paths["extract_readout_cpu"], paths["native_controls_v2"], paths["initial_loss_v2"],
                 base + "Y42_native_control_amendment_062ae665.md"],
    "native_backward": [paths["physical_native"], paths["native_controls_v2"]],
    "resources": [paths["resources"], base + "Y42_resource_freeze_062ae665.md"],
    "independence": [paths["source_bundle"], paths["physical_native"], paths["parent_semantics_v3"]],
    "constraints": [motivation, design, review],
}
receipt = {
    "id": "AdaRES-Y42", "config": "configs_song/model_paper5/adares_y42_point_covariance_res.yaml",
    "code_commit": args.commit, "goal_sha256": goal_sha, "decision": evidence(args.decision),
    "gates": {key: {"passed": True, "evidence": [evidence(p) for p in refs]} for key, refs in gates.items()},
    "scientific_status": "admitted_for_falsifiable_training_not_efficacy",
    "frozen_protocol": {"epochs": 30, "seed": 1999, "server": "server-2", "allowed_gpus": [1],
                        "post_training_design": evidence(design)},
    "limitations": "All adapter failures, adverse released-init loss, wrong-control learning and numerical continuation differences retained. No efficacy, necessity, novelty, no-harm, Pareto or human approval claim.",
}
task = {
    "id": "AdaRES-Y42", "label": "Y42 full-channel point covariance confidence",
    "config": receipt["config"], "status": "waiting", "depends_on": [], "code_commit": args.commit,
    "admission_receipt": "pipeline/admissions/AdaRES-Y42.json", "requires_exclusive_ownership": True,
    "min_epochs": 30, "patience": 30, "total_epochs": 30,
}
old_receipt = json.loads((ROOT / "pipeline/admissions/AdaRES-Y41.json").read_text(encoding="utf-8"))
old_bundle = json.loads((ROOT / old_receipt["gates"]["independence"]["evidence"][0]["path"]).read_text(encoding="utf-8"))
terminal = json.loads((ROOT / (base + "y39_terminal_current_" + args.token + ".json")).read_text(encoding="utf-8"))
original_ssh = monitor.ssh


def scoped_ssh(host, user, command, *positional, **keywords):
    assert host == server["host"] and user == server["user"]
    inventory = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total"
    if command.startswith(inventory):
        command = command.replace("nvidia-smi ", "nvidia-smi -i 1 ", 1)
    if "nvidia-smi" in command:
        assert "nvidia-smi -i 1 " in command
    return original_ssh(host, user, command, *positional, **keywords)


monitor.ssh = scoped_ssh
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue = monitor.load_queue(str(ROOT))
    active = [t for entries in queue["gpu_queues"].values() for t in entries if t["status"] in ("running", "waiting", "claiming")]
    assert len(active) == 1 and active[0]["id"] == "AdaRES-Y41"
    assert active[0]["pid"] == 116906 and str(active[0]["gpu"]) == "0"
    assert not any(t["id"] == task["id"] for entries in queue["gpu_queues"].values() for t in entries)
    preserved_queue = copy.deepcopy(queue)
    program = r'''
import hashlib,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
cmd=pathlib.Path('/proc/116906/cmdline').read_bytes().split(b'\0')
assert b'configs_song/model_paper5/adares_y41_boundary_probability_res.yaml' in cmd
assert cmd[cmd.index(b'--gpu_ids')+1]==b'0'
assert not pathlib.Path('/proc/102267').exists()
for row in P['old']+P['new']:
    assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
for path,sha in P['terminal_sources'].items():
    assert hashlib.sha256((root/'AdaRES'/path).read_bytes()).hexdigest()==sha
print('SOURCE_AND_Y41_IDENTITY_VERIFIED')
'''
    payload = {"old": old_bundle["files"], "new": bundle["files"], "terminal_sources": terminal["source_hashes"]}
    command = server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)
    assert scoped_ssh(server["host"], server["user"], command).strip() == "SOURCE_AND_Y41_IDENTITY_VERIFIED"
    receipt_path = ROOT / task["admission_receipt"]
    assert not receipt_path.exists(), "Do not replay admission"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    closure.admission(ROOT, task)
    queue["gpu_queues"]["server-2"].append(task)
    monitor.save_queue(str(ROOT), queue)
    events = []
    monitor._process_queue(queue, "server-2", server["host"], server["user"], str(ROOT), {"0"}, {"1"},
                           events, max_tasks_per_gpu=1, remote_code_root=server["remote_code_root"],
                           current_tasks=1, max_tasks_total=2)
    latest = monitor.load_queue(str(ROOT))
    for server_id, entries in preserved_queue["gpu_queues"].items():
        assert [t for t in latest["gpu_queues"][server_id] if t["id"] != task["id"]] == entries
    selected = next(t for t in latest["gpu_queues"]["server-2"] if t["id"] == task["id"])
    output = ROOT / (base + "y42_dispatch_" + args.token + ".json")
    assert not output.exists()
    output.write_text(json.dumps({
        "observed_at_local": datetime.now().astimezone().isoformat(),
        "task": selected, "events": events, "receipt": evidence(task["admission_receipt"]),
        "preserved_y41_exact": True, "scope": "Canonical GPU1-only dispatch, not preflight.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"task": selected, "events": events, "preserved_y41_exact": True}, indent=2))
finally:
    monitor.ssh = original_ssh
    lock.close()
