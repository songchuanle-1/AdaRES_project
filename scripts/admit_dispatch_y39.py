"""Admit the actual frozen Y39 gates and dispatch only GPU1, preserving Y37."""
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
commit = closure.git(ROOT, "rev-parse", "HEAD")
assert commit == args.commit and len(commit) == 40
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["user"] == "songcl" and server["allowed_gpus"] == [0, 1]


def evidence(relative):
    return dict(path=relative, sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())


assert evidence("pipeline/goal.json")["sha256"] == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
base = "pipeline/analyses/"
bundle_path = base + "y39_source_bundle_lf_" + args.token + ".json"
bundle = json.loads((ROOT / bundle_path).read_text(encoding="utf-8"))
assert bundle["complete"] and len(bundle["files"]) == 7
for item in bundle["files"]:
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + item["path"]])
    assert hashlib.sha256(blob).hexdigest() == item["sha256"] and evidence(item["path"]) == item
names = ("pooling_cpu_v2", "train_ceiling", "readout_cpu_lf", "extract_cpu_lf",
         "native_controls", "physical_native", "resources", "initial_loss")
paths = {name: base + "y39_" + name + "_" + args.token + ".json" for name in names}
results = {name: json.loads((ROOT / path).read_text(encoding="utf-8")) for name, path in paths.items()}
assert all(item["complete"] for item in results.values())
r = results["train_ceiling"]
assert r["scans"] == 64 and r["expressions"] == 256
r = results["readout_cpu_lf"]
assert r["fixtures"] == 32 and r["minimum_gradient"] > 0
assert r["repeat_max"] == r["padding_poison_max"] == 0 and r["permutation_max"] < 1e-5
assert min(r["wrong_mask_min"], r["wrong_variance_min"]) > 0
r = results["extract_cpu_lf"]
assert r["algebra_max"] < 1e-5 and r["minimum_raw_gradient"] > 0
r = results["native_controls"]
assert r["added_trainable_tensors"] == 8 and r["minimum_added_gradient"] > 0
assert r["cached_repeat_max"] == r["nonindicator_loss_delta_max"] == 0
assert min(r["wrong_masks_min"], r["wrong_variance_min"]) > 0 and r["permutation_max"] < 1e-5
assert r["y37_source_unchanged"]
r = results["physical_native"]
assert r["native_optimizer_steps"] == 2 and len(r["peer_parameter_changes"]) == 8
assert all(value > 0 for value in r["peer_parameter_changes"].values())
assert all(r["resume_before_continuation"][key]["equal"] for key in ("model", "optimizer", "scheduler"))
assert all(row["disabled_parent_mask_max"] == row["disabled_parent_indicator_max"] == row["disabled_parent_sample_mismatches"] == 0 for row in r["cached_inference_rows"])
r = results["resources"]
assert r["summary"]["candidate"]["stress_train_reserved"] < 24
assert r["parameters_added"] == 271618 < .01 * r["summary"]["parent"]["parameters"]
assert not results["initial_loss"]["source_weights_or_production_checkpoint_changed"]
review = base + "Y39_admission_review_48678b81.md"
design = "pipeline/designs/exp_design_Y39_48678b81.md"
motivation = "pipeline/reviews/motivation_review_Y39_48678b81.md"
assert args.decision.startswith("pipeline/decisions/")
for path in (review, design, motivation, args.decision):
    assert (ROOT / path).is_file()
gates = dict(
    diagnosis=[paths["pooling_cpu_v2"], paths["train_ceiling"], paths["extract_cpu_lf"],
               base + "Y39_diagnosis_prereg_48678b81.md", base + "Y39_realization_freeze_48678b81.md"],
    controls=[paths["native_controls"], paths["initial_loss"], paths["readout_cpu_lf"]],
    native_backward=[paths["physical_native"], paths["native_controls"]],
    resources=[paths["resources"], base + "Y39_resource_freeze_48678b81.md"],
    independence=[bundle_path, paths["extract_cpu_lf"], paths["readout_cpu_lf"]],
    constraints=[review, design, motivation],
)
receipt = dict(
    id="AdaRES-Y39", config="configs_song/model_paper5/adares_y39_point_distribution_res.yaml",
    code_commit=commit, goal_sha256=evidence("pipeline/goal.json")["sha256"],
    decision=evidence(args.decision),
    gates={key: dict(passed=True, evidence=[evidence(path) for path in refs]) for key, refs in gates.items()},
    scientific_status="admitted_for_falsifiable_training_not_efficacy",
    frozen_protocol=dict(epochs=30, seed=1999, server="server-2", allowed_gpus=[1],
                         post_training_design=evidence(design)),
    limitations="Near-perfect TRAIN partition ceiling is negative evidence, not a main quantization-bottleneck claim. Initial adverse loss and all wrong controls retained; learned means may encode heterogeneity. No trained benefit, novelty, necessity, no-harm, Pareto or human-approval claim.",
)
receipt_path = ROOT / "pipeline/admissions/AdaRES-Y39.json"
assert not receipt_path.exists(), "Never replay an admission"
task = dict(
    id="AdaRES-Y39", label="Y39 final within-superpoint feature-distribution confidence",
    config=receipt["config"], status="waiting", depends_on=[], code_commit=commit,
    admission_receipt="pipeline/admissions/AdaRES-Y39.json",
    requires_exclusive_ownership=True, min_epochs=30, patience=30, total_epochs=30,
)
old_receipt = json.loads((ROOT / "pipeline/admissions/AdaRES-Y37.json").read_text())
old_bundle = json.loads((ROOT / old_receipt["gates"]["independence"]["evidence"][0]["path"]).read_text())
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
    assert not any(entry["id"] == task["id"] for entries in queue["gpu_queues"].values() for entry in entries)
    active = [entry for entries in queue["gpu_queues"].values() for entry in entries
              if entry["status"] in ("running", "waiting", "claiming")]
    assert len(active) == 1 and active[0]["id"] == "AdaRES-Y37"
    assert active[0]["pid"] == 20348 and str(active[0]["gpu"]) == "0"
    preserved = copy.deepcopy(active[0])
    program = """
import hashlib,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
assert pathlib.Path('/proc/20348/cwd').resolve()==root/'AdaRES'
argv=pathlib.Path('/proc/20348/cmdline').read_bytes().split(b'\\0')
assert b'tools/train.py' in argv and b'configs_song/model_paper5/adares_y37_pair_distribution_res.yaml' in argv
assert argv[argv.index(b'--gpu_ids')+1]==b'0'
for row in P['old']:
    assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
print(json.dumps([dict(path=row['path'],sha256=hashlib.sha256((root/row['path']).read_bytes()).hexdigest()) for row in P['new']]))
"""
    payload = dict(old=old_bundle["files"], new=bundle["files"])
    command = server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)
    assert json.loads(scoped_ssh(server["host"], server["user"], command)) == bundle["files"]
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    closure.admission(ROOT, task)
    queue["gpu_queues"]["server-2"].append(task)
    monitor.save_queue(str(ROOT), queue)
    events = []
    monitor._process_queue(queue, "server-2", server["host"], server["user"], str(ROOT),
                           {"0"}, {"1"}, events, max_tasks_per_gpu=1,
                           remote_code_root=server["remote_code_root"],
                           current_tasks=1, max_tasks_total=2)
    latest = monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
    assert next(entry for entry in latest if entry["id"] == "AdaRES-Y37") == preserved
    selected = next(entry for entry in latest if entry["id"] == "AdaRES-Y39")
    result = dict(observed_at_local=datetime.now().astimezone().isoformat(), events=events,
                  task=selected, receipt=evidence("pipeline/admissions/AdaRES-Y39.json"),
                  preserved_y37_exact=True,
                  scope="Canonical gated GPU1-only dispatcher; not another preflight.")
    output = ROOT / (base + "y39_dispatch_" + args.token + ".json")
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
finally:
    monitor.ssh = original_ssh
    lock.close()
