"""Validate reviewed Y31 artifacts and call the canonical gated dispatcher."""

import argparse
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
assert server["host"] == "192.168.1.163" and server["user"] == "songcl" and server["allowed_gpus"] == [1]


def evidence(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


assert evidence("pipeline/goal.json")["sha256"] == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
base = "pipeline/analyses/"
bundle_path = base + "y31_source_bundle_lf_" + args.token + ".json"
bundle = json.loads((ROOT / bundle_path).read_text())
assert bundle["complete"] and len(bundle["files"]) == 7
for item in bundle["files"]:
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + item["path"]])
    assert hashlib.sha256(blob).hexdigest() == item["sha256"] and evidence(item["path"]) == item
support = base + "y31_shape_diagnosis_" + args.token + ".json"
controls = base + "y31_native_controls_" + args.token + ".json"
native = base + "y31_physical_native_" + args.token + ".json"
resources = base + "y31_resources_" + args.token + ".json"
structure = base + "y31_structure_cpu_lf_" + args.token + ".json"
investigation = base + "y31_initial_loss_" + args.token + ".json"
for path in (support, controls, native, resources, structure, investigation):
    assert json.loads((ROOT / path).read_text())["complete"]
actual = json.loads((ROOT / support).read_text())
assert actual["scans"] == 64 and actual["expressions"] == 256
assert actual["expressions_with_oracle_gap"] > 0 and actual["missed_with_distinct_shape"] > 0
actual = json.loads((ROOT / controls).read_text())
assert actual["added_trainable_tensors"] == 6 and actual["minimum_added_gradient"] > 0
assert actual["cached_repeat_max"] == 0
assert actual["mask_gradient_delta_min"] > 0 and actual["mask_detach_forward_max"] == 0
assert actual["nonindicator_loss_delta_max"] == 0
assert min(actual["wrong_shape_min"], actual["uniform_shape_min"]) > 0
assert actual["permutation_translation_max"] < 1e-5
actual = json.loads((ROOT / native).read_text())
assert actual["native_optimizer_steps"] == 2
assert len(actual["peer_parameter_changes"]) == 6
assert all(v > 0 for v in actual["peer_parameter_changes"].values())
assert all(actual["resume_before_continuation"][k]["equal"] for k in ("model", "optimizer", "scheduler"))
actual = json.loads((ROOT / resources).read_text())
assert actual["summary"]["candidate"]["stress_train_reserved"] < 24
assert actual["parameters_added"] == 133634
assert actual["parameters_added"] < actual["summary"]["parent"]["parameters"] * .01
actual = json.loads((ROOT / structure).read_text())
assert actual["parent_explicit_shape_input_absent"] and actual["cpu_cases"] == 32
actual = json.loads((ROOT / investigation).read_text())
assert not actual["source_weights_or_production_checkpoint_changed"]
review = base + "Y31_admission_review_626ebd89.md"
design = "pipeline/designs/exp_design_Y31_626ebd89.md"
motivation = "pipeline/reviews/motivation_review_Y31_626ebd89.md"
decision = args.decision
assert decision.startswith("pipeline/decisions/") and (ROOT / decision).is_file()
assert (ROOT / review).is_file() and (ROOT / design).is_file()
names = dict(
    diagnosis=[support, base + "Y31_diagnosis_prereg_626ebd89.md"],
    controls=[controls, investigation, structure],
    native_backward=[native, controls],
    resources=[resources],
    independence=[bundle_path, structure],
    constraints=[review, design, motivation])
receipt = dict(
    id="AdaRES-Y31", config="configs_song/model_paper5/adares_y31_mask_shape_res.yaml",
    code_commit=commit, goal_sha256=evidence("pipeline/goal.json")["sha256"],
    decision=evidence(decision),
    gates={name: dict(passed=True, evidence=[evidence(p) for p in paths])
           for name, paths in names.items()},
    scientific_status="admitted_for_falsifiable_training_not_efficacy",
    frozen_protocol=dict(epochs=30, seed=1999, server="server-2", allowed_gpus=[1],
                         post_training_design=evidence(design)),
    limitations="Superpoint-centroid covariance approximation, adverse initial losses, wrong/uniform shape learnability, resume numerical differences and measured cost regressions retained. No trained efficacy, necessity, Pareto or novelty claim.")
receipt_path = ROOT / "pipeline/admissions/AdaRES-Y31.json"
assert not receipt_path.exists(), "Existing admission: reconcile, never replay blindly"
task = dict(id="AdaRES-Y31", label="Y31 mask-supported spatial shape confidence",
            config=receipt["config"], status="waiting", depends_on=[], code_commit=commit,
            admission_receipt="pipeline/admissions/AdaRES-Y31.json",
            requires_exclusive_ownership=True, min_epochs=30, patience=30, total_epochs=30)
original_ssh = monitor.ssh


def scoped_ssh(host, user, command, *positional, **keywords):
    assert host == server["host"] and user == server["user"]
    inventory = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total"
    if command.startswith(inventory):
        command = command.replace("nvidia-smi ", "nvidia-smi -i 1 ", 1)
    if "nvidia-smi" in command:
        assert "nvidia-smi -i 1 " in command, "Refuse unscoped GPU query"
    return original_ssh(host, user, command, *positional, **keywords)


monitor.ssh = scoped_ssh
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None, "Writer busy"
try:
    queue = monitor.load_queue(str(ROOT))
    assert not any(t["id"] == task["id"] for tasks in queue["gpu_queues"].values() for t in tasks)
    assert not any(t["status"] in ("running", "waiting", "claiming")
                   for tasks in queue["gpu_queues"].values() for t in tasks), "Capacity changed"
    processes = monitor.find_running_processes(server["host"], server["user"], server["remote_code_root"])
    assert not processes, "Project training already present"
    verify = ("import hashlib,json,pathlib,subprocess;"
              "assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60';"
              "root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project');rows=" + repr(bundle["files"]) +
              ";print(json.dumps([dict(path=r['path'],sha256=hashlib.sha256((root/r['path']).read_bytes()).hexdigest()) for r in rows]))")
    assert json.loads(scoped_ssh(server["host"], server["user"],
                                server["python"] + " -c " + shlex.quote(verify))) == bundle["files"]
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    closure.admission(ROOT, task)
    queue["gpu_queues"]["server-2"].append(task)
    monitor.save_queue(str(ROOT), queue)
    events = []
    monitor._process_queue(queue, "server-2", server["host"], server["user"], str(ROOT),
                           set(), {"1"}, events, max_tasks_per_gpu=1,
                           remote_code_root=server["remote_code_root"], current_tasks=0,
                           max_tasks_total=1)
    selected = next(t for t in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
                    if t["id"] == task["id"])
    result = dict(observed_at_local=datetime.now().astimezone().isoformat(), events=events,
                  task=selected, receipt=evidence("pipeline/admissions/AdaRES-Y31.json"),
                  scope="Canonical gated dispatcher; GPU inventory constrained to1, no repeated preflight.")
    out = ROOT / (base + "y31_dispatch_" + args.token + ".json")
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
finally:
    monitor.ssh = original_ssh
    lock.close()
