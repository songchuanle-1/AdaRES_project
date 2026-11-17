"""Admit only actual Y37 gates, then dispatch GPU0 while preserving Y36."""

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
assert (datetime.fromisoformat(owner["deadline_at"])-datetime.now(timezone.utc)).total_seconds() > 360
commit = closure.git(ROOT, "rev-parse", "HEAD")
assert commit == args.commit and len(commit) == 40
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["user"] == "songcl"
assert server["allowed_gpus"] == [0, 1]


def evidence(relative):
    return dict(path=relative, sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())


assert evidence("pipeline/goal.json")["sha256"] == "53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966"
base = "pipeline/analyses/"
bundle_path = base+"y37_source_bundle_lf_"+args.token+".json"
bundle = json.loads((ROOT / bundle_path).read_text())
assert bundle["complete"] and len(bundle["files"]) == 7
for item in bundle["files"]:
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", commit+":"+item["path"]])
    assert hashlib.sha256(blob).hexdigest() == item["sha256"] and evidence(item["path"]) == item
paths = {key: base+"y37_"+name+"_"+args.token+".json" for key, name in (
    ("diagnosis", "pair_diagnosis"), ("controls", "native_controls_v3"),
    ("native", "physical_native"), ("resources", "resources"),
    ("structure", "structure_cpu_lf"), ("investigation", "initial_loss"),
)}
results = {k: json.loads((ROOT / p).read_text()) for k, p in paths.items()}
assert all(r["complete"] and r["id"] == "AdaRES-Y37" for r in results.values())
r = results["diagnosis"]
assert r["scans"] == 64 and r["expressions"] == 256
assert r["expressions_with_oracle_gap"] > 0
assert r["positive_pair_contrast_scans"] == r["wrong_masks_nonzero_scans"] == r["wrong_pairs_nonzero_scans"] == 64
assert r["repeat_max"] == 0 and r["permutation_max"] < 1e-5
r = results["controls"]
assert r["added_trainable_tensors"] == 8 and r["minimum_added_gradient"] > 0
assert r["cached_repeat_max"] == r["mask_detach_forward_max"] == r["nonindicator_loss_delta_max"] == 0
assert min(r["mask_gradient_delta_min"], r["wrong_masks_min"], r["wrong_pairs_min"]) > 0
assert r["permutation_max"] < 1e-5
r = results["native"]
assert r["native_optimizer_steps"] == 2 and len(r["peer_parameter_changes"]) == 8
assert all(v > 0 for v in r["peer_parameter_changes"].values())
assert all(r["resume_before_continuation"][k]["equal"] for k in ("model", "optimizer", "scheduler"))
r = results["resources"]
assert r["summary"]["candidate"]["stress_train_reserved"] < 24
assert r["parameters_added"] == 264194 < .01*r["summary"]["parent"]["parameters"]
r = results["structure"]
assert r["cpu_cases"] == 32 and r["new_tensors"] == 8
fixture = r["equal_moment_fixture"]
assert fixture["pair_readout_delta"] > 0 and fixture["w2_delta"] == fixture["y31_delta"] == fixture["y36_delta"] == 0
assert r["invariance_max"] < 1e-5 and r["minimum_gradient"] > 0
assert not results["investigation"]["source_weights_or_production_checkpoint_changed"]
review = base+"Y37_admission_review_e636ce91.md"
design = "pipeline/designs/exp_design_Y37_e636ce91.md"
motivation = "pipeline/reviews/motivation_review_Y37_e636ce91.md"
assert args.decision.startswith("pipeline/decisions/")
for path in (review, design, motivation, args.decision):
    assert (ROOT / path).is_file()
gates = dict(
    diagnosis=[paths["diagnosis"], base+"Y37_diagnosis_prereg_e636ce91.md", base+"Y37_realization_freeze_e636ce91.md"],
    controls=[paths["controls"], paths["investigation"], paths["structure"]],
    native_backward=[paths["native"], paths["controls"]],
    resources=[paths["resources"]],
    independence=[bundle_path, paths["structure"]],
    constraints=[review, design, motivation],
)
receipt = dict(
    id="AdaRES-Y37", config="configs_song/model_paper5/adares_y37_pair_distribution_res.yaml",
    code_commit=commit, goal_sha256=evidence("pipeline/goal.json")["sha256"],
    decision=evidence(args.decision),
    gates={k: dict(passed=True, evidence=[evidence(p) for p in refs]) for k, refs in gates.items()},
    scientific_status="admitted_for_falsifiable_training_not_efficacy",
    frozen_protocol=dict(epochs=30, seed=1999, server="server-2", allowed_gpus=[0],
                         post_training_design=evidence(design)),
    limitations="Fixed kernel orders not a uniquely derived optimum; adverse/wrong-control losses, numerical continuation and resource costs retained. No trained efficacy, novelty, semantic necessity, Pareto or human-approval assertion.",
)
receipt_path = ROOT / "pipeline/admissions/AdaRES-Y37.json"
assert not receipt_path.exists(), "Never replay an admission"
task = dict(
    id="AdaRES-Y37", label="Y37 final mask global-pair-distribution confidence",
    config=receipt["config"], status="waiting", depends_on=[], code_commit=commit,
    admission_receipt="pipeline/admissions/AdaRES-Y37.json",
    requires_exclusive_ownership=True, min_epochs=30, patience=30, total_epochs=30,
)
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
    assert not any(t["id"] == task["id"] for tasks in queue["gpu_queues"].values() for t in tasks)
    active = [t for tasks in queue["gpu_queues"].values() for t in tasks
              if t["status"] in ("running", "waiting", "claiming")]
    assert len(active) == 1 and active[0]["id"] == "AdaRES-Y36"
    assert active[0]["pid"] == 100018 and str(active[0]["gpu"]) == "1"
    preserved = copy.deepcopy(active[0])
    verify = """
import hashlib,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
assert pathlib.Path('/proc/100018/cwd').resolve()==root/'AdaRES'
argv=pathlib.Path('/proc/100018/cmdline').read_bytes().split(b'\\0')
assert b'tools/train.py' in argv and b'configs_song/model_paper5/adares_y36_local_adjacency_res.yaml' in argv
assert argv[argv.index(b'--gpu_ids')+1]==b'1'
print(json.dumps([dict(path=r['path'],sha256=hashlib.sha256((root/r['path']).read_bytes()).hexdigest()) for r in rows]))
"""
    command = server["python"]+" -c "+shlex.quote("rows="+repr(bundle["files"])+"\n"+verify)
    assert json.loads(scoped_ssh(server["host"], server["user"], command)) == bundle["files"]
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    closure.admission(ROOT, task)
    queue["gpu_queues"]["server-2"].append(task)
    monitor.save_queue(str(ROOT), queue)
    events = []
    monitor._process_queue(
        queue, "server-2", server["host"], server["user"], str(ROOT), {"1"}, {"0"}, events,
        max_tasks_per_gpu=1, remote_code_root=server["remote_code_root"],
        current_tasks=1, max_tasks_total=2,
    )
    latest = monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
    assert next(t for t in latest if t["id"] == "AdaRES-Y36") == preserved
    selected = next(t for t in latest if t["id"] == "AdaRES-Y37")
    result = dict(observed_at_local=datetime.now().astimezone().isoformat(), events=events,
                  task=selected, receipt=evidence("pipeline/admissions/AdaRES-Y37.json"),
                  preserved_y36_exact=True,
                  scope="Canonical gated GPU0-only dispatcher, not a repeated preflight.")
    output = ROOT / (base+"y37_dispatch_"+args.token+".json")
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
finally:
    monitor.ssh = original_ssh
    lock.close()
