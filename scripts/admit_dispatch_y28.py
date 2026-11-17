"""Bind reviewed Y28 evidence, then run canonical whitelist-scoped dispatch."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
    return dict(path=path, sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())


bundle = json.loads((ROOT/("pipeline/analyses/y28_source_bundle_"+args.token+".json")).read_text())
assert bundle["complete"] and len(bundle["files"]) == 7
for item in bundle["files"]:
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", commit+":"+item["path"]])
    assert hashlib.sha256(blob).hexdigest() == item["sha256"] and evidence(item["path"]) == item
base = "pipeline/analyses/"
support = base+"y28_readback_diagnosis_"+args.token+".json"
controls = base+"y28_native_controls_"+args.token+".json"
native = base+"y28_physical_native_"+args.token+".json"
resources = base+"y28_resources_"+args.token+".json"
structure = base+"y28_structure_cpu_v2_"+args.token+".json"
investigation = base+"y28_initial_loss_"+args.token+".json"
for path in (support, controls, native, resources, structure, investigation):
    assert json.loads((ROOT/path).read_text())["complete"]
actual = json.loads((ROOT/controls).read_text())
assert actual["added_trainable_tensors"] == 4 and actual["minimum_added_gradient"] > 0
assert actual["cached_repeat_max"] == actual["cached_detach_max"] == 0
assert actual["pool_gradient_delta_min"] > 0 and actual["pool_detach_forward_max"] == 0
assert actual["nonindicator_loss_delta_max"] == 0
assert actual["cached_roll_min"] > 0 and actual["cached_uniform_min"] > 0
actual = json.loads((ROOT/native).read_text())
assert actual["native_optimizer_steps"] == 2
assert all(actual["resume_before_continuation"][k]["equal"] for k in ("model", "optimizer", "scheduler"))
actual = json.loads((ROOT/resources).read_text())
assert actual["summary"]["candidate"]["stress_train_reserved"] < 24
assert actual["parameters_added"] == 262656
assert actual["parameters_added"] < actual["summary"]["parent"]["parameters"] * .01
actual = json.loads((ROOT/structure).read_text())
assert actual["structural_new_mask_visual_aggregation"] and actual["cpu_cases"] == 32
actual = json.loads((ROOT/investigation).read_text())
assert not actual["source_weights_or_production_checkpoint_changed"]
review = base+"Y28_admission_review_7f5e7b9d.md"
design = "pipeline/designs/exp_design_Y28_7f5e7b9d.md"
decision = args.decision
assert decision.startswith("pipeline/decisions/") and (ROOT/decision).is_file()
names = dict(diagnosis=[support, "pipeline/designs/Y28_diagnosis_prereg_7f5e7b9d.md"],
             controls=[controls, investigation, native],
             native_backward=[native],
             resources=[resources],
             independence=["pipeline/analyses/y28_source_bundle_"+args.token+".json", structure],
             constraints=[review, design, "pipeline/reviews/motivation_review_Y28_7f5e7b9d.md"])
receipt = dict(id="AdaRES-Y28", config="configs_song/model_paper5/adares_y28_mask_readback_res.yaml",
               code_commit=commit, goal_sha256=evidence("pipeline/goal.json")["sha256"],
               decision=evidence(decision),
               gates={name: dict(passed=True, evidence=[evidence(p) for p in paths])
                      for name, paths in names.items()},
               scientific_status="admitted_for_falsifiable_training_not_efficacy",
               frozen_protocol=dict(epochs=30, seed=1999, server="server-2", allowed_gpus=[1],
                                    post_training_design=evidence(design)),
               limitations="Mixed initial losses and wrong-content controls also learning, native resume noise, CPU newline assertion failure and measured inference regression preserved. No trained benefit/harm/necessity/Pareto/novelty claim.")
receipt_path = ROOT/"pipeline/admissions/AdaRES-Y28.json"
assert not receipt_path.exists(), "Existing admission: reconcile, never replay blindly"
task = dict(id="AdaRES-Y28", label="Y28 predicted-mask visual-content readback",
            config=receipt["config"], status="waiting", depends_on=[], code_commit=commit,
            admission_receipt="pipeline/admissions/AdaRES-Y28.json",
            requires_exclusive_ownership=True, min_epochs=30, patience=30, total_epochs=30)
original_ssh = monitor.ssh


def scoped_ssh(host, user, command, *positional, **keywords):
    assert host == server["host"] and user == server["user"]
    inventory = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total"
    if command.startswith(inventory):
        command = command.replace("nvidia-smi ", "nvidia-smi -i 1 ", 1)
    if "nvidia-smi" in command:
        assert "nvidia-smi -i 1 " in command, "Refuse any unscoped GPU query"
    return original_ssh(host, user, command, *positional, **keywords)


monitor.ssh = scoped_ssh
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None, "Writer busy"
try:
    queue = monitor.load_queue(str(ROOT))
    assert not any(t["id"] == task["id"] for tasks in queue["gpu_queues"].values() for t in tasks)
    active = [t for tasks in queue["gpu_queues"].values() for t in tasks
              if t["status"] in ("running", "waiting", "claiming")]
    assert not active, "Capacity changed; do not displace a job"
    processes = monitor.find_running_processes(server["host"], server["user"], server["remote_code_root"])
    assert not processes, "Actual project training already present"
    # Re-read exact deployed candidate bytes immediately before admitting it.
    verify = "import hashlib,json,pathlib,subprocess;assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60';root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project');rows="+repr(bundle["files"])+";print(json.dumps([dict(path=r['path'],sha256=hashlib.sha256((root/r['path']).read_bytes()).hexdigest()) for r in rows]))"
    import shlex
    assert json.loads(scoped_ssh(server["host"], server["user"],
                                server["python"]+" -c "+shlex.quote(verify))) == bundle["files"]
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
                  task=selected, receipt=evidence("pipeline/admissions/AdaRES-Y28.json"),
                  scope="Canonical gated dispatcher; GPU inventory constrained to1, no monitor/goal preflight repeated.")
    out = ROOT/(base+"y28_dispatch_"+args.token+".json")
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
finally:
    monitor.ssh = original_ssh
    lock.close()
