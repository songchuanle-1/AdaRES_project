"""Reconcile exact Y45 continuation and persist Y46/Y47 honest dispositions."""

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--decision", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 150


def evidence(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def save(path, value):
    assert not (ROOT / path).exists()
    (ROOT / path).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return evidence(path)


def read_result(stem, cid):
    path = "pipeline/analyses/" + stem + "_" + args.token + ".json"
    value = json.loads((ROOT / path).read_text())
    assert value["complete"] and value["id"] == cid
    return evidence(path), value


y47_ref, y47_result = read_result("y47_actual_overlap_3c6a", "AdaRES-Y47")
y46_ref, y46_result = read_result("y46_native_hook_cpu_3c6a", "AdaRES-Y46")
assert y47_result["all_actual_parent_distinguish_equal_marginal_fixtures"]
assert y46_result["CPU_cases"] == 32 and y46_result["native_backward_executed"] is False
old_source = "scripts/probe_y46_component_cpu_d179_v2.py"
new_source = "scripts/probe_y46_native_gate_3c6a.py"
old_tree, new_tree = [ast.parse((ROOT / path).read_text()) for path in (old_source, new_source)]
for name in ("component_inputs", "AllComponentReadout"):
    extract = lambda tree: next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    assert ast.dump(extract(old_tree)) == ast.dump(extract(new_tree))
compile(new_tree, new_source, "exec")
assert args.decision.startswith("pipeline/decisions/")
decision_ref = evidence(args.decision)
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["allowed_gpus"] == [0, 1]
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text())
    candidates = {c["id"]: c for c in work["candidates"]}
    assert candidates["AdaRES-Y47"]["stage"] == "diagnosis"
    assert candidates["AdaRES-Y46"]["stage"] == "blocked"
    task = next(t for t in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y45")
    assert task["status"] == "running" and task["launch_verified"] and int(task["gpu"]) == 1
    bundle = json.loads((ROOT / "pipeline/analyses/y45_source_bundle_392a_392a5a9f5fe242d7a3109a50e6060fda.json").read_text())
    program = r'''
import hashlib,json,pathlib,re,subprocess,time
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
code=root/"AdaRES"
t=P["task"]
pid=t["pid"]
assert pathlib.Path("/proc/"+str(pid)+"/cwd").resolve()==code
argv=pathlib.Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(b"\0")
assert b"tools/train.py" in argv and t["config"].encode() in argv
assert argv[argv.index(b"--gpu_ids")+1]==b"1"
assert (code/".codex_source_commit").read_text().strip()==t["code_commit"]
for row in P["files"]:
    assert hashlib.sha256((root/row["path"]).read_bytes()).hexdigest()==row["sha256"]
captures=[]
until=time.monotonic()+90
while len(captures)<2:
    assert pathlib.Path("/proc/"+str(pid)).exists()
    with open(t["log"],"rb") as stream:
        stream.seek(0,2)
        stream.seek(max(0,stream.tell()-32000))
        text=stream.read().decode("utf-8",errors="replace")
    assert not any(s in text for s in ("Traceback","CUDA out of memory","RuntimeError"))
    steps=[tuple(map(int,m)) for m in re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)",text)]
    if steps:
        epoch,total,step,totalsteps=steps[-1]
        assert total==30
        if not captures or (epoch,step)>(captures[0]["epoch"],captures[0]["step"]):
            captures.append(dict(epoch=epoch,epochs=total,step=step,total_steps=totalsteps))
    if len(captures)<2:
        if time.monotonic()>=until:
            break
        time.sleep(5)
rows=[]
for gpu in (0,1):
    util,used,total=map(int,subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"]).decode().strip().split(","))
    raw=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader,nounits"]).decode().strip()
    rows.append(dict(gpu=gpu,utilization=util,memory_MiB=used,total_MiB=total,pids=[int(p) for p in raw.splitlines() if p.strip()]))
assert pid in rows[1]["pids"]
print(json.dumps(dict(complete=True,id="AdaRES-Y45",pid=pid,config=t["config"],gpu=1,
                     source_hashes_verified=True,captures=captures,
                     native_steps_increased=len(captures)==2,GPUs=rows,
                     limitations=["Bounded log and native steps, not whole-log crash absence or trained efficacy.",
                                  "Server wall-clock not used for deadline/retry/observation timing."])))
'''
    started = datetime.now().astimezone().isoformat()
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "songcl@192.168.1.163",
         server["python"] + " -c " + shlex.quote("P=" + repr(dict(task=task, files=bundle["files"])) + "\n" + program)],
        capture_output=True, timeout=120)
    raw_path = "pipeline/analyses/y45_continuation_3c6a_" + args.token + ".log"
    assert not (ROOT / raw_path).exists()
    (ROOT / raw_path).write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0, "Preserve observation failure; never blind restart"
    observed = json.loads(response.stdout)
    observed.update(observed_start_local=started, observed_end_local=datetime.now().astimezone().isoformat())
    live_ref = save("pipeline/analyses/y45_continuation_3c6a_" + args.token + ".json", observed)
    assert observed["native_steps_increased"], "No fresh increasing steps; reconcile without relaunch"
    assert all(row["pids"] for row in observed["GPUs"]), "GPU opening: do not record fake external block"
    assert task["pid"] not in observed["GPUs"][0]["pids"]
    freeze_ref = save("pipeline/analyses/y46_native_gate_freeze_3c6a_" + args.token + ".json", dict(
        complete=True, id="AdaRES-Y46", code=evidence(new_source), CPU_hook_test=y46_ref,
        frozen_prototype=evidence(old_source), frozen_class_and_component_function_AST_identical=True,
        previous_freeze=evidence("pipeline/analyses/y46_next_native_freeze_d179_d179d824a64943e0b06bfa90de0caf52.json"),
        native_branch_executed=False, GPU_occupation=live_ref,
        next_test="First live-idle authorized server2 GPU with exclusive diagnostic lease: SAME reviewed native loss/backward paired-cost branch TRAIN8. Then actual tools.train optimizer update, released-init controls and physical independent package/gate review; no admission from CPU surrogate.",
        limitations=["CPU FakeDecoder/synthetic CE is instrumentation only, not native backward.",
                     "No weight/threshold/geometry tuning; inherited all-component prototype unchanged.",
                     "No task-level effect-size threshold or trained resource claim."]))
    review_path = "pipeline/analyses/Y47_rejection_3c6a.md"
    assert not (ROOT / review_path).exists()
    (ROOT / review_path).write_text(
        "# Y47 Missing-Overlap Premise Rejection\n\n"
        "- Actual W2 already computes weighted pairwise intersections, unions,\n"
        "  off-diagonal overlap and overlap-weighted normalized query context.\n"
        "- Preregistered full128/equal-row-mass/equal-point-coverage fixtures:\n"
        "  actual W2 distinguishes all32 seeds; minimum score difference\n"
        "  0.01413108246 versus prospective bound at most2.2737367544e-13.\n"
        "- Actual formula reconstructs outputs exactly; cached repeats exact;\n"
        "  joint point permutation error at most1.8041124150e-16.\n"
        "- Reject this missing-information realization; hard versus soft alone\n"
        "  changes numerics, not the claimed absent correspondence path.\n"
        "- This is not a trained benefit/novelty claim or a rejection of every\n"
        "  pairwise hypothesis. Y29/Y40 remain closed; no renamed admission.\n"
        "- Evidence: " + y47_ref["path"] + " SHA256 " + y47_ref["sha256"] + ".\n",
        encoding="utf-8")
    rejection_ref = evidence(review_path)
    candidates["AdaRES-Y47"].update(
        stage="rejected", scientific_status="missing_correspondence_premise_disconfirmed",
        next_action="Do not admit or rename this duplicate information path. Only a different evidence-backed hypothesis warrants a new diagnosis.",
        rejection_reason="Actual W2 pairwise point-mass overlap/context already distinguishes all32 preregistered equal-marginal fixtures.",
        pending_gates=["native_backward", "resources", "independence", "constraints"])
    candidates["AdaRES-Y47"]["evidence"].extend(
        [y47_ref, rejection_ref, evidence("scripts/probe_y47_actual_overlap_3c6a.py"), decision_ref])
    retry = (datetime.now().astimezone() + timedelta(hours=2)).isoformat()
    y46 = candidates["AdaRES-Y46"]
    y46.update(
        scientific_status="CPU_hook_validated_native_feasibility_blocked",
        blocker="Fresh live GPU0 external compute PID and GPU1 original verified Y45 occupy both authorized cards. New native loss/cost branch is prepared but unrun.",
        retry_after=retry, blocker_kind="live_external_gpu_occupation",
        next_action="Run SAME frozen native loss/cost branch on first actual idle/exclusively leased authorized GPU; then tools.train update/released-init/independence gates before admission.")
    y46["evidence"].extend([y46_ref, freeze_ref, live_ref, evidence(new_source), decision_ref])
    y46["attempted_alternatives"].append(
        "This tick32 CPU final-indicator hook tests: unchanged six auxiliaries, disabled-parent equality, cached repeats, syntheticCE8tensor gradients, wrong correspondence; frozen prototype AST unchanged.")
    candidates["AdaRES-Y45"]["evidence"].append(live_ref)
    candidates["AdaRES-Y45"]["next_action"] = "Preserve SAME verified GPU1 Y45/releasedSPFormer/seed1999/fixed30. Reconcile real changes; terminal frozen full9508/141 twice EACH/FIRST/W2+B1/scene+subgroup harm/repeat/trained cost, no retune or blind restart."
    work["design_cycle"] = dict(
        stop_reason="external_resource_blocked", attempted_ids=["AdaRES-Y46", "AdaRES-Y47"],
        evidence=[live_ref, freeze_ref, y46_ref, y47_ref, rejection_ref, decision_ref],
        resume_condition=y46["unblock_condition"] + " Resume SAME Y46 native gate, do not repeat closed Y47 or unchanged CPU controls.")
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(complete=True, observed=observed, Y46="blocked", Y47="rejected",
                          retry_after=retry, freeze=freeze_ref, rejection=rejection_ref), indent=2))
finally:
    writer.close()
