"""Persist partial Y48 diagnosis and exact continuation without new dispatch."""

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


def result(stem):
    path = "pipeline/analyses/" + stem + "_" + args.token + ".json"
    value = json.loads((ROOT / path).read_text())
    assert value["complete"] and value["id"] == "AdaRES-Y48"
    return evidence(path), value


cpu_ref, cpu = result("y48_exchangeability_0cf9_v2")
bookkeeping_ref, bookkeeping = result("y48_joint_bookkeeping_cpu_0cf9")
assert cpu["maximum_actual_gradient_error"] == 0
assert cpu["nonempty_duplicate_groups"] > 0
assert cpu["actual_native_mixed_label_groups"] is None
assert bookkeeping["native_joint_capture_executed"] is False
native_source = "scripts/probe_y48_joint_native_0cf9.py"
compile((ROOT / native_source).read_text(), native_source, "exec")
# Reuse the exact prior bounded continuation observer, not another monitor run.
observer_source = "scripts/record_adares_3c6a.py"
nodes = [node for node in ast.walk(ast.parse((ROOT / observer_source).read_text()))
         if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "program" for t in node.targets)]
assert len(nodes) == 1 and isinstance(nodes[0].value, ast.Constant)
program = ast.literal_eval(nodes[0].value)
assert isinstance(program, str) and 'root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")' in program
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
    candidate = candidates["AdaRES-Y48"]
    assert candidate["stage"] == "diagnosis"
    assert candidates["AdaRES-Y46"]["stage"] == "blocked"
    task = next(t for t in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y45")
    assert task["status"] == "running" and task["launch_verified"] and int(task["gpu"]) == 1
    bundle = json.loads((ROOT / "pipeline/analyses/y45_source_bundle_392a_392a5a9f5fe242d7a3109a50e6060fda.json").read_text())
    start = datetime.now().astimezone().isoformat()
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "songcl@192.168.1.163",
         server["python"] + " -c " + shlex.quote("P=" + repr(dict(task=task, files=bundle["files"])) + "\n" + program)],
        capture_output=True, timeout=120)
    raw = "pipeline/analyses/y45_continuation_0cf9_" + args.token + ".log"
    assert not (ROOT / raw).exists()
    (ROOT / raw).write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0, "Preserve failed observation, never relaunch blindly"
    observed = json.loads(response.stdout)
    observed.update(observed_start_local=start, observed_end_local=datetime.now().astimezone().isoformat())
    observed_ref = save("pipeline/analyses/y45_continuation_0cf9_" + args.token + ".json", observed)
    assert observed["native_steps_increased"]
    assert all(row["pids"] for row in observed["GPUs"]), "GPU opening: do not fabricate a blocker"
    assert task["pid"] not in observed["GPUs"][0]["pids"]
    supporting_paths = [
        "pipeline/analyses/Y48_exchangeability_prereg_0cf9.md",
        "pipeline/analyses/Y48_native_matching_clarification_0cf9.md",
        "pipeline/analyses/Y48_dtype_contract_v2_0cf9.md",
        "scripts/probe_y48_exchangeability_0cf9.py",
        "pipeline/analyses/y48_exchangeability_0cf9_" + args.token + ".log",
        "scripts/probe_y48_exchangeability_0cf9_v2.py", native_source]
    supports = [evidence(path) for path in supporting_paths]
    freeze_ref = save("pipeline/analyses/y48_joint_native_freeze_0cf9_" + args.token + ".json", dict(
        complete=True, id="AdaRES-Y48", native_capture_source=evidence(native_source),
        source_CPU_controls=cpu_ref, bookkeeping_CPU_controls=bookkeeping_ref,
        source_and_failure_evidence=supports, occupation=observed_ref,
        native_joint_branch_executed=False, actual_native_mixed_label_groups=None,
        nonempty_duplicate_groups=cpu["nonempty_duplicate_groups"],
        nonempty_duplicate_queries=cpu["nonempty_duplicate_queries"],
        initial_single_match_premise="disconfirmed; actual nearest plus all in-instance query matching",
        next_test="First live-idle authorized/exclusively leased server2 GPU: SAME native TRAIN8 joint mask/criterion-match capture. Decide whether nonempty identical-mask groups actually contain both native labels, checking identical literal point TP/FP/FN. No cross-capture inference.",
        limitations="Partial source/fixture/cached mask incidence, not an established native contradiction. If joint capture disconfirms useful mixed-label incidence, reject this realization with the actual result. No production target/weight/mask/selector/model changes or admission."))
    retry = (datetime.now().astimezone() + timedelta(minutes=30)).isoformat()
    candidate.update(
        stage="blocked", scientific_status="partial_diagnosis_native_ownership_conflict_unknown",
        blocker="Both live authorized GPUs are occupied by real compute PIDs: external GPU0 and original Y45 GPU1. Joint native mask/ownership capture requires an idle exclusive CUDA slot; CPU/source/incidence/bookkeeping work executed.",
        blocker_kind="live_external_gpu_occupation", required_server="server-2",
        retry_after=retry,
        unblock_condition="An authorized server2 GPU0/1 becomes live-idle with no compute PID; acquire exclusive diagnostic lease and preserve all existing jobs.",
        next_action="Run SAME frozen joint native TRAIN8 mask/matching capture; actual mixed-label nonempty-group incidence is unknown. Then reject or substantiate diagnosis before target-closure implementation/native gates/admission.",
        attempted_alternatives=["Actual native matcher/loss/indicator source audit, single-winner premise corrected additively",
                                "32 actual-source CPU ownership/gradient/repeat/permutation controls, literal gradient error0",
                                "All32 frozen TRAIN expressions/128query exact bitset duplicate incidence, no query-label joins",
                                "32 joint-capture bookkeeping CPU count/repeat/permutation tests and compiled unrun native branch"],
        diagnostic_results={"native_matching": "already one-to-many",
                            "CPU_fixture_opposition": "observed, not native population evidence",
                            "nonempty_duplicates": cpu["nonempty_duplicate_groups"],
                            "actual_native_mixed_labels": None},
        pending_gates=["diagnosis", "controls", "native_backward", "resources", "independence", "constraints"])
    candidate["evidence"].extend([cpu_ref, bookkeeping_ref, freeze_ref, observed_ref, *supports, decision_ref])
    candidates["AdaRES-Y45"]["evidence"].append(observed_ref)
    # Y46 is not due; preserve its freeze/retry and do not repeat its probes.
    work["design_cycle"] = dict(
        stop_reason="external_resource_blocked", attempted_ids=["AdaRES-Y48"],
        evidence=[cpu_ref, bookkeeping_ref, freeze_ref, observed_ref, decision_ref],
        resume_condition="Observed idle/exclusive authorized GPU opens SAME Y48 joint native capture and SAME Y46 native gate, with stable IDs and original freezes; unchanged CPU/source reviews are not repeated.")
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(complete=True, observed=observed, Y48=candidate["stage"],
                          Y46="unchanged_blocked_not_due", retry_after=retry, freeze=freeze_ref), indent=2))
finally:
    writer.close()
