"""Bind fresh Y46 evidence and live GPU blockers to this held lease."""

import argparse
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
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["allowed_gpus"] == [0, 1]


def evidence(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def save(path, value):
    assert not (ROOT / path).exists()
    (ROOT / path).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return evidence(path)


def result(stem):
    path = "pipeline/analyses/" + stem + "_" + args.token + ".json"
    value = json.loads((ROOT / path).read_text())
    assert value["complete"] and value["id"] == "AdaRES-Y46"
    return evidence(path), value


capture_ref, capture = result("y46_native_capture_d179")
cpu_ref, cpu = result("y46_component_cpu_d179_v2")
parent_ref, parent = result("y46_parent_controls_d179")
assert capture["native_scenes"] == cpu["native_scenes"] == 8
assert cpu["native_expressions"] == 32 and cpu["minimum_new_tensor_gradient"] > 0
assert parent["maximum_actual_parent_difference"] == 0
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text())
    y46 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y46")
    assert y46["stage"] == "diagnosis"
    queue = monitor.load_queue(str(ROOT))
    task = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y45")
    assert task["status"] == "running" and task["launch_verified"] and int(task["gpu"]) == 1
    program = """
import json,subprocess,datetime
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
rows=[]
for gpu in (0,1):
    raw=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"]).decode().strip()
    util,used,total=map(int,raw.split(","))
    pids=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader,nounits"]).decode().strip()
    rows.append(dict(gpu=gpu,utilization=util,memory_MiB=used,total_MiB=total,pids=[int(p) for p in pids.splitlines() if p.strip()]))
print(json.dumps(dict(complete=True,hostname="dl01-MS-7C60",GPUs=rows,observed_at_server=datetime.datetime.now().astimezone().isoformat())))
"""
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "songcl@192.168.1.163",
         server["python"] + " -c " + shlex.quote(program)], capture_output=True, timeout=25)
    assert response.returncode == 0
    live = json.loads(response.stdout)
    assert all(row["pids"] for row in live["GPUs"]), "GPU opening: do not fabricate external block"
    assert task["pid"] in live["GPUs"][1]["pids"]
    assert task["pid"] not in live["GPUs"][0]["pids"]
    live.update(observed_at_local=datetime.now().astimezone().isoformat(),
                scientific_status="GPU0_external_job_preserved_GPU1_verified_Y45")
    live_ref = save("pipeline/analyses/y46_live_block_d179_" + args.token + ".json", live)
    sources = [evidence("scripts/probe_y46_component_cpu_d179_v2.py"),
               evidence("scripts/probe_y46_native_capture_d179.py"),
               evidence("scripts/probe_y46_parent_controls_d179.py"),
               evidence("pipeline/analyses/Y46_topology_prereg_d179.md"),
               evidence("pipeline/analyses/Y46_parent_controls_prereg_d179.md")]
    failed = [evidence("scripts/probe_y46_component_cpu_d179.py"),
              evidence("pipeline/analyses/y46_component_cpu_d179_" + args.token + ".log")]
    freeze_ref = save("pipeline/analyses/y46_next_native_freeze_d179_" + args.token + ".json", dict(
        complete=True, id="AdaRES-Y46", source_evidence=sources,
        capture=capture_ref, CPU_controls=cpu_ref, actual_parent_controls=parent_ref,
        retained_serialization_failure=failed,
        next_test="On the first live-idle authorized server2 GPU, test SAME frozen all-component prototype attached to final W2 indicator's native loss on TRAIN8: all8 finite nonzero gradients, disable-parent/cyclic-wrong/cached-repeat, paired time/memory and real tools.train optimizer update. Then build/review independent production package only if that native feasibility is supported.",
        protocol="Keep every FG/BG component; inherited exact-nearest-tie graph, masks>0, point mass/native256; no validation tuning, component filtering, diagnostic weight deployment or admission.",
        constraints=evidence("pipeline/goal.json"),
        limitations="CPU prototype square-loss gradients are not native backward. CPU302.85s on32 expressions is not paired deployment latency; native overhead is unmeasured. Physical production package, full backward and resource gates remain pending."))
    retry = (datetime.now().astimezone() + timedelta(hours=1)).isoformat()
    refs = [capture_ref, cpu_ref, parent_ref, live_ref, freeze_ref, *sources, *failed]
    y46.update(
        stage="blocked", scientific_status="diagnostic_information_supported_native_feasibility_blocked",
        blocker="Both authorized server2 GPUs have real compute processes: GPU0 external job, GPU1 verified Y45. The next discriminating full native loss/resource test needs an idle exclusively leased GPU.",
        blocker_kind="live_external_gpu_occupation", required_server="server-2",
        retry_after=retry,
        unblock_condition="An authorized server2 GPU0/1 becomes live-idle with no compute PID; acquire exclusive diagnostic lease while preserving all other jobs.",
        next_action="Resume SAME Y46 frozen full native loss/resource test before production package or admission; do not repeat unchanged CPU/incidence controls.",
        attempted_alternatives=["32 actual Y36 local graph witnesses", "native TRAIN8/32 expressions/128 queries capture",
                                "32 all-component versus pooled CPU fixtures and all32 captured expressions",
                                "32 actual W2/Y31/Y36 normalized-shape/local-summary controls"],
        pending_gates=["native_backward", "resources", "independence", "constraints"],
        diagnostic_gates={"diagnosis": "supported on registered controls, not universal blindness",
                          "controls": "CPU prototype separated and permutation/repeat checked; native intervention still pending"})
    y46["evidence"].extend(ref for ref in refs if ref not in y46["evidence"])
    work["design_cycle"] = dict(
        stop_reason="external_resource_blocked", attempted_ids=["AdaRES-Y45", "AdaRES-Y46"],
        evidence=[live_ref, freeze_ref, cpu_ref, parent_ref],
        resume_condition=y46["unblock_condition"] + " Native loss/resources decide feasibility, not more cached reviews.")
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(id=y46["id"], stage=y46["stage"], retry_after=retry,
                          evidence=refs, GPUs=live["GPUs"], observed_at_local=live["observed_at_local"]), indent=2))
finally:
    writer.close()
