"""Continue the two admitted candidates through the canonical gated dispatcher."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import authorize_prepared_candidates as scoped
import closure
import foreground_tick as tick
import pipeline_monitor as monitor

IDS = {"AdaRES-Y52", "AdaRES-Y54"}
CHINA = timezone(timedelta(hours=8))


def snapshot(server, tasks, files):
    program = """
import hashlib,json,pathlib,subprocess
root=pathlib.Path(ROOT)
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
assert root.is_dir() and pathlib.Path(DATA).is_dir()
hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest()
        for p in FILES if (root/p).is_file()}
rows={}
for task in TASKS:
    log=pathlib.Path(task['log']) if task.get('log') else None
    text=''
    if log and log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-24000))
            text=stream.read().decode('utf-8',errors='replace')
    pid=task.get('pid')
    proc=pathlib.Path('/proc')/str(pid) if pid else None
    live=bool(proc and (proc/'cmdline').is_file())
    argv=(proc/'cmdline').read_bytes().split(b'\\0') if live else []
    rows[task['id']]=dict(pid=pid,log=str(log) if log else None,
        process_live=live,cmdline=[v.decode(errors='replace') for v in argv if v],
        cwd=str((proc/'cwd').resolve()) if live else None,tail=text)
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total',
    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
compute=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
print(json.dumps(dict(hostname='dl01-MS-7C60',source_marker=(root/'.codex_source_commit').read_text().strip(),
    source_hashes=hashes,authorized_device_inventory=gpu.stdout,gpu_query_rc=gpu.returncode,
    compute=compute.stdout,compute_query_rc=compute.returncode,tasks=rows)))
"""
    program = "\n".join([
        "ROOT=" + repr(server["remote_code_root"]),
        "DATA=" + repr(server["data_root"]),
        "FILES=" + repr(files), "TASKS=" + repr(tasks), program,
    ])
    output = monitor.ssh(server["host"], server["user"],
                         shlex.quote(server["python"]) + " -B -c " + shlex.quote(program), timeout=30)
    return json.loads(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("operation", choices=("dispatch", "dispatch-waiting", "reconcile"))
    args = parser.parse_args()
    scoped.lease(args.token)
    lock = monitor.acquire_lock(str(ROOT))
    if lock is None:
        raise ValueError("Project writer is busy")
    try:
        scoped.lease(args.token)
        queue = monitor.load_queue(str(ROOT))
        work = tick.load(ROOT / "pipeline/research_work.json")
        tasks = [t for t in queue["gpu_queues"]["server-2"] if t["id"] in IDS]
        candidates = [c for c in work["candidates"] if c["id"] in IDS]
        if (len(tasks) != 2 or {t["id"] for t in tasks} != IDS
                or len(candidates) != 2 or {c["id"] for c in candidates} != IDS):
            raise ValueError("Expected exactly the two admitted IDs")
        server = monitor.load_servers(str(ROOT))["server-2"]
        if (server["allowed_gpus"] != [0, 1]
                or server["max_tasks_per_gpu"] != 1
                or server["host"] != "192.168.1.163"):
            raise ValueError("Server authorization changed")
        head = closure.git(ROOT, "rev-parse", "HEAD")
        files = {}
        for candidate in candidates:
            task = next(t for t in tasks if t["id"] == candidate["id"])
            if task["code_commit"] != head:
                raise ValueError("Admitted HEAD changed")
            closure.admission(ROOT, task)
            _, rows = scoped.checked_bundle(candidate["id"], candidate)
            for row in rows:
                data = scoped.git("show", head + ":" + row["path"])
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError("Committed bytes differ from tested bytes")
                files[row["path"].removeprefix("AdaRES/")] = row["sha256"]
        started = datetime.now(CHINA).isoformat()
        before = snapshot(server, tasks, list(files))
        events = []
        affinity = []
        if args.operation in ("dispatch", "dispatch-waiting"):
            allowed_states = {"waiting"} if args.operation == "dispatch" else {"waiting", "running"}
            if (not any(t["status"] == "waiting" for t in tasks)
                    or any(t["status"] not in allowed_states or t.get("dispatch_hold") for t in tasks)):
                raise ValueError("Existing process/hold must be reconciled, not relaunched")
            procs = monitor.find_project_processes("AdaRES_project", "server-2", server, queue)
            expected_live = {t["config"] for t in tasks if t["status"] == "running"}
            if set(procs) != expected_live:
                raise ValueError("Existing project processes must be reconciled first")
            for task in tasks:
                if task["status"] == "running":
                    process = before["tasks"][task["id"]]
                    if (not process["process_live"] or process["cwd"] != server["remote_code_root"]
                            or task["config"] not in process["cmdline"]):
                        raise ValueError("Existing running identity changed; do not dispatch")
            visible = {line.split(",", 1)[0].strip()
                       for line in before["authorized_device_inventory"].splitlines()}
            if before["gpu_query_rc"] != 0 or not visible.intersection({"0", "1"}):
                raise ValueError("No verified authorized GPU is visible")
            for task in tasks:
                if task.get("gpu") is not None and str(task["gpu"]) not in visible:
                    affinity.append(dict(id=task["id"],previous=task["gpu"],current=None,
                        reason="Soft preference absent from live device inventory; retain authorized pool only"))
                    task["gpu"] = None
            if affinity:
                monitor.save_queue(str(ROOT), queue)
                monitor.log_decision(str(ROOT), "manual",
                    "Only a subset of the authorized GPU indices is currently visible.",
                    "Remove unavailable soft affinity only; preserve one task per authorized GPU.",
                    "User requested continuation; no fixed protocol or admission changed.", json.dumps(affinity))
            monitor._process_queue(
                queue, "server-2", server["host"], server["user"], str(ROOT),
                {str(proc[2]) for proc in procs.values()}, {"0", "1"}, events, max_tasks_per_gpu=1,
                allow_external_overlay=False, remote_code_root=server["remote_code_root"],
                current_tasks=len(procs), max_tasks_total=2,
            )
            queue = monitor.load_queue(str(ROOT))
            tasks = [t for t in queue["gpu_queues"]["server-2"] if t["id"] in IDS]
        else:
            for task in tasks:
                if task.get("pid") and task.get("log") and not task.get("launch_verified"):
                    if closure.verify_progress(server, task, monitor.ssh, wait_s=90):
                        task.update(launch_verified=True, launch_verification="training_progress")
                        task.pop("dispatch_hold", None)
                        exp = monitor.load_experiment(str(ROOT), task["id"])
                        if exp:
                            exp.update(launch_verified=True, launch_verification="training_progress")
                            monitor.save_experiment(str(ROOT), exp)
            monitor.save_queue(str(ROOT), queue)
        after = snapshot(server, tasks, list(files))
        if after["source_marker"] == head and after["source_hashes"] != files:
            raise ValueError("Deployed candidate bytes do not match admission")
        for task in tasks:
            if task["status"] != "running":
                continue
            process = after["tasks"][task["id"]]
            if (not process["process_live"]
                    or process["cwd"] != server["remote_code_root"]
                    or "tools/train.py" not in process["cmdline"]
                    or task["config"] not in process["cmdline"]):
                raise ValueError("Live process identity changed; reconcile, do not relaunch")
            if any(marker in process["tail"] for marker in (
                    "Traceback", "CUDA out of memory", "RuntimeError")):
                raise ValueError("Crash evidence in current log; preserve and reconcile")
            if task.get("launch_verified") and len(set(closure.steps(process["tail"]))) < 2:
                raise ValueError("Verified launch lacks current native-step evidence")
        now = datetime.now(CHINA)
        path = f"pipeline/analyses/prepared_{args.operation}_{args.token}_{now.strftime('%H%M%S')}.json"
        tick.save(ROOT / path, dict(complete=True, operation=args.operation, token=args.token,
            observed_from=started, observed_until=now.isoformat(), code_commit=head,
            before=before, after=after, tasks=tasks, events=events, affinity_changes=affinity,
            exact_deployed_candidate_bytes=(after["source_hashes"] == files),
            limitations=["Native steps verify execution, not trained benefit.",
                         "Device indices/occupation are current observations, not permanent reservations."]))
        evidence = scoped.ref(path)
        for candidate in candidates:
            task = next(t for t in tasks if t["id"] == candidate["id"])
            candidate["evidence"].append(evidence)
            if task["status"] == "running" and task.get("launch_verified") is True:
                if candidate.get("blocker"):
                    previous = {k: candidate[k] for k in (
                        "blocker_kind", "blocker", "retry_after", "unblock_condition")
                        if k in candidate}
                    previous.update(resolved_at=now.isoformat(), resolved_by=evidence)
                    candidate.setdefault("blocker_history", []).append(previous)
                for key in ("blocker_kind", "blocker", "retry_after", "unblock_condition"):
                    candidate.pop(key, None)
                candidate.update(stage="training", experiment_id=task["id"],
                    next_action="Continue this SAME fixed30 single-GPU run; monitor native progress and apply frozen post-training evaluation. No tuning or blind restart.")
            elif task.get("pid") or task.get("dispatch_hold"):
                candidate.update(stage="deployment",
                    next_action="Reconcile the exact existing config/PID/log and increasing native steps; do not relaunch.")
            else:
                candidate.update(stage="blocked", blocker_kind="external_resource",
                    required_server="server-2", retry_after=(now+timedelta(hours=1)).isoformat(),
                    blocker="Mount/source restored; waiting for a genuinely idle visible authorized GPU after canonical dispatch. Use current inventory, preserve existing jobs, and never oversubscribe.",
                    unblock_condition="An authorized visible GPU is genuinely idle and canonical source/lease/progress guards pass.",
                    next_action="Keep the SAME admitted waiting task; canonical dispatch may use any available authorized GPU0/1 without oversubscription or repeated commit consent.")
        closure.validate_work(ROOT, tick.load(ROOT/"pipeline/headless/.tick_context.json"), work["candidates"], queue)
        scoped.lease(args.token)
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(evidence=evidence, tasks=[{k:t.get(k) for k in (
            "id","status","gpu","pid","log","launch_verified","launch_verification","dispatch_hold")}
            for t in tasks], events=events, affinity_changes=affinity), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
