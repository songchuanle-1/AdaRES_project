"""Restart only the interrupted, checkpoint-free Y52 under explicit user consent."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import authorize_prepared_candidates as scoped
import closure
import dispatch_prepared_candidates as prepared
import foreground_tick as tick
import pipeline_monitor as monitor

CHINA = timezone(timedelta(hours=8))
REQUEST = "\u6302\u8f7d\u4e86 \u63d0\u4ea4\u8bad\u7ec3\u5427"
ID = "AdaRES-Y52"


def inspect_storage(server, config, original_log):
    program = "\n".join([
        "ROOT=" + repr(server["remote_code_root"]),
        "WORK=" + repr(server["work_dir"]),
        "DATA=" + repr(server["data_root"]),
        "CONFIG=" + repr(config), "LOG=" + repr(original_log),
        r'''
import hashlib,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path(ROOT)
mount=json.loads(subprocess.check_output(['findmnt','-J','-o','SOURCE,TARGET,FSTYPE,UUID','--target',ROOT]))
assert len(mount['filesystems'])==1
fs=mount['filesystems'][0]
assert fs['target']=='/bak_home' and fs['source']=='/dev/sda1'
assert fs['uuid']=='459a8db5-5b91-446b-a501-eed399c6a45d'
assert root.is_dir() and pathlib.Path(DATA).is_dir() and pathlib.Path(WORK).is_dir()
parent=pathlib.Path(WORK)/pathlib.Path(CONFIG).stem
assert parent.is_dir()
runs=sorted(p for p in parent.iterdir() if p.is_dir())
assert len(runs)<=8, 'Unexpected additional attempts; inspect explicitly'
checkpoints=[str(p) for run in runs for p in run.glob('*.pth')]
logs=[]
for p in [pathlib.Path(LOG)]+[p for run in runs for p in run.glob('*.log')]:
    assert p.is_file() and p.stat().st_size<4*1024*1024
    raw=p.read_bytes()
    logs.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(dict(mount=mount,checkpoints=checkpoints,original_logs=logs,
    source_marker=(root/'.codex_source_commit').read_text().strip(),
    original_pid_exists=pathlib.Path('/proc/15934/cmdline').is_file())))
''',
    ])
    command = shlex.quote(server["python"]) + " -B -S -c " + shlex.quote(program)
    return json.loads(monitor.ssh(server["host"], server["user"], command, timeout=30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    if args.request != REQUEST:
        raise ValueError("The direct owning user's current training request is required")
    scoped.lease(args.token)
    writer = monitor.acquire_lock(str(ROOT))
    if writer is None:
        raise ValueError("Project writer is busy")
    try:
        scoped.lease(args.token)
        queue = monitor.load_queue(str(ROOT))
        tasks = queue["gpu_queues"]["server-2"]
        task = next(t for t in tasks if t["id"] == ID)
        other = next(t for t in tasks if t["id"] == "AdaRES-Y54")
        if (len(tasks) != 2 or task.get("pid") != 15934
                or task.get("restart_attempt_token") or task["status"] != "running"
                or other["status"] != "running" or str(other["gpu"]) != "0"):
            raise ValueError("Existing task identities changed; reconcile rather than replay")
        server = monitor.load_servers(str(ROOT))["server-2"]
        if (server["host"] != "192.168.1.163" or server["allowed_gpus"] != [0, 1]
                or server["max_tasks_per_gpu"] != 1):
            raise ValueError("Resource authorization changed")
        work = tick.load(ROOT / "pipeline/research_work.json")
        head = closure.git(ROOT, "rev-parse", "HEAD")
        files = {}
        for current in tasks:
            closure.admission(ROOT, current)
            if current["code_commit"] != head:
                raise ValueError("Admitted commit changed")
            candidate = next(c for c in work["candidates"] if c["id"] == current["id"])
            _, rows = scoped.checked_bundle(current["id"], candidate)
            files.update({r["path"].removeprefix("AdaRES/"): r["sha256"] for r in rows})
        before = prepared.snapshot(server, tasks, list(files))
        storage = inspect_storage(server, task["config"], task["log"])
        if (storage["original_pid_exists"] or storage["checkpoints"]
                or before["source_marker"] != head or before["source_hashes"] != files):
            raise ValueError("Original checkpoint/process exists or source changed; do not fresh-start")
        old_process = before["tasks"][ID]
        live_other = before["tasks"][other["id"]]
        if old_process["process_live"] or not live_other["process_live"]:
            raise ValueError("Process identity requires reconciliation")
        steps = re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)", old_process["tail"])
        if not steps or tuple(map(int, steps[-1][:2])) != (1, 30):
            raise ValueError("Not the interrupted first epoch; preserve and inspect")
        inventory = [row.split(",") for row in before["authorized_device_inventory"].splitlines()]
        gpu = next([value.strip() for value in row] for row in inventory if row[0].strip() == "1")
        if (before["gpu_query_rc"] or before["compute_query_rc"]
                or int(gpu[3]) >= 10 or int(gpu[4]) >= 1024
                or gpu[1] in before["compute"]):
            raise ValueError("GPU1 is occupied; do not interrupt its owner")
        procs = monitor.find_project_processes("AdaRES_project", "server-2", server, queue)
        if task["config"] in procs or set(procs) != {other["config"]}:
            raise ValueError("Unexpected live project process; reconcile rather than duplicate")
        now = datetime.now(CHINA).isoformat()
        original = deepcopy(task)
        initial = f"pipeline/analyses/y52_mount_restart_{args.token}_before.json"
        if (ROOT / initial).exists():
            raise ValueError("Restart evidence exists; do not replay")
        tick.save(ROOT / initial, dict(complete=True,observed_local=now,token=args.token,
            human_request=args.request,authorization="Direct current owning-user training request",
            original_task=original,original_experiment=monitor.load_experiment(str(ROOT), ID),
            original_deathwatch=monitor.load_deathwatch(str(ROOT)),storage=storage,before=before,
            old_last_progress=list(map(int, steps[-1])),
            disposition="Interrupted before first checkpoint; fresh fixed30 attempt, NOT a stateful resume",
            remote_clock="Use local observation time; server log names/clock are not Shanghai timestamps"))
        initial_ref = scoped.ref(initial)
        exp = monitor.load_experiment(str(ROOT), ID)
        exp.setdefault("history", []).append(dict(ts=now,
            action="interrupted_before_first_checkpoint_user_requested_restart",
            original=original,evidence=initial_ref,human_request=args.request))
        exp["status"] = "waiting"
        for key in ("pid", "log", "launched", "launch_verified", "launch_verification"):
            exp.pop(key, None)
        monitor.save_experiment(str(ROOT), exp)
        task.setdefault("attempt_history", []).append(dict(recorded_at=now,
            reason="Original mount/storage interruption; no checkpoint exists",original=original,
            evidence=initial_ref))
        for key in ("pid", "log", "launched", "launch_verified", "launch_verification",
                    "dispatch_hold", "dispatch_phase", "claimed_gpu"):
            task.pop(key, None)
        task.update(status="waiting", gpu="1", restart_attempt_token=args.token)
        scoped.lease(args.token)
        monitor.save_queue(str(ROOT), queue)
        events = []
        monitor._process_queue(queue,"server-2",server["host"],server["user"],str(ROOT),
            {"0"},{"0","1"},events,max_tasks_per_gpu=1,allow_external_overlay=False,
            remote_code_root=server["remote_code_root"],current_tasks=1,max_tasks_total=2)
        queue = monitor.load_queue(str(ROOT))
        tasks = queue["gpu_queues"]["server-2"]
        current = next(t for t in tasks if t["id"] == ID)
        after_path = f"pipeline/analyses/y52_mount_restart_{args.token}_after.json"
        tick.save(ROOT / after_path, dict(complete=True,observed_local=datetime.now(CHINA).isoformat(),
            token=args.token,original=initial_ref,tasks=tasks,events=events,
            attempt="Fresh fixed30 from admitted released initialization; prior partial attempt preserved"))
        candidate = next(c for c in work["candidates"] if c["id"] == ID)
        candidate["evidence"].extend([initial_ref,scoped.ref(after_path)])
        tick.save(ROOT / "pipeline/research_work.json",work)
        monitor.log_decision(str(ROOT),"manual","User confirms mount restored and requests training.",
            "Preserve interrupted checkpoint-free Y52 attempt and use canonical dispatcher on idle GPU1; preserve Y54 GPU0.",
            "Original PID absent, first epoch535/4686, zero checkpoint files, exact admitted source and idle card rechecked.",initial)
        print(json.dumps(dict(id=ID,status=current["status"],pid=current.get("pid"),
            gpu=current.get("gpu"),launch_verified=current.get("launch_verified"),
            before=initial_ref,after=scoped.ref(after_path)),indent=2))
    finally:
        writer.close()


if __name__ == "__main__":
    main()
