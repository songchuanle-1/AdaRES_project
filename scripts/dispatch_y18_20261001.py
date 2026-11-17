"""Invoke the canonical gated queue dispatcher without repeating preflight."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--output", default="pipeline/analyses/y18_dispatch_3efff5f3.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    owner = json.loads((root / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
    if owner["token"] != args.token or owner["thread_id"] != os.environ["CODEX_THREAD_ID"]:
        raise RuntimeError("Foreground ownership mismatch")
    sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
    import pipeline_monitor as monitor
    import closure
    lock = monitor.acquire_lock(str(root))
    if lock is None:
        raise RuntimeError("Canonical monitor lock unavailable")
    try:
        queue = monitor.load_queue(str(root))
        task = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y18")
        closure.admission(root, task)
        if closure.git(root, "rev-parse", "HEAD") != task["code_commit"]:
            raise RuntimeError("Admitted HEAD mismatch")
        server = monitor.load_servers(str(root))["server-2"]
        if server["host"] != "192.168.1.163" or server["allowed_gpus"] != [0, 1]:
            raise RuntimeError("Resource authorization changed")
        processes = monitor.find_running_processes(
            server["host"], server["user"], server["remote_code_root"])
        used = {str(p[2]) for p in processes.values()}
        events = []
        monitor._process_queue(
            queue, "server-2", server["host"], server["user"], str(root),
            used, {"0", "1"}, events, max_tasks_per_gpu=1,
            remote_code_root=server["remote_code_root"],
            current_tasks=len(processes), max_tasks_total=2)
        current = monitor.load_queue(str(root))
        result = {"at_local": datetime.now().astimezone().isoformat(),
                  "candidate": "AdaRES-Y18", "events": events,
                  "tasks": current["gpu_queues"]["server-2"],
                  "scope": "Canonical dispatch only; no repeated monitor/goal preflight"}
        path = root / args.output
        if path.exists():
            raise RuntimeError("Refusing overwrite dispatch record")
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
