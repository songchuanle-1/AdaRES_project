"""Guarded continuation of the same Y43/Y44 after the authorized shutdown."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
sys.path.insert(0, str(SKILL))
import closure
import pipeline_monitor as monitor

BUNDLES = {
    "AdaRES-Y43": "pipeline/analyses/y43_source_bundle_0422d0d8ad4a4a0288d9e5d7e4e53f73.json",
    "AdaRES-Y44": "pipeline/analyses/y44_source_bundle_ff3476973900467da752992f8195291b.json",
}
GPUS = {"AdaRES-Y43": "0", "AdaRES-Y44": "1"}
DISPATCHES = {
    "AdaRES-Y43": "pipeline/analyses/y43_dispatch_0422_0422d0d8ad4a4a0288d9e5d7e4e53f73.json",
    "AdaRES-Y44": "pipeline/analyses/y44_dispatch_ff34_ff3476973900467da752992f8195291b.json",
}

REMOTE = r'''
import fcntl,gc,hashlib,json,os,pathlib,re,shutil,subprocess,time
P=__PAYLOAD__
root=pathlib.Path(P["server"]["remote_code_root"]).resolve()
work=pathlib.Path(P["server"]["work_dir"]).resolve()
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
assert subprocess.check_output(["findmnt","-n","-o","SOURCE","--target","/bak_home"]).decode().strip()=="/dev/sda1"
assert root.is_dir() and work.is_dir() and pathlib.Path(P["server"]["data_root"]).is_dir()
assert (root/".codex_source_commit").read_text().strip()==P["source_marker"]
def digest(p):
    h=hashlib.sha256()
    with open(str(p),"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()
def tail(p,n):
    with open(str(p),"rb") as f:
        f.seek(max(0,os.fstat(f.fileno()).st_size-n))
        return f.read(n).decode("utf-8",errors="replace")
for e in P["files"]:
    p=root.parent/e["path"]
    p.resolve().relative_to(root)
    assert digest(p)==e["sha256"],"Frozen source changed: "+e["path"]
cfg_path=root/P["config"]
import yaml
cfg=yaml.safe_load(cfg_path.read_text())
assert cfg["train"]["epochs"]==30 and cfg["train"]["seed"]==1999
assert pathlib.Path(cfg["work_dir"]).resolve()==work
assert pathlib.Path(cfg["data"]["train"]["data_root"]).is_dir()
def no_duplicate():
    for p in pathlib.Path("/proc").glob("[0-9]*/cmdline"):
        try:cmd=p.read_bytes().split(b"\0")
        except OSError:continue
        assert not (b"tools/train.py" in cmd and P["config"].encode() in cmd),"Already running; reconcile, never relaunch"
def gpu_idle():
    raw=subprocess.check_output(["nvidia-smi","-i",P["gpu"],
        "--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"]).decode().strip()
    memory,util=map(int,raw.split(","))
    apps=subprocess.check_output(["nvidia-smi","-i",P["gpu"],
        "--query-compute-apps=pid","--format=csv,noheader,nounits"]).decode().strip()
    assert memory<1024 and util<10 and not apps,"Authorized GPU occupied"
    return {"memory_mib":memory,"util_percent":util,"compute_processes":[]}
if P["action"]=="verify":
    pid=int(P["pid"])
    cmd=pathlib.Path("/proc")/str(pid)/"cmdline"
    assert cmd.is_file(),"Resume process is absent"
    argv=cmd.read_bytes().split(b"\0")
    assert P["config"].encode() in argv and b"--resume" in argv and P["checkpoint"].encode() in argv
    assert argv[argv.index(b"--gpu_ids")+1]==P["gpu"].encode()
    assert argv[argv.index(b"--num_gpus")+1]==b"1"
    assert pathlib.Path("/proc",str(pid),"cwd").resolve()==root
    text=tail(P["log"],24000)
    header=pathlib.Path(P["log"]).open("rb").read(64000).decode("utf-8",errors="replace")
    errors=[v for v in ("Traceback","CUDA out of memory","RuntimeError") if v in text]
    matches=re.findall(r"Train\[(\d+)/30\]:[^\r\n]*?\|\s*(\d+)/",text)
    samples=[(int(e),int(step)) for e,step in matches]
    assert not errors,"Resume log contains "+",".join(errors)
    native_logs=[]
    for fd in pathlib.Path("/proc",str(pid),"fd").iterdir():
        try:target=fd.resolve()
        except OSError:continue
        if target.suffix==".log":
            try:target.relative_to(work/pathlib.Path(P["config"]).stem)
            except ValueError:continue
            native_logs.append(str(target))
    for native in sorted(set(native_logs)):
        with open(native,"rb") as stream:
            header+="\n"+stream.read(64000).decode("utf-8",errors="replace")
    restored=("Resume from "+P["checkpoint"] in header and
              "resume start_epoch:"+str(P["resume_epoch"]) in header)
    print(json.dumps({"pid":pid,"restored":restored,"progress":max(samples) if samples else None,
                     "native_logs":sorted(set(native_logs)),"errors":errors}))
else:
    if P["action"] in ("inspect","launch"):
        no_duplicate()
        gpu_before=gpu_idle()
    if P["action"]=="inspect":
        text=tail(P["original_log"],4*1024*1024)
        paths=re.findall(r"save last pth to ([^\s]+/last\.pth)",text)
        assert paths,"Original bounded log has no last checkpoint"
        checkpoint=pathlib.Path(paths[-1]).resolve()
        checkpoint.relative_to(work/pathlib.Path(P["config"]).stem)
        assert checkpoint.name=="last.pth" and checkpoint.is_file()
        best=checkpoint.with_name("best.pth")
        assert best.is_file(),"Original best checkpoint missing"
        import torch
        c=torch.load(str(checkpoint),map_location="cpu")
        assert all(k in c for k in ("model","optimizer","scheduler","meta"))
        epoch=int(c["meta"]["epoch"])
        assert 0<epoch<30,"Already terminal or invalid checkpoint; do not restart"
        assert c["optimizer"]["state"] and c["optimizer"]["param_groups"]
        scheduler=c["scheduler"]
        assert isinstance(scheduler,dict) and scheduler
        assert int(scheduler["last_epoch"])==epoch,"Epoch/scheduler mismatch"
        meta={"epoch":epoch,"best_miou":float(c["meta"]["best_miou"])}
        optimizer_steps=[float(s["step"]) for s in c["optimizer"]["state"].values() if "step" in s]
        result={"checkpoint":str(checkpoint),"saved_epoch":epoch,"resume_epoch":epoch+1,
                "best_checkpoint":str(best),"meta":meta,"scheduler":scheduler,
                "optimizer_state_count":len(c["optimizer"]["state"]),
                "optimizer_step_range":[min(optimizer_steps),max(optimizer_steps)],
                "learning_rates":[g["lr"] for g in c["optimizer"]["param_groups"]],
                "original_last_progress":re.findall(r"Train\[(\d+)/30\]:[^\r\n]*?\|\s*(\d+)/",text)[-1],
                "gpu_before":gpu_before}
        del c
        gc.collect()
        result.update(checkpoint_sha256=digest(checkpoint),best_sha256=digest(best))
        print(json.dumps(result))
    elif P["action"]=="launch":
        checkpoint=pathlib.Path(P["checkpoint"]).resolve()
        checkpoint.relative_to(work/pathlib.Path(P["config"]).stem)
        assert digest(checkpoint)==P["checkpoint_sha256"],"Original last checkpoint changed"
        assert digest(P["best_checkpoint"])==P["best_sha256"],"Original best checkpoint changed"
        lock_name=".y15_gpu1_diagnostic.lock" if P["gpu"]=="1" else ".foreground_gpu0_diagnostic.lock"
        gpu_lease=open(str(root.parent/lock_name),"a")
        fcntl.flock(gpu_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        no_duplicate()
        gpu_before=gpu_idle()
        env=os.environ.copy()
        env["PYTHONPATH"]=":".join([str(root),str(root/"ipdn/lib"),
            P["server"]["site_packages"],P["server"]["pointnet_egg"]])
        env["PYTHONDONTWRITEBYTECODE"]="1"
        env.pop("CUDA_VISIBLE_DEVICES",None)
        with open(P["log"],"xb",buffering=0) as log:
            proc=subprocess.Popen([P["server"]["python"],"-S","tools/train.py",P["config"],
                "--num_gpus","1","--gpu_ids",P["gpu"],"--resume",str(checkpoint)],
                cwd=str(root),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(gpu_lease.fileno(),))
        print(json.dumps({"pid":proc.pid,"log":P["log"],"gpu_before":gpu_before,
                          "result":"launched_pending_progress"}),flush=True)
    elif P["action"]=="carry_best":
        native=pathlib.Path(P["native_log"]).resolve()
        native.relative_to(work/pathlib.Path(P["config"]).stem)
        assert native.is_file() and native.parent!=pathlib.Path(P["checkpoint"]).parent
        original=pathlib.Path(P["best_checkpoint"])
        target=native.with_name("best.pth")
        assert digest(original)==P["best_sha256"]
        if not target.exists():
            # Preserve the pre-shutdown best if later resumed epochs do not beat it.
            with original.open("rb") as src,target.open("xb") as dst:
                shutil.copyfileobj(src,dst,8*1024*1024)
        assert digest(target)==P["best_sha256"],"Refuse to overwrite a new trained best"
        assert digest(P["checkpoint"])==P["checkpoint_sha256"]
        print(json.dumps({"best_carried_to":str(target),"best_sha256":P["best_sha256"],
                          "original_last_unchanged":True,"original_best_unchanged":True}))
'''


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lease(token, remaining=0):
    owner = load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
    assert owner["token"] == token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (datetime.fromisoformat(owner["deadline_at"]) -
            datetime.now(timezone.utc)).total_seconds() > remaining


def remote(payload, timeout=180):
    server = payload["server"]
    program = REMOTE.replace("__PAYLOAD__", repr(payload))
    env = (f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={shlex.quote(server['site_packages'])} "
           f"{shlex.quote(server['python'])} -S -")
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
         "-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=1",
         f"{server['user']}@{server['host']}", env],
        input=program.encode("utf-8"), capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-4000:])
    return json.loads(result.stdout)


def persist(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"evidence": str(path),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "result": record.get("result", record.get("observation"))}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--id", choices=sorted(GPUS), required=True)
    parser.add_argument("--action", choices=("inspect", "launch", "verify"), required=True)
    args = parser.parse_args()
    lease(args.token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None, "Pipeline writer busy"
    try:
        server = monitor.load_servers(str(ROOT))["server-2"]
        assert server["host"] == "192.168.1.163" and server["user"] == "songcl"
        assert server["allowed_gpus"] == [0, 1] and server["max_tasks_per_gpu"] == 1
        queue = monitor.load_queue(str(ROOT))
        task = next((t for t in queue["gpu_queues"]["server-2"] if t["id"] == args.id), None)
        restored_task = task is None
        if restored_task:
            experiment = monitor.load_experiment(str(ROOT), args.id)
            assert experiment["status"] == "stopped_unknown", "Only confirmed shutdown interruptions may resume"
            work = load(ROOT / "pipeline/research_work.json")
            candidate = next(c for c in work["candidates"] if c["id"] == args.id)
            original_dispatch = next(e for e in candidate["evidence"] if e["path"] == DISPATCHES[args.id])
            closure.artifact(ROOT, original_dispatch)
            task = load(ROOT / DISPATCHES[args.id])["task"]
            for key in ("id", "config", "server", "gpu", "pid", "log"):
                assert task[key] == experiment[key], "Interrupted task identity changed: " + key
            task["status"] = "stopped_unknown"
        assert task["server"] == "server-2" and str(task["gpu"]) == GPUS[args.id]
        assert task["total_epochs"] == 30 and task["status"] in ("running", "claiming", "stopped_unknown")
        closure.admission(str(ROOT), task)
        base = ROOT / "pipeline/reports" / ("shutdown_resume_" + args.id + "_" + args.token)
        inspect_path = base.with_name(base.name + "_inspect.json")
        launch_path = base.with_name(base.name + "_launch.json")
        verify_path = base.with_name(base.name + "_verify.json")
        if args.action == "inspect":
            assert not inspect_path.exists()
            files = load(ROOT / BUNDLES[args.id])["files"]
            for item in files:
                assert closure.digest(ROOT / item["path"]) == item["sha256"]
            pinned = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", task["code_commit"] + ":AdaRES/tools/train.py"])
            train_sha = hashlib.sha256(pinned).hexdigest()
            assert closure.digest(ROOT / "AdaRES/tools/train.py") == train_sha
            files.append({"path": "AdaRES/tools/train.py", "sha256": train_sha})
            payload = dict(action="inspect", server=server, config=task["config"], id=args.id,
                           gpu=GPUS[args.id], files=files, original_log=task["log"],
                           original_pid=task["pid"], source_marker=closure.git(ROOT, "rev-parse", "HEAD"))
            observation = remote(payload)
            persist(inspect_path, dict(payload=payload, observation=observation,
                                      observed_local=datetime.now().astimezone().isoformat()))
            return
        record = load(inspect_path)
        payload = {**record["payload"], **record["observation"], "action": args.action}
        if args.action == "launch":
            assert not launch_path.exists() and not task.get("resume_attempt_token")
            payload["log"] = (server["remote_code_root"] + "/logs/resume_" +
                              Path(task["config"]).stem + "_" + args.token + ".log")
            previous = {key: task.get(key) for key in ("pid", "log", "status", "launched")}
            task.update(status="claiming", resume_attempt_token=args.token,
                        launch_verified=False, launch_verification="pending_progress",
                        resume=payload["checkpoint"], resume_checkpoint_sha256=payload["checkpoint_sha256"])
            if restored_task:
                queue["gpu_queues"]["server-2"].append(task)
            monitor.save_queue(str(ROOT), queue)
            result = remote(payload)
            task.update(pid=result["pid"], log=result["log"])
            monitor.save_queue(str(ROOT), queue)
            persist(launch_path, dict(payload=payload, result=result, previous=previous,
                                     observed_local=datetime.now().astimezone().isoformat()))
            return
        launch = load(launch_path)
        assert task["resume_attempt_token"] == args.token and task["pid"] == launch["result"]["pid"]
        if verify_path.exists():
            previous_verification = load(verify_path)
            assert previous_verification["result"] == "pending_progress"
            assert all(s["pid"] == task["pid"] for s in previous_verification["samples"])
            archived = verify_path.with_name(verify_path.stem + "_pending.json")
            assert not archived.exists()
            verify_path.rename(archived)
        payload.update(action="verify", pid=task["pid"], log=task["log"])
        deadline = time.monotonic() + 180
        samples = []
        while time.monotonic() < deadline:
            lease(args.token, 120)
            observation = remote(payload, timeout=30)
            samples.append(dict(observed_local=datetime.now().astimezone().isoformat(),
                                **observation))
            progressing = [s for s in samples if s["restored"] and s["progress"] is not None]
            if (len(progressing) >= 2 and tuple(progressing[-1]["progress"]) >
                    tuple(progressing[0]["progress"])):
                break
            time.sleep(6)
        else:
            persist(verify_path, dict(result="pending_progress", samples=samples))
            raise RuntimeError("Original resume PID launched but increasing steps not verified; reconcile, never relaunch")
        native_logs = samples[-1]["native_logs"]
        assert len(native_logs) == 1, "Cannot uniquely identify resumed native checkpoint directory"
        carried = remote({**payload, "action": "carry_best", "native_log": native_logs[0]})
        task.update(status="running", launch_verified=True, launch_verification="training_progress",
                    resumed_at=datetime.now().astimezone().isoformat(),
                    resume_epoch=payload["resume_epoch"], resumed_work_dir=str(PurePosixPath(native_logs[0]).parent),
                    carried_best=carried["best_carried_to"])
        monitor.save_queue(str(ROOT), queue)
        running = monitor.load_running(str(ROOT))
        running.setdefault("server-2", {})[GPUS[args.id]] = args.id
        monitor.save_running(str(ROOT), running)
        experiment = monitor.load_experiment(str(ROOT), args.id)
        experiment.update({key: task[key] for key in (
            "status", "pid", "log", "gpu", "server", "resume", "resume_checkpoint_sha256",
            "resume_epoch", "resume_attempt_token", "resumed_at", "resumed_work_dir",
            "carried_best", "launch_verified", "launch_verification")})
        experiment.setdefault("history", []).append(dict(
            ts=task["resumed_at"], action="resumed_after_user_shutdown",
            original=launch["previous"], checkpoint=task["resume"], pid=task["pid"],
            gpu=task["gpu"], evidence=verify_path.relative_to(ROOT).as_posix()))
        interruption = {key: experiment.pop(key) for key in
                        ("finished_at", "final_best", "epochs_final", "log_local") if key in experiment}
        experiment["history"][-1]["interruption_record"] = interruption
        monitor.save_experiment(str(ROOT), experiment)
        deathwatch = monitor.load_deathwatch(str(ROOT))
        deathwatch.setdefault("server-2", {}).pop(Path(task["config"]).stem, None)
        monitor.save_deathwatch(str(ROOT), deathwatch)
        persist(verify_path, dict(result="resumed_training_progress_verified",
                                 samples=samples, checkpoint_preservation=carried,
                                 resume_epoch=payload["resume_epoch"], pid=task["pid"], gpu=task["gpu"]))
        work_path = ROOT / "pipeline/research_work.json"
        work = load(work_path)
        candidate = next(c for c in work["candidates"] if c["id"] == args.id)
        assert candidate["stage"] == "training"
        for path in (inspect_path, launch_path, verify_path):
            candidate["evidence"].append(dict(
                path=path.relative_to(ROOT).as_posix(), sha256=closure.digest(path)))
        candidate.update(scientific_status="resumed_training_not_efficacy",
                         next_action="Continue this same fixed30 run from the preserved checkpoint; "
                         "retain the frozen terminal evaluation and harm rules. No retuning or fresh restart.")
        work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
