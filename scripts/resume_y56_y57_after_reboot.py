"""Resume the exact admitted Y56/Y57 attempts after the authorized GPU reboot."""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time

import authorize_prepared_candidates as scoped
import resume_shutdown_training as common

ROOT = common.ROOT
monitor = common.monitor
closure = common.closure
REQUEST = "\u7ee7\u7eed\u8bad\u7ec3"
BOOT = "81549f32-5f49-4ef5-b91e-2b1144a6bfb3"
MARKER = "05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea"
OLD_TOKEN = "dd8e4cf0cbdc46169e8b3eadac13593d"
SPECS = {
    "AdaRES-Y56": dict(
        package="adares_y56_unique_competitors", gpu="0", pid=83638,
        commit="200486ee12bcf3d7e82393d009d06214dfe83e38", epoch=15,
        run="20261020_111322", log="train_adares_y56_unique_competitors_res_20261020_231317.log",
        review="y56_gate_review_" + OLD_TOKEN + ".json",
        dispatch="y56_dispatch_" + OLD_TOKEN + ".json"),
    "AdaRES-Y57": dict(
        package="adares_y57_peer_second_moment", gpu="1", pid=95947,
        commit=MARKER, epoch=17, run="20261020_112850",
        log="train_adares_y57_peer_second_moment_res_20261020_232844.log",
        review="y57_anchor_gate_review_" + OLD_TOKEN + ".json",
        dispatch="y57_dispatch_" + OLD_TOKEN + ".json"),
}


def stamp():
    return datetime.now().astimezone().isoformat()


def validate_task(candidate_id, task, experiment, receipt, proof):
    spec = SPECS[candidate_id]
    config = "configs_song/model_paper5/" + spec["package"] + "_res.yaml"
    assert task["id"] == experiment["id"] == receipt["id"] == candidate_id
    assert task["status"] in ("running", "stopped_unknown", "crashed")
    assert experiment["status"] in ("running", "stopped_unknown", "crashed")
    assert task["server"] == experiment["server"] == "server-2"
    assert str(task["gpu"]) == str(experiment["gpu"]) == spec["gpu"]
    assert task["pid"] == experiment["pid"] == proof["pid"] == spec["pid"]
    assert task["config"] == experiment["config"] == receipt["config"] == config
    assert task["log"] == experiment["log"] == proof["log"]
    assert Path(task["log"]).name == spec["log"]
    assert task["code_commit"] == receipt["code_commit"] == spec["commit"]
    assert all(task[k] == 30 for k in ("fixed_epochs", "min_epochs", "patience"))
    assert task.get("total_epochs", 30) == 30
    argv = proof["cmdline"]
    assert proof["process_live"] and "tools/train.py" in argv and config in argv
    assert argv[argv.index("--gpu_ids") + 1] == spec["gpu"]
    assert argv[argv.index("--num_gpus") + 1] == "1"
    assert not task.get("resume_attempt_token"), "Already attempted; reconcile instead"


def checkpoint_matches(candidate_id, observation, server):
    spec = SPECS[candidate_id]
    expected = str(PurePosixPath(server["work_dir"]) /
                   (spec["package"] + "_res") / spec["run"] / "last.pth")
    assert observation["checkpoint"] == expected, "Unexpected checkpoint attempt"
    assert observation["saved_epoch"] == spec["epoch"]
    assert observation["resume_epoch"] == spec["epoch"] + 1
    assert observation["optimizer_state_count"] > 0
    assert observation["scheduler"]["last_epoch"] == spec["epoch"]


def remote(payload, timeout=180):
    guard = """
import json,pathlib,socket,subprocess
assert socket.gethostname()=='dl01-MS-7C60'
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT_ID
mount=json.loads(subprocess.check_output(['findmnt','--json','--target','/bak_home',
    '--output','TARGET,SOURCE,FSTYPE,UUID']).decode())['filesystems']
assert len(mount)==1 and mount[0]['target']=='/bak_home'
assert mount[0]['uuid']=='459a8db5-5b91-446b-a501-eed399c6a45d'
assert mount[0]['fstype']=='ext4'
assert not pathlib.Path('/proc',str(OLD_PID)).exists(),'Old PID exists; reconcile'
"""
    program = ("BOOT_ID=" + repr(BOOT) + "\nOLD_PID=" + repr(payload["original_pid"]) +
               "\n" + guard + common.REMOTE.replace("__PAYLOAD__", repr(payload)))
    server = payload["server"]
    command = ("PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=" + shlex.quote(server["site_packages"]) +
               " " + shlex.quote(server["python"]) + " -B -S -")
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
         "-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=1",
         server["user"] + "@" + server["host"], command],
        input=program.encode("utf-8"), capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-4000:])
    import json
    return json.loads(result.stdout)


def inspected_files(candidate, task):
    spec = SPECS[task["id"]]
    review_ref = scoped.ref("pipeline/analyses/" + spec["review"])
    assert review_ref in candidate["evidence"]
    rows = deepcopy(common.load(closure.artifact(ROOT, review_ref))["files"])
    assert len(rows) == 7
    for row in rows:
        assert scoped.ref(row["path"]) == row, "Tested working bytes changed"
        blob = scoped.git("show", task["code_commit"] + ":" + row["path"])
        assert hashlib.sha256(blob).hexdigest() == row["sha256"]
    native = "AdaRES/tools/train.py"
    blob = scoped.git("show", MARKER + ":" + native).replace(b"\r\n", b"\n")
    assert (ROOT / native).read_bytes().replace(b"\r\n", b"\n") == blob
    rows.append(dict(path=native, sha256=hashlib.sha256(blob).hexdigest()))
    return review_ref, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("--id", choices=tuple(SPECS), required=True)
    parser.add_argument("--action", choices=("inspect", "launch", "verify"), required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    assert args.request == REQUEST, "Exact current owning-user recovery request required"
    scoped.lease(args.token)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None, "Project writer busy"
    try:
        server = monitor.load_servers(str(ROOT))["server-2"]
        assert server["host"] == "192.168.1.163" and server["user"] == "songcl"
        assert server["allowed_gpus"] == [0, 1] and server["max_tasks_per_gpu"] == 1
        queue = monitor.load_queue(str(ROOT))
        tasks = queue["gpu_queues"]["server-2"]
        matches = [t for t in tasks if t["id"] == args.id]
        assert len(matches) == 1, "Original queue identity must remain unambiguous"
        task = matches[0]
        work_path = ROOT / "pipeline/research_work.json"
        work = common.load(work_path)
        candidate = next(c for c in work["candidates"] if c["id"] == args.id)
        assert candidate["stage"] == "training", "Do not reopen historical candidates"
        base = ROOT / "pipeline/reports" / ("gpu_reboot_resume_" + args.id + "_" + args.token)
        paths = {a: base.with_name(base.name + "_" + a + ".json")
                 for a in ("inspect", "launch", "verify")}
        if args.action == "inspect":
            assert not paths["inspect"].exists()
            receipt = closure.admission(ROOT, task)
            proof_ref = scoped.ref("pipeline/analyses/" + SPECS[args.id]["dispatch"])
            assert proof_ref in candidate["evidence"]
            proof = common.load(closure.artifact(ROOT, proof_ref))["after"]["tasks"][args.id]
            experiment = monitor.load_experiment(str(ROOT), args.id)
            validate_task(args.id, task, experiment, receipt, proof)
            review, files = inspected_files(candidate, task)
            closure.source_integrity.verify_marker(ROOT, MARKER, server, monitor.ssh)
            payload = dict(action="inspect", server=server, id=args.id,
                           config=task["config"], gpu=str(task["gpu"]), files=files,
                           original_log=task["log"], original_pid=task["pid"],
                           source_marker=MARKER)
            observation = remote(payload)
            checkpoint_matches(args.id, observation, server)
            scoped.lease(args.token)
            common.persist(paths["inspect"], dict(
                complete=True, token=args.token, human_request=args.request,
                observed_local=stamp(), boot_id=BOOT, task=deepcopy(task),
                original_experiment=experiment, original_dispatch=proof_ref,
                tested_review=review, source_manifest_verified=True,
                payload=payload, observation=observation))
            return

        record = common.load(paths["inspect"])
        assert record["token"] == args.token and record["human_request"] == args.request
        assert record["boot_id"] == BOOT
        closure.admission(ROOT, task)
        inspected_files(candidate, task)
        payload = {**record["payload"], **record["observation"], "action": args.action}
        checkpoint_matches(args.id, record["observation"], server)
        if args.action == "launch":
            assert not paths["launch"].exists() and task == record["task"]
            assert monitor.load_experiment(str(ROOT), args.id) == record["original_experiment"]
            payload["log"] = (server["remote_code_root"] + "/logs/gpu_reboot_resume_" +
                              Path(task["config"]).stem + "_" + args.token + ".log")
            monitor.log_decision(
                str(ROOT), "manual", "Owning user requests continued training after GPU reboot.",
                "Resume SAME " + args.id + " from epoch " + str(SPECS[args.id]["epoch"]) +
                " last checkpoint on GPU" + str(task["gpu"]) + "; keep optimizer/scheduler, seed1999/fixed30.",
                "Original admitted/tested source and attempt, new boot/mounted UUID, "
                "CPU-loaded checkpoint and exclusive idle GPU checked; no fresh restart.",
                paths["inspect"].relative_to(ROOT).as_posix())
            task.update(status="claiming", pid=None, log=payload["log"],
                        total_epochs=30, resume_attempt_token=args.token,
                        launch_verified=False, launch_verification="pending_progress",
                        dispatch_phase="launch_started", resume=payload["checkpoint"],
                        resume_checkpoint_sha256=payload["checkpoint_sha256"],
                        reboot_boot_id=BOOT)
            scoped.lease(args.token)
            monitor.save_queue(str(ROOT), queue)
            result = remote(payload)
            task.update(pid=result["pid"], dispatch_phase="progress_pending")
            monitor.save_queue(str(ROOT), queue)
            common.persist(paths["launch"], dict(complete=True, token=args.token,
                payload=payload, result=result, previous=record["task"], observed_local=stamp()))
            return

        assert task["status"] == "claiming" and task["resume_attempt_token"] == args.token
        assert not paths["verify"].exists()
        launch = common.load(paths["launch"])
        assert task["pid"] == launch["result"]["pid"]
        payload.update(pid=task["pid"], log=task["log"])
        samples = []
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            scoped.lease(args.token)
            sample = remote(payload, timeout=30)
            samples.append(dict(observed_local=stamp(), **sample))
            ready = [s for s in samples if s["restored"] and s["progress"] is not None]
            if len(ready) >= 2 and tuple(ready[-1]["progress"]) > tuple(ready[0]["progress"]):
                break
            time.sleep(6)
        else:
            common.persist(paths["verify"].with_name(base.name + "_pending.json"),
                           dict(token=args.token, result="pending_progress", samples=samples))
            raise RuntimeError("Launched PID progress unverified; reconcile, never relaunch")
        native_logs = samples[-1]["native_logs"]
        assert len(native_logs) == 1
        assert samples[-1]["progress"][0] == payload["resume_epoch"]
        carried = remote({**payload, "action": "carry_best", "native_log": native_logs[0]})
        scoped.lease(args.token)
        task.update(status="running", launch_verified=True, launch_verification="training_progress",
                    dispatch_phase="progress_verified", resumed_at=stamp(),
                    resume_epoch=payload["resume_epoch"],
                    resumed_work_dir=str(PurePosixPath(native_logs[0]).parent),
                    carried_best=carried["best_carried_to"])
        monitor.save_queue(str(ROOT), queue)
        running = monitor.load_running(str(ROOT))
        assert running.get("server-2", {}).get(str(task["gpu"])) in (None, args.id)
        running.setdefault("server-2", {})[str(task["gpu"])] = args.id
        monitor.save_running(str(ROOT), running)
        experiment = monitor.load_experiment(str(ROOT), args.id)
        assert experiment == record["original_experiment"]
        original = deepcopy(experiment)
        experiment.update(task)
        for key in ("finished_at", "final_best", "epochs_final", "log_local", "archive_id"):
            experiment.pop(key, None)
        experiment.setdefault("history", []).append(dict(
            ts=stamp(), action="resumed_after_authorized_gpu_reboot", interruption=original,
            human_request=args.request, boot_id=BOOT, checkpoint=task["resume"],
            evidence=paths["verify"].relative_to(ROOT).as_posix()))
        monitor.save_experiment(str(ROOT), experiment)
        deathwatch = monitor.load_deathwatch(str(ROOT))
        deathwatch.setdefault("server-2", {}).pop(Path(task["config"]).stem, None)
        monitor.save_deathwatch(str(ROOT), deathwatch)
        common.persist(paths["verify"], dict(
            complete=True, result="resumed_training_progress_verified", token=args.token,
            samples=samples, checkpoint_preservation=carried, resume_epoch=payload["resume_epoch"],
            pid=task["pid"], gpu=task["gpu"], historical_gpu_fault_preserved=True))
        for path in list(paths.values()) + [Path(__file__)]:
            reference = scoped.ref(path.relative_to(ROOT).as_posix())
            if reference not in candidate["evidence"]:
                candidate["evidence"].append(reference)
        candidate.update(scientific_status="resumed_training_not_efficacy",
                         next_action="Continue SAME fixed30 run from preserved optimizer/scheduler; "
                         "retain GPU fault history, original frozen evaluation and no tuning.")
        scoped.lease(args.token)
        scoped.tick.save(work_path, work)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
