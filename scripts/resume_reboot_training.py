"""Continue only the latest verified Y52/Y54 attempts after an explicit reboot."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import subprocess
import time

import authorize_prepared_candidates as scoped
import resume_mounted_training as previous

common = previous.common
closure = common.closure
monitor = common.monitor
ROOT = common.ROOT
SPECS = previous.SPECS
REQUEST = "\u4e0d\u5c0f\u5fc3\u91cd\u542f\u673a\u5668\u4e86 \u4f60\u91cd\u65b0resume\u5427"
MOUNT_UUID = "459a8db5-5b91-446b-a501-eed399c6a45d"


def stamp():
    return datetime.now().astimezone().isoformat()


def restored_task(candidate_id, experiment, launch, verification, receipt):
    stem, gpu, admission = SPECS[candidate_id]
    config = "configs_song/model_paper5/" + stem + ".yaml"
    assert experiment["id"] == receipt["id"] == candidate_id
    assert experiment["status"] in ("running", "stopped_unknown")
    assert experiment["server"] == "server-2" and str(experiment["gpu"]) == gpu
    assert experiment["config"] == receipt["config"] == config
    assert experiment["code_commit"] == receipt["code_commit"]
    assert all(experiment[key] == 30 for key in
               ("fixed_epochs", "min_epochs", "total_epochs", "patience"))
    assert launch["result"]["pid"] == experiment["pid"] == verification["pid"]
    assert launch["result"]["log"] == experiment["log"]
    assert launch["token"] == experiment["resume_attempt_token"]
    assert verification["token"] == experiment["resume_attempt_token"]
    assert launch["payload"]["config"] == config
    assert str(launch["payload"]["gpu"]) == str(verification["gpu"]) == gpu
    assert launch["payload"]["checkpoint"] == experiment["resume"]
    assert verification["result"] == "resumed_training_progress_verified"
    samples = verification["samples"]
    assert len(samples) >= 2
    assert all(sample["pid"] == experiment["pid"] and sample["restored"] and
               not sample["errors"] and sample["progress"] is not None for sample in samples)
    assert tuple(samples[-1]["progress"]) > tuple(samples[0]["progress"])
    native_logs = samples[-1]["native_logs"]
    assert len(native_logs) == 1
    assert str(PurePosixPath(native_logs[0]).parent) == experiment["resumed_work_dir"]
    assert verification["resume_epoch"] == experiment["resume_epoch"]
    return dict(
        id=candidate_id, label=candidate_id + " fixed30", config=config,
        server="server-2", gpu=gpu, status=experiment["status"],
        code_commit=receipt["code_commit"],
        admission_receipt="pipeline/admissions/" + admission,
        patience=30, min_epochs=30, fixed_epochs=30, total_epochs=30,
        scientific_candidate_id=candidate_id, requires_exclusive_ownership=True,
        pid=experiment["pid"], log=experiment["log"],
    )


def boot_check(server, old_pid, boot_id):
    payload = dict(pid=old_pid, boot_id=boot_id, mount_uuid=MOUNT_UUID)
    program = r'''
import json,pathlib,socket,subprocess
P=__PAYLOAD__
assert socket.gethostname()=="dl01-MS-7C60","Unexpected server identity"
boot_id=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
assert boot_id==P["boot_id"],"Server rebooted again; inspect before continuing"
mount=json.loads(subprocess.check_output(["findmnt","--json","--target","/bak_home",
    "--output","TARGET,SOURCE,FSTYPE,UUID"]).decode())["filesystems"]
assert len(mount)==1 and mount[0]["target"]=="/bak_home"
assert mount[0]["source"]=="/dev/sda1" and mount[0]["uuid"]==P["mount_uuid"]
assert not pathlib.Path("/proc",str(P["pid"])).exists(),"Previous PID remains; reconcile"
print(json.dumps({"hostname":socket.gethostname(),"boot_id":boot_id,"mount":mount[0],
    "previous_pid":P["pid"],"previous_pid_absent":True}))
'''.replace("__PAYLOAD__", repr(payload))
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
         "-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=1",
         f"{server['user']}@{server['host']}", "/usr/bin/python3 -B -S -"],
        input=program.encode("utf-8"), capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-2000:])
    return json.loads(result.stdout)


def prior_evidence(candidate, experiment):
    relative = experiment["history"][-1]["evidence"]
    prefix = "pipeline/reports/"
    assert relative.startswith(prefix) and relative.endswith("_verify.json")
    assert any(relative.startswith(prefix + kind + experiment["id"] + "_")
               for kind in ("mount_resume_", "reboot_resume_"))
    assert experiment["resume_attempt_token"] in relative
    names = [relative[:-len("_verify.json")] + "_" + action + ".json"
             for action in ("inspect", "launch", "verify")]
    refs = [next(entry for entry in candidate["evidence"] if entry["path"] == name)
            for name in names]
    records = [common.load(closure.artifact(ROOT, entry)) for entry in refs]
    return refs, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("--id", choices=tuple(SPECS), required=True)
    parser.add_argument("--action", choices=("inspect", "launch", "verify"), required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--boot-id", required=True)
    args = parser.parse_args()
    assert args.request == REQUEST, "Current owning-user reboot/resume request required"
    scoped.lease(args.token)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None, "Project writer busy"
    try:
        server = monitor.load_servers(str(ROOT))["server-2"]
        assert server["host"] == "192.168.1.163" and server["user"] == "songcl"
        assert server["allowed_gpus"] == [0, 1] and server["max_tasks_per_gpu"] == 1
        assert server.get("dispatch_enabled", True)
        queue = monitor.load_queue(str(ROOT))
        tasks = queue["gpu_queues"]["server-2"]
        matches = [task for task in tasks if task["id"] == args.id]
        assert len(matches) <= 1, "Ambiguous queue identity"
        work_path = ROOT / "pipeline/research_work.json"
        work = common.load(work_path)
        candidate = next(item for item in work["candidates"] if item["id"] == args.id)
        assert candidate["stage"] == "training"
        base = ROOT / "pipeline/reports" / ("reboot_resume_" + args.id + "_" + args.token)
        paths = {action: base.with_name(base.name + "_" + action + ".json")
                 for action in ("inspect", "launch", "verify")}
        if args.action == "inspect":
            assert not paths["inspect"].exists()
            experiment = monitor.load_experiment(str(ROOT), args.id)
            refs, records = prior_evidence(candidate, experiment)
            task = restored_task(args.id, experiment, records[1], records[2],
                                 common.load(ROOT / "pipeline/admissions" / SPECS[args.id][2]))
            if matches:
                assert all(matches[0][key] == task[key] for key in
                           ("id", "config", "server", "gpu", "pid", "log", "status"))
                assert matches[0].get("launch_verified")
            else:
                assert experiment["status"] == "stopped_unknown"
            closure.admission(ROOT, task)
            bundle_ref, rows = scoped.checked_bundle(args.id, candidate)
            assert closure.git(ROOT, "rev-parse", "HEAD") == task["code_commit"]
            pinned = subprocess.check_output([
                "git", "-C", str(ROOT), "show", task["code_commit"] + ":AdaRES/tools/train.py"])
            import hashlib
            train_sha = hashlib.sha256(pinned).hexdigest()
            assert closure.digest(ROOT / "AdaRES/tools/train.py") == train_sha
            boot = boot_check(server, task["pid"], args.boot_id)
            payload = dict(
                action="inspect", server=server, config=task["config"], id=args.id,
                gpu=task["gpu"], files=list(rows) + [
                    {"path": "AdaRES/tools/train.py", "sha256": train_sha}],
                original_log=task["log"], original_pid=task["pid"],
                source_marker=task["code_commit"])
            observation = common.remote(payload)
            assert str(PurePosixPath(observation["checkpoint"]).parent) == experiment[
                "resumed_work_dir"], "Refuse fallback to a pre-resume checkpoint"
            scoped.lease(args.token)
            common.persist(paths["inspect"], dict(
                complete=True, token=args.token, human_request=args.request,
                boot_id=args.boot_id, reboot_observation=boot, observed_local=stamp(),
                task=task, original_queue_task=deepcopy(matches[0]) if matches else None,
                original_experiment=experiment, previous_attempt_evidence=refs,
                tested_bundle=bundle_ref, payload=payload, observation=observation))
            return

        record = common.load(paths["inspect"])
        assert record["token"] == args.token and record["human_request"] == args.request
        assert record["boot_id"] == args.boot_id
        assert monitor.load_experiment(str(ROOT), args.id) == record["original_experiment"]
        payload = {**record["payload"], **record["observation"], "action": args.action}
        assert payload["server"] == server
        if args.action == "launch":
            assert not paths["launch"].exists()
            original = record["original_queue_task"]
            assert (matches == [original]) if original is not None else not matches
            closure.admission(ROOT, record["task"])
            scoped.checked_bundle(args.id, candidate)
            boot = boot_check(server, record["task"]["pid"], args.boot_id)
            payload["log"] = server["remote_code_root"] + "/logs/reboot_resume_" + Path(
                record["task"]["config"]).stem + "_" + args.token + ".log"
            monitor.log_decision(
                str(ROOT), "manual", "User reports accidental reboot and requests resume.",
                "Continue SAME " + args.id + " from latest verified-attempt last checkpoint "
                "on GPU" + record["task"]["gpu"] + "; preserve optimizer/scheduler and fixed30.",
                "Verified boot, mounted disk, absent former PID, frozen source/admission, "
                "latest attempt and idle exclusive GPU; never a fresh start.",
                paths["inspect"].relative_to(ROOT).as_posix())
            task = deepcopy(record["task"])
            task.update(
                status="claiming", pid=None, log=payload["log"],
                resume_attempt_token=args.token, launch_verified=False,
                launch_verification="pending_progress", dispatch_phase="launch_started",
                resume=payload["checkpoint"], resume_checkpoint_sha256=payload["checkpoint_sha256"],
                reboot_boot_id=args.boot_id,
                interrupted_attempt_evidence=scoped.ref(paths["inspect"].relative_to(ROOT).as_posix()))
            scoped.lease(args.token)
            if matches:
                tasks[tasks.index(matches[0])] = task
            else:
                tasks.append(task)
            monitor.save_queue(str(ROOT), queue)
            result = common.remote(payload)
            task.update(pid=result["pid"], dispatch_phase="progress_pending")
            monitor.save_queue(str(ROOT), queue)
            common.persist(paths["launch"], dict(
                token=args.token, payload=payload, result=result, reboot_observation=boot,
                previous=record["original_experiment"], observed_local=stamp()))
            return

        assert len(matches) == 1 and not paths["verify"].exists()
        task = matches[0]
        launch = common.load(paths["launch"])
        assert task["resume_attempt_token"] == args.token and task["pid"] == launch["result"]["pid"]
        assert task["status"] == "claiming"
        closure.admission(ROOT, task)
        boot_check(server, record["task"]["pid"], args.boot_id)
        payload.update(action="verify", pid=task["pid"], log=task["log"])
        samples = []
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            scoped.lease(args.token)
            sample = common.remote(payload, timeout=30)
            samples.append(dict(observed_local=stamp(), **sample))
            progressing = [s for s in samples if s["restored"] and s["progress"] is not None]
            if len(progressing) >= 2 and tuple(progressing[-1]["progress"]) > tuple(
                    progressing[0]["progress"]):
                break
            time.sleep(6)
        else:
            common.persist(paths["verify"], dict(
                result="pending_progress", token=args.token, samples=samples))
            raise RuntimeError("Exact PID launched but progress unverified; never relaunch blindly.")
        native_logs = samples[-1]["native_logs"]
        assert len(native_logs) == 1
        carried = common.remote({**payload, "action": "carry_best", "native_log": native_logs[0]})
        scoped.lease(args.token)
        task.update(
            status="running", launch_verified=True, launch_verification="training_progress",
            resumed_at=stamp(), resume_epoch=payload["resume_epoch"],
            resumed_work_dir=str(PurePosixPath(native_logs[0]).parent),
            carried_best=carried["best_carried_to"], dispatch_phase="progress_verified")
        monitor.save_queue(str(ROOT), queue)
        running = monitor.load_running(str(ROOT))
        assert running.get("server-2", {}).get(task["gpu"]) in (None, args.id)
        running.setdefault("server-2", {})[task["gpu"]] = args.id
        monitor.save_running(str(ROOT), running)
        experiment = monitor.load_experiment(str(ROOT), args.id)
        interrupted = {key: experiment.get(key) for key in
                       ("status", "pid", "log", "resume", "resume_epoch", "resumed_work_dir")}
        experiment.update({key: value for key, value in task.items() if key not in
                           ("label", "interrupted_attempt_evidence")})
        for key in ("finished_at", "final_best", "epochs_final", "log_local", "archive_id"):
            experiment.pop(key, None)
        experiment.setdefault("history", []).append(dict(
            ts=task["resumed_at"], action="resumed_after_user_reboot",
            interruption=interrupted, human_request=args.request, boot_id=args.boot_id,
            checkpoint=task["resume"], evidence=paths["verify"].relative_to(ROOT).as_posix()))
        monitor.save_experiment(str(ROOT), experiment)
        deathwatch = monitor.load_deathwatch(str(ROOT))
        deathwatch.setdefault("server-2", {}).pop(Path(task["config"]).stem, None)
        monitor.save_deathwatch(str(ROOT), deathwatch)
        common.persist(paths["verify"], dict(
            result="resumed_training_progress_verified", token=args.token,
            samples=samples, checkpoint_preservation=carried,
            resume_epoch=payload["resume_epoch"], pid=task["pid"], gpu=task["gpu"]))
        for path in paths.values():
            candidate["evidence"].append(scoped.ref(path.relative_to(ROOT).as_posix()))
        candidate.update(
            scientific_status="resumed_training_not_efficacy",
            next_action="Continue SAME fixed30 from preserved optimizer/scheduler after reboot; "
                        "retain frozen terminal evaluation and harm rules, no restart or tuning.")
        scoped.lease(args.token)
        scoped.tick.save(work_path, work)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
