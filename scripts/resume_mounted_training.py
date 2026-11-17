"""Resume only the interrupted Y52/Y54 runs after the user's mount recovery."""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath
import subprocess
import time

import authorize_prepared_candidates as scoped
import resume_shutdown_training as common

closure = common.closure
monitor = common.monitor
ROOT = common.ROOT
REQUEST = "\u6302\u8f7d\u4e86 \u63d0\u4ea4\u8bad\u7ec3\u5427"
PROOF = "pipeline/analyses/mounted_training_verified_097ff69b175b49c59fbcf414fb0bd4f5.json"
SPECS = {
    "AdaRES-Y52": ("adares_y52_point_target_res", "1", "AdaRES-Y52_v2.json"),
    "AdaRES-Y54": ("adares_y54_fine_support_res", "0", "AdaRES-Y54.json"),
}


def restored_task(candidate_id, experiment, proof, receipt):
    stem, gpu, admission = SPECS[candidate_id]
    config = "configs_song/model_paper5/" + stem + ".yaml"
    assert experiment["id"] == candidate_id
    assert experiment["status"] == "stopped_unknown", "Only an interrupted run may resume"
    assert experiment["server"] == "server-2" and str(experiment["gpu"]) == gpu
    assert experiment["config"] == config == receipt["config"]
    assert receipt["id"] == candidate_id
    assert experiment["pid"] == proof["pid"] and experiment["log"] == proof["log"]
    argv = proof["cmdline"]
    assert "tools/train.py" in argv and config in argv
    assert argv[argv.index("--gpu_ids") + 1] == gpu
    assert argv[argv.index("--num_gpus") + 1] == "1"
    return dict(
        id=candidate_id, label=candidate_id + " fixed30", config=config,
        server="server-2", gpu=gpu, status="stopped_unknown",
        code_commit=receipt["code_commit"],
        admission_receipt="pipeline/admissions/" + admission,
        patience=30, min_epochs=30, fixed_epochs=30, total_epochs=30,
        scientific_candidate_id=candidate_id, requires_exclusive_ownership=True,
        pid=experiment["pid"], log=experiment["log"],
    )


def stamp():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("--id", choices=tuple(SPECS), required=True)
    parser.add_argument("--action", choices=("inspect", "launch", "verify"), required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    assert args.request == REQUEST, "Direct current owning-user request required"
    scoped.lease(args.token)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None, "Pipeline writer busy"
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
        candidate = next(c for c in work["candidates"] if c["id"] == args.id)
        assert candidate["stage"] == "training", "Do not reopen another research stage"
        base = ROOT / "pipeline/reports" / ("mount_resume_" + args.id + "_" + args.token)
        paths = {action: base.with_name(base.name + "_" + action + ".json")
                 for action in ("inspect", "launch", "verify")}
        if args.action == "inspect":
            assert not matches and not paths["inspect"].exists()
            proof_ref = next(e for e in candidate["evidence"] if e["path"] == PROOF)
            proof_record = common.load(closure.artifact(ROOT, proof_ref))
            proof = proof_record["after"]["tasks"][args.id]
            experiment = monitor.load_experiment(str(ROOT), args.id)
            admission = "pipeline/admissions/" + SPECS[args.id][2]
            task = restored_task(args.id, experiment, proof, common.load(ROOT / admission))
            closure.admission(ROOT, task)
            bundle_ref, rows = scoped.checked_bundle(args.id, candidate)
            assert closure.git(ROOT, "rev-parse", "HEAD") == task["code_commit"]
            pinned = subprocess.check_output([
                "git", "-C", str(ROOT), "show",
                task["code_commit"] + ":AdaRES/tools/train.py"])
            import hashlib
            train_sha = hashlib.sha256(pinned).hexdigest()
            assert closure.digest(ROOT / "AdaRES/tools/train.py") == train_sha
            files = list(rows) + [{"path": "AdaRES/tools/train.py", "sha256": train_sha}]
            payload = dict(
                action="inspect", server=server, config=task["config"], id=args.id,
                gpu=task["gpu"], files=files, original_log=task["log"],
                original_pid=task["pid"], source_marker=task["code_commit"])
            observation = common.remote(payload)
            scoped.lease(args.token)
            common.persist(paths["inspect"], dict(
                complete=True, token=args.token, human_request=args.request,
                authorization="Current direct owning-user mount recovery and training request",
                observed_local=stamp(), task=task, original_experiment=experiment,
                prior_verified_run=proof_ref, tested_bundle=bundle_ref,
                payload=payload, observation=observation))
            return

        record = common.load(paths["inspect"])
        assert record["token"] == args.token and record["human_request"] == args.request
        payload = {**record["payload"], **record["observation"], "action": args.action}
        assert payload["server"] == server
        if args.action == "launch":
            assert not matches and not paths["launch"].exists()
            assert monitor.load_experiment(str(ROOT), args.id) == record["original_experiment"]
            task = deepcopy(record["task"])
            closure.admission(ROOT, task)
            scoped.checked_bundle(args.id, candidate)
            payload["log"] = server["remote_code_root"] + "/logs/resume_" + Path(
                task["config"]).stem + "_" + args.token + ".log"
            monitor.log_decision(
                str(ROOT), "manual", "Owning user confirms mount and requests training.",
                "Resume the SAME " + args.id + " from its intact last checkpoint on GPU" +
                task["gpu"] + "; preserve optimizer/scheduler and fixed30, never fresh-start.",
                "Frozen admission/source, original attempt, saved state and idle GPU checked.",
                paths["inspect"].relative_to(ROOT).as_posix())
            scoped.lease(args.token)
            task.update(
                status="claiming", resume_attempt_token=args.token,
                launch_verified=False, launch_verification="pending_progress",
                dispatch_phase="launch_started", resume=payload["checkpoint"],
                resume_checkpoint_sha256=payload["checkpoint_sha256"],
                interrupted_attempt_evidence=scoped.ref(paths["inspect"].relative_to(ROOT).as_posix()))
            tasks.append(task)
            monitor.save_queue(str(ROOT), queue)
            result = common.remote(payload)
            task.update(pid=result["pid"], log=result["log"], dispatch_phase="progress_pending")
            monitor.save_queue(str(ROOT), queue)
            common.persist(paths["launch"], dict(
                token=args.token, payload=payload, result=result,
                previous=record["original_experiment"], observed_local=stamp()))
            return

        assert len(matches) == 1 and not paths["verify"].exists()
        task = matches[0]
        launch = common.load(paths["launch"])
        assert task["resume_attempt_token"] == args.token
        assert task["pid"] == launch["result"]["pid"]
        assert task["status"] == "claiming"
        closure.admission(ROOT, task)
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
            raise RuntimeError("Exact PID launched; progress unverified. Never relaunch blindly.")
        native_logs = samples[-1]["native_logs"]
        assert len(native_logs) == 1
        carried = common.remote({
            **payload, "action": "carry_best", "native_log": native_logs[0]})
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
        assert experiment == record["original_experiment"]
        original = deepcopy(experiment)
        experiment.update({key: value for key, value in task.items() if key not in (
            "label", "interrupted_attempt_evidence")})
        for key in ("finished_at", "final_best", "epochs_final", "log_local", "archive_id"):
            experiment.pop(key, None)
        experiment.setdefault("history", []).append(dict(
            ts=task["resumed_at"], action="resumed_after_mount_recovery",
            interruption=original, human_request=args.request,
            checkpoint=task["resume"], pid=task["pid"], gpu=task["gpu"],
            evidence=paths["verify"].relative_to(ROOT).as_posix()))
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
            next_action="Continue the SAME fixed30 run from preserved optimizer/scheduler state; "
                        "retain frozen terminal evaluation and harm rules. No fresh restart or tuning.")
        scoped.lease(args.token)
        scoped.tick.save(work_path, work)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
