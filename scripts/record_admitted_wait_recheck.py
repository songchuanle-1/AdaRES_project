"""Record a due admitted candidate's measured unreachable-server dependency."""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import authorize_prepared_candidates as scoped
import closure
import foreground_tick as tick
import pipeline_monitor as monitor

CHINA = timezone(timedelta(hours=8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("--candidate-id", choices=("AdaRES-Y54",), required=True)
    args = parser.parse_args()
    scoped.lease(args.token)
    lock = monitor.acquire_lock(str(ROOT))
    if lock is None:
        raise ValueError("Project writer is busy; no mutation performed")
    try:
        scoped.lease(args.token)
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        if args.candidate_id not in {c["id"] for c in view["research"]["shown"]}:
            raise ValueError("Only this tick's bound due work may be recorded")
        observed = datetime.fromisoformat(view["generated_at"]).astimezone(CHINA)
        retry = observed + timedelta(hours=1)
        if retry <= datetime.now(CHINA):
            raise ValueError("The preflight observation is too old for this recheck")
        text = (ROOT / "pipeline/headless/.foreground_monitor.txt").read_text(encoding="utf-8")
        section = text.split("--- server-2 (192.168.1.163) ---", 1)[1].split("---", 1)[0]
        if "[不可达]" not in section:
            raise ValueError("Current monitor does not establish an unreachable server")
        work = tick.load(ROOT / "pipeline/research_work.json")
        queue = monitor.load_queue(str(ROOT))
        matches = [c for c in work["candidates"] if c["id"] == args.candidate_id]
        tasks = [t for t in queue["gpu_queues"]["server-2"] if t["id"] == args.candidate_id]
        if len(matches) != 1 or len(tasks) != 1:
            raise ValueError("Candidate/queue binding is missing or ambiguous")
        candidate, task = matches[0], tasks[0]
        if candidate["stage"] != "blocked" or task["status"] != "waiting":
            raise ValueError("Reconcile existing execution instead of recording a wait")
        closure.admission(ROOT, task)
        path = f"pipeline/analyses/y54_wait_recheck_{args.token}.json"
        if (ROOT / path).exists():
            raise ValueError("Do not replay an already recorded recheck")
        source_root = ROOT / "AdaRES/ipdn"
        facts = dict(
            complete=True, token=args.token, id=args.candidate_id,
            observed_at=observed.isoformat(), recorded_at=datetime.now(CHINA).isoformat(),
            retry_after=retry.isoformat(), kind="external_dependency_not_scientific_test",
            preflight=view["preflight"], server="server-2", host="192.168.1.163",
            monitor=scoped.ref("pipeline/headless/.foreground_monitor.txt"),
            archived_monitor_text=text,
            current_live_gpu_state="unknown", current_y52_progress="unknown",
            registry=[{k: t.get(k) for k in ("id", "status", "gpu", "pid", "launch_verified")}
                      for t in queue["gpu_queues"]["server-2"] if t["id"] in ("AdaRES-Y52", "AdaRES-Y54")],
            last_verified_y52_evidence=scoped.ref(
                "pipeline/analyses/prepared_progress_b5a843848cd34b83bf362acf98c88634_140133.json"),
            local_capabilities=dict(
                python=sys.executable,
                torch_available=importlib.util.find_spec("torch") is not None,
                numpy_available=importlib.util.find_spec("numpy") is not None,
                scipy_available=importlib.util.find_spec("scipy") is not None,
                source_directories=sorted(p.name for p in source_root.iterdir() if p.is_dir()),
                dataset_directory_exists=(source_root / "data").is_dir(),
            ),
            budgets=dict(training_admission=view["queue"]["candidate_budget"],
                         reserve_budget=view["research"]["reserve_budget"],
                         reserve_only=view["research"]["reserve_only"]),
            new_candidates=0, new_scientific_tests=0, model_revisions=0,
            new_admissions=0, remote_mutations=0, launches=0,
            disposition=("Resume the admitted Y54 dependency only. One reserve slot is "
                "available but not filled by a placeholder. The inspected local mirror "
                "has model packages, not shared dataset/collation implementations, and "
                "the configured inspected local Python has no Torch. This does not "
                "claim every independent research direction is exhausted. No native "
                "test or novel mechanism is certified by source inspection."),
        )
        tick.save(ROOT / path, facts)
        evidence = scoped.ref(path)
        previous = {k: candidate[k] for k in (
            "blocker_kind", "blocker", "retry_after", "unblock_condition") if k in candidate}
        previous.update(superseded_at=observed.isoformat(), superseded_by=evidence)
        candidate.setdefault("blocker_history", []).append(previous)
        candidate["evidence"].append(evidence)
        resume = ("Restore authenticated server-2 connectivity, then reconcile existing "
            "Y52 config/PID/log and current native progress without a blind restart. "
            "Continue the SAME admitted Y54 through canonical dispatch only when an "
            "authorized visible GPU is really idle and source/lease/progress checks "
            "pass. Original commit consent is already resolved.")
        candidate.update(
            blocker_kind="external_resource", required_server="server-2",
            blocker=("Current canonical preflight reports server-2 unreachable. Live "
                "GPU availability, source state and Y52 progress are unknown; the "
                "earlier single visible GPU is cached evidence, not proof of current "
                "availability. Local scoped commits/admission are already resolved."),
            retry_after=retry.isoformat(), unblock_condition=resume, next_action=resume,
            last_resource_check_at=observed.isoformat(),
        )
        work["design_cycle"] = dict(
            stop_reason="external_resource_blocked", attempted_ids=[args.candidate_id],
            evidence=[evidence], resume_condition=resume,
            clarification="Real admitted dependency check; no route-exhaustion or scientific PASS claim.",
        )
        closure.validate_work(
            ROOT, tick.load(ROOT / "pipeline/headless/.tick_context.json"),
            work["candidates"], queue,
        )
        scoped.lease(args.token)
        tick.save(ROOT / "pipeline/research_work.json", work)
        decision = monitor.log_decision(
            str(ROOT), "other", "Y54 woke after queue/source changes but server-2 is unreachable.",
            "Preserve Y52 registry and the same admitted waiting Y54; await verified connectivity.",
            "Current preflight cannot confirm live resources; no new reserve placeholder or native PASS.", path,
        )
        print(json.dumps(dict(evidence=evidence, decision=decision,
            retry_after=retry.isoformat(), stage=candidate["stage"],
            new_candidates=0, new_scientific_tests=0, launches=0), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
