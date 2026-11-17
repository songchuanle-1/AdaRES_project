"""Bounded due-only commit checks for prepared Y52/Y54; no commits or dispatch."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
sys.path.insert(0, str(SKILL))
import foreground_tick as tick
import pipeline_monitor as monitor


def ref(relative):
    path = (ROOT / relative).resolve()
    assert ROOT in path.parents and path.is_file()
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def stable_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=30,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--candidate-id", choices=("AdaRES-Y52", "AdaRES-Y54"), required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--review", required=True)
    args = parser.parse_args()
    owner_path = ROOT / "pipeline/headless/.foreground.lockdir/owner.json"
    owner = tick.load(owner_path)
    assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 240
    for name, key in ((".tick_context.json", "context_sha256"), (".tick_context_state.next.json", "pending_sha256")):
        assert ref("pipeline/headless/" + name)["sha256"] == owner[key]
    view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
    assert args.candidate_id in {row["id"] for row in view["research"]["shown"]}
    assert view["research"]["due_count"] > 0 and view["research"]["reserve_budget"] == 0
    result_path = "pipeline/analyses/" + args.candidate_id.split("-", 1)[1].lower() + "_authorization_recheck_" + args.token + ".json"
    assert not (ROOT / result_path).exists()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == args.candidate_id)
        assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "human_authorization"
        old_retry = candidate["retry_after"]
        assert datetime.fromisoformat(old_retry) <= datetime.now(timezone.utc)
        others = lambda: [row for row in work["candidates"] if row["id"] != args.candidate_id]
        other_before = stable_digest(others())
        original_evidence = list(candidate["evidence"])
        known = {row["path"]: row for row in original_evidence}
        assert ref(args.bundle) == known[args.bundle] and ref(args.review) == known[args.review]
        bundle, review = tick.load(ROOT / args.bundle), tick.load(ROOT / args.review)
        assert bundle["id"] == review["id"] == args.candidate_id
        assert (review.get("tested_bundle") or review.get("bundle")) == ref(args.bundle)
        assert bundle["complete"] and review["complete"]
        files = bundle["files"]
        assert len(files) == 7 and len({row["path"] for row in files}) == 7
        observed_files = []
        for row in files:
            actual = ref(row["path"])
            observed_files.append(dict(**actual, expected_sha256=row["sha256"],
                matches_tested_bytes=actual["sha256"] == row["sha256"]))
        auth_path = "pipeline/authorizations/" + args.candidate_id + "_commit.json"
        assert auth_path in candidate["wake_paths"]
        auth_exists = (ROOT / auth_path).exists()
        assert Path(git("rev-parse", "--show-toplevel")).resolve() == ROOT
        head = git("rev-parse", "HEAD")
        paths = [row["path"] for row in files]
        index = git("ls-files", "--stage", "--", *paths)
        status = git("status", "--porcelain=v1", "--untracked-files=all", "--", *paths)
        observed_at = datetime.now().astimezone()
        new_retry = (observed_at + timedelta(hours=4)).isoformat()
        helper = "scripts/recheck_candidate_commit_dependency.py"
        evidence = dict(schema_version=1, complete=True, id=args.candidate_id, token=args.token,
            checked_at=observed_at.isoformat(), snapshot_at=view["generated_at"],
            kind="due_authorization_dependency_recheck_not_scientific_test",
            prior_retry_after=old_retry, next_retry_after_if_still_blocked=new_retry,
            scoped_authorization=dict(path=auth_path, exists=auth_exists),
            tested_bundle=ref(args.bundle), retained_native_review=ref(args.review), files=observed_files,
            git_head=head, git_optional_locks="0", git_index_entries=index.splitlines(),
            scoped_git_status=status.splitlines(), no_new_supplied_matching_commit=True,
            all_seven_bytes_unchanged=all(row["matches_tested_bytes"] for row in observed_files),
            all_seven_untracked=set(status.splitlines()) == {"?? " + path for path in paths},
            original_unblock_condition=candidate["unblock_condition"], other_candidates_sha256_before=other_before,
            native_or_resource_gates_rerun=False, GPU_probed_or_used=False,
            new_scientific_candidates=0, model_changes=0, commits=0, admissions=0, training_launches=0,
            interpretation="A due four-hour dependency check, not a new gate PASS, trained gain or GPU shortage. Existing Y50-only permission is not extended; no historical candidate or tested source changed.",
            helper=ref(helper))
        tick.save(ROOT / result_path, evidence)
        assert not auth_exists, "Authorization intake changed; inspect it rather than extending a block"
        assert evidence["all_seven_bytes_unchanged"] and evidence["all_seven_untracked"] and not index
        assert tick.load(owner_path)["token"] == args.token
        candidate["evidence"].extend([ref(result_path), ref(helper)])
        candidate["retry_after"] = new_retry
        candidate.setdefault("authorization_rechecks", []).append(dict(checked_at=observed_at.isoformat(),
            previous_retry_after=old_retry, retry_after=new_retry, result="same_scoped_permission_missing",
            evidence=ref(result_path)))
        assert candidate["evidence"][:len(original_evidence)] == original_evidence
        assert stable_digest(others()) == other_before
        work["design_cycle"] = dict(stop_reason="external_resource_blocked", attempted_ids=[args.candidate_id],
            evidence=[ref(result_path)],
            resume_condition="Explicit scoped tested-byte commit authorization or a supplied matching commit for a prepared stable candidate; then recheck current goal/source/gates/commit and use the existing dispatcher after live authorized GPU checks. Preserve wake paths and four-hour dependency retries.",
            clarification="Only the due stable ID " + args.candidate_id + " received a real bounded authorization/source/index check. Its seven tested files remain unchanged and untracked without scoped permission. Every other candidate, historical closed/rejected record and native gate remains unchanged. No new prototype, equivalent scientific probe, fake GPU blocker or route-exhaustion claim.")
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(id=args.candidate_id, stage=candidate["stage"], evidence=ref(result_path),
            checked_at=observed_at.isoformat(), old_retry_after=old_retry, retry_after=new_retry,
            scoped_authorization_exists=False, seven_files_unchanged_untracked=True,
            git_head=head, other_candidates_unchanged=True, admissions=0, training_launches=0), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
