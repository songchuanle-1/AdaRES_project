"""Recheck only the due Y52 commit dependency under its foreground lease."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "259fbac1521b4b0aa547fe01ef96e23a"
ID = "AdaRES-Y52"
SKILL = Path("C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
sys.path.insert(0, str(SKILL))
import foreground_tick as tick
import pipeline_monitor as monitor

RESULT = "pipeline/analyses/y52_authorization_recheck_" + TOKEN + ".json"
BUNDLE = "pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json"
REVIEW = "pipeline/analyses/y52_native_gate_review_89ffd8db36cf43ed8cfd75a68c77b4b9.json"


def ref(relative):
    path = (ROOT / relative).resolve()
    assert ROOT in path.parents and path.is_file()
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git(*args):
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
        text=True, encoding="utf-8", errors="replace", env=environment, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def main():
    owner_path = ROOT / "pipeline/headless/.foreground.lockdir/owner.json"
    owner = tick.load(owner_path)
    assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 240
    for name, key in ((".tick_context.json", "context_sha256"), (".tick_context_state.next.json", "pending_sha256")):
        assert ref("pipeline/headless/" + name)["sha256"] == owner[key]
    view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
    assert view["research"]["due_count"] == 1
    assert [row["id"] for row in view["research"]["shown"]] == [ID]
    assert view["research"]["reserve_count"] == 2 and view["research"]["reserve_budget"] == 0
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == ID)
        assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "human_authorization"
        old_retry = candidate["retry_after"]
        assert datetime.fromisoformat(old_retry) <= datetime.now(timezone.utc)
        other_before = stable_digest([row for row in work["candidates"] if row["id"] != ID])
        original_evidence = list(candidate["evidence"])
        known = {row["path"]: row for row in original_evidence}
        assert ref(BUNDLE) == known[BUNDLE] and ref(REVIEW) == known[REVIEW]
        bundle = tick.load(ROOT / BUNDLE)
        review = tick.load(ROOT / REVIEW)
        assert bundle["id"] == review["id"] == ID and review["tested_bundle"] == ref(BUNDLE)
        files = bundle["files"]
        assert len(files) == 7
        observations = [dict(**ref(row["path"]), expected_sha256=row["sha256"],
            matches_tested_bytes=ref(row["path"])["sha256"] == row["sha256"]) for row in files]
        authorization_path = "pipeline/authorizations/AdaRES-Y52_commit.json"
        assert authorization_path in candidate["wake_paths"]
        authorization_exists = (ROOT / authorization_path).exists()
        head = git("rev-parse", "HEAD")
        paths = [row["path"] for row in files]
        index = git("ls-files", "--stage", "--", *paths)
        status = git("status", "--porcelain=v1", "--untracked-files=all", "--", *paths)
        observed_at = datetime.now().astimezone()
        new_retry = (observed_at + timedelta(hours=4)).isoformat()
        evidence = dict(schema_version=1, complete=True, id=ID, token=TOKEN,
            checked_at=observed_at.isoformat(), snapshot_at=view["generated_at"],
            kind="due_authorization_dependency_recheck_not_scientific_test",
            prior_retry_after=old_retry, next_retry_after_if_still_blocked=new_retry,
            scoped_authorization=dict(path=authorization_path, exists=authorization_exists),
            tested_bundle=ref(BUNDLE), retained_native_review=ref(REVIEW), files=observations,
            git_head=head, git_optional_locks="0", git_index_entries=index.splitlines(),
            scoped_git_status=status.splitlines(), no_commit_supplied_in_new_human_message=True,
            all_seven_bytes_unchanged=all(row["matches_tested_bytes"] for row in observations),
            all_seven_untracked=set(status.splitlines()) == {"?? " + path for path in paths},
            original_unblock_condition=candidate["unblock_condition"],
            other_candidates_sha256_before=other_before,
            native_or_resource_gates_rerun=False, GPU_probed_or_used=False,
            new_scientific_candidates=0, model_changes=0, commits=0, admissions=0, training_launches=0,
            interpretation="A four-hour dependency check, not a new gate PASS, trained gain or resource shortage. Preserve the same seven-file candidate and all frozen evidence; do not extend the old Y50-only permission.",
            helper=ref("scripts/recheck_y52_authorization_259f.py"))
        assert not (ROOT / RESULT).exists()
        tick.save(ROOT / RESULT, evidence)
        assert not authorization_exists, "Authorization intake changed; inspect it instead of extending a block"
        assert evidence["all_seven_bytes_unchanged"] and evidence["all_seven_untracked"] and not index
        assert tick.load(owner_path)["token"] == TOKEN
        candidate["evidence"].extend([ref(RESULT), ref("scripts/recheck_y52_authorization_259f.py")])
        candidate["retry_after"] = new_retry
        candidate.setdefault("authorization_rechecks", []).append(dict(checked_at=observed_at.isoformat(),
            previous_retry_after=old_retry, retry_after=new_retry, result="same_scoped_permission_missing",
            evidence=ref(RESULT)))
        assert candidate["evidence"][:len(original_evidence)] == original_evidence
        assert stable_digest([row for row in work["candidates"] if row["id"] != ID]) == other_before
        work["design_cycle"] = dict(stop_reason="external_resource_blocked", attempted_ids=[ID],
            evidence=[ref(RESULT)],
            resume_condition="Explicit scoped tested-byte commit authorization or a supplied matching commit for Y52/Y54; then recheck goal/source/gates/commit and use the canonical dispatcher after live authorized GPU checks. Existing wake paths can wake sooner; unchanged authorization dependencies wait their four-hour retry.",
            clarification="The only due stable ID Y52 received a real bounded authorization/source/index check. Its tested seven files remain unchanged and untracked with no scoped authorization, so no admission. Y54 and every other candidate are byte-for-byte structurally unchanged. No synthetic candidate, repeated native gate, fake GPU blocker or route-exhaustion assertion.")
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(id=ID, stage=candidate["stage"], evidence=ref(RESULT),
            checked_at=observed_at.isoformat(), old_retry_after=old_retry, retry_after=new_retry,
            scoped_authorization_exists=False, seven_files_unchanged_untracked=True,
            git_head=head, other_candidates_unchanged=True, admissions=0, training_launches=0), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
