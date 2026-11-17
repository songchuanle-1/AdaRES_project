"""Apply the owning user's scoped continuation to the tested Y52/Y54 files."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import closure
import foreground_tick as tick
import pipeline_monitor as monitor

THREAD = "01a07e32-78bf-7670-925b-370911ea48db"
MESSAGE_ID = "01a15264-f969-74c2-9640-2edb5406a06e"
SPECS = {
    "AdaRES-Y52": (
        "adares_y52_point_target",
        "pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json",
    ),
    "AdaRES-Y54": (
        "adares_y54_fine_support",
        "pipeline/analyses/y54_prototype_bundle_fde01739a9864bae9bb342a577235e66.json",
    ),
}


def ref(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("Missing or outside-project evidence: " + relative)
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "-c", "core.autocrlf=false", *args]
    )


def lease(token):
    headless = ROOT / "pipeline/headless"
    owner = tick.load(headless / ".foreground.lockdir/owner.json")
    if (owner["token"] != token or owner["thread_id"] != THREAD
            or os.environ.get("CODEX_THREAD_ID") != THREAD):
        raise ValueError("The current owning foreground lease is required")
    if (datetime.fromisoformat(owner["deadline_at"])
            - datetime.now(timezone.utc)).total_seconds() <= 180:
        raise ValueError("Insufficient time for a scoped operation and closeout")
    for name, key in ((".tick_context.json", "context_sha256"),
                      (".tick_context_state.next.json", "pending_sha256")):
        if tick.digest(headless / name) != owner[key]:
            raise ValueError("The original tick snapshot changed")
    return owner


def checked_bundle(candidate_id, candidate):
    package, bundle_path = SPECS[candidate_id]
    bundle_ref = ref(bundle_path)
    if bundle_ref not in candidate["evidence"]:
        raise ValueError("Tested bundle is not bound to the stable candidate")
    bundle = tick.load(ROOT / bundle_path)
    files = bundle["files"]
    expected = {
        f"AdaRES/ipdn/model_paper5/{package}/{name}"
        for name in ("__init__.py", "backbone.py", "dec.py", "ipdn.py",
                     "loss.py", "sample_model.py")
    } | {f"AdaRES/configs_song/model_paper5/{package}_res.yaml"}
    if (not bundle["complete"] or bundle["id"] != candidate_id
            or len(files) != 7 or {row["path"] for row in files} != expected):
        raise ValueError("Authorization is restricted to the exact seven-file bundle")
    if any(ref(row["path"]) != row for row in files):
        raise ValueError("Source bytes no longer match the tested bundle")
    for entry in candidate["evidence"]:
        closure.artifact(ROOT, entry)
    return bundle_ref, files


def unrelated_index(files):
    return sorted(
        row for row in git("ls-files", "--stage", "-z").split(b"\0")
        if row and row.split(b"\t", 1)[1].decode("utf-8") not in files
    )


def apply(candidate_id, token, operation):
    owner = lease(token)
    lock = monitor.acquire_lock(str(ROOT))
    if lock is None:
        raise ValueError("Project writer is busy; no mutation performed")
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        matches = [c for c in work["candidates"] if c["id"] == candidate_id]
        if len(matches) != 1:
            raise ValueError("Candidate is missing or duplicated")
        candidate = matches[0]
        bundle_ref, rows = checked_bundle(candidate_id, candidate)
        files = [row["path"] for row in rows]
        if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT.resolve():
            raise ValueError("Unexpected Git repository")
        head = git("rev-parse", "HEAD").decode().strip()
        auth_path = f"pipeline/authorizations/{candidate_id}_commit.json"
        if operation == "inspect":
            result = dict(id=candidate_id, tested_files=rows, code_commit=head,
                          git_status=git("status", "--porcelain=v1",
                                         "--untracked-files=all", "--", *files).decode(),
                          authorization_exists=(ROOT / auth_path).exists())
        elif operation == "authorize":
            if (candidate["stage"] != "blocked"
                    or candidate.get("blocker_kind") != "human_authorization"
                    or (ROOT / auth_path).exists()):
                raise ValueError("Authorization state changed; inspect before proceeding")
            if git("ls-files", "--", *files).strip():
                raise ValueError("Candidate files are unexpectedly tracked")
            authorization = dict(
                complete=True, id=candidate_id,
                kind="explicit_human_scoped_commit_authorization",
                recorded_at=datetime.now().astimezone().isoformat(),
                thread_id=owner["thread_id"], human_message_id=MESSAGE_ID,
                human_message="继续",
                context=(
                    "Direct owning-user continuation after the assistant identified Y52/Y54 "
                    "as the two prepared candidates awaiting local commits of their seven "
                    "tested files each. The earlier confirmation question linked these "
                    "exact bundles. Only that existing local-commit scope is accepted."
                ),
                files=rows, tested_bundle=bundle_ref, starting_HEAD=head,
                permits=["Local Git add/commit of only the seven exact tested files."],
                excludes=[
                    "Unrelated files, other projects, remote Git push or scheduler changes.",
                    "Scientific PASS, admission or training permission inferred from this file.",
                    "Changed goals/gates/protocols, historical reopening or diagnostic weights.",
                ],
            )
            lease(token)
            tick.save(ROOT / auth_path, authorization)
            prior = {key: candidate[key] for key in (
                "blocker_kind", "blocker", "retry_after", "unblock_condition"
            ) if key in candidate}
            candidate.setdefault("blocker_history", []).append(dict(
                **prior, resolved_at=authorization["recorded_at"],
                resolved_by=ref(auth_path)))
            for key in ("blocker_kind", "blocker", "retry_after",
                        "unblock_condition", "required_server"):
                candidate.pop(key, None)
            candidate.update(
                stage="validation",
                next_action=(
                    "Owning human confirmed scoped local commit. Commit only the exact "
                    "seven tested files, revalidate current evidence/goal/commit and all "
                    "six gates before any existing canonical gated training continuation."
                ))
            candidate["evidence"].append(ref(auth_path))
            work.pop("design_cycle", None)
            tick.save(ROOT / "pipeline/research_work.json", work)
            result = dict(id=candidate_id, authorization=ref(auth_path),
                          stage=candidate["stage"], committed=False)
        else:
            authorization = tick.load(ROOT / auth_path)
            if (authorization["kind"] != "explicit_human_scoped_commit_authorization"
                    or authorization["human_message_id"] != MESSAGE_ID
                    or authorization["files"] != rows
                    or candidate["stage"] != "validation"
                    or head != authorization["starting_HEAD"]):
                raise ValueError("Scoped authorization or starting commit changed")
            before = unrelated_index(files)
            lease(token)
            git("add", "--", *files)
            for row in rows:
                if hashlib.sha256(git("show", ":" + row["path"])).hexdigest() != row["sha256"]:
                    raise ValueError("Staged bytes differ from tested bytes")
            git("commit", "--only", "-m",
                f"Add tested {candidate_id} independent candidate", "--", *files)
            commit = git("rev-parse", "HEAD").decode().strip()
            changes = git("diff-tree", "--no-commit-id", "--name-status", "-r", commit).decode().splitlines()
            if sorted(changes) != sorted("A\t" + path for path in files):
                raise ValueError("Commit contains unexpected paths or change kinds")
            for row in rows:
                if hashlib.sha256(git("show", commit + ":" + row["path"])).hexdigest() != row["sha256"]:
                    raise ValueError("Committed bytes differ from tested bytes")
            if unrelated_index(files) != before:
                raise ValueError("Unrelated index entries changed")
            binding = f"pipeline/analyses/{candidate_id}_commit_binding_{token}.json"
            if (ROOT / binding).exists():
                raise ValueError("A commit binding already exists; reconcile instead of replaying")
            tick.save(ROOT / binding, dict(
                complete=True, id=candidate_id,
                observed_at=datetime.now().astimezone().isoformat(),
                authorization=ref(auth_path), tested_bundle=bundle_ref,
                code_commit=commit, parent_commit=head, files=rows,
                exact_committed_tested_bytes=True, only_seven_additions=True,
                unrelated_index_unchanged=True, no_admission_or_launch_yet=True))
            candidate["evidence"].append(ref(binding))
            candidate.update(
                code_commit=commit,
                next_action="Exact tested files committed; revalidate all six gates and current goal, "
                            "then use existing canonical admission/dispatch only within prior permissions.")
            tick.save(ROOT / "pipeline/research_work.json", work)
            result = dict(id=candidate_id, code_commit=commit, binding=ref(binding),
                          committed_files=7, unrelated_index_unchanged=True)
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("inspect", "authorize", "commit"))
    parser.add_argument("--candidate-id", choices=tuple(SPECS), required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    apply(args.candidate_id, args.token, args.operation)


if __name__ == "__main__":
    main()
