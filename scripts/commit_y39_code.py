"""Commit exactly the seven tested Y39 files, preserving unrelated changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 420
bundle = json.loads((ROOT / ("pipeline/analyses/y39_source_bundle_" + args.token + ".json")).read_text())
paths = [item["path"] for item in bundle["files"]]
assert len(paths) == 7
for item in bundle["files"]:
    assert item["path"].startswith("AdaRES/ipdn/model_paper5/adares_y39_point_distribution/") or item["path"] == "AdaRES/configs_song/model_paper5/adares_y39_point_distribution_res.yaml"
    assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


before = git("rev-parse", "HEAD")
assert before == "67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773"
assert not git("diff", "--cached", "--name-only"), "Preserve unrelated staged changes"
git("add", "--", *paths)
assert set(git("diff", "--cached", "--name-only").splitlines()) == set(paths)
output = git("commit", "--only", "-m", "Add Y39 point-distribution confidence experiment", "--", *paths)
after = git("rev-parse", "HEAD")
changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", after).splitlines()
assert set(changed) == set(paths) and not git("diff", "--cached", "--name-only")
result = dict(complete=True, id="AdaRES-Y39", previous_head=before, code_commit=after,
              changed_paths=changed, output=output, origin_push=False,
              observed_at_local=datetime.now().astimezone().isoformat())
path = ROOT / ("pipeline/analyses/y39_commit_" + args.token + ".json")
assert not path.exists()
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
