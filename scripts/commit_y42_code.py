"""Commit exactly the seven tested Y42 files, preserving unrelated changes."""
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
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text(encoding="utf-8"))
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 420
bundle = json.loads((ROOT / ("pipeline/analyses/y42_source_bundle_" + args.token + ".json")).read_text(encoding="utf-8"))
paths = [item["path"] for item in bundle["files"]]
assert len(paths) == 7
for item in bundle["files"]:
    assert item["path"].startswith("AdaRES/ipdn/model_paper5/adares_y42_point_covariance/") or item["path"] == "AdaRES/configs_song/model_paper5/adares_y42_point_covariance_res.yaml"
    assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


before = git("rev-parse", "HEAD")
assert before == "79ce2de9582c369422facb27c3cbe85559bf227c"
assert not git("diff", "--cached", "--name-only"), "Preserve unrelated staged changes"
git("add", "--", *paths)
assert set(git("diff", "--cached", "--name-only").splitlines()) == set(paths)
output = git("commit", "--only", "-m", "Add Y42 point-covariance confidence experiment", "--", *paths)
after = git("rev-parse", "HEAD")
changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", after).splitlines()
assert set(changed) == set(paths) and not git("diff", "--cached", "--name-only")
result = dict(complete=True, id="AdaRES-Y42", previous_head=before, code_commit=after,
              changed_paths=changed, output=output, origin_push=False,
              observed_at_local=datetime.now().astimezone().isoformat())
path = ROOT / ("pipeline/analyses/y42_commit_" + args.token + ".json")
assert not path.exists()
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
