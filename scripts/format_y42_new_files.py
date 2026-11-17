"""Normalize only the new seven files to LF before their first native tests."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
old = json.loads((ROOT / ("pipeline/analyses/y42_local_bundle_" + args.token + ".json")).read_text())
assert len(old["files"]) == 7
records = []
for item in old["files"]:
    path = (ROOT / item["path"]).resolve()
    assert ROOT.resolve() in path.parents
    assert "/adares_y42_point_covariance/" in path.as_posix() or path.name == "adares_y42_point_covariance_res.yaml"
    before = path.read_bytes()
    assert hashlib.sha256(before).hexdigest() == item["sha256"]
    after = before.replace(b"\r\n", b"\n")
    if path.suffix == ".py":
        assert ast.dump(ast.parse(before.decode("utf-8"))) == ast.dump(ast.parse(after.decode("utf-8")))
    path.write_bytes(after)
    records.append({"path": item["path"], "sha256": hashlib.sha256(after).hexdigest()})
output = ROOT / ("pipeline/analyses/y42_local_bundle_lf_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps({"complete": True, "id": "AdaRES-Y42", "files": records,
                             "only_new_seven_lf_format": True,
                             "native_tests_not_yet_run": True}, indent=2), encoding="utf-8")
print(json.dumps({"path": str(output), "files": len(records)}))
