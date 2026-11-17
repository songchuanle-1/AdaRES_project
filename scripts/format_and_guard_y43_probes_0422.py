"""LF-format new sources before tests; protect the original GPU1 job in each probe."""

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
bundle = json.loads((ROOT / ("pipeline/analyses/y43_local_bundle_" + args.token + ".json")).read_text())
records = []
for item in bundle["files"]:
    path = (ROOT / item["path"]).resolve()
    assert ROOT.resolve() in path.parents
    assert "/adares_y43_point_asymmetry/" in path.as_posix() or path.name == "adares_y43_point_asymmetry_res.yaml"
    before = path.read_bytes()
    assert hashlib.sha256(before).hexdigest() == item["sha256"]
    after = before.replace(b"\r\n", b"\n")
    if path.suffix == ".py":
        assert ast.dump(ast.parse(before.decode())) == ast.dump(ast.parse(after.decode()))
    path.write_bytes(after)
    records.append({"path": item["path"], "sha256": hashlib.sha256(after).hexdigest()})

preamble = '''
from pathlib import Path
import hashlib
assert P["physical_gpu"] == 0
_y42_command = Path("/proc/91215/cmdline").read_bytes().split(b"\\0")
assert b"tools/train.py" in _y42_command
assert b"configs_song/model_paper5/adares_y42_point_covariance_res.yaml" in _y42_command
assert _y42_command[_y42_command.index(b"--gpu_ids")+1] == b"1"
_y42_folder = Path("ipdn/model_paper5/adares_y42_point_covariance")
_y42_sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in _y42_folder.glob("*.py")}
assert len(_y42_sources) == 6
'''
suffix = '''
assert Path("/proc/91215/cmdline").read_bytes().split(b"\\0") == _y42_command
assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in _y42_folder.glob("*.py")} == _y42_sources
'''
probes = []
for name in ("probe_y43_native_controls_0422.py", "probe_y43_initial_loss_0422.py",
             "probe_y43_physical_native_0422.py", "probe_y43_resources_0422.py"):
    path = ROOT / "scripts" / name
    before = path.read_bytes()
    text = preamble + "\n" + before.decode() + "\n" + suffix
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8", newline="\n")
    probes.append({"path": path.relative_to(ROOT).as_posix(),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "previous_sha256": hashlib.sha256(before).hexdigest()})
output = ROOT / ("pipeline/analyses/y43_lf_and_preservation_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps({"complete": True, "files": records, "guarded_probes": probes,
    "only_new_seven_lf_format": True, "native_tests_not_yet_run": True}, indent=2), encoding="utf-8")
print(json.dumps({"files": len(records), "guarded_probes": len(probes)}))
