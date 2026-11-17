"""Frozen native train diagnostics on the authorized GPU under an owned lease."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "d20c71e418354a6b818d9f37ab72fc7b"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN
out = ROOT / "pipeline/analyses/y22_y23_diagnosis_d20c71e4.json"
log = out.with_suffix(".log")
assert not out.exists() and not log.exists()
source = ROOT / "scripts/y22_y23_remote_d20c71e4.py"
program = source.read_text()
compile(program, str(source), "exec")
response = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
     "timeout --signal=TERM --kill-after=10s 360 /home/songcl/.conda/envs/adares_project/bin/python -S -"],
    input=program.encode(), capture_output=True, timeout=390)
log.write_bytes(response.stdout + b"\nSTDERR:\n" + response.stderr)
if response.returncode:
    raise RuntimeError("Failed diagnosis; preserved " + str(log))
subprocess.run(["scp", "-q", "songcl@192.168.1.163:/tmp/adares_y22_y23_d20c71e4.json",
                str(out)], check=True, timeout=60)
result = json.loads(out.read_text())
result["local_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result["summary"], indent=2))
