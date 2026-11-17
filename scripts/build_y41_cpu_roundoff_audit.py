"""Retain the failed finite control and add exact-binary arithmetic diagnosis."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "5e172bd47ee3450dbee928bdecaede7f"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
source = ROOT / "scripts/probe_y41_boundary_cpu.py"
text = ("P['readout_path']='/tmp/adares_y41_readout_stable_' + P['token'] + '.py'\n"
        + source.read_text())
target = ROOT / "scripts/probe_y41_boundary_cpu_nonempty_audit.py"
assert not target.exists()
compile(text, str(target), "exec")
target.write_text(text, encoding="utf-8", newline="\n")
print(json.dumps({"path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
