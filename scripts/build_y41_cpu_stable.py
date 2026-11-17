"""Embed the pre-test stable direct-edge arithmetic without changing controls."""

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
readout = ROOT / "scripts/y41_boundary_readout.py"
text = ("from pathlib import Path\n"
        "P['readout_path']='/tmp/adares_y41_readout_stable_' + P['token'] + '.py'\n"
        "assert not Path(P['readout_path']).exists()\n"
        "Path(P['readout_path']).write_text(" + repr(readout.read_text()) +
        ", encoding='utf-8')\n" + source.read_text())
target = ROOT / "scripts/probe_y41_boundary_cpu_stable.py"
assert not target.exists()
compile(text, str(target), "exec")
target.write_text(text, encoding="utf-8", newline="\n")
print(json.dumps({"path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "readout_sha256": hashlib.sha256(readout.read_bytes()).hexdigest()}))
