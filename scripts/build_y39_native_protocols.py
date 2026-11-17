"""Mechanically specialize existing fixed native and resource protocols."""
import argparse
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
for source_name in ("probe_y37_physical_native.py", "probe_y37_resources.py"):
    source = ROOT / "scripts" / source_name
    text = source.read_text(encoding="utf-8")
    text = text.replace("Y37", "Y39").replace("adares_y37_pair_distribution", "adares_y39_point_distribution")
    text = text.replace("pair_readout", "point_readout")
    if source_name.endswith("physical_native.py"):
        anchor = "        original = model.dec(*args)"
        assert text.count(anchor) == 1
        text = text.replace(anchor, "        original = model.dec(*args[:-1])")
    text = "assert __import__('pathlib').Path('.codex_source_commit').read_text().strip()=='67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773'\n" + text
    target = ROOT / "scripts" / source_name.replace("y37", "y39")
    assert not target.exists()
    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps(dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
