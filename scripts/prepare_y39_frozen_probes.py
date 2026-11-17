"""Bind existing frozen scientific protocols to original Y39 and live Y41."""

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
terminal = json.loads(
    (ROOT / ("pipeline/analyses/y39_terminal_current_" + args.token + ".json")).read_text()
)
best = next(row for row in terminal["checkpoints"] if row["path"].endswith("/best.pth"))
for source_name, target_name in (
    ("probe_y37_frozen_full.py", "probe_y39_frozen_full_current.py"),
    ("probe_y37_trained_cost_current.py", "probe_y39_trained_cost_current.py"),
):
    text = (ROOT / "scripts" / source_name).read_text()
    if source_name == "probe_y37_frozen_full.py":
        text = text.replace("GPU0", "GPU1").replace('P["physical_gpu"] == 0', 'P["physical_gpu"] == 1')
        text = text.replace('"physical_gpu": 0', '"physical_gpu": 1')
        text = text.replace('"--gpu-id", "0"', '"--gpu-id", "1"')
        text = text.replace("20261009_235404/best.pth", "20261010_074811/best.pth")
        text = text.replace(
            "6993fc5dc67f8a7d5c2f5a1d3209a9e59c2b09b5952b64d7b6d8acb1aa1702ca",
            best["sha256"],
        )
    text = text.replace("102267", "116906")
    text = text.replace("adares_y39_point_distribution_res.yaml", "adares_y41_boundary_probability_res.yaml")
    text = text.replace("cmd.index(b'--gpu_ids')+1] == b'1'", "cmd.index(b'--gpu_ids')+1] == b'0'")
    text = text.replace('cmd.index(b"--gpu_ids") + 1] == b"1"', 'cmd.index(b"--gpu_ids") + 1] == b"0"')
    text = text.replace('"gpu": 1, "config": "adares_y39_point_distribution_res"',
                        '"gpu": 0, "config": "adares_y41_boundary_probability_res"')
    text = text.replace("AdaRES-Y37", "AdaRES-Y39")
    text = text.replace("adares_y37_pair_distribution", "adares_y39_point_distribution")
    text = text.replace("y37_frozen_full_retry_", "y39_frozen_full_current_")
    text = text.replace("y37_terminal_current_", "y39_terminal_current_")
    text = text.replace('("y37",', '("y39",').replace('["y37"]', '["y39"]')
    text = text.replace("['y37']", "['y39']")
    compile(text, target_name, "exec")
    target = ROOT / "scripts" / target_name
    assert not target.exists()
    target.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"path": target_name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
