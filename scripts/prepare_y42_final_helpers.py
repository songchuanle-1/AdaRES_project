"""Generate narrowly adapted existing commit/launch helpers, retaining originals."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "062ae665ba824deba180a65efb4c44cf"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text(encoding="utf-8"))
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])

commit = (ROOT / "scripts/commit_y39_code.py").read_text(encoding="utf-8")
commit = commit.replace("Y39", "Y42").replace("y39", "y42")
commit = commit.replace("point_distribution", "point_covariance")
commit = commit.replace("point-distribution", "point-covariance")
commit = commit.replace("67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773", "79ce2de9582c369422facb27c3cbe85559bf227c")

launch = (ROOT / "scripts/capture_finalize_y41_launch.py").read_text(encoding="utf-8")
launch = launch.replace("5e172bd47ee3450dbee928bdecaede7f", TOKEN)
launch = launch.replace("Y41", "Y42").replace("y41", "y42")
launch = launch.replace("Y39", "Y41").replace("y39", "y41")
launch = launch.replace("Y37", "Y39").replace("y37", "y39")
launch = launch.replace("102267", "116906")
launch = launch.replace("adares_y41_point_distribution_res.yaml", "adares_y41_boundary_probability_res.yaml")
launch = launch.replace('str(task["gpu"]) == "0"', 'str(task["gpu"]) == "1"')
launch = launch.replace("argv[argv.index(b'--gpu_ids')+1]==b'0'", "argv[argv.index(b'--gpu_ids')+1]==b'1'")
launch = launch.replace("old[old.index(b'--gpu_ids')+1]==b'1'", "old[old.index(b'--gpu_ids')+1]==b'0'")
launch = launch.replace("'nvidia-smi','-i','0'", "'nvidia-smi','-i','1'")
launch = launch.replace("actual_gpu=0", "actual_gpu=1").replace("gpu0_device", "gpu1_device")
launch = launch.replace("preserved_y41_gpu=1", "preserved_y41_gpu=0")
launch = launch.replace("Y42 GPU0 and Y41 GPU1", "Y42 GPU1 and Y41 GPU0")
launch = launch.replace('"pipeline/decisions/20261011-162854_next_batch.md",', "DECISION,")
launch = launch.replace('"pipeline/analyses/y42_parent_semantics_" + TOKEN', '"pipeline/analyses/y42_parent_semantics_v3_" + TOKEN')
launch = launch.replace('TOKEN = "' + TOKEN + '"', 'import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument("--decision", required=True)\nargs = parser.parse_args()\nDECISION = args.decision\nassert DECISION.startswith("pipeline/decisions/")\nTOKEN = "' + TOKEN + '"')
launch = launch.replace(".read_text())", '.read_text(encoding="utf-8"))')
commit = commit.replace(".read_text())", '.read_text(encoding="utf-8"))')

outputs = {"scripts/commit_y42_code.py": commit, "scripts/capture_finalize_y42_launch.py": launch}
for path, text in outputs.items():
    target = ROOT / path
    assert not target.exists()
    compile(text, path, "exec")
    target.write_text(text, encoding="utf-8", newline="\n")
manifest = ROOT / ("pipeline/analyses/y42_final_helper_binding_" + TOKEN + ".json")
assert not manifest.exists()
manifest.write_text(json.dumps({
    "complete": True, "token": TOKEN,
    "files": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in outputs],
    "scope": "Existing exact-seven-file commit and independent increasing-step launch helpers, Y42 GPU1 and preserved Y41 GPU0. No model/protocol edits.",
}, indent=2), encoding="utf-8")
print(json.dumps({"generated": list(outputs)}))
