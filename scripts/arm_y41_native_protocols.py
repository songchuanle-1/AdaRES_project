"""Bind existing fixed protocols to exact Y41 bytes and preserve Y39."""

from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "5e172bd47ee3450dbee928bdecaede7f"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
source = ROOT / "scripts/probe_y37_initial_loss_investigation.py"
text = source.read_text(encoding="utf-8").replace("Y37", "Y41").replace(
    "adares_y37_pair_distribution", "adares_y41_boundary_probability"
).replace("pair_readout", "boundary_readout").replace("new8", "new4")
tree = ast.parse(text)
for node in ast.walk(tree):
    if isinstance(node, ast.Assert):
        for part in ast.walk(node.test):
            if isinstance(part, ast.Constant) and part.value == 8:
                part.value = 4
target = ROOT / "scripts/probe_y41_initial_loss.py"
initial_text = ast.unparse(tree) + "\n"
if target.exists():
    assert target.read_text(encoding="utf-8") == initial_text
else:
    target.write_text(initial_text, encoding="utf-8", newline="\n")
bundle = json.loads((ROOT / ("pipeline/analyses/y41_source_bundle_" + TOKEN + ".json")).read_text())
assert bundle["complete"] and len(bundle["files"]) == 7
prefix = (
    "import hashlib\nfrom pathlib import Path\n"
    "manifest=" + repr(bundle["files"]) + "\n"
    "root=Path('/bak_home/songcl/Project/AdaRES_project')\n"
    "assert all(hashlib.sha256((root / row['path']).read_bytes()).hexdigest()==row['sha256'] for row in manifest)\n"
    "preserved_cmd=Path('/proc/102267/cmdline').read_bytes().split(b'\\0')\n"
    "assert b'configs_song/model_paper5/adares_y39_point_distribution_res.yaml' in preserved_cmd\n"
    "assert preserved_cmd[preserved_cmd.index(b'--gpu_ids')+1]==b'1'\n"
    "assert P['physical_gpu']==0\n"
)
suffix = (
    "\nassert Path('/proc/102267/cmdline').read_bytes().split(b'\\0')==preserved_cmd\n"
    "assert all(hashlib.sha256((root / row['path']).read_bytes()).hexdigest()==row['sha256'] for row in manifest)\n"
)
for name in ("native_controls", "physical_native", "resources", "initial_loss"):
    source = ROOT / ("scripts/probe_y41_" + name + ".py")
    target = ROOT / ("scripts/probe_y41_" + name + "_bound.py")
    assert not target.exists()
    text = prefix + source.read_text(encoding="utf-8") + suffix
    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
