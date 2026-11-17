"""Mechanically specialize the existing guarded Y45 operator for tested Y48."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
token = sys.argv[sys.argv.index("--token") + 1]
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
source = ROOT / "scripts/operate_y45_d179.py"
tree = ast.parse(source.read_text())
for node in ast.walk(tree):
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        continue
    value = node.value
    value = value.replace("392a5a9f5fe242d7a3109a50e6060fda", token)
    value = value.replace("adares_y45_metric_decomposition", "adares_y48_mask_equivalence")
    value = value.replace("y45_source_bundle_392a", "y48_source_bundle_601b")
    value = value.replace("exp_design_Y45_392a", "exp_design_Y48_601b")
    value = value.replace("y45_", "y48_").replace("Y45", "Y48")
    value = value.replace("_d179", "_601b")
    value = value.replace("c65ad2fcba4ef10b71ea6ae58ed01109568bd0cd",
                          "338a3d7a5b21fa463d116e566b482f0733fde9c2")
    value = value.replace("Add Y48 native metric error decomposition",
                          "Add Y48 final mask equivalence targets")
    value = value.replace("Y48 point-confusion decomposition confidence",
                          "Y48 final-mask equivalent targets")
    if value.startswith("Native local objective conflicts, wrong-control learnability"):
        value = "Original one-to-many matching, uncertainty and all harness failures retained. Target closure is not efficacy/necessity/novelty/no-harm/Phase2 evidence; no diagnostic weights deployed."
    node.value = value
destination = ROOT / "scripts/operate_y48_601b.py"
text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
compile(text, str(destination), "exec")
with destination.open("x", encoding="utf-8") as stream:
    stream.write(text)
print(json.dumps(dict(
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    generated_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
    path=str(destination), scope="Same commit/admit/verify controls, only candidate bindings updated.")))
