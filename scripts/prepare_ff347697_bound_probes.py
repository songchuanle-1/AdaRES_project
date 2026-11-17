"""Bind existing unchanged protocols to Y42/Y44 and protect original Y43."""
import argparse
import ast
from datetime import datetime, timezone
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
guard = '''
from pathlib import Path
import hashlib
assert P["physical_gpu"] == 1
assert Path(".codex_source_commit").read_text().strip() == "6fc7d4e46b6c26a11fede7f12d4e79b3471cc1cc"
_terminal = json.loads(Path("/tmp/adares_y42_terminal_ff34_" + P["token"] + ".json").read_text())
_y43_command = Path("/proc/13155/cmdline").read_bytes()
assert b"configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml" in _y43_command
assert _y43_command.split(b"\\0")[_y43_command.split(b"\\0").index(b"--gpu_ids")+1] == b"0"
_protected = dict(_terminal["source_hashes"], **_terminal["protected_source_hashes"])
def check_originals():
    assert Path("/proc/13155/cmdline").read_bytes() == _y43_command
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in _protected.items())
check_originals()
'''


class Bind(ast.NodeTransformer):
    def __init__(self, replacements):
        self.replacements = replacements

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value
            for old, new in self.replacements.items():
                value = value.replace(old, new)
            return ast.copy_location(ast.Constant(value=value), node)
        return node


def save(tree, name):
    ast.fix_missing_locations(tree)
    compile(tree, name, "exec")
    path = ROOT / "scripts" / name
    assert not path.exists()
    path.write_text(ast.unparse(tree) + "\n", encoding="utf-8", newline="\n")


for source, name in (
    ("probe_y43_physical_native_0422.py", "probe_y44_physical_native_ff34.py"),
    ("probe_y43_resources_0422.py", "probe_y44_resources_ff34.py"),
    ("probe_y43_initial_loss_0422.py", "probe_y44_initial_loss_ff34.py"),
):
    tree = ast.parse((ROOT / "scripts" / source).read_text())
    start = next(i for i, n in enumerate(tree.body)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                 and isinstance(n.value.value, str))
    assert all(isinstance(n, ast.Assert) for n in tree.body[-2:])
    tree.body = tree.body[start:-2]
    tree = Bind({
        "adares_y43_point_asymmetry": "adares_y44_coordinate_feature",
        "AdaRES-Y43": "AdaRES-Y44", "Y42": "Y44",
    }).visit(tree)
    assert not any(isinstance(n, ast.Name) and n.id.startswith("_y42") for n in ast.walk(tree))
    tree.body = ast.parse(guard).body + tree.body + ast.parse("check_originals()").body
    save(tree, name)

for source, name in (
    ("analyze_y41_frozen_0422.py", "analyze_y42_frozen_ff34.py"),
    ("analyze_y41_object_groups_0422.py", "analyze_y42_objects_ff34.py"),
):
    tree = ast.parse((ROOT / "scripts" / source).read_text())
    tree = Bind({
        "y41_frozen_full_0422_": "y42_frozen_full_ff34_",
        "y41_clustered_comparison_": "y42_clustered_comparison_",
        "y41_object_groups_": "y42_object_groups_",
        "y41": "y42", "AdaRES-Y41": "AdaRES-Y42",
    }).visit(tree)
    save(tree, name)

tree = ast.parse((ROOT / "scripts/probe_y41_trained_cost_0422_v2.py").read_text())
removed = []
kept = []
for node in tree.body:
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    if "cmd" in names:
        removed.append(node)
    else:
        kept.append(node)
assert len(removed) == 4
tree.body = kept
tree = Bind({
    "y41_frozen_full_0422_": "y42_frozen_full_ff34_",
    "y41_terminal_0422_": "y42_terminal_ff34_",
    "y41": "y42", "AdaRES-Y41": "AdaRES-Y42",
}).visit(tree)
tree.body = ast.parse(guard).body + tree.body + ast.parse("check_originals()").body
save(tree, "probe_y42_trained_cost_ff34.py")
print(json.dumps({"complete": True, "created_protocols": 6,
                  "scope": "Only explicit ID/config/protected-job/source bindings; no scientific thresholds changed."}))
