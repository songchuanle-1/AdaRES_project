"""Mechanically bind the frozen serial cost protocol to trained Y36."""

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
source_path = ROOT / "scripts/probe_y35_trained_cost_fixed.py"
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "e3859b80ef211cf7d0d0d15a1cfc4441288935a9098a2b71256602ebf189e035"
bundle = (ROOT / "pipeline/analyses/y36_source_bundle_lf_8fd5e469dee241da8f21da7c870c0d3f.json").read_bytes()
assert hashlib.sha256(bundle).hexdigest() == "96a2e9621d96847b8621438813dc3a1cbc07e3ac883a61f163a526db04cf5f39"
manifest = [(row["path"][7:], row["sha256"]) for row in json.loads(bundle)["files"]]


class Identity(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "source_manifest"
               for target in node.targets):
            return ast.copy_location(ast.parse("source_manifest=" + repr(manifest)).body[0], node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "check_output"
                and node.args and isinstance(node.args[0], ast.List)):
            items = node.args[0].elts
            if [getattr(item, "value", None) for item in items[:3]] == ["nvidia-smi", "-i", "0"]:
                items[2] = ast.Constant(value="1")
        return node

    def visit_Constant(self, node):
        if node.value == 164958087:
            return ast.copy_location(ast.Constant(value=165089671), node)
        if isinstance(node.value, str):
            value = node.value.replace("Y35", "Y36").replace("y35", "y36")
            value = value.replace("108331", "100018").replace("GPU0", "GPU1")
            value = value.replace(
                "6725f9bd6a8f9b47addae6a4b8336d004f73f1ae",
                "67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773",
            )
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
guard = ast.parse(
    "old = Path('/proc/20348/cmdline').read_bytes().split(b'\\0')\n"
    "assert b'tools/train.py' in old and "
    "b'configs_song/model_paper5/adares_y37_pair_distribution_res.yaml' in old\n"
    "assert old[old.index(b'--gpu_ids')+1] == b'0'\n"
)
index = next(i for i, node in enumerate(tree.body)
             if isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == "captures"
                 for target in node.targets))
tree.body[index:index] = guard.body
tree = ast.fix_missing_locations(tree)
output = ROOT / "scripts/probe_y36_trained_cost_fixed.py"
assert not output.exists()
output.write_text(ast.unparse(tree) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "sampling_and_cost_protocol": "Unchanged; identity/source/GPU/parameter binding only"}))
