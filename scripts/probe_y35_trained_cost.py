"""Build a scoped adaptation of the preserved paired trained-cost protocol."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "e636ce91a12e45488093b07fa2fc092b"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)
original = ROOT / "scripts/probe_y33_trained_cost.py"
outer = ast.parse(original.read_text(encoding="utf-8"))
source = next(ast.literal_eval(n.value) for n in outer.body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "source" for t in n.targets))
bundle = (ROOT / "pipeline/analyses/y35_source_bundle_lf_a8d0c77d45d142c0a6c46b163b956253.json").read_bytes()
assert hashlib.sha256(bundle).hexdigest() == "7dc4bbfd4fbca46cbda803818c77b76e08b2b01b9201ed956bd7cabcffd3bc0a"
manifest = json.loads(bundle)["files"]
files = [(r["path"][7:], r["sha256"]) for r in manifest]


class Identity(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "source_manifest" for t in node.targets):
            return ast.copy_location(ast.parse("source_manifest=" + repr(files)).body[0], node)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            if node.value == "1":
                return ast.copy_location(ast.Constant(value="0"), node)
            value = node.value.replace("Y31", "Y35").replace("y31", "y35").replace("66677", "108331")
            value = value.replace("5f2489de78196e8d89c004e452a665c0946f2a94", "6725f9bd6a8f9b47addae6a4b8336d004f73f1ae")
            value = value.replace("GPU1", "GPU0")
            return ast.copy_location(ast.Constant(value=value), node)
        if node.value == 164763271:
            return ast.copy_location(ast.Constant(value=164958087), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
target = ROOT / "scripts/probe_y35_trained_cost_fixed.py"
assert not target.exists()
text = ast.unparse(tree) + "\n"
compile(text, str(target), "exec")
target.write_text(text, encoding="utf-8", newline="\n")
print(json.dumps(dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
