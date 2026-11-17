"""Mechanical identity changes to fixed native protocols, preserving originals."""

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


class Identity(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "edge_readout":
            node.attr = "pair_readout"
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y36", "Y37").replace(
                "y36_local_adjacency", "y37_pair_distribution").replace(
                "edge_readout", "pair_readout").replace("edge_gradients", "pair_gradients")
            value = value.replace("wrong_edges", "wrong_pairs")
            value = value.replace("local-adjacency", "global-pair-distribution")
            value = value.replace("GPU0 query", "GPU1 query")
            return ast.copy_location(ast.Constant(value=value), node)
        return node


rows = []
for task, original_task in (
    ("native_controls", "native_controls_v2"), ("physical_native", "physical_native"),
    ("resources", "resources"), ("initial_loss_investigation", "initial_loss_investigation"),
):
    original = ROOT / ("scripts/probe_y36_" + original_task + ".py")
    target = ROOT / ("scripts/probe_y37_" + task + ".py")
    assert not target.exists()
    data = original.read_bytes()
    tree = ast.fix_missing_locations(Identity().visit(ast.parse(data.decode("utf-8"))))
    source = ast.unparse(tree) + "\n"
    compile(source, str(target), "exec")
    target.write_text(source, encoding="utf-8", newline="\n")
    rows.append(dict(original=str(original.relative_to(ROOT)),
                     original_sha256=hashlib.sha256(data).hexdigest(),
                     target=str(target.relative_to(ROOT)),
                     sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
output = ROOT / ("pipeline/analyses/y37_probe_protocols_" + TOKEN + ".json")
assert not output.exists()
output.write_text(json.dumps(dict(complete=True, files=rows,
    preserved="Native entry, TRAIN8/init4 controls, two steps, AdamW/PolyLR resume, resource manifest.",
    changed="Only candidate identity, final-readout attribute and wrong-coordinate naming.",
    limitations="Historical inherited prose is not a new semantic proof; actual new path has no visual feature gradient."),
    indent=2), encoding="utf-8")
print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
