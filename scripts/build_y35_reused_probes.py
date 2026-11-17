"""Reuse frozen native protocols by AST-transforming candidate identifiers only."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((root/"pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)

class CandidateOnly(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "token_readout":
            node.attr = "language_readout"
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("Y33", "Y35").replace(
                "y33_token_attention", "y35_language_evidence").replace(
                "token_readout", "language_readout")
        return node

rows = []
for task in ("physical_native", "resources"):
    original = root/("scripts/probe_y33_"+task+".py")
    target = root/("scripts/probe_y35_"+task+".py")
    assert not target.exists()
    data = original.read_bytes()
    tree = CandidateOnly().visit(ast.parse(data.decode("utf-8")))
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)+"\n"
    compile(source, str(target), "exec")
    target.write_text(source, encoding="utf-8")
    rows.append(dict(original=str(original.relative_to(root)),
                     original_sha256=hashlib.sha256(data).hexdigest(),
                     target=str(target.relative_to(root)),
                     sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
out = root/("pipeline/analyses/y35_reused_probe_sources_"+args.token+".json")
assert not out.exists()
out.write_text(json.dumps(dict(
    complete=True, changes="AST string candidate IDs/package/member names only",
    preserved="native train entry/two steps/modelAdamWPolyLR resume/dataset/stress/timing",
    files=rows), indent=2), encoding="utf-8")
print(json.dumps(dict(path=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
