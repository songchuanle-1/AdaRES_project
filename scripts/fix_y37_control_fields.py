"""Preserve the original generated protocol; correct output field spelling only."""

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
source = ROOT / "scripts/probe_y37_native_controls.py"
target = ROOT / "scripts/probe_y37_native_controls_v2.py"
assert not target.exists()


class Fields(ast.NodeTransformer):
    def visit_keyword(self, node):
        if node.arg == "wrong_edges_max":
            node.arg = "wrong_pairs_max"
        elif node.arg == "wrong_edges_min":
            node.arg = "wrong_pairs_min"
        return self.generic_visit(node)


text = ast.unparse(ast.fix_missing_locations(Fields().visit(ast.parse(source.read_text(encoding="utf-8")))))+"\n"
compile(text, str(target), "exec")
target.write_text(text, encoding="utf-8", newline="\n")
print(json.dumps(dict(original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      target=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                      changed="Two output-keyword names only; original was never executed.")))
