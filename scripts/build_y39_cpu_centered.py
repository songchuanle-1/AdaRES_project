"""Build a mechanical centered-moment derivative; preserve the failed probe."""

import argparse
import ast
import copy
from datetime import datetime, timezone
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
source = Path(__file__).with_name("probe_y39_pooling_cpu.py")
target = Path(__file__).with_name("probe_y39_pooling_cpu_v2.py")
assert not target.exists()


class CenteredMoment(ast.NodeTransformer):
    replacements = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "var":
            assert len(node.args) == 1 and node.args[0].value == 1
            assert len(node.keywords) == 1 and node.keywords[0].arg == "unbiased"
            assert node.keywords[0].value.value is False
            expression = ast.parse(
                "((VALUE - VALUE.mean(1, keepdim=True)) ** 2).mean(1)", mode="eval").body

            class Bind(ast.NodeTransformer):
                def visit_Name(self, name):
                    return copy.deepcopy(node.func.value) if name.id == "VALUE" else name

            self.replacements += 1
            return ast.copy_location(Bind().visit(expression), node)
        return node


transform = CenteredMoment()
tree = ast.fix_missing_locations(transform.visit(ast.parse(source.read_bytes())))
assert transform.replacements == 3
target.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
print(str(target))
