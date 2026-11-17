"""Safety and identity adaptation of the unchanged trained paired-cost body."""

import argparse
import ast
import copy
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
source = ROOT / "scripts/probe_y42_trained_cost_ff34.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
cut = next(i for i, node in enumerate(tree.body)
           if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
           and node.value.value == "Frozen serial inference and separate native stress with trained weights.")
body = tree.body[cut:]
assert isinstance(body[-1], ast.Expr) and body[-1].value.func.id == "check_originals"
body = body[:-1]
preamble = '''
import hashlib,json
from pathlib import Path
assert P["physical_gpu"]==1
terminal=json.loads(Path("/tmp/adares_TERMINAL_"+P["token"]+".json").read_text())
protected=terminal["source_hashes"]
def check_originals():
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    for row in terminal["checkpoints"]:
        digest=hashlib.sha256()
        with open(row["path"],"rb") as stream:
            for block in iter(lambda:stream.read(8388608),b""):digest.update(block)
        assert digest.hexdigest()==row["sha256"]
check_originals()
'''
manifest = []
for name, terminal_name in (("y43", "y43_terminal_7a40"), ("y44", "y44_terminal_392a")):
    replacements = {
        "/tmp/adares_y42_frozen_full_ff34_": "/tmp/adares_" + name + "_frozen_392a_",
        "/tmp/adares_y42_terminal_ff34_": "/tmp/adares_" + terminal_name + "_",
        "y42": name, "AdaRES-Y42": "AdaRES-" + name.upper(),
    }

    class Identity(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in replacements:
                return ast.copy_location(ast.Constant(value=replacements[node.value]), node)
            return node

    adapted = Identity().visit(ast.Module(body=copy.deepcopy(body), type_ignores=[]))
    prefix = ast.parse(preamble.replace("TERMINAL", terminal_name))
    generated = ast.fix_missing_locations(ast.Module(
        body=prefix.body + adapted.body + ast.parse("check_originals()").body, type_ignores=[]))
    path = ROOT / ("scripts/probe_" + name + "_trained_cost_392a.py")
    assert not path.exists()
    path.write_text(ast.unparse(generated) + "\n", encoding="utf-8")
    manifest.append(dict(path=path.relative_to(ROOT).as_posix(),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         changes="Identity and terminal safety only; trained inference/stress body unchanged."))
output = ROOT / ("pipeline/analyses/cost_adapter_392a_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps(dict(complete=True, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                 manifest=manifest), indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
