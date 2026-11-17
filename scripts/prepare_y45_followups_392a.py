"""Identity-only adaptation of existing native resume and resource protocols."""

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
records = []
for source_name, output_name, marker in (
        ("probe_y44_physical_native_ff34.py", "probe_y45_physical_native_392a.py",
         "Physical Y44 native two-step entry, checkpoint resume and cached controls."),
        ("probe_y43_resources_0422.py", "probe_y45_resources_392a.py",
         "Matched serial native Y42/W2 training and inference resource probes.")):
    source = ROOT / "scripts" / source_name
    tree = ast.parse(source.read_text(encoding="utf-8"))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Expr)
                 and isinstance(n.value, ast.Constant) and n.value.value == marker)
    body = tree.body[start:]
    if output_name == "probe_y45_physical_native_392a.py":
        assert isinstance(body[-1], ast.Expr) and body[-1].value.func.id == "check_originals"
        body = body[:-1]
    else:
        # The final two historical safety assertions reference a departed peer.
        assert all(isinstance(n, ast.Assert) for n in body[-2:])
        body = body[:-2]

    class Identity(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if node.attr == "point_readout":
                node.attr = "metric_readout"
            return node

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                value = node.value.replace("adares_y44_coordinate_feature", "adares_y45_metric_decomposition")
                value = value.replace("adares_y43_point_asymmetry", "adares_y45_metric_decomposition")
                value = value.replace("point_readout", "metric_readout")
                value = value.replace("Y44", "Y45").replace("Y42", "Y45")
                return ast.copy_location(ast.Constant(value=value), node)
            if output_name == "probe_y45_physical_native_392a.py" and node.value == 8:
                return ast.copy_location(ast.Constant(value=4), node)
            return node

    adapted = Identity().visit(ast.Module(body=body, type_ignores=[]))
    if output_name == "probe_y45_physical_native_392a.py":
        for n in ast.walk(adapted):
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "h":
                n.targets = [ast.parse("model.dec.metric_readout.enabled", mode="eval").body]
                n.targets[0].ctx = ast.Store()
                n.value = ast.Constant(False)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if isinstance(n.func.value, ast.Name) and n.func.value.id == "h" and n.func.attr == "remove":
                    n.func = ast.parse("setattr", mode="eval").body
                    n.args = [ast.parse("model.dec.metric_readout", mode="eval").body,
                              ast.Constant("enabled"), ast.Constant(True)]
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "dec":
                if n.args and isinstance(n.args[0], ast.Starred) and isinstance(n.args[0].value, ast.Subscript):
                    sub = n.args[0].value
                    assert sub.value.id == "args"
                    n.args[0].value = ast.Name(id="args", ctx=ast.Load())
    preamble = ast.parse('''
import hashlib,json
from pathlib import Path
assert P["physical_gpu"]==1
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
           for folder in ("adares_w2_pointmass_competition","adares_y43_point_asymmetry","adares_y44_coordinate_feature","adares_y45_metric_decomposition")
           for p in (Path("ipdn/model_paper5")/folder).glob("*.py")}
def check_originals():
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
check_originals()
''').body
    full = ast.fix_missing_locations(ast.Module(
        body=preamble + adapted.body + ast.parse("check_originals()").body, type_ignores=[]))
    output = ROOT / "scripts" / output_name
    assert not output.exists()
    output.write_text(ast.unparse(full) + "\n", encoding="utf-8")
    records.append(dict(path=output.relative_to(ROOT).as_posix(),
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                        original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        changes="Candidate identity, four new tensors, disabled categorical path and terminal-source safety; protocol unchanged."))
proof = ROOT / ("pipeline/analyses/y45_followup_adapters_392a_" + args.token + ".json")
assert not proof.exists()
proof.write_text(json.dumps(dict(complete=True, files=records), indent=2), encoding="utf-8")
print(json.dumps(records, indent=2))
