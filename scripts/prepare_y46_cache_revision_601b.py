"""Mechanically bind an exact cache replacement to unchanged frozen probes."""

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
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 900
replacement_path = ROOT / "scripts/y46_component_cache_impl_601b.py"
replacement = ast.parse(replacement_path.read_text()).body[-1]
assert isinstance(replacement, ast.FunctionDef) and replacement.name == "component_inputs"
bindings = [
    ("probe_y46_component_cpu_d179_v2.py", "probe_y46_cached_cpu_601b.py",
     "713ad86a68f51142776be20a50295597aa07e06b7fff6d146797f91f6c5a5ecf"),
    ("probe_y46_native_gate_3c6a.py", "probe_y46_cached_native_601b.py",
     "78a9a9dd13134f23c2062e997c02303b776d552f845068ea9a4998e1a11036ad"),
]
outputs = []
for old_name, new_name, expected in bindings:
    source = ROOT / "scripts" / old_name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    tree = ast.parse(source.read_text())
    rewritten = []
    changed = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "component_inputs":
            original = ast.parse(ast.unparse(node)).body[0]
            original.name = "reference_component_inputs"
            rewritten.extend([
                original, ast.parse("component_cache_checks = []").body[0],
                ast.parse(ast.unparse(replacement)).body[0]])
            changed.append("component_inputs")
        elif (old_name == "probe_y46_component_cpu_d179_v2.py"
              and isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "capture_path"
                      for target in node.targets)):
            rewritten.append(ast.parse(
                "capture_path = Path('/tmp/adares_y46_native_capture_d179_"
                "d179d824a64943e0b06bfa90de0caf52.json')").body[0])
            changed.append("capture_path_to_unchanged_hashed_fixture")
        else:
            rewritten.append(node)
    assert "component_inputs" in changed
    if old_name == "probe_y46_component_cpu_d179_v2.py":
        final = rewritten.pop()
        assert isinstance(final, ast.Expr)
        rewritten.extend(ast.parse(
            "result['implementation_revision'] = 1\n"
            "result['cache_checks'] = component_cache_checks\n"
            "result['cache_all_exact'] = all(v['exact_rows_weights_groups'] for v in component_cache_checks)\n"
        ).body)
        rewritten.append(final)
    tree.body = rewritten
    ast.fix_missing_locations(tree)
    text = ast.unparse(tree) + "\n"
    compile(text, new_name, "exec")
    destination = ROOT / "scripts" / new_name
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
    outputs.append(dict(
        path=destination.relative_to(ROOT).as_posix(),
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        original_source=dict(path=source.relative_to(ROOT).as_posix(), sha256=expected),
        changed_AST_nodes=changed))
result = dict(
    complete=True, id="AdaRES-Y46", implementation_revision=1,
    preregistered_at=datetime.now().astimezone().isoformat(),
    failure_scope="implementation",
    reason="Native candidate costs 3.49-18.66 seconds on seven warmed scenes versus 0.21-0.31 seconds parent; mechanism remains unfalsified.",
    change="Only memoize identical selected point sets within one component_inputs call; no cross-expression/forward cache.",
    prediction="Rows/weights/group ordering exactly equal original on all frozen CPU controls; native new gradients remain finite/nonzero; measure cost without fitted success threshold.",
    failure_evidence=dict(
        path="pipeline/analyses/y46_native_gate_601b_" + args.token + ".json",
        sha256=hashlib.sha256((ROOT / ("pipeline/analyses/y46_native_gate_601b_" + args.token + ".json")).read_bytes()).hexdigest()),
    outputs=outputs, result="pending",
    limitations="No model, graph, target, dimensions, thresholds, weights, training protocol or historical source changed; still no production admission.")
manifest = ROOT / ("pipeline/analyses/y46_cache_revision_prereg_601b_" + args.token + ".json")
with manifest.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
