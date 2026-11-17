"""Create scoped AST adaptations of existing fixed native probe protocols."""

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
assert datetime.fromisoformat(owner["deadline_at"]) > datetime.now(timezone.utc)


class Candidate(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "token_readout":
            node.attr = "edge_readout"
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("Y33", "Y36").replace(
                "y33_token_attention", "y36_local_adjacency").replace(
                "token_readout", "edge_readout").replace(
                "MaskTokenAttentionReadout", "MaskLocalAdjacencyReadout").replace(
                "5f2489de78196e8d89c004e452a665c0946f2a94",
                "b49b744dc2943a38ab4c3aeb788acc6a2cb90011")
        elif node.value == 7:
            node.value = 8
        return node


class Controls(Candidate):
    def visit_Constant(self, node):
        node = super().visit_Constant(node)
        if isinstance(node.value, str):
            node.value = node.value.replace("wrong_tokens", "wrong_edges").replace(
                "token_gradients", "edge_gradients")
        return node

    def visit_FunctionDef(self, node):
        if node.name == "readout_pre":
            return ast.parse("""
def readout_pre(module, args):
    captured["readout"] = clone(args)
    q, masks, pad, mass, tokens, coords = args
    if mode == "mask_detach":
        return q, masks.detach(), pad, mass, tokens, coords
    if mode == "wrong_masks":
        return q, masks.roll(1, 1), pad, mass, tokens, coords
    if mode == "wrong_edges":
        changed = coords.clone()
        for b in range(coords.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = coords[b, :count].roll(1, 0)
        return q, masks, pad, mass, tokens, changed
""").body[0]
        return self.generic_visit(node)

    def visit_Assign(self, node):
        self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Tuple) and [getattr(n, "id", "") for n in target.elts] == [
                    "q", "masks", "pad", "mass", "tokens"]:
                target.elts.append(ast.Name(id="coords", ctx=ast.Store()))
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "readout" and len(node.args) == 5:
            expression = "coords[:, sp]" if isinstance(node.args[-1], ast.Subscript) else "coords"
            node.args.append(ast.parse(expression).body[0].value)
        return node


class Diagnosis(Candidate):
    def visit_Assign(self, node):
        self.generic_visit(node)
        if any(isinstance(t, ast.Name) and t.id == "hooks" for t in node.targets):
            node.value.elts.append(ast.parse(
                "model.dec.register_forward_pre_hook(coordinate_pre)").body[0].value)
        if any(isinstance(t, ast.Name) and t.id == "tokens" for t in node.targets):
            return [node, ast.parse("coords = cache['coordinates']").body[0]]
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "readout" and len(node.args) == 5:
            expression = "coords[:, perm]" if isinstance(node.args[-1], ast.Subscript) else "coords"
            node.args.append(ast.parse(expression).body[0].value)
        return node


rows = []
for task in ("physical_native", "resources", "native_controls", "token_diagnosis"):
    original = ROOT / ("scripts/probe_y33_" + task + ".py")
    target_task = "edge_diagnosis" if task == "token_diagnosis" else task
    target = ROOT / ("scripts/probe_y36_" + target_task + ".py")
    assert not target.exists()
    data = original.read_bytes()
    transform = Controls() if task == "native_controls" else Diagnosis() if task == "token_diagnosis" else Candidate()
    tree = transform.visit(ast.parse(data.decode("utf-8")))
    if task == "token_diagnosis":
        tree.body.insert(0, ast.parse("""
def coordinate_pre(module, args):
    cache["coordinates"] = module.get_batches(args[5], args[2])[0].detach()
""").body[0])
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    compile(source, str(target), "exec")
    target.write_text(source, encoding="utf-8")
    rows.append(dict(original=str(original.relative_to(ROOT)),
                     original_sha256=hashlib.sha256(data).hexdigest(),
                     target=str(target.relative_to(ROOT)), sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
cpu = ROOT / "scripts/probe_y36_structure_cpu.py"
adapter = ROOT / "scripts/probe_y36_structure_cpu_v2.py"
source = adapter.read_text(encoding="utf-8")
assert source.count("LOCAL_PROBE_SOURCE") == 1
source = source.replace("LOCAL_PROBE_SOURCE", repr(cpu.read_text(encoding="utf-8")))
adapter.write_text(source, encoding="utf-8")
result = dict(complete=True, files=rows,
              preserved="native entry/2steps/modelAdamWPolyLR restoration/TRAIN sampling/resource protocol",
              changed="candidate identity/new8tensors/6-input edge path/coordinate interventions only",
              cpu_fixture_fix="Exact equal-feature-marginal fixture now uses dyadic integers; no gate tolerance relaxation.")
out = ROOT / ("pipeline/analyses/y36_probe_protocols_" + args.token + ".json")
assert not out.exists()
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(dict(path=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
