"""Bind existing frozen native protocols to third moments without changing samples."""

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
replacements = {
    "adares_y42_point_covariance": "adares_y43_point_asymmetry",
    "adares_y42_point_covariance_res.yaml": "adares_y43_point_asymmetry_res.yaml",
    "79ce2de9582c369422facb27c3cbe85559bf227c": "243e0b6827073831e0d68090857755d575efbf16",
    "configs_song/model_paper5/adares_y42_point_covariance_res.yaml":
        "configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml",
    "AdaRES-Y42": "AdaRES-Y43",
    "ipdn/model_paper5/adares_y41_boundary_probability":
        "ipdn/model_paper5/adares_y42_point_covariance",
}


class Bind(ast.NodeTransformer):
    def visit_Name(self, node):
        if "covariance" in node.id:
            node.id = node.id.replace("covariance", "third")
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            if node.value in replacements:
                node.value = replacements[node.value]
                return node
            node.value = node.value.replace("covariance", "third").replace("diagonal_only", "zero_third")
            node.value = node.value.replace("Y42", "Y43").replace("Y41", "Y42").replace("y41_", "y42_")
        return node


records = []
for source_name, target_name in (
    ("probe_y42_native_controls_v2.py", "probe_y43_native_controls_0422.py"),
    ("probe_y42_initial_loss_investigation_v2.py", "probe_y43_initial_loss_0422.py"),
    ("probe_y42_parent_semantics_v3.py", "probe_y43_parent_semantics_0422.py"),
):
    source = ROOT / "scripts" / source_name
    tree = Bind().visit(ast.parse(source.read_text(encoding="utf-8")))
    if "native_controls" in source_name:
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "readout_pre")
        replacement = ast.parse("""
def readout_pre(module, args):
    if mode == "enabled":
        captured["readout"] = clone(args)
    q, masks, pad, mass, third = args
    if mode == "wrong_masks":
        return q, masks.roll(1, 1), pad, mass, third
    if mode == "wrong_third":
        changed = third.clone()
        for b in range(third.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = third[b, :count].roll(1, 0)
        return q, masks, pad, mass, changed
    if mode == "zero_third":
        return q, masks, pad, mass, torch.zeros_like(third)
    if mode == "reflected_third":
        return q, masks, pad, mass, -third
""").body[0]
        tree.body[tree.body.index(function)] = replacement
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Subscript):
                if ast.unparse(node.left) == "native_third.shape[1]":
                    assert isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 528
                    node.comparators[0].value = 32
    elif "initial_loss" in source_name:
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "shape_input")
        tree.body[tree.body.index(function)] = ast.parse("""
def shape_input(module, args):
    q, masks, pad, mass, third = args
    if mode == "rolled":
        return q, masks.roll(1, 1), pad, mass, third
    if mode == "wrong_third":
        changed = third.clone()
        for b in range(third.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = third[b, :count].roll(1, 0)
        return q, masks, pad, mass, changed
""").body[0]
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "MaskPointCovarianceReadout":
                node.value = "MaskPointAsymmetryReadout"
            if isinstance(node, ast.Tuple) and len(node.elts) == 3:
                if [ast.literal_eval(x) for x in node.elts if isinstance(x, ast.Set)] == [
                    {"mean_features"}, {"centered"}, {"joint_rows"}]:
                    node.elts = [ast.Set(elts=[ast.Constant(value="mean_features")])]
            if isinstance(node, ast.Dict):
                keys = [ast.literal_eval(k) for k in node.keys if isinstance(k, ast.Constant)]
                if "constructor" in keys and "extract_statistic" in keys:
                    index = keys.index("extract_statistic")
                    assert node.values[index].value == 3
                    node.values[index].value = 1
    ast.fix_missing_locations(tree)
    text = ast.unparse(tree) + "\n"
    target = ROOT / "scripts" / target_name
    assert not target.exists()
    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8", newline="\n")
    records.append({"path": target.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "parent_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
output = ROOT / ("pipeline/analyses/y43_control_bindings_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps({"complete": True, "files": records,
    "frozen_samples": "TRAIN first8/init4; whole-module AST stripping only",
    "limits": ["Unexecuted bindings are not gate passes."]}, indent=2), encoding="utf-8")
print(json.dumps({"files": len(records)}))
