"""Whole-parent AST identity and actual new-class pairing controls on CPU."""
import ast
import hashlib
from pathlib import Path
import sys
import yaml
import torch
from torch import nn
from torch_scatter import scatter_mean

base = Path("ipdn/model_paper5")
parent = base / "adares_w2_pointmass_competition"
candidate = base / "adares_y44_coordinate_feature"
counts = {}


def record(key):
    counts[key] = counts.get(key, 0) + 1


class StripCall(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if node.args and isinstance(node.args[-1], ast.Name):
            argument = node.args[-1].id
            if argument == "point_cross":
                assert isinstance(node.func, ast.Attribute) and node.func.attr in ("forward_iter_pred", "dec")
                node.args.pop()
                record("cross_call")
            elif argument == "coords_float" and isinstance(node.func, ast.Attribute) and node.func.attr == "extract_feat":
                node.args.pop()
                record("coord_call")
        return node


for name in ("dec.py", "ipdn.py"):
    original = ast.parse((parent / name).read_text())
    changed = ast.parse((candidate / name).read_text())
    changed.body = [n for n in changed.body if not (isinstance(n, ast.ClassDef) and n.name == "MaskCoordinateFeatureReadout")]
    for cls in changed.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in ("DEC", "IPDN"):
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.args.args and method.args.args[-1].arg == "point_cross":
                method.args.args.pop()
                assert isinstance(method.args.defaults[-1], ast.Constant) and method.args.defaults[-1].value is None
                method.args.defaults.pop()
                record("cross_signature")
            elif method.name == "extract_feat":
                assert method.args.args[-1].arg == "coords_float"
                method.args.args.pop()
                record("coord_signature")
            kept = []
            for statement in method.body:
                if isinstance(statement, ast.With) and any(isinstance(n, ast.Attribute) and n.attr == "point_readout" for n in ast.walk(statement)):
                    record("constructor")
                    continue
                if isinstance(statement, ast.If) and any(isinstance(n, ast.Name) and n.id == "point_cross" for n in ast.walk(statement.test)):
                    record("required_input")
                    continue
                if isinstance(statement, ast.Assign):
                    targets = {n.id for t in statement.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
                    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "point_readout" for n in ast.walk(statement.value)):
                        record("final_addition")
                        continue
                    if "sp_feats" in targets and "point_cross" in targets:
                        assert statement.value.func.attr == "extract_feat"
                        statement.targets = [ast.Name(id="sp_feats", ctx=ast.Store())]
                        record("extract_call")
                    elif "point_cross" in targets:
                        record("cross_assignment")
                        continue
                    elif method.name == "extract_feat" and targets in ({"mean_features"}, {"mean_coords"}):
                        record("extract_statistic")
                        continue
                if method.name == "extract_feat" and isinstance(statement, ast.Return) and isinstance(statement.value, ast.Tuple):
                    assert [n.id for n in statement.value.elts] == ["x", "point_cross"]
                    statement.value = ast.Name(id="x", ctx=ast.Load())
                    record("extract_return")
                kept.append(statement)
            method.body = kept
    changed = StripCall().visit(changed)
    assert ast.dump(original) == ast.dump(changed), name
expected = {"constructor": 1, "cross_signature": 2, "required_input": 1,
            "cross_assignment": 4, "final_addition": 1, "cross_call": 3,
            "extract_call": 2, "extract_statistic": 2, "extract_return": 1,
            "coord_call": 2, "coord_signature": 1}
assert counts == expected, counts
configs = Path("configs_song/model_paper5")
assert yaml.safe_load((configs / "adares_w2_pointmass_competition_res.yaml").read_text()) == yaml.safe_load((configs / "adares_y44_coordinate_feature_res.yaml").read_text())
tree = ast.parse((candidate / "dec.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MaskCoordinateFeatureReadout")
namespace = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=[cls], type_ignores=[]), "<actual-y44-class>", "exec"), namespace)
torch.set_num_threads(4)
gaps, floors, gradients = [], [], []
for seed in range(32):
    torch.manual_seed(seed)
    readout = namespace["MaskCoordinateFeatureReadout"](256, 32).double()
    sp = torch.arange(6).repeat_interleave(4)
    coords = torch.randn(24, 3, dtype=torch.double)
    features = torch.rand(24, 32, dtype=torch.double)
    permutation = torch.arange(24).reshape(6, 4).flip(1).flatten()

    def statistic(c, f, labels):
        return scatter_mean(((c - scatter_mean(c, labels, dim=0)[labels]).unsqueeze(-1) *
                             (f - scatter_mean(f, labels, dim=0)[labels]).unsqueeze(1)).flatten(1), labels, dim=0)

    cross = statistic(coords, features, sp).unsqueeze(0).requires_grad_()
    wrong = statistic(coords, features[permutation], sp).unsqueeze(0)
    q = torch.randn(1, 8, 256, dtype=torch.double)
    masks = torch.randn(1, 8, 6, dtype=torch.double, requires_grad=True)
    pad = torch.tensor([[False, False, False, False, False, True]])
    mass = torch.full((1, 6), 4.0, dtype=torch.double)
    output = readout(q, masks, pad, mass, cross)
    output.square().sum().backward()
    norms = {n: float(p.grad.norm()) for n, p in readout.named_parameters()}
    assert len(norms) == 8 and all(v > 0 for v in norms.values())
    assert masks.grad is None and torch.isfinite(cross.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in readout.parameters())
    with torch.no_grad():
        gaps.append(float((output - readout(q, masks, pad, mass, wrong)).abs().max()))
        permutation2 = torch.arange(23, -1, -1)
        joint = statistic(coords[permutation2], features[permutation2], sp[permutation2]).unsqueeze(0)
        translated = statistic(coords + 5, features, sp).unsqueeze(0)
        poison = cross.detach().clone()
        poison[pad] = 100000.0
        floors.extend([
            float((output - readout(q, masks, pad, mass, cross)).abs().max()),
            float((output - readout(q, masks, pad, mass, joint)).abs().max()),
            float((output - readout(q, masks, pad, mass, translated)).abs().max()),
            float((output - readout(q, masks, pad, mass, poison)).abs().max()),
        ])
    gradients.extend(norms.values())
assert min(gaps) > max(floors)
result = {"complete": True, "id": "AdaRES-Y44", "whole_modules_parent_identity": True,
          "removed_exactly": counts, "config_differences": [], "cpu_cases": 32,
          "minimum_pairing_response": min(gaps), "control_floor": max(floors),
          "minimum_new_gradient": min(gradients), "all_feature_arms_nonnegative": True,
          "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for folder in (parent, candidate) for p in folder.glob("*.py")},
          "limitations": ["Actual-class CPU controlled mechanism, not native incidence or task efficacy.",
                          "Shared whole-module parent equality after removing exactly new path; no runtime weights copied."]}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
