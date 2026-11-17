"""Whole-parent AST identity and actual independent readout controls."""

import ast
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
import yaml

root = Path(P["root"])
base = root / "ipdn/model_paper5"
parent = base / "adares_w2_pointmass_competition"
candidate = base / "adares_y45_metric_decomposition"
removed = {}


def count(key):
    removed[key] = removed.get(key, 0) + 1


for name in ("dec.py", "ipdn.py", "loss.py"):
    original = ast.parse((parent / name).read_text())
    changed = ast.parse((candidate / name).read_text())
    changed.body = [n for n in changed.body if not (
        isinstance(n, ast.ClassDef) and n.name == "MetricDecompositionReadout")]
    for cls in changed.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        kept_methods = []
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name in ("point_positives", "loss_decomposition"):
                count(method.name)
                continue
            kept_methods.append(method)
            if not isinstance(method, ast.FunctionDef):
                continue
            body = []
            for statement in method.body:
                if isinstance(statement, ast.With) and any(
                        isinstance(n, ast.Attribute) and n.attr == "metric_readout" for n in ast.walk(statement)):
                    count("constructor")
                    continue
                if isinstance(statement, ast.If) and any(
                        isinstance(n, ast.Attribute) and n.attr == "metric_readout" for n in ast.walk(statement.test)):
                    count("final_readout")
                    continue
                if isinstance(statement, ast.If) and any(
                        isinstance(n, ast.Constant) and n.value == "error_logits" for n in ast.walk(statement.test)):
                    count("new_likelihood")
                    continue
                if isinstance(statement, ast.Assign):
                    target = statement.targets[0]
                    if isinstance(target, ast.Name) and target.id in ("error_logits", "point_positive"):
                        count(target.id)
                        continue
                    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and target.slice.value == "point_positive":
                        count("positive_padding")
                        continue
                if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Dict):
                    pairs = [(k, v) for k, v in zip(statement.value.keys, statement.value.values)
                             if not (isinstance(k, ast.Constant) and k.value in ("error_logits", "point_mass"))]
                    statement.value.keys = [k for k, _ in pairs]
                    statement.value.values = [v for _, v in pairs]
                body.append(statement)
            method.body = body
        cls.body = kept_methods
    assert ast.dump(original) == ast.dump(changed), name
assert removed == dict(constructor=1, error_logits=1, final_readout=1,
                       point_positive=1, positive_padding=1, point_positives=1,
                       new_likelihood=1, loss_decomposition=1)
for name in ("__init__.py", "backbone.py", "sample_model.py"):
    assert ast.dump(ast.parse((parent / name).read_text())) == ast.dump(ast.parse((candidate / name).read_text()))
configs = root / "configs_song/model_paper5"
assert yaml.safe_load((configs / "adares_w2_pointmass_competition_res.yaml").read_text()) == yaml.safe_load((configs / "adares_y45_metric_decomposition_res.yaml").read_text())
tree = ast.parse((candidate / "dec.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MetricDecompositionReadout")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(candidate / "dec.py"), "exec"), namespace)
torch.set_num_threads(4)
gaps, floors, gradients = [], [], []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    readout = namespace["MetricDecompositionReadout"](256).double()
    query = torch.randn(2, 8, 256, dtype=torch.double)
    masks = torch.randn(2, 8, 16, dtype=torch.double, requires_grad=True)
    padding = torch.zeros(2, 16, dtype=torch.bool)
    padding[:, -1] = True
    mass = torch.randint(1, 16, (2, 16)).double()
    features = torch.randn(2, 16, 256, dtype=torch.double, requires_grad=True)
    output = readout(query, masks, padding, mass, features)
    output.square().sum().backward()
    assert masks.grad is None
    assert torch.isfinite(features.grad).all()
    norms = [float(p.grad.norm()) for p in readout.parameters()]
    assert len(norms) == 4 and all(v > 0 for v in norms)
    assert all(torch.isfinite(p.grad).all() for p in readout.parameters())
    with torch.no_grad():
        wrong = readout(query, masks.roll(1, 1), padding, mass, features)
        gaps.append(float((wrong - output).abs().max()))
        order = torch.arange(15, -1, -1)
        joint = readout(query, masks[:, :, order], padding[:, order], mass[:, order], features[:, order])
        poisoned = features.detach().clone()
        poisoned[:, -1] = 1e5
        poison = readout(query, masks, padding, mass, poisoned)
        floors.extend([float((output - readout(query, masks, padding, mass, features)).abs().max()),
                       float((output - joint).abs().max()), float((output - poison).abs().max())])
    gradients.extend(norms)
assert min(gaps) > max(floors)
result = dict(complete=True, id="AdaRES-Y45", whole_parent_AST_identity=True,
              stripped_exactly=removed, config_differences=[], cases=32,
              minimum_wrong_mask_response=min(gaps), observed_control_floor=max(floors),
              minimum_new_CPU_gradient=min(gradients), detached_mask_verified=True,
              source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for folder in (parent, candidate) for p in folder.glob("*.py")},
              scientific_status="actual_class_CPU_controls_native_graph_pending",
              limitations=["Independent physical source equality after removing exact additions.",
                           "Actual readout class CPU controls, not native TRAIN diagnosis or useful learnability.",
                           "No gate receipt, diagnostic weight deployment or efficacy."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
