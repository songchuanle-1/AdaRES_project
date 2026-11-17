"""Actual-source AST and CPU module contracts, not synthetic efficacy."""

import ast
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(4)
assert not torch.cuda.is_available(), "CPU-only audit must not expose a GPU"
folder = Path("ipdn/model_paper5/adares_y28_mask_readback")
parent = folder.parent / "adares_w2_pointmass_competition"
new_tree = ast.parse((folder / "dec.py").read_text())
old_tree = ast.parse((parent / "dec.py").read_text())
classes = {n.name: n for n in new_tree.body if isinstance(n, ast.ClassDef)}
old_classes = {n.name: n for n in old_tree.body if isinstance(n, ast.ClassDef)}
for name in old_classes.keys() - {"DEC", "QueryCompetitionIndicator"}:
    assert ast.dump(classes[name]) == ast.dump(old_classes[name])
for name in old_classes["DEC"].body:
    if isinstance(name, ast.FunctionDef) and name.name == "__init__":
        new_init = next(n for n in classes["DEC"].body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        assert ast.dump(name) == ast.dump(new_init)
calls = [n for n in ast.walk(classes["DEC"])
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "indi_embedding"]
assert len(calls) == 2 and all(isinstance(n.args[-1], ast.Name) and n.args[-1].id == "inst_feats" for n in calls)
assert all((folder / f).read_bytes() == (parent / f).read_bytes()
           for f in ("__init__.py", "backbone.py", "ipdn.py", "loss.py", "sample_model.py"))
assert Path("configs_song/model_paper5/adares_y28_mask_readback_res.yaml").read_bytes() == Path(
    "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml").read_bytes()
namespace = dict(torch=torch, nn=nn, F=F)
exec(compile(ast.Module(body=[classes["MaskContentReadback"], classes["QueryCompetitionIndicator"]],
                        type_ignores=[]), str(folder / "dec.py"), "exec"), namespace)
parent_ns = dict(torch=torch, nn=nn, F=F)
exec(compile(ast.Module(body=[old_classes["QueryCompetitionIndicator"]], type_ignores=[]),
             str(parent / "dec.py"), "exec"), parent_ns)
torch.manual_seed(1999)
candidate = namespace["QueryCompetitionIndicator"](256)
torch.manual_seed(1999)
baseline = parent_ns["QueryCompetitionIndicator"](256)
new_keys = {k for k in candidate.state_dict() if ".mask_readback." in "." + k}
assert len(new_keys) == 4
assert all(torch.equal(v, baseline.state_dict()[k]) for k, v in candidate.state_dict().items() if k not in new_keys)
assert sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in baseline.parameters()) == 262656
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    q = torch.randn(2, 8, 256)
    masks = torch.randn(2, 8, 16)
    features = torch.randn(2, 16, 256)
    padding = torch.zeros(2, 16, dtype=torch.bool)
    padding[:, -3:] = True
    mass = torch.rand(2, 16) + 1
    enabled = candidate(q, masks, padding, mass, features)
    repeated = candidate(q, masks, padding, mass, features)
    perm = torch.arange(15, -1, -1)
    permuted = candidate(q, masks[:, :, perm], padding[:, perm], mass[:, perm], features[:, perm])
    wrong = candidate(q, masks, padding, mass, features.roll(1, 1))
    changed_features, changed_masks, changed_mass = features.clone(), masks.clone(), mass.clone()
    changed_features[:, -3:] = 1e8
    changed_masks[:, :, -3:] = 1e8
    changed_mass[:, -3:] = 1e8
    bad = candidate(q, changed_masks, padding, changed_mass, changed_features)
    handle = candidate.mask_readback.register_forward_hook(lambda module, args, out: torch.zeros_like(out))
    disabled = candidate(q, masks, padding, mass, features)
    handle.remove()
    original = baseline(q, masks, padding, mass)
    candidate.zero_grad(set_to_none=True)
    loss = F.cross_entropy(enabled.reshape(-1, 2), torch.arange(16) % 2)
    loss.backward()
    gradients = {k: float(p.grad.norm()) for k, p in candidate.named_parameters()
                 if ".mask_readback." in "." + k}
    assert len(gradients) == 4 and all(v > 0 for v in gradients.values())
    row = dict(seed=seed, repeat=float((enabled-repeated).abs().max()),
               padding=float((enabled-bad).abs().max()),
               disabled_parent=float((disabled-original).abs().max()),
               permutation=float((enabled-permuted).abs().max()),
               wrong_content=float((enabled-wrong).abs().max()), gradients=gradients)
    assert row["repeat"] == row["padding"] == row["disabled_parent"] == 0
    assert row["permutation"] < 1e-5 and row["wrong_content"] > 0
    rows.append(row)
result = dict(complete=True, id="AdaRES-Y28", cases=rows,
              structural_new_mask_visual_aggregation=True,
              shared_initialization_exact=True, config_exact_parent=True,
              source_sha256=hashlib.sha256((folder / "dec.py").read_bytes()).hexdigest(),
              cpu_cases=32, added_parameters=262656, new_trainable_tensors=4,
              repeat_max=max(r["repeat"] for r in rows),
              padding_max=max(r["padding"] for r in rows),
              disabled_parent_max=max(r["disabled_parent"] for r in rows),
              permutation_max=max(r["permutation"] for r in rows),
              wrong_content_min=min(r["wrong_content"] for r in rows),
              limitations=["Actual physical source classes extracted unchanged via AST; CPU module contracts, not population efficacy.",
                           "Full-model native TRAIN gradients and shared initialization require separate GPU evidence.",
                           "No claim of literature novelty or trained Pareto benefit."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
