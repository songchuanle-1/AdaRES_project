"""Actual independent extraction/readout, using nonnegative matched fixtures."""

import ast
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace
import torch
from torch import nn
from torch_scatter import scatter_max, scatter_mean

torch.set_num_threads(4)
assert not torch.cuda.is_available()
base = Path("ipdn/model_paper5/adares_y43_point_asymmetry")
ipdn, dec = base / "ipdn.py", base / "dec.py"
extract = next(n for n in ast.walk(ast.parse(ipdn.read_text()))
               if isinstance(n, ast.FunctionDef) and n.name == "extract_feat")
scope = {"torch": torch, "scatter_mean": scatter_mean, "scatter_max": scatter_max}
exec(compile(ast.Module(body=[extract], type_ignores=[]), str(ipdn), "exec"), scope)
module = SimpleNamespace(input_conv=lambda x: x, unet=lambda x: (x, None),
                         output_layer=lambda x: x, pool="mean")
module.extract_feat = MethodType(scope["extract_feat"], module)
node = next(n for n in ast.parse(dec.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "MaskPointAsymmetryReadout")
scope = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(dec), "exec"), scope)
labels, mapping = torch.arange(10).repeat_interleave(6), torch.arange(60)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    mean = torch.randint(10, 14, (10, 32)).double()
    scale = torch.randint(1, 4, (10, 32)).double()
    signs = 2 * torch.randint(0, 2, (10, 32)).double() - 1
    centered = torch.tensor([-1., -1., 2., -1., -1., 2.], dtype=torch.float64)
    centered = centered[None, :, None] * (scale * signs)[:, None]
    a = (mean[:, None] + centered).reshape(60, 32).requires_grad_()
    b = (mean[:, None] - centered).reshape(60, 32)
    assert min(float(a.min()), float(b.min())) >= 0
    ma, ta = module.extract_feat(SimpleNamespace(features=a), labels, mapping)
    mb, tb = module.extract_feat(SimpleNamespace(features=b), labels, mapping)
    ca = torch.einsum("snc,snd->scd", centered, centered) / 6
    cb = torch.einsum("snc,snd->scd", -centered, -centered) / 6
    assert torch.equal(ma, mb) and torch.equal(ca, cb) and torch.equal(ta, -tb)
    reference = 2 * (scale * signs) ** 3
    assert torch.equal(ta, reference)
    q = torch.randn(1, 8, 256, dtype=torch.float64, requires_grad=True)
    masks = torch.randn(1, 8, 10, dtype=torch.float64, requires_grad=True)
    pad = torch.zeros(1, 10, dtype=torch.bool)
    pad[:, -2:] = True
    mass = torch.full((1, 10), 6., dtype=torch.float64)
    readout = scope["MaskPointAsymmetryReadout"](256, 32).double().eval()
    out = readout(q, masks, pad, mass, ta[None])
    repeat = readout(q, masks, pad, mass, ta[None])
    reflected = readout(q, masks, pad, mass, tb[None])
    wrong = readout(q, masks.roll(1, 1), pad, mass, ta[None])
    order = torch.randperm(60)
    pm, pt = module.extract_feat(SimpleNamespace(features=a), labels[order], mapping[order])
    same_third = readout(q, masks, pad, mass, pt[None])
    sp = torch.randperm(10)
    permuted = readout(q, masks[:, :, sp], pad[:, sp], mass[:, sp], ta[None, sp])
    poison = ta.detach().clone()
    poison[pad[0]] = 1e5
    poisoned = readout(q, masks, pad, mass, poison[None])
    out.square().mean().backward()
    gradients = {name: float(p.grad.norm()) for name, p in readout.named_parameters()}
    assert len(gradients) == 8 and all(v > 0 for v in gradients.values())
    assert all(torch.isfinite(p.grad).all() for p in readout.parameters())
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.norm() > 0
    assert masks.grad is None
    zero = torch.zeros(1, 10, 32, dtype=torch.float64, requires_grad=True)
    readout(q.detach(), masks.detach(), pad, mass, zero).square().mean().backward()
    assert torch.isfinite(zero.grad).all()
    row = {
        "seed": seed, "minimum_fixture_feature": min(float(a.min()), float(b.min())),
        "repeat_max": float((out - repeat).abs().max()),
        "equal_third_moment_max": float((out - same_third).abs().max()),
        "support_permutation_max": float((out - permuted).abs().max()),
        "padding_poison_max": float((out - poisoned).abs().max()),
        "reflection_delta": float((out - reflected).abs().max()),
        "wrong_masks_delta": float((out - wrong).abs().max()),
        "added_gradients": gradients, "raw_gradient": float(a.grad.norm()),
        "zero_moment_finite_gradient": True, "mask_gradient_none": True,
    }
    assert row["repeat_max"] == row["equal_third_moment_max"] == row["padding_poison_max"] == 0
    assert row["reflection_delta"] > row["support_permutation_max"]
    assert row["wrong_masks_delta"] > row["support_permutation_max"]
    rows.append(row)
result = {
    "complete": True, "id": "AdaRES-Y43", "cpu_cases": 32, "channels": 32, "rows": rows,
    "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ipdn, dec)},
    "all_fixture_features_nonnegative": True,
    "minimum_reflection_delta": min(r["reflection_delta"] for r in rows),
    "minimum_wrong_mask_delta": min(r["wrong_masks_delta"] for r in rows),
    "minimum_added_gradient": min(v for r in rows for v in r["added_gradients"].values()),
    "repeat_max": max(r["repeat_max"] for r in rows),
    "permutation_floor": max(r["support_permutation_max"] for r in rows),
    "all_zero_moment_gradients_finite": True,
    "limits": ["Actual new independent branch, synthetic fixed-feature representability only.",
               "Initial reflection witness allowed some negative features; this retained confirmation confines both arms to ReLU codomain.",
               "No task efficacy or native backbone invariance claim."],
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
