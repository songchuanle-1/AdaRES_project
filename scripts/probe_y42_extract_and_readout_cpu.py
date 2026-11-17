"""Execute actual covariance extraction, pair separation and invariances."""

import ast
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace
import torch
from torch import nn
from torch_scatter import scatter_max, scatter_mean

torch.set_num_threads(4)
assert not torch.cuda.is_available()
ipdn_path = Path("ipdn/model_paper5/adares_y42_point_covariance/ipdn.py")
dec_path = Path("ipdn/model_paper5/adares_y42_point_covariance/dec.py")
extract = next(
    n for n in ast.walk(ast.parse(ipdn_path.read_text(encoding="utf-8")))
    if isinstance(n, ast.FunctionDef) and n.name == "extract_feat"
)
namespace = {"torch": torch, "scatter_mean": scatter_mean, "scatter_max": scatter_max}
exec(compile(ast.Module(body=[extract], type_ignores=[]), str(ipdn_path), "exec"), namespace)
module = SimpleNamespace(input_conv=lambda x: x, unet=lambda x: (x, None),
                         output_layer=lambda x: x, pool="mean")
module.extract_feat = MethodType(namespace["extract_feat"], module)
node = next(
    n for n in ast.parse(dec_path.read_text(encoding="utf-8")).body
    if isinstance(n, ast.ClassDef) and n.name == "MaskPointCovarianceReadout"
)
scope = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(dec_path), "exec"), scope)
tri = torch.triu_indices(32, 32)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    features = torch.rand(60, 32, requires_grad=True)
    labels = torch.arange(10).repeat_interleave(6)
    order = torch.randperm(60)
    inverse = torch.argsort(order)
    x = SimpleNamespace(features=features[order])
    pooled, covariance = module.extract_feat(x, labels, inverse)
    expected_mean = features.reshape(10, 6, 32).mean(1)
    centered = features.reshape(10, 6, 32) - expected_mean[:, None]
    expected = torch.einsum("snc,snd->scd", centered, centered) / 6
    expected = expected[:, tri[0], tri[1]]
    point_order = torch.randperm(60)
    pm, pc = module.extract_feat(x, labels[point_order], inverse[point_order])
    covariance.square().sum().backward()
    readout = scope["MaskPointCovarianceReadout"](256, 32)
    q = torch.randn(1, 8, 256, requires_grad=True)
    masks = torch.randn(1, 8, 10, requires_grad=True)
    pad = torch.zeros(1, 10, dtype=torch.bool)
    pad[:, -2:] = True
    mass = torch.full((1, 10), 6.)
    cov = covariance.detach()[None].requires_grad_()
    out = readout(q, masks, pad, mass, cov)
    repeat = readout(q, masks, pad, mass, cov)
    wrong = cov.detach().clone()
    offdiag = tri[0] != tri[1]
    wrong[:, :, offdiag] = wrong[:, :, offdiag].roll(1, 1)
    changed = readout(q, masks, pad, mass, wrong)
    sp = torch.arange(9, -1, -1)
    permuted = readout(q, masks[:, :, sp], pad[:, sp], mass[:, sp], cov[:, sp])
    poison = cov.detach().clone()
    poison[pad] = 1e5
    poisoned = readout(q, masks, pad, mass, poison)
    signs = torch.randint(0, 2, (10, 32)).float() * 2 - 1
    points_a = 2 + signs[:, None] * torch.tensor([1., -1., 1., -1.])[None, :, None]
    points_b = points_a.clone()
    points_b[:, :, 1::2] = 4 - points_b[:, :, 1::2]
    pair_labels = torch.arange(10).repeat_interleave(4)
    pair_map = torch.arange(40)
    ma, ca = module.extract_feat(SimpleNamespace(features=points_a.reshape(40, 32)), pair_labels, pair_map)
    mb, cb = module.extract_feat(SimpleNamespace(features=points_b.reshape(40, 32)), pair_labels, pair_map)
    assert torch.equal(ma, mb) and torch.equal(ca[:, tri[0] == tri[1]], cb[:, tri[0] == tri[1]])
    pair_delta = float((readout(q, masks, pad, mass, ca[None])
                        - readout(q, masks, pad, mass, cb[None])).abs().max())
    out.square().sum().backward()
    gradients = {k: float(p.grad.norm()) for k, p in readout.named_parameters()}
    row = {
        "seed": seed, "mean_reference_max": float((pooled - expected_mean).abs().max()),
        "covariance_reference_max": float((covariance - expected).abs().max()),
        "point_permutation_max": float((covariance - pc).abs().max()),
        "support_permutation_max": float((out - permuted).abs().max()),
        "repeat_max": float((out - repeat).abs().max()),
        "padding_poison_max": float((out - poisoned).abs().max()),
        "wrong_offdiagonal_delta": float((out - changed).abs().max()),
        "matched_covariance_only_delta": pair_delta,
        "gradients": gradients, "raw_gradient": float(features.grad.norm()),
        "input_gradient": float(cov.grad.norm()), "mask_gradient_none": masks.grad is None,
    }
    assert covariance.shape[1] == 528 and len(gradients) == 8
    assert all(g > 0 for g in gradients.values()) and row["raw_gradient"] > 0
    assert row["input_gradient"] > 0 and row["mask_gradient_none"]
    assert row["repeat_max"] == row["padding_poison_max"] == 0
    assert all(row[k] < 1e-5 for k in row if k.endswith("_max"))
    assert row["wrong_offdiagonal_delta"] > 0 and pair_delta > 0
    rows.append(row)
result = {
    "complete": True, "id": "AdaRES-Y42", "cpu_cases": len(rows), "rows": rows,
    "source_hashes": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ipdn_path, dec_path)
    },
    "covariance_width": 528, "added_tensors": 8,
    "algebra_and_permutation_max": max(r[k] for r in rows for k in r if k.endswith("_max")),
    "wrong_offdiagonal_min": min(r["wrong_offdiagonal_delta"] for r in rows),
    "matched_covariance_only_min": min(r["matched_covariance_only_delta"] for r in rows),
    "minimum_added_gradient": min(g for r in rows for g in r["gradients"].values()),
    "limits": ["Identity-backbone extraction algebra and actual synthetic readout only.",
               "No sparse-backbone invariance, native utility, efficacy or admission."],
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
