"""Actual mean/variance/covariance extraction under centered reflection."""

import ast
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace
import torch
from torch import nn
from torch_scatter import scatter_max, scatter_mean

torch.set_num_threads(4)
assert not torch.cuda.is_available()
sources = {}


def extraction(package):
    path = Path("ipdn/model_paper5") / package / "ipdn.py"
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    node = next(n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(n, ast.FunctionDef) and n.name == "extract_feat")
    scope = {"torch": torch, "scatter_mean": scatter_mean, "scatter_max": scatter_max}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    module = SimpleNamespace(input_conv=lambda x: x, unet=lambda x: (x, None),
                             output_layer=lambda x: x, pool="mean")
    module.extract_feat = MethodType(scope["extract_feat"], module)
    return module.extract_feat


def readout(package, name):
    path = Path("ipdn/model_paper5") / package / "dec.py"
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    scope = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


variance_extract = extraction("adares_y39_point_distribution")
covariance_extract = extraction("adares_y42_point_covariance")
variance_class = readout("adares_y39_point_distribution", "MaskPointDistributionReadout")
covariance_class = readout("adares_y42_point_covariance", "MaskPointCovarianceReadout")
parent_class = readout("adares_w2_pointmass_competition", "QueryCompetitionIndicator")
labels = torch.arange(10).repeat_interleave(6)
mapping = torch.arange(60)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    mean = torch.randint(4, 9, (10, 32)).double()
    scale = torch.randint(1, 4, (10, 32)).double()
    signs = 2 * torch.randint(0, 2, (10, 32)).double() - 1
    centered = torch.tensor([-1., -1., 2., -1., -1., 2.], dtype=torch.float64)
    centered = centered[None, :, None] * (scale * signs)[:, None]
    a = (mean[:, None] + centered).reshape(60, 32)
    b = (mean[:, None] - centered).reshape(60, 32)
    ma, va = variance_extract(SimpleNamespace(features=a), labels, mapping)
    mb, vb = variance_extract(SimpleNamespace(features=b), labels, mapping)
    mca, ca = covariance_extract(SimpleNamespace(features=a), labels, mapping)
    mcb, cb = covariance_extract(SimpleNamespace(features=b), labels, mapping)
    ta = scatter_mean((a - ma[labels]) ** 3, labels, dim=0)
    tb = scatter_mean((b - mb[labels]) ** 3, labels, dim=0)
    assert torch.equal(ma, mb) and torch.equal(mca, mcb)
    assert torch.equal(va, vb) and torch.equal(ca, cb)
    assert torch.equal(ta, -tb) and torch.all(ta != 0)
    query = torch.randn(1, 8, 256).double()
    masks = torch.randn(1, 8, 10).double()
    pad = torch.zeros(1, 10, dtype=torch.bool)
    pad[:, -2:] = True
    mass = torch.full((1, 10), 6., dtype=torch.float64)
    variance = variance_class(256, 32).double().eval()
    covariance = covariance_class(256, 32).double().eval()
    parent = parent_class(256).double().eval()
    ya = variance(query, masks, pad, mass, va[None])
    yb = variance(query, masks, pad, mass, vb[None])
    za = covariance(query, masks, pad, mass, ca[None])
    zb = covariance(query, masks, pad, mass, cb[None])
    repeat = covariance(query, masks, pad, mass, ca[None])
    point_order = torch.randperm(60)
    pm, pc = covariance_extract(SimpleNamespace(features=a), labels[point_order],
                                mapping[point_order])
    pt = scatter_mean((a[point_order] - pm[labels[point_order]]) ** 3,
                      labels[point_order], dim=0)
    support_order = torch.randperm(10)
    support = covariance(query, masks[:, :, support_order], pad[:, support_order],
                         mass[:, support_order], ca[None, support_order])
    same_third = torch.equal(ta, pt)
    parent_delta = float((parent(query, masks, pad, mass)
                          - parent(query, masks, pad, mass)).abs().max())
    row = {
        "seed": seed, "mean_max": float((ma - mb).abs().max()),
        "variance_max": float((va - vb).abs().max()),
        "covariance_max": float((ca - cb).abs().max()),
        "variance_readout_delta": float((ya - yb).abs().max()),
        "covariance_readout_delta": float((za - zb).abs().max()),
        "parent_delta": parent_delta, "repeat_max": float((za - repeat).abs().max()),
        "point_permutation_max": float((ca - pc).abs().max()),
        "support_permutation_max": float((za - support).abs().max()),
        "equal_third_moment_point_permutation": same_third,
        "third_moment_reflection_min": float((ta - tb).abs().min()),
    }
    assert same_third
    assert row["variance_readout_delta"] == row["covariance_readout_delta"] == 0
    assert row["repeat_max"] == row["point_permutation_max"] == 0
    rows.append(row)
result = {
    "complete": True, "id": "AdaRES-Y43", "seeds": 32, "channels": 32, "rows": rows,
    "source_hashes": sources,
    "matched_mean_variance_covariance_max": max(
        r[k] for r in rows for k in ("mean_max", "variance_max", "covariance_max")),
    "existing_readout_reflection_max": max(
        r[k] for r in rows for k in
        ("variance_readout_delta", "covariance_readout_delta", "parent_delta")),
    "minimum_third_moment_reflection_gap": min(r["third_moment_reflection_min"] for r in rows),
    "repeat_max": max(r["repeat_max"] for r in rows),
    "permutation_floor_max": max(r["support_permutation_max"] for r in rows),
    "scientific_scope": "Missing explicit statistic at fixed post-backbone features only.",
    "pending": ["Native prediction-blind TRAIN incidence", "Candidate readout controls",
                "Native backward/resources", "Independent implementation", "Trained efficacy"],
    "limits": ["No upstream blindness, benefit, necessity or novelty claim.",
               "Integer reflection fixture is a representational witness, not native task utility."],
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
