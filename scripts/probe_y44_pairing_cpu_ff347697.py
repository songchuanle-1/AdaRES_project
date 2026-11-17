"""Actual marginal readouts under fixed-geometry point-feature pairing changes."""

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
    obj = SimpleNamespace(input_conv=lambda x: x, unet=lambda x: (x, None),
                          output_layer=lambda x: x, pool="mean")
    return MethodType(scope["extract_feat"], obj)


def readout(package, name):
    path = Path("ipdn/model_paper5") / package / "dec.py"
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    scope = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def cross(coords, feats, labels):
    cm = scatter_mean(coords, labels, dim=0)
    fm = scatter_mean(feats, labels, dim=0)
    products = (coords - cm[labels])[:, :, None] * (feats - fm[labels])[:, None, :]
    return scatter_mean(products.flatten(1), labels, dim=0)


extractors = {
    "variance": extraction("adares_y39_point_distribution"),
    "covariance": extraction("adares_y42_point_covariance"),
    "third": extraction("adares_y43_point_asymmetry"),
}
classes = {
    "variance": readout("adares_y39_point_distribution", "MaskPointDistributionReadout"),
    "covariance": readout("adares_y42_point_covariance", "MaskPointCovarianceReadout"),
    "third": readout("adares_y43_point_asymmetry", "MaskPointAsymmetryReadout"),
}
parent_class = readout("adares_w2_pointmass_competition", "QueryCompetitionIndicator")
labels = torch.arange(10).repeat_interleave(6)
mapping = torch.arange(60)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    base = torch.randint(4, 9, (10, 32)).double()
    scale = torch.randint(1, 4, (10, 32)).double()
    centered = torch.tensor([-1., -1., 2., -1., -1., 2.], dtype=torch.float64)
    features = (base[:, None] + centered[None, :, None] * scale[:, None]).reshape(60, 32)
    wrong = features.reshape(10, 6, 32).roll(1, dims=1).reshape(60, 32)
    coords = torch.stack((centered, centered.square(), torch.arange(6).double()), dim=1)
    coords = coords.repeat(10, 1) + labels[:, None].double() * 10
    assert torch.all(features >= 0) and torch.all(wrong >= 0)
    query = torch.randn(1, 8, 256).double()
    masks = torch.randn(1, 8, 10).double()
    pad = torch.zeros(1, 10, dtype=torch.bool)
    pad[:, -2:] = True
    mass = torch.full((1, 10), 6., dtype=torch.float64)
    marginal_max = output_max = 0.
    for name, extract in extractors.items():
        a = extract(SimpleNamespace(features=features), labels, mapping)
        b = extract(SimpleNamespace(features=wrong), labels, mapping)
        assert len(a) == len(b) == 2
        marginal_max = max(marginal_max, *(float((x-y).abs().max()) for x,y in zip(a,b)))
        model = classes[name](256, 32).double().eval()
        output_max = max(output_max, float((
            model(query, masks, pad, mass, a[1][None]) -
            model(query, masks, pad, mass, b[1][None])).abs().max()))
    parent = parent_class(256).double().eval()
    parent_repeat = float((parent(query, masks, pad, mass) -
                           parent(query, masks, pad, mass)).abs().max())
    a, b = cross(coords, features, labels), cross(coords, wrong, labels)
    order = torch.randperm(60)
    perm = cross(coords[order], features[order], labels[order])
    shifted = cross(coords + torch.tensor([17., -8., 31.]), features, labels)
    row = {"seed": seed, "existing_marginal_max": marginal_max,
           "existing_readout_max": output_max, "repeat_max": parent_repeat,
           "joint_pairing_gap": float((a-b).abs().max()),
           "joint_order_floor": float((a-perm).abs().max()),
           "translation_floor": float((a-shifted).abs().max())}
    assert marginal_max == output_max == parent_repeat == 0
    assert row["joint_pairing_gap"] > max(row["joint_order_floor"], row["translation_floor"])
    rows.append(row)
result = {"complete": True, "id": "AdaRES-Y44", "seeds": 32,
          "coordinates": 3, "channels": 32, "rows": rows, "source_hashes": sources,
          "existing_marginal_max": max(r["existing_marginal_max"] for r in rows),
          "existing_readout_max": max(r["existing_readout_max"] for r in rows),
          "minimum_pairing_gap": min(r["joint_pairing_gap"] for r in rows),
          "control_floor": max(max(r["joint_order_floor"], r["translation_floor"], r["repeat_max"]) for r in rows),
          "both_feature_arms_nonnegative": True,
          "limits": ["Fixed post-backbone witness, not an input-level backbone equivalence.",
                     "Explicit marginal-path gap only; native incidence/controls/backward/resources/independence pending.",
                     "No efficacy, necessity, upstream blindness or novelty conclusion."]}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
