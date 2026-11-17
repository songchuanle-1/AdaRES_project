"""Actual W2/Y39 discrimination on fixed diagonal covariance controls."""

import ast
import hashlib
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(4)
assert not torch.cuda.is_available()
paths = {
    "w2": Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py"),
    "y39": Path("ipdn/model_paper5/adares_y39_point_distribution/dec.py"),
}


def actual_class(path, name):
    node = next(
        n for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


torch.manual_seed(1999)
w2 = actual_class(paths["w2"], "QueryCompetitionIndicator")(256).double().eval()
y39 = actual_class(paths["y39"], "MaskPointDistributionReadout")(256, 32).double().eval()
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    mean = torch.randint(2, 5, (10, 32)).double()
    signs = torch.randint(0, 2, (10, 32)).double() * 2 - 1
    sequence = torch.tensor([1., -1., 1., -1.]).double().view(1, 4, 1)
    a = mean[:, None, :] + signs[:, None, :] * sequence
    b = a.clone()
    # Reflect alternating channels: all marginal means/variances stay fixed.
    b[:, :, 1::2] = 2 * mean[:, None, 1::2] - b[:, :, 1::2]
    va = ((a - a.mean(1, keepdim=True)) ** 2).mean(1)
    vb = ((b - b.mean(1, keepdim=True)) ** 2).mean(1)
    ca = torch.einsum("snc,snd->scd", a - mean[:, None], a - mean[:, None]) / 4
    cb = torch.einsum("snc,snd->scd", b - mean[:, None], b - mean[:, None]) / 4
    assert torch.equal(a.mean(1), b.mean(1))
    assert torch.equal(va, vb)
    query = torch.randn(1, 8, 256).double()
    masks = torch.randn(1, 8, 10).double()
    padding = torch.zeros(1, 10, dtype=torch.bool)
    mass = torch.full((1, 10), 4.).double()
    with torch.no_grad():
        wa, wb = w2(query, masks, padding, mass), w2(query, masks, padding, mass)
        ya = y39(query, masks, padding, mass, va[None])
        yb = y39(query, masks, padding, mass, vb[None])
        repeat = y39(query, masks, padding, mass, va[None])
    permuted = a[:, [3, 1, 2, 0]]
    cp = torch.einsum(
        "snc,snd->scd", permuted - mean[:, None], permuted - mean[:, None]
    ) / 4
    row = {
        "seed": seed, "mean_difference": float((a.mean(1) - b.mean(1)).abs().max()),
        "diagonal_variance_difference": float((va - vb).abs().max()),
        "covariance_difference": float((ca - cb).abs().max()),
        "w2_difference": float((wa - wb).abs().max()),
        "y39_difference": float((ya - yb).abs().max()),
        "repeat_difference": float((ya - repeat).abs().max()),
        "point_permutation_difference": float((ca - cp).abs().max()),
    }
    assert row["covariance_difference"] == 2
    assert all(row[key] == 0 for key in row if key.endswith("difference")
               and key != "covariance_difference")
    rows.append(row)
result = {
    "complete": True, "id": "AdaRES-Y42", "cpu_only": True, "cpu_cases": len(rows),
    "rows": rows, "source_hashes": {
        key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()
    },
    "w2_max": max(row["w2_difference"] for row in rows),
    "y39_max": max(row["y39_difference"] for row in rows),
    "covariance_min": min(row["covariance_difference"] for row in rows),
    "repeat_max": max(row["repeat_difference"] for row in rows),
    "permutation_max": max(row["point_permutation_difference"] for row in rows),
    "conclusion": "Actual W2/Y39 readouts are blind to the covariance-only "
                  "pair. The joint statistic separates all 32 fixed fixtures.",
    "limitations": ["Synthetic readout inputs, not upstream backbone blindness.",
                    "No native utility, necessity, efficacy or admission PASS."],
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
