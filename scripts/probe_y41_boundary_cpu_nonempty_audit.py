P['readout_path']='/tmp/adares_y41_readout_stable_' + P['token'] + '.py'
"""Actual W2/Y36 path and preregistered boundary-mass discrimination."""

import ast
import hashlib
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(4)
assert not torch.cuda.is_available()


def actual(path, name):
    raw = path.read_bytes()
    node = next(n for n in ast.parse(raw.decode("utf-8")).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], hashlib.sha256(raw).hexdigest(), ast.dump(node)


base = Path("ipdn/model_paper5")
w2_cls, w2_hash, w2_ast = actual(
    base / "adares_w2_pointmass_competition/dec.py", "QueryCompetitionIndicator")
y36_cls, y36_hash, _ = actual(
    base / "adares_y36_local_adjacency/dec.py", "MaskLocalAdjacencyReadout")
prospective = Path(P["readout_path"])
new_cls, new_hash, _ = actual(prospective, "MaskBoundaryProbabilityReadout")
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    w2, existing, new = w2_cls(256), y36_cls(256), new_cls(256)
    q = torch.randn(1, 4, 256)
    xyz = torch.zeros(1, 8, 3)
    xyz[0, :, 0] = torch.arange(8)
    a = torch.tensor([-2., -2., -2., -2., 2., 2., 2., 2.]).view(1, 1, 8).repeat(1, 4, 1)
    b = torch.tensor([-2., 2., -2., 2., -2., 2., -2., 2.]).view(1, 1, 8).repeat(1, 4, 1)
    tokens = torch.randn(1, 1, 256).repeat(1, 8, 1)
    mass, pad = torch.ones(1, 8), torch.zeros(1, 8, dtype=torch.bool)
    da = new.descriptor(a, pad, mass, xyz)
    db = new.descriptor(b, pad, mass, xyz)
    row = {
        "seed": seed,
        "mask_histogram_equal": torch.equal(a.sort(-1).values, b.sort(-1).values),
        "w2_difference": float((w2(q, a, pad, mass) - w2(q, b, pad, mass)).abs().max()),
        "y36_difference": float((existing(q, a, pad, mass, tokens, xyz) -
                                  existing(q, b, pad, mass, tokens, xyz)).abs().max()),
        "descriptor_difference": float((da - db).abs().max()),
        "new_indicator_difference": float((new(q, a, pad, mass, tokens, xyz) -
                                          new(q, b, pad, mass, tokens, xyz)).abs().max()),
        "block_descriptor": da[0, 0].tolist(), "alternating_descriptor": db[0, 0].tolist(),
    }
    print(json.dumps({"original_finite_fixture": row}), flush=True)
    assert row["mask_histogram_equal"]
    binary_a = torch.where(a > 0, float("inf"), -float("inf"))
    binary_b = torch.tensor([
        -float("inf"), -float("inf"), float("inf"), float("inf"),
        -float("inf"), -float("inf"), float("inf"), float("inf"),
    ]).view(1, 1, 8).repeat(1, 4, 1)
    binary = {
        "w2_difference": float((w2(q, binary_a, pad, mass) -
                                w2(q, binary_b, pad, mass)).abs().max()),
        "y36_difference": float((existing(q, binary_a, pad, mass, tokens, xyz) -
                                 existing(q, binary_b, pad, mass, tokens, xyz)).abs().max()),
        "descriptor_difference": float((new.descriptor(binary_a, pad, mass, xyz) -
                                        new.descriptor(binary_b, pad, mass, xyz)).abs().max()),
        "new_indicator_difference": float((new(q, binary_a, pad, mass, tokens, xyz) -
                                          new(q, binary_b, pad, mass, tokens, xyz)).abs().max()),
    }
    row["exact_binary_control"] = binary
    assert binary["w2_difference"] == 0, binary
    assert binary["y36_difference"] < 1e-6 and binary["descriptor_difference"] > 0, binary
    assert binary["new_indicator_difference"] > 0, binary
    masks = torch.randn(2, 6, 16, requires_grad=True)
    coords = torch.randn(2, 16, 3)
    query = torch.randn(2, 6, 256)
    features = torch.randn(2, 16, 256)
    padding = torch.zeros(2, 16, dtype=torch.bool)
    padding[:, -3:] = True
    point_mass = torch.rand(2, 16) + 1
    output = new(query, masks, padding, point_mass, features, coords)
    perm = torch.arange(15, -1, -1)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    variants = {
        "repeat": new(query, masks, padding, point_mass, features, coords),
        "permutation": new(query, masks[:, :, perm], padding[:, perm], point_mass[:, perm],
                           features[:, perm], coords[:, perm]),
        "query_permutation": new(query.roll(1, 1), masks.roll(1, 1), padding,
                                 point_mass, features, coords).roll(-1, 1),
        "translation": new(query, masks, padding, point_mass, features, coords + 8),
        "rotation": new(query, masks, padding, point_mass, features, coords @ rotation),
        "scale": new(query, masks, padding, point_mass, features, coords * 2),
        "wrong_masks": new(query, masks.roll(1, 1), padding, point_mass, features, coords),
        "wrong_coordinates": new(query, masks, padding, point_mass, features, coords.roll(1, 1)),
    }
    poisoned_masks, poisoned_coords = masks.detach().clone(), coords.clone()
    poisoned_masks[:, :, -3:] = poisoned_coords[:, -3:] = 1e8
    variants["padding"] = new(query, poisoned_masks, padding, point_mass, features, poisoned_coords)
    row["differences"] = {k: float((v - output).abs().max()) for k, v in variants.items()}
    new.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(output.reshape(-1, 2), torch.arange(12) % 2).backward()
    row["gradients"] = {k: float(p.grad.norm()) for k, p in new.named_parameters()}
    row["mask_gradient"] = float(masks.grad.norm())
    assert len(row["gradients"]) == 4 and min(row["gradients"].values()) > 0
    assert all(torch.isfinite(p.grad).all() for p in new.parameters())
    assert row["mask_gradient"] > 0
    assert row["differences"]["repeat"] == row["differences"]["padding"] == 0
    assert max(row["differences"][k] for k in
               ("permutation", "query_permutation", "translation", "rotation", "scale")) < 1e-5
    assert min(row["differences"][k] for k in ("wrong_masks", "wrong_coordinates")) > 0
    rows.append(row)
result = {
    "complete": True, "id": "AdaRES-Y41", "rows": rows, "cpu_cases": 32,
    "w2_sha256": w2_hash, "w2_class_ast_sha256": hashlib.sha256(w2_ast.encode()).hexdigest(),
    "y36_sha256": y36_hash, "readout_sha256": new_hash,
    "w2_max": max(r["w2_difference"] for r in rows),
    "exact_binary_w2_max": max(r["exact_binary_control"]["w2_difference"] for r in rows),
    "y36_max": max(r["y36_difference"] for r in rows),
    "descriptor_min": min(r["descriptor_difference"] for r in rows),
    "new_indicator_min": min(r["new_indicator_difference"] for r in rows),
    "invariance_max": max(r["differences"][k] for r in rows for k in
                          ("permutation", "query_permutation", "translation", "rotation", "scale")),
    "minimum_gradient": min(g for r in rows for g in r["gradients"].values()),
    "limitations": ["Synthetic information separation, not native incidence, efficacy or novelty.",
                    "Constant visual features hold normalized Y36 messages equal; arbitrary native "
                    "visual features may correlate with boundary probability mass."],
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
