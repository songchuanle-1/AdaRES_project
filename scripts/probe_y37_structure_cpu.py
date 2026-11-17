"""Actual-class prospective equal-moment and invariance discrimination."""

import ast
import hashlib
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(4)
assert not torch.cuda.is_available()


def actual(folder, name):
    path = Path("ipdn/model_paper5") / folder / "dec.py"
    source = path.read_bytes()
    node = next(n for n in ast.parse(source.decode("utf-8")).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], hashlib.sha256(source).hexdigest()


cls, sha = actual("adares_y37_pair_distribution", "MaskPairDistributionReadout")
torch.manual_seed(1999)
readout = cls(256)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    q = torch.randn(2, 8, 256)
    masks = torch.randn(2, 8, 16, requires_grad=True)
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[:, -3:] = True
    mass = torch.rand(2, 16) + 1
    tokens = torch.randn(2, 16, 256)
    xyz = torch.randn(2, 16, 3)
    output = readout(q, masks, pad, mass, tokens, xyz)
    perm = torch.arange(15, -1, -1)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    readings = dict(
        repeat=readout(q, masks, pad, mass, tokens, xyz),
        permutation=readout(q, masks[:, :, perm], pad[:, perm], mass[:, perm], tokens[:, perm], xyz[:, perm]),
        query_permutation=readout(q.roll(1, 1), masks.roll(1, 1), pad, mass, tokens, xyz).roll(-1, 1),
        translation=readout(q, masks, pad, mass, tokens, xyz + 8),
        rotation=readout(q, masks, pad, mass, tokens, xyz @ rotation),
        scale=readout(q, masks, pad, mass, tokens, xyz * 2),
        detach=readout(q, masks.detach(), pad, mass, tokens, xyz),
        wrong_masks=readout(q, masks.roll(1, 1), pad, mass, tokens, xyz),
        wrong_coordinates=readout(q, masks, pad, mass, tokens, xyz.roll(1, 1)),
    )
    bad_masks, bad_xyz, bad_mass = masks.detach().clone(), xyz.clone(), mass.clone()
    bad_masks[:, :, -3:] = bad_xyz[:, -3:] = bad_mass[:, -3:] = 1e8
    readings["padding"] = readout(q, bad_masks, pad, bad_mass, tokens, bad_xyz)
    readout.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(output.reshape(-1, 2), torch.arange(16) % 2).backward()
    gradients = {k: float(p.grad.norm()) for k, p in readout.named_parameters()}
    row = dict(seed=seed, differences={k: float((v-output).abs().max()) for k, v in readings.items()},
               mask_gradient=float(masks.grad.norm()), gradients=gradients)
    dif = row["differences"]
    assert dif["repeat"] == dif["detach"] == dif["padding"] == 0
    assert max(dif[k] for k in ("permutation", "query_permutation", "translation", "rotation", "scale")) < 1e-5
    assert min(dif["wrong_masks"], dif["wrong_coordinates"], row["mask_gradient"]) > 0
    assert len(gradients) == 8 and min(gradients.values()) > 0
    assert all(torch.isfinite(p.grad).all() for p in readout.parameters())
    rows.append(row)

torch.manual_seed(1999)
q = torch.randn(1, 2, 256)
mask = torch.zeros(1, 2, 8)
pad = torch.zeros(1, 8, dtype=torch.bool)
mass = torch.ones(1, 8)
tokens = torch.zeros(1, 8, 256)
a = torch.zeros(1, 8, 3)
b = a.clone()
a[0, :, 0] = torch.tensor([-3, -3, -1, -1, 1, 1, 3, 3])
b[0, :, 0] = torch.tensor([-4, -2, 0, 0, 0, 0, 2, 4])
assert torch.equal(a.mean(1), b.mean(1))
assert torch.equal(a.transpose(1, 2) @ a, b.transpose(1, 2) @ b)
left, right = readout(q, mask, pad, mass, tokens, a), readout(q, mask, pad, mass, tokens, b)
delta = float((left-right).abs().max())
assert delta > 0
parent_cls, parent_sha = actual("adares_w2_pointmass_competition", "QueryCompetitionIndicator")
shape_cls, shape_sha = actual("adares_y31_mask_shape", "MaskShapeReadout")
edge_cls, edge_sha = actual("adares_y36_local_adjacency", "MaskLocalAdjacencyReadout")
parent = parent_cls(256)
shape = shape_cls(256)
edge = edge_cls(256)
parent_delta = float((parent(q, mask, pad, mass)-parent(q, mask, pad, mass)).abs().max())
shape_delta = float((shape(q, mask, pad, mass, a)-shape(q, mask, pad, mass, b)).abs().max())
edge_delta = float((edge(q, mask, pad, mass, tokens, a)-edge(q, mask, pad, mass, tokens, b)).abs().max())
assert parent_delta == shape_delta == edge_delta == 0
result = dict(complete=True, id="AdaRES-Y37", cpu_cases=32, cases=rows,
              equal_moment_fixture=dict(mean_equal=True, covariance_equal=True,
                  visual_features_equal=True, masks_equal=True, pair_readout_delta=delta,
                  w2_delta=parent_delta, y31_delta=shape_delta, y36_delta=edge_delta),
              source_sha256=sha, parent_sha256=parent_sha, shape_sha256=shape_sha, edge_sha256=edge_sha,
              parameters=sum(p.numel() for p in readout.parameters()), new_tensors=8,
              minimum_gradient=min(v for r in rows for v in r["gradients"].values()),
              invariance_max=max(r["differences"][k] for r in rows for k in
                  ("permutation", "query_permutation", "translation", "rotation", "scale")),
              limitations=["Synthetic information separation, not native incidence or trained efficacy.",
                           "Uniform-mask fixture includes scene shape; native mask-specific controls still required.",
                           "Superpoint centroid pair distribution is not exact raw-point pair geometry.",
                           "No upstream blindness, semantic necessity or novelty claim."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
