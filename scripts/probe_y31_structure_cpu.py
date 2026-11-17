"""Actual Y31 covariance source and deterministic CPU discrimination."""

import ast
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(4)
assert not torch.cuda.is_available()
folder = Path("ipdn/model_paper5/adares_y31_mask_shape")
parent = folder.parent / "adares_w2_pointmass_competition"
new = ast.parse((folder / "dec.py").read_text())
old = ast.parse((parent / "dec.py").read_text())
classes = {n.name: n for n in new.body if isinstance(n, ast.ClassDef)}
original = {n.name: n for n in old.body if isinstance(n, ast.ClassDef)}
assert all(ast.dump(classes[n]) == ast.dump(c)
           for n, c in original.items() if n != "DEC")
head = original["QueryCompetitionIndicator"]
head_forward = next(n for n in head.body
                    if isinstance(n, ast.FunctionDef) and n.name == "forward")
assert [n.arg for n in head_forward.args.args] == [
    "self", "query", "pred_masks", "batch_mask", "point_mass"]
unchanged = []
for name in ("__init__.py", "backbone.py", "ipdn.py", "loss.py", "sample_model.py"):
    a, b = (folder / name).read_bytes(), (parent / name).read_bytes()
    assert a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")
    assert ast.dump(ast.parse(a.decode())) == ast.dump(ast.parse(b.decode()))
    unchanged.append(dict(file=name, candidate_sha256=hashlib.sha256(a).hexdigest(),
                          parent_sha256=hashlib.sha256(b).hexdigest()))
cfg = Path("configs_song/model_paper5/adares_y31_mask_shape_res.yaml").read_bytes()
cfg0 = Path("configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml").read_bytes()
assert cfg.replace(b"\r\n", b"\n") == cfg0.replace(b"\r\n", b"\n")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[classes["MaskShapeReadout"]], type_ignores=[]),
             str(folder / "dec.py"), "exec"), namespace)
torch.manual_seed(1999)
readout = namespace["MaskShapeReadout"](256)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    query = torch.randn(2, 8, 256)
    masks = torch.randn(2, 8, 16, requires_grad=True)
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[:, -3:] = True
    mass = torch.rand(2, 16) + 1
    pos = torch.randn(2, 16, 3)
    output = readout(query, masks, pad, mass, pos)
    perm = torch.arange(15, -1, -1)
    repeated = readout(query, masks, pad, mass, pos)
    point_perm = readout(query, masks[:, :, perm], pad[:, perm], mass[:, perm], pos[:, perm])
    query_perm = readout(query.roll(1, 1), masks.roll(1, 1), pad, mass, pos).roll(-1, 1)
    shifted = readout(query, masks, pad, mass, pos + torch.tensor([3., -2., 4.]))
    shape = readout.descriptor(masks, pad, mass, pos)
    wrong = readout.fusion(torch.cat([
        query, readout.shape_embedding(shape.roll(1, 1))], -1))
    bad_masks, bad_mass, bad_pos = masks.detach().clone(), mass.clone(), pos.clone()
    bad_masks[:, :, -3:] = 1e8
    bad_mass[:, -3:] = 1e8
    bad_pos[:, -3:] = 1e8
    padded = readout(query, bad_masks, pad, bad_mass, bad_pos)
    readout.zero_grad(set_to_none=True)
    F.cross_entropy(output.reshape(-1, 2), torch.arange(16) % 2).backward()
    gradients = {k: float(v.grad.norm()) for k, v in readout.named_parameters()}
    assert all(v.grad is not None and torch.isfinite(v.grad).all()
               for v in readout.parameters())
    assert all(v > 0 for v in gradients.values())
    assert masks.grad is not None and torch.isfinite(masks.grad).all()
    row = dict(seed=seed, repeat=float((output-repeated).abs().max()),
               padding=float((output-padded).abs().max()),
               point_permutation=float((output-point_perm).abs().max()),
               query_permutation=float((output-query_perm).abs().max()),
               translation=float((output-shifted).abs().max()),
               wrong_shape=float((output-wrong).abs().max()),
               gradients=gradients, mask_gradient=float(masks.grad.norm()))
    assert row["repeat"] == row["padding"] == 0
    assert max(row["point_permutation"], row["query_permutation"],
               row["translation"]) < 1e-5
    assert row["wrong_shape"] > 0 and row["mask_gradient"] > 0
    rows.append(row)

points = torch.tensor([[[-2., 0., 0.], [2., 0., 0.],
                        [0., -1., 0.], [0., 1., 0.]]])
masks = torch.tensor([[[20., 20., -20., -20.],
                       [-20., -20., 20., 20.]]])
shape = readout.descriptor(masks, torch.zeros(1, 4, dtype=torch.bool),
                           torch.ones(1, 4), points)
assert not torch.equal(shape[:, 0], shape[:, 1])
result = dict(
    complete=True, id="AdaRES-Y31", cpu_cases=32, cases=rows,
    equal_centroid_equal_mass_shape_difference=float((shape[:, 0]-shape[:, 1]).abs().max()),
    added_parameters=sum(p.numel() for p in readout.parameters()),
    new_trainable_tensors=len(list(readout.parameters())),
    parent_explicit_shape_input_absent=True, unchanged_files=unchanged,
    config_exact_parent_normalized=True,
    source_sha256=hashlib.sha256((folder / "dec.py").read_bytes()).hexdigest(),
    permutation_translation_max=max(max(r["point_permutation"],
        r["query_permutation"], r["translation"]) for r in rows),
    wrong_shape_min=min(r["wrong_shape"] for r in rows),
    limitations=["CPU mechanics only, not native diagnosis, efficacy or necessity.",
                 "Point-mass weighted superpoint-centroid covariance omits within-superpoint variance.",
                 "Orientation retained; rotation invariance is not claimed.",
                 "All six admission gates still need actual native/source/resource evidence."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
