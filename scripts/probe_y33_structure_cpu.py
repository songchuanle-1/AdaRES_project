"""Actual Y33 class discrimination against equal first-moment supports."""

import ast
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(4)
assert not torch.cuda.is_available()
assert Path(".codex_source_commit").read_text().strip() == "5f2489de78196e8d89c004e452a665c0946f2a94"
folder = Path("ipdn/model_paper5/adares_y33_token_attention")
parent = folder.parent / "adares_w2_pointmass_competition"
new = ast.parse((folder / "dec.py").read_text())
old = ast.parse((parent / "dec.py").read_text())
classes = {n.name: n for n in new.body if isinstance(n, ast.ClassDef)}
original = {n.name: n for n in old.body if isinstance(n, ast.ClassDef)}
assert all(ast.dump(classes[n]) == ast.dump(c) for n, c in original.items() if n != "DEC")
unchanged = []
for name in ("__init__.py", "backbone.py", "ipdn.py", "loss.py", "sample_model.py"):
    a, b = (folder / name).read_bytes(), (parent / name).read_bytes()
    assert a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")
    unchanged.append(dict(file=name, candidate_sha256=hashlib.sha256(a).hexdigest(),
                          parent_sha256=hashlib.sha256(b).hexdigest()))
cfg = Path("configs_song/model_paper5/adares_y33_token_attention_res.yaml").read_bytes()
cfg0 = Path("configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml").read_bytes()
assert cfg.replace(b"\r\n", b"\n") == cfg0.replace(b"\r\n", b"\n")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[classes["MaskTokenAttentionReadout"]], type_ignores=[]),
             str(folder / "dec.py"), "exec"), namespace)
torch.manual_seed(1999)
readout = namespace["MaskTokenAttentionReadout"](256)
assert len(list(readout.parameters())) == 7
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    query = torch.randn(2, 8, 256)
    masks = torch.randn(2, 8, 16, requires_grad=True)
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[:, -3:] = True
    mass = torch.rand(2, 16) + 1
    tokens = torch.randn(2, 16, 256, requires_grad=True)
    output = readout(query, masks, pad, mass, tokens)
    perm = torch.arange(15, -1, -1)
    repeated = readout(query, masks, pad, mass, tokens)
    support_perm = readout(query, masks[:, :, perm], pad[:, perm], mass[:, perm], tokens[:, perm])
    query_perm = readout(query.roll(1, 1), masks.roll(1, 1), pad, mass, tokens).roll(-1, 1)
    bad_masks, bad_mass, bad_tokens = masks.detach().clone(), mass.clone(), tokens.detach().clone()
    bad_masks[:, :, -3:] = 1e8
    bad_mass[:, -3:] = 1e8
    bad_tokens[:, -3:] = 1e8
    padded = readout(query, bad_masks, pad, bad_mass, bad_tokens)
    wrong = readout(query, masks, pad, mass, tokens.roll(1, 1))
    detached = readout(query, masks.detach(), pad, mass, tokens)
    readout.zero_grad(set_to_none=True)
    F.cross_entropy(output.reshape(-1, 2), torch.arange(16) % 2).backward()
    gradients = {k: float(v.grad.norm()) for k, v in readout.named_parameters()}
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in readout.parameters())
    assert all(v > 0 for v in gradients.values())
    assert masks.grad is not None and torch.isfinite(masks.grad).all()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    row = dict(seed=seed, repeat=float((output-repeated).abs().max()),
               padding=float((output-padded).abs().max()),
               support_permutation=float((output-support_perm).abs().max()),
               query_permutation=float((output-query_perm).abs().max()),
               mask_detach=float((output-detached).abs().max()),
               wrong_tokens=float((output-wrong).abs().max()),
               gradients=gradients, mask_gradient=float(masks.grad.norm()),
               token_gradient=float(tokens.grad.norm()))
    assert row["repeat"] == row["padding"] == row["mask_detach"] == 0
    assert max(row["support_permutation"], row["query_permutation"]) < 1e-5
    assert min(row["wrong_tokens"], row["mask_gradient"], row["token_gradient"]) > 0
    rows.append(row)

tokens_a = torch.zeros(1, 4, 256)
tokens_a[0, 0, 0], tokens_a[0, 1, 0] = 1., -1.
tokens_a[0, 2, 2], tokens_a[0, 3, 2] = 1., -1.
tokens_b = tokens_a.clone()
tokens_b[0, :2] = 0
tokens_b[0, 0, 1], tokens_b[0, 1, 1] = 1., -1.
masks = torch.tensor([[[20., 20., -20., -20.]]])
pad = torch.zeros(1, 4, dtype=torch.bool)
mass = torch.ones(1, 4)
query = torch.randn(1, 1, 256)
means = lambda x: (torch.bmm(masks.sigmoid(), x), torch.bmm((-masks).sigmoid(), x))
assert all(torch.equal(a, b) for a, b in zip(means(tokens_a), means(tokens_b)))
distribution_delta = float((readout(query, masks, pad, mass, tokens_a)
                            - readout(query, masks, pad, mass, tokens_b)).abs().max())
assert distribution_delta > 0
result = dict(complete=True, id="AdaRES-Y33", cpu_cases=32, cases=rows,
              equal_fg_bg_first_moments=True, distribution_control_delta=distribution_delta,
              added_parameters=sum(p.numel() for p in readout.parameters()),
              new_trainable_tensors=7, unchanged_files=unchanged,
              config_exact_parent_normalized=True,
              source_sha256=hashlib.sha256((folder / "dec.py").read_bytes()).hexdigest(),
              permutation_max=max(max(r["support_permutation"], r["query_permutation"]) for r in rows),
              limitations=["CPU mechanism discrimination only; native incidence/gates and efficacy pending.",
                           "Y28 first-moment realization stays rejected; upstream attention exists.",
                           "No novelty/necessity/admission/trained benefit claim or numerical search."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
