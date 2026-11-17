"""Actual independent temporal source and fixed CPU module controls."""

import ast
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(4)
assert not torch.cuda.is_available()
folder = Path("ipdn/model_paper5/adares_y30_query_trajectory")
parent = folder.parent/"adares_w2_pointmass_competition"
new = ast.parse((folder/"dec.py").read_text())
old = ast.parse((parent/"dec.py").read_text())
classes = {n.name: n for n in new.body if isinstance(n, ast.ClassDef)}
original = {n.name: n for n in old.body if isinstance(n, ast.ClassDef)}
assert all(ast.dump(classes[n]) == ast.dump(c) for n, c in original.items() if n != "DEC")
checks = []
for name in ("__init__.py", "backbone.py", "ipdn.py", "loss.py", "sample_model.py"):
    a, b = (folder/name).read_bytes(), (parent/name).read_bytes()
    assert a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")
    assert ast.dump(ast.parse(a.decode())) == ast.dump(ast.parse(b.decode()))
    checks.append(dict(file=name, candidate_sha256=hashlib.sha256(a).hexdigest(),
                       parent_sha256=hashlib.sha256(b).hexdigest(), ast_exact=True))
cfg = Path("configs_song/model_paper5/adares_y30_query_trajectory_res.yaml").read_bytes()
cfg0 = Path("configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml").read_bytes()
assert cfg.replace(b"\r\n", b"\n") == cfg0.replace(b"\r\n", b"\n")
namespace = dict(torch=torch, nn=nn, F=F)
exec(compile(ast.Module(body=[classes["QueryMaskTrajectoryReadout"]], type_ignores=[]),
             str(folder/"dec.py"), "exec"), namespace)
torch.manual_seed(1999)
readout = namespace["QueryMaskTrajectoryReadout"](256)
assert sum(p.numel() for p in readout.parameters()) == 397314
assert len(list(readout.parameters())) == 8
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    queries = [torch.randn(2, 8, 256) for _ in range(7)]
    masks = [torch.randn(2, 8, 16, requires_grad=True) for _ in range(7)]
    padding = torch.zeros(2, 16, dtype=torch.bool)
    padding[:, -3:] = True
    mass = torch.rand(2, 16)+1
    output = readout(queries, masks, padding, mass)
    repeated = readout(queries, masks, padding, mass)
    rolled = readout([v.roll(1, 1) for v in queries[:-1]]+queries[-1:],
                     [v.roll(1, 1) for v in masks[:-1]]+masks[-1:], padding, mass)
    reversed_ = readout(list(reversed(queries[:-1]))+queries[-1:],
                        list(reversed(masks[:-1]))+masks[-1:], padding, mass)
    terminal = readout([queries[-1]]*7, [masks[-1]]*7, padding, mass)
    perm = torch.arange(15, -1, -1)
    sp_permuted = readout(queries, [v[:, :, perm] for v in masks], padding[:, perm], mass[:, perm])
    q_permuted = readout([v.roll(1, 1) for v in queries], [v.roll(1, 1) for v in masks], padding, mass).roll(-1, 1)
    bad = [v.detach().clone() for v in masks]
    for v in bad:
        v[:, :, -3:] = 1e8
    bad_mass = mass.clone()
    bad_mass[:, -3:] = 1e8
    padded = readout(queries, bad, padding, bad_mass)
    readout.zero_grad(set_to_none=True)
    F.cross_entropy(output.reshape(-1, 2), torch.arange(16)%2).backward()
    grads = {k: float(p.grad.norm()) for k, p in readout.named_parameters()}
    assert len(grads) == 8 and all(v > 0 for v in grads.values())
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in masks)
    row = dict(seed=seed, repeat=float((output-repeated).abs().max()),
               padding=float((output-padded).abs().max()),
               sp_permutation=float((output-sp_permuted).abs().max()),
               query_permutation=float((output-q_permuted).abs().max()),
               wrong_history=float((output-rolled).abs().max()),
               reverse_history=float((output-reversed_).abs().max()),
               repeated_terminal=float((output-terminal).abs().max()), gradients=grads,
               minimum_stage_mask_gradient=min(float(v.grad.norm()) for v in masks))
    assert row["repeat"] == row["padding"] == 0
    assert max(row["sp_permutation"], row["query_permutation"]) < 1e-5
    assert min(row["wrong_history"], row["reverse_history"], row["repeated_terminal"]) > 0
    rows.append(row)
result = dict(complete=True, id="AdaRES-Y30", cases=rows, cpu_cases=32,
              structural_new_ordered_state_mask_readout=True,
              config_exact_parent_normalized=True, unchanged_files=checks,
              source_sha256=hashlib.sha256((folder/"dec.py").read_bytes()).hexdigest(),
              added_parameters=397314, new_trainable_tensors=8,
              repeat_max=max(r["repeat"] for r in rows), padding_max=max(r["padding"] for r in rows),
              permutation_max=max(max(r["sp_permutation"], r["query_permutation"]) for r in rows),
              wrong_history_min=min(r["wrong_history"] for r in rows),
              reverse_history_min=min(r["reverse_history"] for r in rows),
              terminal_repeat_min=min(r["repeated_terminal"] for r in rows),
              limitations=["Actual candidate class, not synthetic efficacy or necessity.",
                           "Parent/current head/source/loss unchanged; native full-model initialization/backward checked separately.",
                           "No validation or numerical search."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
