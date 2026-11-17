"""Discriminate Y34's missing-path premise with the actual W2 indicator."""

import ast
import hashlib
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(4)
assert not torch.cuda.is_available()
path = Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py")
data = path.read_bytes()
tree = ast.parse(data.decode("utf-8"))
node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
            and n.name == "QueryCompetitionIndicator")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    indicator = namespace["QueryCompetitionIndicator"](256)
    query = torch.randn(2, 8, 256)
    masks = torch.randn(2, 8, 16, requires_grad=True)
    padding = torch.zeros(2, 16, dtype=torch.bool)
    padding[:, -3:] = True
    mass = torch.rand(2, 16) + 1
    actual = indicator(query, masks, padding, mass)
    probabilities = masks.sigmoid() * (~padding).unsqueeze(1)
    weighted = probabilities * mass.unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = weighted @ probabilities.transpose(-1, -2)
    overlap = intersection / (
        areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection).clamp_min(1e-6)
    overlap = overlap.masked_fill(torch.eye(8, dtype=torch.bool)[None], 0)
    graph = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    normalized = indicator.query_norm(query)
    context = graph @ normalized
    proposed_evidence = torch.cat(
        [normalized, context, normalized-context, overlap.max(-1).values[..., None]], -1)
    reference = indicator.head(indicator.evidence_norm(proposed_evidence))
    repeat = indicator(query, masks, padding, mass)
    wrong = indicator(query, masks.roll(1, 1), padding, mass)
    joint = indicator(query.roll(1, 1), masks.roll(1, 1), padding, mass).roll(-1, 1)
    actual.square().mean().backward()
    row = dict(seed=seed, reference_max=float((actual-reference).abs().max()),
               repeat_max=float((actual-repeat).abs().max()),
               wrong_mask_max=float((actual-wrong).abs().max()),
               joint_permutation_max=float((actual-joint).abs().max()),
               mask_gradient_norm=float(masks.grad.norm()))
    assert row["reference_max"] < 1e-6 and row["repeat_max"] == 0
    assert row["wrong_mask_max"] > 0 and row["mask_gradient_norm"] > 0
    assert row["joint_permutation_max"] < 1e-5
    rows.append(row)
result = dict(
    complete=True, id="AdaRES-Y34", cpu_cases=32, rows=rows,
    actual_source_sha256=hashlib.sha256(data).hexdigest(),
    existing_path="point-mass soft overlap -> rival context and query-context difference -> indicator",
    proposed_reference_max=max(r["reference_max"] for r in rows),
    repeat_max=max(r["repeat_max"] for r in rows),
    permutation_max=max(r["joint_permutation_max"] for r in rows),
    minimum_mask_response=min(r["wrong_mask_max"] for r in rows),
    minimum_mask_gradient=min(r["mask_gradient_norm"] for r in rows),
    disposition="Missing overlap-conditioned comparison path premise falsified for this realization.",
    limitations=["Not trained harm or numerical identity of every possible relational readout.",
                 "No rejection of nonlinear relation families, no new validation selection.",
                 "Existing X1 and Y8 adverse results remain unchanged; no GPU consumed."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
