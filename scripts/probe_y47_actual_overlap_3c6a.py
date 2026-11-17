"""Disconfirm a missing-overlap premise with actual W2 equal-marginal controls."""

import ast
import hashlib
import json
from pathlib import Path
import torch
from torch import nn

assert P["physical_gpu"] is None
torch.set_num_threads(4)
source = Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py")
tree = ast.parse(source.read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
           and node.name == "QueryCompetitionIndicator")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
mass = torch.ones(1, 8, dtype=torch.double)
padding = torch.zeros(1, 8, dtype=torch.bool)
binary_a = torch.tensor([[1] * 4 + [0] * 4, [0] * 4 + [1] * 4],
                        dtype=torch.bool).repeat(64, 1)
binary_b = torch.stack([torch.tensor([(j - i) % 8 < 4 for j in range(8)])
                        for i in range(8)]).repeat(16, 1)
assert binary_a.shape == binary_b.shape == (128, 8)
assert torch.equal(binary_a.sum(-1), binary_b.sum(-1))
assert torch.equal(binary_a.sum(0), binary_b.sum(0))
masks = [torch.where(binary, 8., -8.).double().unsqueeze(0) for binary in (binary_a, binary_b)]


def parent_context(query, logits):
    probability = logits.sigmoid()
    weighted = probability * mass.unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.einsum("bim,bjm->bij", weighted, probability)
    union = areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection
    overlap = intersection / union.clamp_min(1e-6)
    overlap = overlap.masked_fill(torch.eye(128, dtype=torch.bool).unsqueeze(0), 0)
    competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    return overlap, competition


cases = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    model = namespace["QueryCompetitionIndicator"](256).double()
    query = torch.randn(1, 128, 256, dtype=torch.double)
    with torch.no_grad():
        scores = [model(query, logits, padding, mass) for logits in masks]
        repeated = model(query, masks[0], padding, mass)
        order = torch.arange(7, -1, -1)
        permuted = model(query, masks[0][:, :, order], padding[:, order], mass[:, order])
        matrices = [parent_context(query, logits)[0] for logits in masks]
        reconstructed = []
        for logits in masks:
            overlap, competition = parent_context(query, logits)
            normalized = model.query_norm(query)
            context = torch.einsum("bij,bjd->bid", competition, normalized)
            values = torch.cat([normalized, context, normalized - context,
                                overlap.max(-1).values.unsqueeze(-1)], -1)
            reconstructed.append(model.head(model.evidence_norm(values)))
    assert torch.equal(scores[0], repeated)
    assert all(torch.equal(a, b) for a, b in zip(scores, reconstructed))
    scale = max(1., float(scores[0].abs().max()), float(scores[1].abs().max()))
    prospective = 8 * torch.finfo(torch.double).eps * 128 * scale
    gap = float((scores[0] - scores[1]).abs().max())
    permutation_error = float((scores[0] - permuted).abs().max())
    assert gap > prospective
    assert permutation_error <= prospective
    assert not torch.equal(matrices[0], matrices[1])
    cases.append(dict(seed=seed, actual_W2_score_difference=gap,
                      prospective8eps_queries_bound=prospective,
                      exact_cached_repeat=True, exact_actual_formula_reconstruction=True,
                      joint_superpoint_permutation_error=permutation_error,
                      equal_row_mass=True, equal_point_coverage=True,
                      actual_overlap_matrix_difference=float((matrices[0] - matrices[1]).abs().max())))
result = dict(
    complete=True, id="AdaRES-Y47", cases=cases, CPU_cases=32, queries=128,
    actual_W2_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    actual_W2_class_source=ast.get_source_segment(source.read_text(), cls),
    all_actual_parent_distinguish_equal_marginal_fixtures=True,
    minimum_actual_W2_score_difference=min(c["actual_W2_score_difference"] for c in cases),
    maximum_joint_permutation_error=max(c["joint_superpoint_permutation_error"] for c in cases),
    maximum_prospective_bound=max(c["prospective8eps_queries_bound"] for c in cases),
    proposed_missing_context_premise="disconfirmed_by_actual_parent",
    scientific_status="rejected_missing_information_realization_not_all_pairwise_hypotheses",
    limitations=["Actual W2 already computes the point-mass pairwise intersection/union/competitor context.",
                 "Hard versus soft overlap alone would change numerics, not supply the claimed missing correspondence path.",
                 "Fixed diagnostic mask logits are not deployed or chosen using validation.",
                 "Controlled actual-class differences are not population efficacy.",
                 "No new model/config/checkpoint/loss/admission/queue or GPU operation."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
