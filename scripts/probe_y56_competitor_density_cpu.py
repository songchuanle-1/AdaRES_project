"""Replay the exact native indicator on fixed CPU-only multiplicity controls."""
import hashlib
import json
import os
import subprocess

import torch
import torch.nn as nn

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert subprocess.check_output(["hostname"]).decode().strip() == "dl01-MS-7C60"
torch.set_num_threads(1)
namespace = {"torch": torch, "nn": nn}
exec(compile(PAYLOAD["class_source"], "native_QueryCompetitionIndicator", "exec"), namespace)
Indicator = namespace["QueryCompetitionIndicator"]


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def reference(model, query, masks, mass, inverse_counts):
    probabilities = masks.sigmoid()
    weighted = probabilities * mass.unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.einsum("bim,bjm->bij", weighted, probabilities)
    union = areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection
    overlap = intersection / union.clamp_min(1e-6)
    diagonal = torch.eye(query.shape[1], dtype=torch.bool).unsqueeze(0)
    overlap = overlap.masked_fill(diagonal, 0.0)
    adjusted = overlap * inverse_counts.view(1, 1, -1)
    competition = adjusted / adjusted.sum(-1, keepdim=True).clamp_min(1e-6)
    normalized = model.query_norm(query)
    context = torch.einsum("bij,bjd->bid", competition, normalized)
    evidence = torch.cat([normalized, context, normalized-context,
                          overlap.max(-1).values.unsqueeze(-1)], dim=-1)
    return model.head(model.evidence_norm(evidence)), context


def native(model, query, masks, mass):
    capture = []
    hook = model.evidence_norm.register_forward_pre_hook(
        lambda module, args: capture.append(args[0].detach().clone()))
    try:
        output = model(query, masks, torch.zeros((1, masks.shape[-1]), dtype=torch.bool), mass)
    finally:
        hook.remove()
    return output, capture[0][:, :, 256:512]


def arm(model, base_query, base_masks, mass, counts):
    groups = torch.repeat_interleave(torch.arange(3), torch.tensor(counts))
    query_leaf = base_query.detach().clone().requires_grad_(True)
    query = query_leaf.index_select(0, groups).unsqueeze(0)
    masks = base_masks.index_select(0, groups).unsqueeze(0)
    output, context = native(model, query, masks, mass)
    repeated, repeated_context = native(model, query, masks, mass)
    permutation = torch.arange(127, -1, -1)
    inverse = torch.argsort(permutation)
    permuted, _ = native(model, query[:, permutation], masks[:, permutation], mass)
    inverse_counts = 1.0 / torch.tensor(counts, dtype=query.dtype)[groups]
    corrected, corrected_context = reference(model, query, masks, mass, inverse_counts)
    unchanged, _ = reference(model, query, masks, mass, torch.ones(128, dtype=query.dtype))
    gradient, = torch.autograd.grad(output[0, 0, 1]-output[0, 0, 0], query_leaf)
    eps = torch.finfo(query.dtype).eps
    scale = 1 + max(float(output.detach().abs().max()), float(context.abs().max()))
    numerical_bound = 128 * eps * scale
    return dict(
        output=output.detach(), context=context.detach(), corrected=corrected.detach(),
        corrected_context=corrected_context.detach(),
        focal_query=query[0, 0].detach(), focal_mask=masks[0, 0].detach(),
        repeat_error=float((output-repeated).detach().abs().max()),
        repeat_context_error=float((context-repeated_context).abs().max()),
        permutation_error=float((output-permuted[:, inverse]).detach().abs().max()),
        reference_identity_error=float((output-unchanged).detach().abs().max()),
        gradient_finite=bool(torch.isfinite(gradient).all()),
        gradient_norm=float(gradient.norm()), numerical_bound=numerical_bound,
    )


rows = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999)
    model = Indicator(256).to(dtype=dtype).eval()
    original_state = state_hash(model)
    q = torch.randn(3, 256, dtype=dtype)
    coordinate = torch.linspace(-1.5, 1.5, 64, dtype=dtype)
    masks = torch.stack([coordinate, coordinate.flip(0)+0.4, coordinate*0.5-0.2])
    mass = (1 + torch.arange(64) % 7).to(dtype=dtype).unsqueeze(0)
    for fixture in ("contrastive_competitors", "identical_competitor_features_negative"):
        base_query = q.clone()
        if fixture.endswith("negative"):
            base_query[2] = base_query[1]
        left = arm(model, base_query, masks, mass, [1, 63, 64])
        right = arm(model, base_query, masks, mass, [1, 64, 63])
        unique, unique_context = native(model, base_query.unsqueeze(0), masks.unsqueeze(0), mass)
        bound = max(left["numerical_bound"], right["numerical_bound"])
        row = dict(
            fixture=fixture, dtype=str(dtype), query_count=128, feature_dim=256,
            counts_before=[1, 63, 64], counts_after=[1, 64, 63],
            focal_query_equal=bool(torch.equal(left["focal_query"], right["focal_query"])),
            focal_mask_equal=bool(torch.equal(left["focal_mask"], right["focal_mask"])),
            native_focal_logit_delta=float((left["output"][:, 0]-right["output"][:, 0]).abs().max()),
            native_focal_context_delta=float((left["context"][:, 0]-right["context"][:, 0]).abs().max()),
            corrected_focal_logit_delta=float((left["corrected"][:, 0]-right["corrected"][:, 0]).abs().max()),
            corrected_unique_reference_error=max(
                float((a["corrected"][:, 0]-unique[:, 0]).abs().max()) for a in (left, right)),
            corrected_context_unique_reference_error=max(
                float((a["corrected_context"][:, 0]-unique_context[:, 0]).abs().max()) for a in (left, right)),
            repeat_error=max(left["repeat_error"], right["repeat_error"]),
            permutation_error=max(left["permutation_error"], right["permutation_error"]),
            identity_reference_error=max(left["reference_identity_error"], right["reference_identity_error"]),
            numerical_bound=bound,
            component_backward_finite=left["gradient_finite"] and right["gradient_finite"],
            component_gradient_norms=[left["gradient_norm"], right["gradient_norm"]],
            parameter_state_unchanged=state_hash(model)==original_state,
        )
        assert row["focal_query_equal"] and row["focal_mask_equal"]
        assert row["component_backward_finite"] and row["parameter_state_unchanged"]
        assert max(row["repeat_error"], row["permutation_error"], row["identity_reference_error"],
                   row["corrected_unique_reference_error"],
                   row["corrected_context_unique_reference_error"]) <= bound
        if fixture.endswith("negative"):
            assert row["native_focal_logit_delta"] <= bound
        rows.append(row)

active = [r for r in rows if r["fixture"] == "contrastive_competitors"]
signal = all(r["native_focal_context_delta"] > r["numerical_bound"]
             and r["native_focal_logit_delta"] > r["numerical_bound"] for r in active)
print(json.dumps(dict(
    complete=True, id="AdaRES-Y56", component="exact committed native indicator class",
    torch_version=torch.__version__, device="cpu", cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
    threads=torch.get_num_threads(), structural_fixture_signal=signal, rows=rows,
    formal_model_files=0, training_launches=0, gpu_tests=0, full_native_backward=False,
    limitations=[
        "Fixed counterfactual CPU tensors and a randomly initialized native head, not actual TRAIN incidence or trained efficacy.",
        "Only focal singleton query readout is tested; the inverse-count reference is not a formally implemented model or all-query theorem.",
        "Unique-three-query reference is diagnostic only; both compared multiplicity arms retain the native128-query budget.",
        "Numerical bounds derive from dtype epsilon and128-term arithmetic, not a pre-training task-gain threshold.",
        "No sparse backbone, real FPS, matcher, optimizer/restore/resource gate, checkpoint or novel-method claim.",
        "No changes to Y48 ownership targets, Y52 score targets, Y54 support masks or a running experiment.",
    ],
)))
