"""Actual custom attention controls for the continuous support intervention."""

import ast
import hashlib
from pathlib import Path
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdn.torch.nn_new import MultiheadAttention

torch.set_num_threads(4)
source_path = Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "6362d882662b0e51594cd4af994359b93f99497affc93600322a34ca6c349b01"
tree = ast.parse(source)
layer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CrossAttentionLayer")
get_mask = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_mask")
namespace = dict(torch=torch, nn=nn, F=F, MultiheadAttention=MultiheadAttention)
exec(compile(ast.fix_missing_locations(ast.Module(body=[layer, get_mask], type_ignores=[])),
             str(source_path), "exec"), namespace)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    attention = namespace["CrossAttentionLayer"](d_model=8, nhead=2, dropout=0).double()
    query = torch.zeros(1, 2, 8, dtype=torch.double)
    query[0, 0, 0] = 1.; query[0, 1, 1] = 1.
    query.requires_grad_(True)
    mask_features = torch.randn(1, 4, 8, dtype=torch.double)
    mask_features[0, :, :2] = torch.tensor([[1., 1.], [-1., -1.], [.5, -.5], [100., 100.]])
    padding = torch.tensor([[False, False, False, True]])
    masks, hard = namespace["get_mask"](types.SimpleNamespace(attn_mask=True), query, mask_features, padding)
    source_features = torch.randn(1, 4, 8, dtype=torch.double)
    bias = F.logsigmoid(masks).masked_fill(padding.unsqueeze(1), float("-inf"))
    hard_out, hard_weights, _ = attention(source_features, query, padding, hard)
    soft_out, soft_weights, _ = attention(source_features, query, padding, bias)
    detached_out, _, _ = attention(source_features, query, padding, bias.detach())
    full_out, full_weights, _ = attention(source_features, query, padding, torch.zeros_like(masks))
    unmasked_out, _, _ = attention(source_features, query, padding, None)
    objective = soft_weights[:, :, 1].sum()
    grad = torch.autograd.grad(objective, masks, retain_graph=True)[0]
    detached_grad = torch.autograd.grad(detached_out.sum(), masks, allow_unused=True, retain_graph=True)[0]
    full_grad = torch.autograd.grad(full_out.sum(), masks, allow_unused=True, retain_graph=True)[0]
    assert hard[0, :, 1].all()
    rows.append(dict(seed=seed, hard_target_reading=float(hard_weights[:, :, 1].abs().max()),
                     soft_target_reading_min=float(soft_weights[:, :, 1].min()),
                     soft_target_mask_gradient=float(grad[:, :, 1].abs().min()),
                     pad_attention_max=float(soft_weights[:, :, 3].abs().max()),
                     pad_mask_gradient_max=float(grad[:, :, 3].abs().max()),
                     detach_forward_max=float((soft_out-detached_out).abs().max()),
                     detach_mask_gradient_absent=detached_grad is None,
                     full_mask_gradient_absent=full_grad is None,
                     full_vs_unmasked_max=float((full_out-unmasked_out).abs().max()),
                     finite=bool(torch.isfinite(grad).all())))
assert all(r["hard_target_reading"] == 0 and r["soft_target_reading_min"] > 0 and
           r["soft_target_mask_gradient"] > 0 and r["pad_attention_max"] == 0 and
           r["pad_mask_gradient_max"] == 0 and r["detach_forward_max"] == 0 and
           r["detach_mask_gradient_absent"] and r["full_mask_gradient_absent"] and
           r["full_vs_unmasked_max"] == 0 and r["finite"] for r in rows)
result = dict(complete=True, id="AdaRES-Y25", cases=rows, cases_run=32,
              hard_target_reading_max=max(r["hard_target_reading"] for r in rows),
              soft_target_reading_min=min(r["soft_target_reading_min"] for r in rows),
              soft_target_gradient_min=min(r["soft_target_mask_gradient"] for r in rows),
              detach_forward_max=0, full_identity_max=0, padding_leak_max=0,
              source_sha256=hashlib.sha256(source).hexdigest(),
              limitations=["Actual native custom attention on synthetic CPU controls, not population benefit.",
                           "Finite log prior does not guarantee float-softmax support at arbitrarily extreme logits.",
                           "Detached-prior control distinguishes the new gradient path; full-valid distinguishes unconditional attention."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
