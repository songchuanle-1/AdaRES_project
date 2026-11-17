"""Actual-source CPU gradient controls; no production model changes."""

import ast
import hashlib
from pathlib import Path
import types
from typing import Union

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
path = Path("ipdn/model_paper5/adares_w2_pointmass_competition/loss.py")
source = path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "836ff790a0a8b9c55cf465a472c9462699f0753a7e87d0ef74a8946edf31565a"
tree = ast.parse(source)
names = {"get_iou", "dice_loss", "loss_masks", "loss_indi", "_get_src_permutation_idx"}
nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes} == names
for node in nodes:
    node.decorator_list = []
namespace = dict(torch=torch, F=F, Union=Union)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
criterion = types.SimpleNamespace(eos_coef=.1)
criterion._get_src_permutation_idx = types.MethodType(namespace["_get_src_permutation_idx"], criterion)
rows = []
for seed in range(32):
    torch.manual_seed(seed)
    masks = torch.randn(2, 9, requires_grad=True)
    scores = torch.randn(2, 1, requires_grad=True)
    indi = torch.randn(1, 2, 2, requires_grad=True)
    target = [{"masks": torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0., 0.]])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    indicator = namespace["loss_indi"](criterion, indi, target, indices, 1)
    grads = torch.autograd.grad(indicator, (indi, masks), allow_unused=True, retain_graph=True)
    detached = namespace["loss_indi"](criterion, indi, [{"masks": target[0]["masks"].detach()}], indices, 1)
    detached_grad = torch.autograd.grad(detached, indi, retain_graph=True)[0]
    losses = namespace["loss_masks"](criterion, [masks], [scores], target, indices, 1)
    quality = namespace["get_iou"](masks[:1], target[0]["masks"])
    positive = F.mse_loss(scores[:1], masks[:1].sigmoid().mean(-1, keepdim=True))
    positive_grad = torch.autograd.grad(positive, masks)[0]
    rows.append(dict(seed=seed, native_indicator_mask_gradient_absent=grads[1] is None,
                     indicator_gradient_active=bool(grads[0].abs().sum() > 0),
                     stop_target_value_difference=float((indicator-detached).abs()),
                     stop_target_gradient_difference=float((grads[0]-detached_grad).abs().max()),
                     actual_binary_iou_requires_grad=quality.requires_grad,
                     positive_control_mask_gradient=float(positive_grad.norm()),
                     all_finite=bool(torch.isfinite(grads[0]).all())))
assert all(r["native_indicator_mask_gradient_absent"] and not r["actual_binary_iou_requires_grad"] for r in rows)
assert all(r["stop_target_value_difference"] == r["stop_target_gradient_difference"] == 0 for r in rows)
assert all(r["indicator_gradient_active"] and r["positive_control_mask_gradient"] > 0 and r["all_finite"] for r in rows)
result = dict(complete=True, id="AdaRES-Y24", cases=rows, source_sha256=hashlib.sha256(source).hexdigest(),
              cases_run=32, native_mask_target_gradient_cases=0,
              detach_loss_max=0, detach_gradient_max=0,
              positive_control_min=min(r["positive_control_mask_gradient"] for r in rows),
              status="disconfirmed: proposed stop-target realization is already a no-op",
              limitations=["Actual source CPU controls, not native population efficacy.",
                           "Indicator targets are binary matching labels; separate score IoU target is already no-grad.",
                           "Does not reject every possible confidence-supervision mechanism."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
