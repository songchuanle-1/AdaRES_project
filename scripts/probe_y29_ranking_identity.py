"""Compare a preregistered regret objective to actual historical B0 code."""

import ast
import hashlib
from pathlib import Path
import types

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
marker = Path(".codex_source_commit").read_text().strip()
assert marker == "ce2dc75ee9e706da68c5768d9b3c855f33bda7fb"
paths = {
    "parent": Path("ipdn/model_paper5/adares_w2_pointmass_competition/loss.py"),
    "ranking": Path("ipdn/model_paper5/adares_b0/loss.py"),
}
sources = {}
methods = {}
for name, path in paths.items():
    source = path.read_bytes()
    sources[name] = dict(path=str(path), sha256=hashlib.sha256(source).hexdigest())
    tree = ast.parse(source.decode("utf-8"))
    wanted = "loss_indi" if name == "parent" else "loss_iou_ranking"
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == wanted)
    namespace = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    methods[name] = namespace[wanted]
parent_method = next(n for n in ast.walk(ast.parse(paths["parent"].read_text()))
                     if isinstance(n, ast.FunctionDef) and n.name == "loss_indi")
assert any(isinstance(n, ast.Attribute) and n.attr == "log_softmax"
           for n in ast.walk(parent_method))
assert not any(isinstance(n, ast.Attribute) and n.attr == "mse_loss"
               for n in ast.walk(parent_method))


def proposed(logits, masks, target):
    binary = masks.sigmoid() > .5
    truth = target > .5
    quality = (binary & truth).sum(-1).double() / (binary | truth).sum(-1).clamp_min(1).double()
    score = logits.softmax(-1)[..., 1]
    delta = quality[:, None] - quality[None, :]
    active = delta.abs() > 0
    pair = F.softplus(-(score[:, None] - score[None, :]) * delta.sign())
    if not bool(active.any()):
        return score.sum() * 0
    return (pair * delta.abs())[active].sum() / delta.abs()[active].sum().clamp_min(1e-6)


torch.manual_seed(1999)
obj = types.SimpleNamespace(rank_pair_margin=0.0)
rows = []
for case in range(32):
    logits = torch.randn(1, 8, 2, dtype=torch.float64, requires_grad=True)
    masks = torch.randn(8, 17, dtype=torch.float64)
    truth = torch.arange(17).remainder(3).eq(0).double().unsqueeze(0)
    if case == 0:
        masks[:] = -1
    native = methods["ranking"](obj, logits, [masks], truth, torch.ones_like(truth, dtype=torch.bool)).sum()
    candidate = proposed(logits[0], masks, truth[0])
    old_grad = torch.autograd.grad(native, logits, retain_graph=True)[0]
    new_grad = torch.autograd.grad(candidate, logits)[0]
    rows.append(dict(case=case, loss_max=float((native-candidate).abs()),
                     gradient_max=float((old_grad-new_grad).abs().max()),
                     tie=case == 0))
assert max(r["loss_max"] for r in rows) == 0
assert max(r["gradient_max"] for r in rows) == 0
result = dict(complete=True, id="AdaRES-Y29", cases=rows, case_count=32,
              sources=sources, source_marker=marker,
              parent_is_matched_classification_not_absolute_quality_regression=True,
              proposed_regret_equals_historical_b0_on_equal_point_masses=True,
              loss_max=max(r["loss_max"] for r in rows),
              gradient_max=max(r["gradient_max"] for r in rows),
              scientific_disposition="Reject this realized renamed ranking objective; not a universal ban on structured confidence training.",
              limitations=["No training, validation, inference change or numerical search.",
                           "Only tested objective and source premise are rejected; future distinct causal objectives are not ruled out."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
