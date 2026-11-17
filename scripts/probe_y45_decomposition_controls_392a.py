"""Actual B2/G1/C1 objective controls versus three-category target gradients."""

import ast
import hashlib
import json
from pathlib import Path
import types
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
root = Path(P["root"])
modules = {
    "b2": "adares_b2_metric_indicator",
    "g1": "adares_g1_metric_targets",
    "c1": "adares_c1_listwise_expected_mask",
}
methods, sources = {}, {}
for key, folder in modules.items():
    path = root / "ipdn/model_paper5" / folder / "loss.py"
    sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree = ast.parse(path.read_text())
    names = ("_get_src_permutation_idx", "loss_indi") if key in ("b2", "g1") else ("_query_iou", "loss_listwise")
    obj = types.SimpleNamespace(eos_coef=.1, metric_target=True, target_temperature=.05,
                                rank_temperature=.1)
    for name in names:
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        static = name == "_query_iou"
        assert (len(node.decorator_list) == 1 and node.decorator_list[0].id == "staticmethod") if static else not node.decorator_list
        node.decorator_list = []
        namespace = dict(torch=torch, F=F)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        setattr(obj, name, namespace[name] if static else types.MethodType(namespace[name], obj))
    methods[key] = obj

rows = []
indices = [(torch.tensor([0]), torch.tensor([0]))]
for seed in range(32):
    torch.manual_seed(1999 + seed)
    binary = torch.arange(12) < 6
    masks = torch.where(binary, torch.tensor(8.), torch.tensor(-8.)).repeat(128, 1)
    truths = [(torch.arange(12) < 4).float(), (torch.arange(12) < 9).float()]
    logits2 = torch.randn(1, 128, 2, requires_grad=True)
    logits3 = torch.randn(1, 128, 3, requires_grad=True)
    captured = []
    for truth in truths:
        target = [{"masks": truth.unsqueeze(0)}]
        outputs = {}
        for key in ("b2", "g1"):
            loss = methods[key].loss_indi(logits2, [masks], target, indices, torch.tensor([1.])).sum()
            outputs[key] = (float(loss), torch.autograd.grad(loss, logits2, retain_graph=True)[0])
        c1 = methods["c1"].loss_listwise(logits2, [masks], target).sum()
        outputs["c1"] = (float(c1), torch.autograd.grad(c1, logits2, retain_graph=True)[0])
        confusion = torch.tensor([(binary & truth.bool()).sum(),
                                  (binary & ~truth.bool()).sum(),
                                  (~binary & truth.bool()).sum()], dtype=torch.float32)
        target3 = confusion / confusion.sum()
        loss3 = -(target3 * logits3.log_softmax(-1)).sum(-1).mean()
        gradient3 = torch.autograd.grad(loss3, logits3, retain_graph=True)[0]
        # Collapsing FP/FN recovers the same scalar-IoU supervision.
        collapsed = torch.stack((torch.logsumexp(logits3[..., 1:], -1), logits3[..., 0]), -1)
        collapsed_target = torch.stack((1 - target3[0], target3[0]))
        loss2 = -(collapsed_target * collapsed.log_softmax(-1)).sum(-1).mean()
        gradient2 = torch.autograd.grad(loss2, logits3, retain_graph=True)[0]
        captured.append((outputs, target3, float(loss3), gradient3, float(loss2), gradient2))
    first, second = captured
    gaps = {key: dict(loss=abs(first[0][key][0] - second[0][key][0]),
                      gradient=float((first[0][key][1] - second[0][key][1]).abs().max()))
            for key in modules}
    assert all(v["loss"] == v["gradient"] == 0 for v in gaps.values())
    new_gap = float((first[3] - second[3]).abs().max())
    assert new_gap > 0 and torch.isfinite(first[3]).all() and torch.isfinite(second[3]).all()
    assert first[4] == second[4] and torch.equal(first[5], second[5])
    permutation = torch.randperm(128)
    permuted = -(first[1] * logits3[:, permutation].log_softmax(-1)).sum(-1).mean()
    permutation_gradient = torch.autograd.grad(permuted, logits3, retain_graph=True)[0]
    assert torch.equal(permutation_gradient, first[3])
    rows.append(dict(seed=seed, closed_objective_gaps=gaps,
                     confusion_targets=[first[1].tolist(), second[1].tolist()],
                     decomposition_gradient_gap=new_gap,
                     collapsed_scalar_gradient_gap=0, query_permutation_gradient_gap=0))
result = dict(
    complete=True, id="AdaRES-Y45", rows=rows, cases=32, source_hashes=sources,
    minimum_decomposition_gradient_gap=min(r["decomposition_gradient_gap"] for r in rows),
    all_closed_actual_loss_gradient_gaps_zero=True,
    scientific_status="controlled_objective_difference_only_not_native_new_path",
    limitations=["Actual historical B2/G1/C1 functions, generated equal-IoU witnesses.",
                 "A three-way CE target has extra conditional FP/FN supervision, not demonstrated utility.",
                 "No new model, new-tensor native backward, independence, resources or admission.",
                 "No human novelty judgment or reopened Y29 ranking realization.",
                 "Query permutation gradient equality tested; reduction loss may differ by float roundoff."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
