"""Actual-source CPU target sufficiency test, not a trained-benefit test."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import types
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
root = Path(P["root"])
loss_path = root / "ipdn/model_paper5/adares_w2_pointmass_competition/loss.py"
inference_path = root / "ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py"
tree = ast.parse(loss_path.read_text())
methods = {}
for name in ("get_iou", "_get_src_permutation_idx", "loss_indi", "loss_masks"):
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(torch=torch, F=F, Union=__import__("typing").Union)
    if "get_iou" in methods:
        namespace["get_iou"] = methods["get_iou"]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(loss_path), "exec"), namespace)
    methods[name] = namespace[name]
obj = types.SimpleNamespace(eos_coef=0.1)
obj._get_src_permutation_idx = types.MethodType(methods["_get_src_permutation_idx"], obj)
indices = [(torch.tensor([0]), torch.tensor([0]))]
rows = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    logits = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    score = torch.tensor([[0.3], [0.4]], dtype=torch.float64, requires_grad=True)
    # Identical predicted mask mass6; two truth masks have IoU2/3 but FP/FN differ.
    binary = torch.arange(12) < 6
    masks = torch.where(binary, torch.tensor(8.0), torch.tensor(-8.0)).double().repeat(2, 1)
    truths = [(torch.arange(12) < 4).double(), (torch.arange(12) < 9).double()]
    readouts = []
    for truth in truths:
        target = [{"masks": truth.unsqueeze(0)}]
        classification = methods["loss_indi"](obj, logits, target, indices, 1).sum()
        classification_gradient = torch.autograd.grad(classification, logits, retain_graph=True)[0]
        _, _, quality_loss = methods["loss_masks"](obj, [masks], [score], target, indices, 1)
        quality_gradient = torch.autograd.grad(quality_loss.sum(), score, retain_graph=True)[0]
        tp = int((binary & truth.bool()).sum())
        fp = int((binary & ~truth.bool()).sum())
        fn = int((~binary & truth.bool()).sum())
        readouts.append(dict(tp=tp, fp=fp, fn=fn, point_mask_mass=int(binary.sum()),
                             exact_iou=tp / (tp + fp + fn),
                             native_iou=float(methods["get_iou"](masks[:1], truth.unsqueeze(0))[0]),
                             classification_loss=float(classification),
                             classification_gradient=classification_gradient.detach(),
                             quality_gradient=quality_gradient.detach()))
    a, b = readouts
    assert a["exact_iou"] == b["exact_iou"] == 2 / 3
    assert [a[k] for k in ("tp", "fp", "fn")] == [4, 2, 0]
    assert [b[k] for k in ("tp", "fp", "fn")] == [6, 0, 3]
    loss_gap = abs(a["classification_loss"] - b["classification_loss"])
    gradient_gap = float((a["classification_gradient"] - b["classification_gradient"]).abs().max())
    assert loss_gap == gradient_gap == 0
    native_gap = abs(a["native_iou"] - b["native_iou"])
    # This exact source epsilon changes ratio2/3 with denominators6 versus9.
    epsilon_bound = abs(4 / (6 + 1e-6) - 6 / (9 + 1e-6)) + 4 * torch.finfo(torch.float64).eps
    assert native_gap <= epsilon_bound
    order = torch.randperm(12)
    assert float(methods["get_iou"](masks[:1, order], truths[0][order].unsqueeze(0))[0]) == a["native_iou"]
    rows.append(dict(seed=seed, confusion_partitions=[[4, 2, 0], [6, 0, 3]],
                     exact_iou=2 / 3, classification_loss_gap=loss_gap,
                     classification_gradient_gap=gradient_gap,
                     native_iou_gap=native_gap, epsilon_derived_bound=epsilon_bound,
                     scalar_quality_gradient_gap=float((a["quality_gradient"] - b["quality_gradient"]).abs().max())))
predict = next(n for n in ast.walk(ast.parse(inference_path.read_text()))
               if isinstance(n, ast.FunctionDef) and n.name == "predict")
inference_attributes = sorted({n.attr for n in ast.walk(predict) if isinstance(n, ast.Attribute)})
source_report = {
    "loss_indi": "Matched binary classification, not scalar IoU regression.",
    "loss_masks": "Scalar mask-quality regression plus BCE/Dice mask supervision.",
    "prediction": "Final selected query uses positive indicator softmax/argmax, not the scalar scores branch.",
    "premise_correction": "The original possible scalar-confidence premise is false for actual W2 selection; no source or historical evidence was changed.",
}
assert "softmax" in inference_attributes and "argmax" in inference_attributes
occupation = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits"]).decode().strip()
apps = subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits"]).decode().strip()
result = dict(
    complete=True, id="AdaRES-Y45", case_count=32, cases=rows,
    source_report=source_report,
    sources={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in (loss_path, inference_path)},
    classification_loss_max=max(v["classification_loss_gap"] for v in rows),
    classification_gradient_max=max(v["classification_gradient_gap"] for v in rows),
    native_iou_gap_max=max(v["native_iou_gap"] for v in rows),
    gpu_occupation_csv=occupation, compute_processes_csv=apps,
    scientific_status="synthetic_target_distinction_only_native_training_test_pending",
    limitations=[
        "Confusion counts are not inference inputs and this witness does not establish useful learnability.",
        "Matched indicator classification is already affected by native matching; no claim that the backbone is blind.",
        "The fixture corrects an erroneous premise, does not pass controls/native backward/resources/independence.",
        "No model/config/checkpoint mutation, training, validation tuning or new admission.",
    ])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
