"""Native train-only mask-head gradient feasibility, not a training launch."""

import argparse
import json
import time
from pathlib import Path

import gorilla
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from diagnose_adares_y14_reliability import load_model, scene_indices, weighted_iou
from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_b4_query_competition import IPDN as B4
from ipdn.model_paper5.adares_w2_pointmass_competition import IPDN as W2
from ipdn.utils import get_root_logger


def cosine(a, b):
    denom = a.norm() * b.norm()
    return float((a.flatten() @ b.flatten()) / denom) if denom > 0 else None


def packed_grad(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True,
                                allow_unused=False)
    return torch.cat([g.detach().flatten() for g in grads])


def main():
    parser = argparse.ArgumentParser()
    for key in ("w2-config", "b4-config", "w2-checkpoint", "b4-checkpoint",
                "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--material-diagnosis")
    parser.add_argument("--query-basis", action="store_true")
    parser.add_argument("--material-scans", type=int, default=2)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.w2_config)
    dc = cfg.data.train.copy()
    dc.training, dc.aug, dc.with_elastic, dc.lang_num_max = False, False, False, 32
    dataset = build_dataset(dc, get_root_logger())
    selected = scene_indices(dataset.scene_inputs, 2)
    if args.material_diagnosis:
        diagnosis = json.loads(Path(args.material_diagnosis).read_text())
        material = [r for r in diagnosis["rows"] if r["material"]]
        scans = sorted({r["scan_id"] for r in material})[:args.material_scans]
        selected = []
        for scan in scans:
            manifest = [r for r in material if r["scan_id"] == scan][:4]
            keys = {(str(r["ann_id"]), tuple(r["object_ids"])) for r in manifest}
            index = next(i for i, group in enumerate(dataset.scene_inputs)
                         if group[0]["scene_id"] == scan)
            original = dataset.scene_inputs[index]
            keep = [r for r in original
                    if (str(r["ann_id"]), (int(r["object_id"]),)) in keys]
            if len(keep) != len(manifest):
                raise ValueError("Material expression identity did not resolve")
            anchor_objects = {int(keep[0]["object_id"])}
            control = next((r for r in original
                            if int(r["object_id"]) not in anchor_objects
                            and r not in keep), None)
            if control is not None:
                keep.append(control)
            dataset.scene_inputs[index] = keep
            selected.append(index)
    else:
        # Freeze which expressions will be loaded before examining native outputs.
        for index in selected:
            dataset.scene_inputs[index] = dataset.scene_inputs[index][:4]
    student = load_model(W2, cfg, args.w2_checkpoint)
    teacher = load_model(B4, gorilla.Config.fromfile(args.b4_config),
                         args.b4_checkpoint)
    for p in student.dec.parameters():
        p.requires_grad_(True)
    parameters = [p for name, p in student.dec.named_parameters()
                  if name.startswith("x_mask")]
    if not parameters:
        raise ValueError("No native x_mask tensors")
    native_out = []
    basis_out = []
    teacher_out = []
    hook = student.dec.register_forward_hook(
        lambda module, inputs, output: native_out.append(output))
    basis_hook = student.dec.x_mask.register_forward_hook(
        lambda module, inputs, output: basis_out.append(output))
    teacher_hook = teacher.dec.register_forward_hook(
        lambda module, inputs, output: teacher_out.append(output))
    rows, native = [], []
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for scene, index in enumerate(selected):
        batch = dataset.collate_fn([dataset[index]])
        teacher_out.clear()
        basis_out.clear()
        with torch.no_grad():
            b1 = teacher(batch, mode="predict")["rows"]
            teacher_logits = teacher_out[-1]["masks"].detach().cpu()
            b2 = teacher(batch, mode="predict")["rows"]
        native_out.clear()
        a = student(batch, mode="predict")["rows"]
        out = native_out[-1]
        for expr, (s, t, tr) in enumerate(zip(a, b1, b2)):
            sp_logits = out["masks"][expr]
            mass = s["mass"].to(sp_logits.device)
            points = batch["superpoints"].detach().cpu().long()
            target = torch.bincount(
                points, weights=batch["gt_pmasks"][expr].detach().cpu().gt(0.5).float(),
                minlength=mass.numel()).to(sp_logits.device)
            y = target / mass.clamp_min(1)
            ti = int(t["ious"].argmax())
            left, right = linear_sum_assignment(
                -weighted_iou(s["binary"], t["binary"], s["mass"]).numpy())
            mapping = {int(j): int(i) for i, j in zip(left, right)}
            si = mapping[ti]
            logits = sp_logits[si]
            if args.query_basis:
                raw_feature = basis_out[-1]
                feature = (raw_feature[expr] if raw_feature.ndim == 3
                           else raw_feature[expr*mass.numel():(expr+1)*mass.numel()]
                           ).detach().cpu().double()
                if feature.ndim != 2 or feature.shape[0] != mass.numel():
                    raise ValueError("Native mask basis and point mass shape mismatch")
                weight = mass.detach().cpu().double()
                student_oracle = int(s["ious"].argmax())
                targets = {
                    "teacher": teacher_logits[expr, ti].double(),
                    "shifted_teacher": teacher_logits[expr, ti].double().roll(1),
                    "student_oracle": sp_logits[student_oracle].detach().cpu().double(),
                    "gt_only": (2*y.detach().cpu()-1).double(),
                }
                weighted_features = feature * weight.sqrt()[:, None]
                projection = {}
                for label, value in targets.items():
                    solution = torch.linalg.lstsq(
                        weighted_features, value*weight.sqrt(), driver="gelsd")
                    reconstructed = feature @ solution.solution
                    binary = (reconstructed > 0).float()
                    target_mass = target.detach().cpu()
                    intersection = binary @ target_mass
                    union = binary @ weight.float() + target_mass.sum()-intersection
                    denominator = (weight*value.square()).sum()
                    residual = (weight*(reconstructed-value).square()).sum()
                    projection[label] = {
                        "point_iou": float(intersection/union.clamp_min(1)),
                        "relative_weighted_residual": float(residual/denominator)
                        if denominator > 0 else None,
                        "feature_rank": int(solution.rank),
                    }
                rows.append({
                    "scan_id": s["scan_id"], "object_ids": s["object_ids"],
                    "ann_id": s["ann_id"], "material": float(
                        t["ious"].max()-s["ious"].max()) > 0.05,
                    "student_oracle_iou": float(s["ious"].max()),
                    "teacher_oracle_iou": float(t["ious"].max()),
                    "projection": projection,
                })
                continue
            truth_loss = (F.binary_cross_entropy_with_logits(
                logits, y, reduction="none") * mass).sum() / mass.sum()
            target_gradient = (logits.detach().sigmoid() - y) * mass / mass.sum()
            correct = t["binary"][ti].float().to(logits.device)
            transfer_loss = (F.binary_cross_entropy_with_logits(
                logits, correct, reduction="none") * mass).sum() / mass.sum()
            transfer_gradient = (logits.detach().sigmoid() - correct) * mass / mass.sum()
            gt_param = packed_grad(truth_loss, parameters)
            teacher_param = packed_grad(transfer_loss, parameters)
            wrong = None
            for offset in range(1, len(b1)):
                wt = b1[(expr + offset) % len(b1)]
                if set(wt["object_ids"]) != set(s["object_ids"]):
                    wi = int(wt["ious"].argmax())
                    ll, rr = linear_sum_assignment(
                        -weighted_iou(s["binary"], wt["binary"], s["mass"]).numpy())
                    ws = {int(j): int(i) for i, j in zip(ll, rr)}[wi]
                    wl = sp_logits[ws]
                    wrong_truth_loss = (F.binary_cross_entropy_with_logits(
                        wl, y, reduction="none") * mass).sum() / mass.sum()
                    wrong_loss = (F.binary_cross_entropy_with_logits(
                        wl, wt["binary"][wi].float().to(wl.device),
                        reduction="none") * mass).sum() / mass.sum()
                    wrong_grad = packed_grad(wrong_loss, parameters)
                    wrong_gt = packed_grad(wrong_truth_loss, parameters)
                    wrong = cosine(wrong_grad, wrong_gt)
                    break
            repeat_binary = tr["binary"][int(tr["ious"].argmax())].float().to(logits.device)
            repeat_gradient = (logits.detach().sigmoid()-repeat_binary)*mass/mass.sum()
            rows.append({
                "scan_id": s["scan_id"], "object_ids": s["object_ids"],
                "ann_id": s["ann_id"], "teacher_query": ti, "student_query": si,
                "material": float(t["ious"].max()-s["ious"].max()) > 0.05,
                "logit_gradient_cosine": cosine(target_gradient, transfer_gradient),
                "mask_head_gradient_cosine": cosine(gt_param, teacher_param),
                "teacher_gradient_orthogonal_fraction": max(
                    0, 1-cosine(gt_param, teacher_param)**2)**0.5,
                "wrong_mask_head_gradient_cosine": wrong,
                "correct_minus_wrong_cosine": cosine(gt_param, teacher_param)-wrong
                if wrong is not None else None,
                "teacher_grad_norm": float(teacher_param.norm()),
                "teacher_grad_finite_nonzero": bool(
                    torch.isfinite(teacher_param).all() and teacher_param.norm() > 0),
                "repeat_logit_gradient_max_diff": float(
                    (repeat_gradient-transfer_gradient).abs().max()),
                "target_loss": float(truth_loss.detach()),
                "teacher_loss": float(transfer_loss.detach()),
            })
        native.append({"scan_id": a[0]["scan_id"], "expressions": len(a)})
        del out, a, b1, b2, batch, sp_logits
        native_out.clear()
        student.zero_grad(set_to_none=True)
        print(json.dumps({"scene": scene, "rows": len(rows)}), flush=True)
    hook.remove()
    basis_hook.remove()
    teacher_hook.remove()
    result = {
        "rows": rows, "scans": native, "elapsed_s": time.monotonic()-start,
        "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
        "scope": ("oracle query-basis representability; no inference claim"
                  if args.query_basis else
                  "native decoder autograd probe; no optimizer or tools/train.py"),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
