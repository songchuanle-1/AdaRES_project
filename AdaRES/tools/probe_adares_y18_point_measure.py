"""Frozen analytic and native point-measure controls; never launches training."""

import argparse
import copy
import gc
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
import types

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "y18_point_measure", ROOT / "ipdn/model_paper5/adares_y18_point_measure/point_measure.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, result):
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def cosine(a, b):
    den = a.norm() * b.norm()
    return float((a.flatten() @ b.flatten()) / den) if den > 0 else None


def old_losses(logits, labels):
    target = labels.to(logits).expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dice = (1 - (2 * (p * target).sum(-1) + 1)
            / (p.sum(-1) + target.sum(-1) + 1)).mean()
    return bce, dice


def algebra():
    torch.manual_seed(1999)
    cases, max_value, max_grad = 0, 0.0, 0.0
    equal_unit = 0.0
    mass_differences, label_differences = [], []
    for count_tuple in itertools.product((1, 2, 4), repeat=3):
        counts = torch.tensor(count_tuple, dtype=torch.float64)
        sp = torch.repeat_interleave(torch.arange(3), counts.long())
        for positive_tuple in itertools.product(
                *(range(c + 1) for c in count_tuple)):
            positive = torch.tensor(positive_tuple, dtype=torch.float64)
            labels = torch.cat([
                torch.cat((torch.ones(p), torch.zeros(c - p)))
                for c, p in zip(count_tuple, positive_tuple)]).double()
            z = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
            compressed = pm.point_losses(z, counts, positive)
            expanded_z = z[:, sp]
            expanded_bce = F.binary_cross_entropy_with_logits(
                expanded_z, labels.expand_as(expanded_z))
            p = expanded_z.sigmoid()
            expanded_dice = (1 - (2 * (p * labels).sum(-1) + 1)
                             / (p.sum(-1) + labels.sum() + 1)).mean()
            expanded = expanded_bce, expanded_dice
            for a, b in zip(compressed, expanded):
                delta = float((a - b).abs())
                ga = torch.autograd.grad(a, z, retain_graph=True)[0]
                gb = torch.autograd.grad(b, z, retain_graph=True)[0]
                gd = float((ga - gb).abs().max())
                max_value, max_grad = max(max_value, delta), max(max_grad, gd)
                if delta > 1e-12 or gd > 1e-12:
                    raise RuntimeError("Point expansion mismatch")
            native = old_losses(z, positive / counts >= 0.5)
            if count_tuple == (1, 1, 1):
                equal_unit = max(equal_unit, *(float((a - b).abs())
                                               for a, b in zip(native, compressed)))
            elif len(set(count_tuple)) > 1:
                mass_differences.append(float((native[0] - compressed[0]).abs()))
            if any(0 < p < c for c, p in zip(count_tuple, positive_tuple)):
                label_differences.append(float((native[0] - compressed[0]).abs()))
            cases += 1
    invalid = 0
    if equal_unit > 1e-12:
        raise RuntimeError("Unit-mass unchanged control mismatch")
    for counts, positives in (
            ([0, 1], [0, 0]), ([1, 1], [2, 0]), ([1, 1], [-1, 0])):
        try:
            pm.point_losses(torch.zeros(1, 2),
                            torch.tensor(counts), torch.tensor(positives))
        except ValueError:
            invalid += 1
    sp = torch.tensor([0, 0, 1, 2, 2, 3])
    truths = [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0]),
              torch.tensor([1, 1, 0])]
    counts, pos = pm.batch_point_measure(
        sp, torch.tensor([0, 2, 4]), torch.tensor([2, 1]), truths,
        [[1], [2], [3]])
    if [c.tolist() for c in counts] != [[2, 1], [2, 1], [2, 1]]:
        raise RuntimeError("Scene counts alignment failure")
    if [p.tolist() for p in pos] != [[1, 1], [1, 0], [2, 0]]:
        raise RuntimeError("Scene positives alignment failure")
    return {"complete": True, "cases": cases, "max_value_error": max_value,
            "max_gradient_error": max_grad, "unit_mass_control_error": equal_unit,
            "max_unequal_mass_bce_change": max(mass_differences),
            "max_mixed_label_bce_change": max(label_differences),
            "invalid_cases_rejected": invalid, "multi_scene_alignment": True,
            "scope": "Exact arithmetic equivalence, not trained benefit",
            "smoothing_caveat": "Equal superpoint sizes >1 change Dice +1 smoothing; "
            "only unit masses recover the unchanged native Dice exactly."}


def acquire(gpu):
    import fcntl
    handle = open("/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock", "a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], universal_newlines=True)
    used, util = (int(v.strip()) for v in raw.strip().split(","))
    if used >= 1024 or util >= 10 or gpu != 1:
        raise RuntimeError("Authorized GPU1 not idle: " + raw.strip())
    return handle, {"memory_mib": used, "utilization": util}


def native(args, result, path):
    import gorilla
    import numpy as np
    from ipdn.dataset import build_dataset
    from ipdn.utils import get_root_logger
    from train import resolve_model_class, load_shape_compatible_pretrain
    from diagnose_adares_y14_reliability import scene_indices

    lease, occupation = acquire(args.gpu_id)
    result["gpu_precheck"] = occupation
    torch.set_num_threads(4)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    dc = cfg.data.train.copy()
    if args.mode == "diagnosis":
        dc.training, dc.aug, dc.with_elastic = False, False, False
        dc.lang_num_max = 4
    dataset = build_dataset(dc, get_root_logger())
    selected = scene_indices(dataset.scene_inputs, args.scans)
    result["selection"] = [
        {"item": i, "scan": str(dataset.scene_inputs[i][0]["scene_id"])}
        for i in selected]
    result["scope"] = "Train-only fixed distinct scans; no validation data"
    model = resolve_model_class(args.config)(**cfg.model).cuda()
    if args.mode == "diagnosis":
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        load = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        allowed = {"text_encoder.embeddings.position_ids",
                   "text_encoder.embeddings.token_type_ids"}
        if load.missing_keys or set(load.unexpected_keys) - allowed:
            raise RuntimeError(str(load))
        result["checkpoint_sha256"] = digest(args.checkpoint)
        del checkpoint
        model.eval()
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
        model.train()
    original = model.criterion.loss_masks
    captures = []

    def record(self, pred_masks, pred_scores, targets, indices, num_boxes):
        for mask, target, (qidx, _) in zip(pred_masks, targets, indices):
            if target["masks"] is None:
                continue
            z = mask[qidx]
            counts, pos = target["point_counts"], target["point_positives"]
            point = pm.point_losses(z, counts, pos)
            labels = target["masks"]
            # RES has one target repeated for all native matched queries.
            if not torch.equal(labels[0].expand_as(labels), labels):
                raise RuntimeError("Nonidentical RES repeated targets")
            old = old_losses(z, labels[0])
            old_grad = torch.autograd.grad(sum(old), z, retain_graph=True)[0]
            point_grad = torch.autograd.grad(sum(point), z, retain_graph=True)[0]
            controls = {
                "mass_only": pm.point_losses(z, counts, counts * labels[0].to(counts)),
                "occupancy_only": pm.point_losses(z, torch.ones_like(counts), pos / counts),
                "equal_size": pm.point_losses(
                    z, torch.full_like(counts, counts.mean()),
                    pos / counts * counts.mean()),
            }
            prob = z.detach().sigmoid()
            intersection = (prob * pos).sum(-1)
            union = (prob * counts).sum(-1) + pos.sum() - intersection
            soft_iou = (intersection / union.clamp_min(1)).mean()
            iou_grad = torch.autograd.grad(
                -((z.sigmoid() * pos).sum(-1) /
                  ((z.sigmoid() * counts).sum(-1) + pos.sum()
                   - (z.sigmoid() * pos).sum(-1)).clamp_min(1)).mean(),
                z, retain_graph=True)[0]
            entry = {
                "matched": len(qidx), "points": int(counts.sum()),
                "superpoints": counts.numel(), "count_cv": float(counts.std() / counts.mean()),
                "mixed_superpoints": int(((pos > 0) & (pos < counts)).sum()),
                "point_positive": int(pos.sum()), "old_bce": float(old[0]),
                "point_bce": float(point[0]), "old_dice": float(old[1]),
                "point_dice": float(point[1]),
                "gradient_change_norm": float((point_grad - old_grad).norm()),
                "old_vs_point_gradient_cos": cosine(old_grad, point_grad),
                "old_vs_negative_point_iou_cos": cosine(old_grad, iou_grad),
                "point_vs_negative_point_iou_cos": cosine(point_grad, iou_grad),
                "soft_point_iou": float(soft_iou),
                "finite": bool(torch.isfinite(point_grad).all()),
            }
            for name, pair in controls.items():
                grad = torch.autograd.grad(sum(pair), z, retain_graph=True)[0]
                entry[name] = {"bce": float(pair[0]), "dice": float(pair[1]),
                               "vs_negative_point_iou_cos": cosine(grad, iou_grad),
                               "gradient_distance_from_point": float((grad-point_grad).norm())}
            captures.append(entry)
        return original(pred_masks, pred_scores, targets, indices, num_boxes)

    model.criterion.loss_masks = types.MethodType(record, model.criterion)
    result["rows"] = []
    optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
    step_start = time.monotonic()
    for offset in range(0, len(selected), 2):
        gorilla.set_random_seed(1999 + offset)
        batch = dataset.collate_fn([dataset[i] for i in selected[offset:offset + 2]])
        captures.clear()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        loss, values = model(batch, mode="loss")
        loss.backward()
        grads = {n: {"norm": float(p.grad.norm()),
                     "finite": bool(torch.isfinite(p.grad).all())}
                 for n, p in model.named_parameters()
                 if n.startswith("dec.") and p.grad is not None}
        if not torch.isfinite(loss).all() or not all(v["finite"] for v in grads.values()):
            raise RuntimeError("Nonfinite native loss/backward")
        if args.mode == "native":
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
        torch.cuda.synchronize()
        row = {"scans": batch["scan_ids"], "expressions": len(batch["ann_ids"]),
               "loss": float(loss), "values": {k: float(v) for k, v in values.items()},
               "captures": copy.deepcopy(captures), "decoder_gradients": grads,
               "elapsed_s": time.monotonic() - start,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        result["rows"].append(row)
        save(path, result)
        print(json.dumps({"batch": offset // 2, "loss": row["loss"],
                          "captures": len(captures), "scans": row["scans"],
                          "elapsed_s": row["elapsed_s"]}), flush=True)
        del loss, values, batch
    result["elapsed_s"] = time.monotonic() - step_start
    result["parameters"] = sum(p.numel() for p in model.parameters())
    result["complete"] = True
    lease.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("algebra", "diagnosis", "native"), required=True)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--scans", type=int, default=8)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise RuntimeError("Refusing to overwrite evidence")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    result = {"complete": False, "mode": args.mode, "tool_sha256": digest(__file__)}
    if args.config:
        result["config_sha256"] = digest(args.config)
    try:
        if args.mode == "algebra":
            result.update(algebra())
        else:
            native(args, result, path)
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        save(path, result)
        raise
    save(path, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
