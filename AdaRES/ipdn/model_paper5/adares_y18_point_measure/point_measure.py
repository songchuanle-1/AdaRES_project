"""Exact binary point losses without expanding superpoint logits."""

import torch
import torch.nn.functional as F


def point_losses(logits, counts, positives):
    if logits.ndim != 2 or counts.ndim != 1 or positives.ndim != 1:
        raise ValueError("Expected matched masks and one expression point measure")
    if logits.shape[-1] != counts.numel() or counts.shape != positives.shape:
        raise ValueError("Point measure and mask shape mismatch")
    if (not torch.isfinite(counts).all() or not torch.isfinite(positives).all()
            or (counts <= 0).any() or (positives < 0).any()
            or (positives > counts).any()):
        raise ValueError("Invalid point measure")
    counts = counts.to(logits)
    positives = positives.to(logits)
    total = counts.sum()
    bce = ((F.softplus(logits) * counts - logits * positives).sum(-1)
           / total).mean()
    prob = logits.sigmoid()
    dice = (1 - (2 * (prob * positives).sum(-1) + 1)
            / ((prob * counts).sum(-1) + positives.sum() + 1)).mean()
    return bce, dice


def batch_point_measure(superpoints, offsets, scenes_len, gt_pmasks, object_idss):
    counts, positives = [], []
    expression = 0
    for scene in range(len(offsets) - 1):
        lo, hi = int(offsets[scene]), int(offsets[scene + 1])
        keep = (superpoints >= lo) & (superpoints < hi)
        local = superpoints[keep].long() - lo
        mass = torch.bincount(local, minlength=hi - lo).float()
        for _ in range(int(scenes_len[scene])):
            truth = gt_pmasks[expression]
            if len(object_idss[expression]) != 1:
                raise ValueError("Y18 point supervision requires single-target RES")
            if truth.ndim != 1 or truth.numel() != local.numel():
                raise ValueError("Expression point labels do not align with scene")
            positive = torch.bincount(
                local, weights=truth.gt(0.5).float(), minlength=hi - lo)
            counts.append(mass)
            positives.append(positive)
            expression += 1
    if expression != len(gt_pmasks) or expression != len(object_idss):
        raise ValueError("Scene/expression alignment mismatch")
    return counts, positives
