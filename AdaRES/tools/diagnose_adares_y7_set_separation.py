import argparse
import json
import math
import statistics
import sys
import types
from collections import defaultdict
from pathlib import Path

import gorilla
import torch
from torch_scatter import scatter_mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_w2_pointmass_competition import IPDN
from ipdn.utils import get_root_logger


def sign_test_p_value(positive, negative):
    trials = positive + negative
    if trials == 0:
        return 1.0, 0
    tail = sum(math.comb(trials, k) for k in range(0, min(positive, negative) + 1))
    p_value = min(1.0, 2.0 * tail / (2 ** trials))
    return p_value, trials


def membership_measure(membership, scores):
    count = int(membership.sum())
    if count == 0 or count == membership.numel():
        return None
    positive_scores = scores[membership]
    negative_scores = scores[~membership]
    best_positive = positive_scores.max()
    best_negative = negative_scores.max()
    margin = best_positive - best_negative
    return {
        "count": count,
        "best_positive_score": float(best_positive),
        "best_negative_score": float(best_negative),
        "margin": float(margin),
        "separated": bool(margin > 0),
        "best_positive_rank": 1 + int((scores > best_positive).sum()),
    }


def shifted_measure(membership, scores):
    query_count = scores.numel()
    margins = []
    separated = []
    ranks = []
    best_positive_scores = []
    for offset in range(1, query_count):
        measure = membership_measure(membership.roll(offset), scores)
        if measure is None:
            continue
        margins.append(measure["margin"])
        separated.append(measure["separated"])
        ranks.append(measure["best_positive_rank"])
        best_positive_scores.append(measure["best_positive_score"])
    if not margins:
        return None
    return {
        "count": int(membership.sum()),
        "shift_count": len(margins),
        "mean_margin": statistics.fmean(margins),
        "median_margin": statistics.median(margins),
        "separated_rate": statistics.fmean(separated),
        "mean_best_positive_rank": statistics.fmean(ranks),
        "mean_best_positive_score": statistics.fmean(best_positive_scores),
    }


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    _, expanded_pos, cached_fps, offsets, expanded_labels, _ = self.probe_expand
    if not torch.equal(cached_fps, fps_seed_sp):
        raise RuntimeError("Expanded FPS indices changed before predictor")
    if len(expanded_labels) != len(gt_pmasks):
        raise RuntimeError("Expanded instance labels do not align with expressions")
    rows = []
    for index, truth in enumerate(gt_pmasks):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        sp_pos = expanded_pos[start:end]
        sp_labels = expanded_labels[index].to(sp_pos.device)
        if sp_pos.shape[0] != sp_labels.numel():
            raise RuntimeError("Expanded superpoint labels/positions mismatch")
        seed_pos = sp_pos[fps_seed_sp[index].long()]
        sampled = out["sample_inds"][index].long()
        sampled_pos = seed_pos[sampled]
        sampled_labels = sp_labels[fps_seed_sp[index].long()][sampled]
        obj_ids = object_idss[index]
        if not obj_ids:
            raise RuntimeError("No instance ID for a ScanRefer expression")
        masks = torch.stack([sp_labels == obj_id for obj_id in obj_ids])
        if not masks.any(-1).all():
            raise RuntimeError("Target instance absent from native superpoints")
        centroids = torch.stack([
            scatter_mean(sp_pos, mask.long(), dim=0)[1] for mask in masks
        ])
        _, _, indices, _ = self.criterion.match_sampled_ins(
            sampled_labels, obj_ids, sampled_pos, centroids
        )
        scores = out["indis"][index].softmax(-1)[:, 1]
        selected = int(scores.argmax())
        points = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(points.device)
        ious = (points & target).sum(-1).float() / (
            (points | target).sum(-1).clamp_min(1).float())
        oracle = int(ious.argmax())
        positive = torch.zeros_like(scores, dtype=torch.bool)
        positive[indices[0].long()] = True
        true_measure = membership_measure(positive, scores)
        shift_measure = shifted_measure(positive, scores)
        if true_measure is None or shift_measure is None:
            raise RuntimeError("Y7 requires both positive and negative queries")
        rows.append({
            "scene_index": self.probe_scene,
            "expression_index": index,
            "scan_id": str(scan_ids[0]),
            "ann_id": str(ann_ids[index]),
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "material_error": bool(ious[oracle] - ious[selected] > 0.05),
            "oracle_positive": bool(positive[oracle]),
            "selected_positive": bool(positive[selected]),
            "selected_score": float(scores[selected]),
            "positive_count": true_measure["count"],
            "true_margin": true_measure["margin"],
            "true_separated": true_measure["separated"],
            "true_best_positive_rank": true_measure["best_positive_rank"],
            "true_best_positive_score": true_measure["best_positive_score"],
            "shift_mean_margin": shift_measure["mean_margin"],
            "shift_separated_rate": shift_measure["separated_rate"],
            "shift_mean_best_positive_rank": (
                shift_measure["mean_best_positive_rank"]),
            "shift_mean_best_positive_score": (
                shift_measure["mean_best_positive_score"]),
            "shift_count": shift_measure["shift_count"],
        })
    return {"rows": rows}


def select_cohort(rows, name):
    if name == "all":
        return rows
    if name == "material_errors":
        return [row for row in rows if row["material_error"]]
    if name == "oracle_positive_errors":
        return [
            row for row in rows
            if row["material_error"] and row["oracle_positive"]
        ]
    raise ValueError(name)


def paired_scene_summary(rows, true_key, shift_key):
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene_index"]].append(row)
    paired = []
    for scene, scene_rows in scenes.items():
        true_value = statistics.fmean(row[true_key] for row in scene_rows)
        shift_value = statistics.fmean(
            row[shift_key] for row in scene_rows)
        paired.append(true_value - shift_value)
    positive = sum(delta > 0 for delta in paired)
    negative = sum(delta < 0 for delta in paired)
    ties = len(paired) - positive - negative
    p_value, trials = sign_test_p_value(positive, negative)
    return {
        "scenes": len(paired),
        "scene_mean_true": statistics.fmean(
            statistics.fmean(row[true_key] for row in scene_rows)
            for scene_rows in scenes.values()),
        "scene_mean_shift": statistics.fmean(
            statistics.fmean(row[shift_key] for row in scene_rows)
            for scene_rows in scenes.values()),
        "scene_mean_paired_delta": statistics.fmean(paired),
        "positive_scenes": positive,
        "negative_scenes": negative,
        "tie_scenes": ties,
        "sign_test_trials": trials,
        "sign_test_p_two_sided": p_value,
    }


def summarize(rows):
    result = {}
    for cohort in ("all", "material_errors", "oracle_positive_errors"):
        subset = select_cohort(rows, cohort)
        scenes = defaultdict(list)
        for row in subset:
            scenes[row["scene_index"]].append(row)
        entry = {
            "expressions": len(subset),
            "scenes": len(scenes),
            "mean_positive_count": statistics.fmean(
                row["positive_count"] for row in subset) if subset else None,
            "true_separated": sum(row["true_separated"] for row in subset),
            "true_separated_rate": statistics.fmean(
                row["true_separated"] for row in subset) if subset else None,
            "shift_separated_rate": statistics.fmean(
                row["shift_separated_rate"] for row in subset) if subset else None,
            "scene_mean_true_separated": statistics.fmean(
                statistics.fmean(row["true_separated"] for row in scene)
                for scene in scenes.values()) if scenes else None,
            "scene_mean_shift_separated": statistics.fmean(
                statistics.fmean(row["shift_separated_rate"] for row in scene)
                for scene in scenes.values()) if scenes else None,
            "mean_true_margin": statistics.fmean(
                row["true_margin"] for row in subset) if subset else None,
            "mean_shift_margin": statistics.fmean(
                row["shift_mean_margin"] for row in subset) if subset else None,
            "oracle_positive": sum(row["oracle_positive"] for row in subset),
            "selected_positive": sum(row["selected_positive"] for row in subset),
            "paired": {
                "separated": paired_scene_summary(
                    subset, "true_separated", "shift_separated_rate")
                if subset else None,
                "margin": paired_scene_summary(
                    subset, "true_margin", "shift_mean_margin")
                if subset else None,
            },
        }
        result[cohort] = entry
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--scenes", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = getattr(cfg.data, args.split).copy()
    data_cfg.training = False
    data_cfg.aug = False
    data_cfg.with_elastic = False
    data_cfg.lang_num_max = 32
    dataset = build_dataset(data_cfg, get_root_logger())
    model = IPDN(**cfg.model).cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loaded = model.load_state_dict(
        checkpoint.get("model", checkpoint), strict=False)
    allowed = {
        "text_encoder.embeddings.position_ids",
        "text_encoder.embeddings.token_type_ids",
    }
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.eval()

    def capture(module, inputs, output):
        model.probe_expand = output

    original_expand = model.expand_and_fps

    def expand_and_capture(self, *args, **kwargs):
        result = original_expand(*args, **kwargs)
        capture(self, args, result)
        return result

    model.expand_and_fps = types.MethodType(expand_and_capture, model)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    mismatch = 0
    decision_mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        if len(first) != len(second):
            raise RuntimeError("Repeated expression count mismatch")
        mismatch += sum(a != b for a, b in zip(first, second))
        for left, right in zip(first, second):
            if any(
                left[key] != right[key]
                for key in (
                    "material_error",
                    "oracle_positive",
                    "selected_positive",
                    "true_separated",
                )
            ):
                decision_mismatch += 1
        rows.extend(first)
    summary = summarize(rows)
    primary = summary["oracle_positive_errors"]["paired"]
    passed = bool(
        primary["separated"]["scene_mean_paired_delta"] > 0
        and primary["separated"]["sign_test_p_two_sided"] < 0.05
        and primary["margin"]["scene_mean_paired_delta"] > 0
    )
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "repeat_mismatch": mismatch,
        "repeat_decision_mismatch": decision_mismatch,
        "passed": passed,
        "summary": summary,
        "peak_inference_reserved_gib": (
            torch.cuda.max_memory_reserved() / 2**30),
        "rows": rows,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "split": result["split"],
        "scenes": result["scenes"],
        "expressions": result["expressions"],
        "repeat_mismatch": result["repeat_mismatch"],
        "repeat_decision_mismatch": result["repeat_decision_mismatch"],
        "passed": result["passed"],
        "oracle_positive_errors": summary["oracle_positive_errors"],
        "peak_inference_reserved_gib": result["peak_inference_reserved_gib"],
    }, indent=2))


if __name__ == "__main__":
    main()
