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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_w2_pointmass_competition import IPDN
from ipdn.utils import get_root_logger


CONTROL_PERMUTATIONS = 8


def sign_test_p_value(positive, negative):
    trials = positive + negative
    if trials == 0:
        return 1.0, 0
    tail = sum(math.comb(trials, k) for k in range(0, min(positive, negative) + 1))
    p_value = min(1.0, 2.0 * tail / (2 ** trials))
    return p_value, trials


def make_control_permutations(query_count, device):
    generator = torch.Generator()
    generator.manual_seed(1999)
    return [
        torch.randperm(query_count, generator=generator).to(device)
        for _ in range(CONTROL_PERMUTATIONS)
    ]


def build_row_normalized_overlap(mask_logits, point_mass):
    probabilities = mask_logits.sigmoid()
    weighted = probabilities * point_mass.unsqueeze(0)
    areas = weighted.sum(-1)
    intersection = torch.einsum(
        "im,jm->ij", weighted, probabilities
    )
    union = areas.unsqueeze(-1) + areas.unsqueeze(0) - intersection
    overlap = intersection / union.clamp_min(1e-6)
    overlap = overlap.masked_fill(
        torch.eye(
            overlap.shape[0], dtype=torch.bool, device=overlap.device
        ),
        0.0,
    )
    return overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    point_mass = torch.bincount(superpoints.long()).float()
    rows = []
    for index, truth in enumerate(gt_pmasks):
        scores = out["indis"][index].softmax(-1)[:, 1]
        mask_logits = out["masks"][index]
        if mask_logits.shape[1] != point_mass.numel():
            raise RuntimeError("Query mask and point-mass dimensions disagree")
        overlap = build_row_normalized_overlap(mask_logits, point_mass)
        diffused = 0.5 * scores + 0.5 * torch.matmul(overlap, scores)
        permutations = make_control_permutations(scores.numel(), scores.device)
        control_scores = [
            torch.matmul(overlap[permutation], scores)
            for permutation in permutations
        ]
        points = mask_logits.sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(points.device)
        ious = (points & target).sum(-1).float() / (
            (points | target).sum(-1).clamp_min(1).float())
        raw_index = int(scores.argmax())
        diffused_index = int(diffused.argmax())
        control_indices = [int(value.argmax()) for value in control_scores]
        oracle_index = int(ious.argmax())
        rows.append({
            "scene_index": self.probe_scene,
            "expression_index": index,
            "scan_id": str(scan_ids[0]),
            "ann_id": str(ann_ids[index]),
            "oracle_iou": float(ious[oracle_index]),
            "raw_iou": float(ious[raw_index]),
            "diffused_iou": float(ious[diffused_index]),
            "control_mean_iou": statistics.fmean(
                float(ious[value]) for value in control_indices),
            "raw_selected_index": raw_index,
            "diffused_selected_index": diffused_index,
            "control_selected_indices": control_indices,
            "raw_material_error": bool(
                ious[oracle_index] - ious[raw_index] > 0.05),
            "diffused_material_error": bool(
                ious[oracle_index] - ious[diffused_index] > 0.05),
        })
    return {"rows": rows}


def select_cohort(rows, name):
    if name == "all":
        return rows
    if name == "raw_material_errors":
        return [row for row in rows if row["raw_material_error"]]
    raise ValueError(name)


def paired_scene_summary(rows, true_key, control_key):
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene_index"]].append(row)
    deltas = []
    for scene_rows in scenes.values():
        true_value = statistics.fmean(row[true_key] for row in scene_rows)
        control_value = statistics.fmean(row[control_key] for row in scene_rows)
        deltas.append(true_value - control_value)
    positive = sum(delta > 0 for delta in deltas)
    negative = sum(delta < 0 for delta in deltas)
    ties = len(deltas) - positive - negative
    p_value, trials = sign_test_p_value(positive, negative)
    return {
        "scenes": len(deltas),
        "scene_mean_true_iou": statistics.fmean(
            statistics.fmean(row[true_key] for row in scene_rows)
            for scene_rows in scenes.values()),
        "scene_mean_control_iou": statistics.fmean(
            statistics.fmean(row[control_key] for row in scene_rows)
            for scene_rows in scenes.values()),
        "scene_mean_paired_delta": statistics.fmean(deltas),
        "positive_scenes": positive,
        "negative_scenes": negative,
        "tie_scenes": ties,
        "sign_test_trials": trials,
        "sign_test_p_two_sided": p_value,
    }


def summarize(rows):
    result = {}
    for cohort in ("all", "raw_material_errors"):
        subset = select_cohort(rows, cohort)
        result[cohort] = {
            "expressions": len(subset),
            "scenes": len({row["scene_index"] for row in subset}),
            "mean_raw_iou": statistics.fmean(
                row["raw_iou"] for row in subset) if subset else None,
            "mean_diffused_iou": statistics.fmean(
                row["diffused_iou"] for row in subset) if subset else None,
            "mean_control_iou": statistics.fmean(
                row["control_mean_iou"] for row in subset) if subset else None,
            "diffused_material_errors": sum(
                row["diffused_material_error"] for row in subset),
            "selected_flip_count": sum(
                row["raw_selected_index"] != row["diffused_selected_index"]
                for row in subset),
            "paired_vs_raw": paired_scene_summary(
                subset, "diffused_iou", "raw_iou") if subset else None,
            "paired_vs_control": paired_scene_summary(
                subset, "diffused_iou", "control_mean_iou") if subset else None,
        }
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
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    repeat_identity_mismatch = 0
    repeat_material_mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        if len(first) != len(second):
            raise RuntimeError("Repeated expression count mismatch")
        for left, right in zip(first, second):
            if (
                left["raw_selected_index"] != right["raw_selected_index"]
                or left["diffused_selected_index"]
                != right["diffused_selected_index"]
                or left["control_selected_indices"]
                != right["control_selected_indices"]
            ):
                repeat_identity_mismatch += 1
            if (
                left["raw_material_error"] != right["raw_material_error"]
                or left["diffused_material_error"]
                != right["diffused_material_error"]
            ):
                repeat_material_mismatch += 1
        rows.extend(first)
    summary = summarize(rows)
    all_summary = summary["all"]
    passed = bool(
        all_summary["paired_vs_raw"]["scene_mean_paired_delta"] > 0
        and all_summary["paired_vs_control"]["scene_mean_paired_delta"] > 0
        and all_summary["paired_vs_raw"]["sign_test_p_two_sided"] < 0.05
        and repeat_identity_mismatch == 0
    )
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "control_permutations": CONTROL_PERMUTATIONS,
        "repeat_identity_mismatch": repeat_identity_mismatch,
        "repeat_material_mismatch": repeat_material_mismatch,
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
        "control_permutations": result["control_permutations"],
        "repeat_identity_mismatch": result["repeat_identity_mismatch"],
        "repeat_material_mismatch": result["repeat_material_mismatch"],
        "passed": result["passed"],
        "summary": summary,
        "peak_inference_reserved_gib": result["peak_inference_reserved_gib"],
    }, indent=2))


if __name__ == "__main__":
    main()
