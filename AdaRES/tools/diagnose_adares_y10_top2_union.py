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
    tail = sum(
        math.comb(trials, k)
        for k in range(0, min(positive, negative) + 1)
    )
    return min(1.0, 2.0 * tail / (2 ** trials)), trials


def iou(binary, target):
    numerator = (binary & target).sum(-1).float()
    denominator = (binary | target).sum(-1).clamp_min(1).float()
    return numerator / denominator


def make_second_permutations(query_count, device):
    generator = torch.Generator()
    generator.manual_seed(1999)
    return [
        torch.randperm(query_count, generator=generator).to(device)
        for _ in range(CONTROL_PERMUTATIONS)
    ]


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    rows = []
    for index, truth in enumerate(gt_pmasks):
        scores = out["indis"][index].softmax(-1)[:, 1]
        probabilities = out["masks"][index].sigmoid()[:, superpoints]
        target = truth.gt(0.5).to(probabilities.device)
        top_two = torch.topk(scores, 2).indices
        primary = int(top_two[0])
        secondary = int(top_two[1])
        top1_binary = probabilities[primary] > 0.5
        top2_union = 1.0 - (
            (1.0 - probabilities[primary]) * (1.0 - probabilities[secondary])
        )
        top2_binary = top2_union > 0.5
        permutations = make_second_permutations(
            scores.numel(), scores.device)
        control_ious = []
        control_secondaries = []
        for permutation in permutations:
            candidate = int(permutation[0])
            if candidate == primary and scores.numel() > 1:
                candidate = int(permutation[1])
            control_union = 1.0 - (
                (1.0 - probabilities[primary])
                * (1.0 - probabilities[candidate])
            )
            control_ious.append(float(iou(control_union > 0.5, target)))
            control_secondaries.append(candidate)
        rows.append({
            "scene_index": self.probe_scene,
            "expression_index": index,
            "scan_id": str(scan_ids[0]),
            "ann_id": str(ann_ids[index]),
            "oracle_iou": float(iou(probabilities > 0.5, target).max()),
            "top1_iou": float(iou(top1_binary, target)),
            "top2_union_iou": float(iou(top2_binary, target)),
            "control_mean_iou": statistics.fmean(control_ious),
            "primary_index": primary,
            "secondary_index": secondary,
            "control_secondary_indices": control_secondaries,
        })
    return {"rows": rows}


def paired_scene_summary(rows, true_key, control_key):
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene_index"]].append(row)
    deltas = []
    for scene_rows in scenes.values():
        true_value = statistics.fmean(row[true_key] for row in scene_rows)
        control_value = statistics.fmean(
            row[control_key] for row in scene_rows)
        deltas.append(true_value - control_value)
    positive = sum(value > 0 for value in deltas)
    negative = sum(value < 0 for value in deltas)
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
    return {
        "expressions": len(rows),
        "scenes": len({row["scene_index"] for row in rows}),
        "mean_top1_iou": statistics.fmean(
            row["top1_iou"] for row in rows) if rows else None,
        "mean_top2_union_iou": statistics.fmean(
            row["top2_union_iou"] for row in rows) if rows else None,
        "mean_control_iou": statistics.fmean(
            row["control_mean_iou"] for row in rows) if rows else None,
        "secondary_flip_count": sum(
            row["secondary_index"] not in row["control_secondary_indices"]
            for row in rows),
        "paired_vs_top1": paired_scene_summary(
            rows, "top2_union_iou", "top1_iou") if rows else None,
        "paired_vs_control": paired_scene_summary(
            rows, "top2_union_iou", "control_mean_iou") if rows else None,
    }


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
                left["primary_index"] != right["primary_index"]
                or left["secondary_index"] != right["secondary_index"]
                or left["control_secondary_indices"]
                != right["control_secondary_indices"]
            ):
                repeat_identity_mismatch += 1
        rows.extend(first)
    summary = summarize(rows)
    passed = bool(
        summary["paired_vs_top1"]["scene_mean_paired_delta"] > 0
        and summary["paired_vs_control"]["scene_mean_paired_delta"] > 0
        and summary["paired_vs_top1"]["sign_test_p_two_sided"] < 0.05
        and repeat_identity_mismatch == 0
    )
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "control_permutations": CONTROL_PERMUTATIONS,
        "repeat_identity_mismatch": repeat_identity_mismatch,
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
        "passed": result["passed"],
        "summary": summary,
        "peak_inference_reserved_gib": result["peak_inference_reserved_gib"],
    }, indent=2))


if __name__ == "__main__":
    main()
