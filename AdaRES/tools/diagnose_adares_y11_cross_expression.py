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


def mask_iou(left, right):
    return float(
        (left & right).sum().float()
        / (left | right).sum().clamp_min(1).float()
    )


def make_target_permutations(target_sets, device):
    generator = torch.Generator()
    generator.manual_seed(1999)
    permutations = []
    count = len(target_sets)
    for _ in range(CONTROL_PERMUTATIONS):
        order = torch.randperm(count, generator=generator).tolist()
        permutations.append([target_sets[index] for index in order])
    return permutations


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    records = []
    for index, truth in enumerate(gt_pmasks):
        scores = out["indis"][index].softmax(-1)[:, 1]
        points = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(points.device)
        ious = (points & target).sum(-1).float() / (
            (points | target).sum(-1).clamp_min(1).float())
        selected_index = int(scores.argmax())
        oracle_index = int(ious.argmax())
        records.append({
            "expression_index": index,
            "ann_id": str(ann_ids[index]),
            "target_ids": tuple(sorted(int(value) for value in object_idss[index])),
            "selected_index": selected_index,
            "oracle_index": oracle_index,
            "selected_iou": float(ious[selected_index]),
            "oracle_iou": float(ious[oracle_index]),
            "selected_mask": points[selected_index].detach(),
            "oracle_mask": points[oracle_index].detach(),
        })

    target_sets = [record["target_ids"] for record in records]
    permutations = make_target_permutations(target_sets, points.device)
    pair_rows = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            true_disjoint = set(records[left]["target_ids"]).isdisjoint(
                records[right]["target_ids"])
            control_disjoint = statistics.fmean(
                float(set(permutation[left]).isdisjoint(permutation[right]))
                for permutation in permutations
            )
            pair_rows.append({
                "scene_index": self.probe_scene,
                "left": left,
                "right": right,
                "left_ann_id": records[left]["ann_id"],
                "right_ann_id": records[right]["ann_id"],
                "true_disjoint": true_disjoint,
                "control_disjoint_rate": control_disjoint,
                "selected_overlap": mask_iou(
                    records[left]["selected_mask"],
                    records[right]["selected_mask"]),
                "oracle_overlap": mask_iou(
                    records[left]["oracle_mask"],
                    records[right]["oracle_mask"]),
                "left_selected_index": records[left]["selected_index"],
                "right_selected_index": records[right]["selected_index"],
                "left_oracle_index": records[left]["oracle_index"],
                "right_oracle_index": records[right]["oracle_index"],
            })
    return {"rows": pair_rows}


def scene_weighted_overlap(rows, key, disjoint_key):
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = float(row[disjoint_key])
        numerator += weight * row[key]
        denominator += weight
    if denominator == 0:
        return None
    return numerator / denominator


def summarize(pair_rows):
    scenes = defaultdict(list)
    for row in pair_rows:
        scenes[row["scene_index"]].append(row)
    scene_rows = []
    for scene, rows in scenes.items():
        true_pairs = [row for row in rows if row["true_disjoint"]]
        if not true_pairs:
            continue
        true_selected = statistics.fmean(
            row["selected_overlap"] for row in true_pairs)
        true_oracle = statistics.fmean(
            row["oracle_overlap"] for row in true_pairs)
        control_selected = scene_weighted_overlap(
            rows, "selected_overlap", "control_disjoint_rate")
        control_oracle = scene_weighted_overlap(
            rows, "oracle_overlap", "control_disjoint_rate")
        scene_rows.append({
            "scene_index": scene,
            "true_pairs": len(true_pairs),
            "true_selected_overlap": true_selected,
            "true_oracle_overlap": true_oracle,
            "control_selected_overlap": control_selected,
            "control_oracle_overlap": control_oracle,
            "selected_collisions": sum(
                row["selected_overlap"] > 0.5 for row in true_pairs),
            "oracle_collisions": sum(
                row["oracle_overlap"] > 0.5 for row in true_pairs),
        })

    def compare(left, right):
        deltas = [
            row[left] - row[right] for row in scene_rows
            if row[left] is not None and row[right] is not None
        ]
        positive = sum(value > 0 for value in deltas)
        negative = sum(value < 0 for value in deltas)
        ties = len(deltas) - positive - negative
        p_value, trials = sign_test_p_value(positive, negative)
        return {
            "scenes": len(deltas),
            "scene_mean_left": statistics.fmean(
                row[left] for row in scene_rows
                if row[left] is not None),
            "scene_mean_right": statistics.fmean(
                row[right] for row in scene_rows
                if row[right] is not None),
            "scene_mean_paired_delta": statistics.fmean(deltas),
            "positive_scenes": positive,
            "negative_scenes": negative,
            "tie_scenes": ties,
            "sign_test_trials": trials,
            "sign_test_p_two_sided": p_value,
        }

    true_pairs = [row for row in pair_rows if row["true_disjoint"]]
    return {
        "scenes": len(scene_rows),
        "all_pairs": len(pair_rows),
        "true_disjoint_pairs": len(true_pairs),
        "selected_collisions": sum(
            row["selected_overlap"] > 0.5 for row in true_pairs),
        "oracle_collisions": sum(
            row["oracle_overlap"] > 0.5 for row in true_pairs),
        "selected_vs_control": compare(
            "true_selected_overlap", "control_selected_overlap"),
        "oracle_vs_selected": compare(
            "true_oracle_overlap", "true_selected_overlap"),
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
    pair_rows = []
    repeat_identity_mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        if len(first) != len(second):
            raise RuntimeError("Repeated pair count mismatch")
        for left, right in zip(first, second):
            if (
                left["left_selected_index"] != right["left_selected_index"]
                or left["right_selected_index"] != right["right_selected_index"]
                or left["left_oracle_index"] != right["left_oracle_index"]
                or left["right_oracle_index"] != right["right_oracle_index"]
                or left["true_disjoint"] != right["true_disjoint"]
            ):
                repeat_identity_mismatch += 1
        pair_rows.extend(first)
    summary = summarize(pair_rows)
    passed = bool(
        summary["selected_vs_control"]["scene_mean_paired_delta"] > 0
        and summary["selected_vs_control"]["sign_test_p_two_sided"] < 0.05
        and summary["oracle_vs_selected"]["scene_mean_paired_delta"] < 0
        and summary["oracle_vs_selected"]["sign_test_p_two_sided"] < 0.05
        and repeat_identity_mismatch == 0
    )
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "control_permutations": CONTROL_PERMUTATIONS,
        "repeat_identity_mismatch": repeat_identity_mismatch,
        "passed": passed,
        "summary": summary,
        "peak_inference_reserved_gib": (
            torch.cuda.max_memory_reserved() / 2**30),
        "rows": pair_rows,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "split": result["split"],
        "scenes": result["scenes"],
        "control_permutations": result["control_permutations"],
        "repeat_identity_mismatch": result["repeat_identity_mismatch"],
        "passed": result["passed"],
        "summary": summary,
        "peak_inference_reserved_gib": result["peak_inference_reserved_gib"],
    }, indent=2))


if __name__ == "__main__":
    main()
