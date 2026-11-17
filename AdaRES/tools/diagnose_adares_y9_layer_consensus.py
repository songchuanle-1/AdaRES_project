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


CONTROL_SETS = 8


def sign_test_p_value(positive, negative):
    trials = positive + negative
    if trials == 0:
        return 1.0, 0
    tail = sum(
        math.comb(trials, k)
        for k in range(0, min(positive, negative) + 1)
    )
    return min(1.0, 2.0 * tail / (2 ** trials)), trials


def margin(logits):
    return logits[:, 1] - logits[:, 0]


def make_permutation_sets(layer_count, query_count, device):
    generator = torch.Generator()
    generator.manual_seed(1999)
    return [
        [
            torch.randperm(query_count, generator=generator).to(device)
            for _ in range(layer_count)
        ]
        for _ in range(CONTROL_SETS)
    ]


def build_predictor():
    def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
                  gt_pmasks, gt_spmasks, fps_seed_sp):
        final_indis = out["indis"]
        if final_indis.dim() != 3:
            raise RuntimeError("Expected batched indicator logits")
        auxiliary = out.get("aux_outputs")
        if not auxiliary:
            raise RuntimeError("Y9 requires auxiliary decoder outputs")
        margins_by_layer = [
            margin(item["indis"]) for item in auxiliary
        ] + [margin(final_indis)]
        if any(value.shape != margins_by_layer[0].shape
               for value in margins_by_layer):
            raise RuntimeError("Indicator layer shapes disagree")

        rows = []
        for index, truth in enumerate(gt_pmasks):
            query_count = margins_by_layer[0].shape[1]
            per_layer = [
                value[index] for value in margins_by_layer
            ]
            consensus = torch.stack(per_layer).mean(0)
            permutation_sets = make_permutation_sets(
                len(per_layer), query_count, consensus.device)
            control_scores = []
            for permutations in permutation_sets:
                aligned = [
                    layer[permutation]
                    for layer, permutation in zip(per_layer, permutations)
                ]
                control_scores.append(torch.stack(aligned).mean(0))

            points = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            target = truth.gt(0.5).to(points.device)
            ious = (points & target).sum(-1).float() / (
                (points | target).sum(-1).clamp_min(1).float())
            final_selected = int(per_layer[-1].argmax())
            consensus_selected = int(consensus.argmax())
            control_selected = [
                int(value.argmax()) for value in control_scores
            ]
            rows.append({
                "scene_index": self.probe_scene,
                "expression_index": index,
                "scan_id": str(scan_ids[0]),
                "ann_id": str(ann_ids[index]),
                "layer_count": len(per_layer),
                "oracle_iou": float(ious.max()),
                "final_iou": float(ious[final_selected]),
                "consensus_iou": float(ious[consensus_selected]),
                "control_mean_iou": statistics.fmean(
                    float(ious[value]) for value in control_selected),
                "final_selected_index": final_selected,
                "consensus_selected_index": consensus_selected,
                "control_selected_indices": control_selected,
            })
        return {"rows": rows}
    return predictor


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
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene_index"]].append(row)
    return {
        "expressions": len(rows),
        "scenes": len(scenes),
        "layer_count": rows[0]["layer_count"] if rows else None,
        "mean_final_iou": statistics.fmean(
            row["final_iou"] for row in rows) if rows else None,
        "mean_consensus_iou": statistics.fmean(
            row["consensus_iou"] for row in rows) if rows else None,
        "mean_control_iou": statistics.fmean(
            row["control_mean_iou"] for row in rows) if rows else None,
        "selected_flip_count": sum(
            row["final_selected_index"] != row["consensus_selected_index"]
            for row in rows),
        "paired_vs_final": paired_scene_summary(
            rows, "consensus_iou", "final_iou") if rows else None,
        "paired_vs_control": paired_scene_summary(
            rows, "consensus_iou", "control_mean_iou") if rows else None,
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
    model.predict_by_feat = types.MethodType(build_predictor(), model)
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
                left["final_selected_index"] != right["final_selected_index"]
                or left["consensus_selected_index"]
                != right["consensus_selected_index"]
                or left["control_selected_indices"]
                != right["control_selected_indices"]
            ):
                repeat_identity_mismatch += 1
        rows.extend(first)
    summary = summarize(rows)
    passed = bool(
        summary["paired_vs_final"]["scene_mean_paired_delta"] > 0
        and summary["paired_vs_control"]["scene_mean_paired_delta"] > 0
        and summary["paired_vs_final"]["sign_test_p_two_sided"] < 0.05
        and repeat_identity_mismatch == 0
    )
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "control_sets": CONTROL_SETS,
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
        "control_sets": result["control_sets"],
        "repeat_identity_mismatch": result["repeat_identity_mismatch"],
        "passed": result["passed"],
        "summary": summary,
        "peak_inference_reserved_gib": result["peak_inference_reserved_gib"],
    }, indent=2))


if __name__ == "__main__":
    main()
