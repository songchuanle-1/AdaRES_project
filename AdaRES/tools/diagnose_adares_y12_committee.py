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
from ipdn.model_paper5.adares_b4_query_competition import IPDN as B4IPDN
from ipdn.model_paper5.adares_w2_pointmass_competition import IPDN as W2IPDN
from ipdn.utils import get_root_logger


CONTROL_PERMUTATIONS = 8
ALLOWED_MISSING_KEYS = {
    "text_encoder.embeddings.position_ids",
    "text_encoder.embeddings.token_type_ids",
}


def sign_test_p_value(positive, negative):
    trials = positive + negative
    if trials == 0:
        return 1.0, 0
    tail = sum(
        math.comb(trials, k)
        for k in range(0, min(positive, negative) + 1)
    )
    return min(1.0, 2.0 * tail / (2 ** trials)), trials


def standardize(values):
    mean = values.mean()
    std = values.std(unbiased=False)
    if not torch.isfinite(std) or float(std) == 0.0:
        raise RuntimeError("Cannot standardize a constant indicator margin")
    return (values - mean) / std


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys or set(loaded.unexpected_keys) - ALLOWED_MISSING_KEYS:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")


def make_control_permutations(query_count, device):
    generator = torch.Generator()
    generator.manual_seed(1999)
    return [
        torch.randperm(query_count, generator=generator).to(device)
        for _ in range(CONTROL_PERMUTATIONS)
    ]


def w2_predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
                 gt_pmasks, gt_spmasks, fps_seed_sp):
    rows = []
    for index, truth in enumerate(gt_pmasks):
        logits = out["indis"][index]
        margins = (logits[:, 1] - logits[:, 0]).detach()
        points = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(points.device)
        ious = (points & target).sum(-1).float() / (
            (points | target).sum(-1).clamp_min(1).float())
        rows.append({
            "expression_index": index,
            "scan_id": str(scan_ids[0]),
            "ann_id": str(ann_ids[index]),
            "query_count": int(margins.numel()),
            "margins": margins.cpu().tolist(),
            "ious": ious.cpu().tolist(),
        })
    return {"rows": rows}


def b4_predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
                 gt_pmasks, gt_spmasks, fps_seed_sp):
    rows = []
    for index, _ in enumerate(gt_pmasks):
        logits = out["indis"][index]
        margins = (logits[:, 1] - logits[:, 0]).detach()
        rows.append({
            "expression_index": index,
            "scan_id": str(scan_ids[0]),
            "ann_id": str(ann_ids[index]),
            "query_count": int(margins.numel()),
            "margins": margins.cpu().tolist(),
        })
    return {"rows": rows}


def combine_rows(w2_rows, b4_rows, permutations):
    if len(w2_rows) != len(b4_rows):
        raise RuntimeError("W2 and B4 expression counts disagree")
    combined = []
    for w2_row, b4_row in zip(w2_rows, b4_rows):
        if w2_row["ann_id"] != str(b4_row["ann_id"]):
            raise RuntimeError("W2 and B4 ann_id order disagree")
        if w2_row["query_count"] != b4_row["query_count"]:
            raise RuntimeError("W2 and B4 query counts disagree")
        w2_margin = torch.tensor(
            w2_row["margins"], dtype=torch.float32,
            device=permutations[0].device)
        b4_margin = torch.tensor(
            b4_row["margins"], dtype=torch.float32,
            device=permutations[0].device)
        ious = torch.tensor(
            w2_row["ious"], dtype=torch.float32,
            device=permutations[0].device)
        w2_standard = standardize(w2_margin)
        b4_standard = standardize(b4_margin)
        committee = w2_standard + b4_standard
        w2_selected = int(w2_standard.argmax())
        b4_selected = int(b4_standard.argmax())
        committee_selected = int(committee.argmax())
        control_selected = [
            int((w2_standard + b4_standard[permutation]).argmax())
            for permutation in permutations
        ]
        combined.append({
            "expression_index": w2_row["expression_index"],
            "scan_id": w2_row["scan_id"],
            "ann_id": w2_row["ann_id"],
            "query_count": w2_row["query_count"],
            "w2_selected_index": w2_selected,
            "b4_selected_index": b4_selected,
            "committee_selected_index": committee_selected,
            "control_selected_indices": control_selected,
            "w2_iou": float(ious[w2_selected]),
            "b4_query_on_w2_iou": float(ious[b4_selected]),
            "committee_iou": float(ious[committee_selected]),
            "control_mean_iou": statistics.fmean(
                float(ious[index]) for index in control_selected),
            "oracle_iou": float(ious.max()),
            "w2_b4_disagree": int(w2_selected != b4_selected),
            "committee_equals_w2": int(committee_selected == w2_selected),
        })
    return combined


def paired_scene_summary(rows, true_key, control_key):
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene_index"]].append(row)
    deltas = []
    for scene_rows in scenes.values():
        true_value = statistics.fmean(row[true_key] for row in scene_rows)
        control_value = statistics.fmean(row[control_key] for row in scene_rows)
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
        "query_count": rows[0]["query_count"] if rows else None,
        "mean_w2_iou": statistics.fmean(row["w2_iou"] for row in rows),
        "mean_b4_query_on_w2_iou": statistics.fmean(
            row["b4_query_on_w2_iou"] for row in rows),
        "mean_committee_iou": statistics.fmean(
            row["committee_iou"] for row in rows),
        "mean_control_iou": statistics.fmean(
            row["control_mean_iou"] for row in rows),
        "mean_oracle_iou": statistics.fmean(
            row["oracle_iou"] for row in rows),
        "w2_b4_disagree_count": sum(
            row["w2_b4_disagree"] for row in rows),
        "committee_equals_w2_count": sum(
            row["committee_equals_w2"] for row in rows),
        "paired_vs_w2": paired_scene_summary(
            rows, "committee_iou", "w2_iou"),
        "paired_vs_control": paired_scene_summary(
            rows, "committee_iou", "control_mean_iou"),
        "informational_b4_vs_w2": paired_scene_summary(
            rows, "b4_query_on_w2_iou", "w2_iou"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2-config", required=True)
    parser.add_argument("--w2-checkpoint", required=True)
    parser.add_argument("--b4-config", required=True)
    parser.add_argument("--b4-checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--scenes", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    w2_cfg = gorilla.Config.fromfile(args.w2_config)
    b4_cfg = gorilla.Config.fromfile(args.b4_config)
    data_cfg = getattr(w2_cfg.data, args.split).copy()
    data_cfg.training = False
    data_cfg.aug = False
    data_cfg.with_elastic = False
    data_cfg.lang_num_max = 32
    dataset = build_dataset(data_cfg, get_root_logger())

    w2_model = W2IPDN(**w2_cfg.model).cuda()
    load_model(w2_model, args.w2_checkpoint)
    w2_model.eval()
    w2_model.predict_by_feat = types.MethodType(w2_predictor, w2_model)
    b4_model = B4IPDN(**b4_cfg.model).cuda()
    load_model(b4_model, args.b4_checkpoint)
    b4_model.eval()
    b4_model.predict_by_feat = types.MethodType(b4_predictor, b4_model)

    rows = []
    repeat_identity_mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            w2_first = w2_model(batch, mode="predict")["rows"]
            b4_first = b4_model(batch, mode="predict")["rows"]
            w2_second = w2_model(batch, mode="predict")["rows"]
            b4_second = b4_model(batch, mode="predict")["rows"]
        query_count = w2_first[0]["query_count"] if w2_first else 0
        permutations = make_control_permutations(
            query_count, torch.device("cuda"))
        first = combine_rows(w2_first, b4_first, permutations)
        second = combine_rows(w2_second, b4_second, permutations)
        if len(first) != len(second):
            raise RuntimeError("Repeated expression count mismatch")
        for left, right in zip(first, second):
            if (
                left["w2_selected_index"] != right["w2_selected_index"]
                or left["b4_selected_index"] != right["b4_selected_index"]
                or left["committee_selected_index"]
                != right["committee_selected_index"]
                or left["control_selected_indices"]
                != right["control_selected_indices"]
            ):
                repeat_identity_mismatch += 1
        for row in first:
            row["scene_index"] = scene
        rows.extend(first)

    summary = summarize(rows)
    passed = bool(
        summary["paired_vs_w2"]["scene_mean_paired_delta"] > 0
        and summary["paired_vs_control"]["scene_mean_paired_delta"] > 0
        and summary["paired_vs_w2"]["sign_test_p_two_sided"] < 0.05
        and repeat_identity_mismatch == 0
    )
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "control_permutations": CONTROL_PERMUTATIONS,
        "standardization": "population std (unbiased=False)",
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
