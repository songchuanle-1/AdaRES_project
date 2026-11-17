"""Train-only mask-pool complementarity versus repeated and wrong-language pools."""

import argparse
import json
import statistics
import time
import types
from collections import defaultdict
from pathlib import Path

import gorilla
import torch

from diagnose_adares_y14_reliability import (
    capture_predictor, load_model, scene_indices, sha256, weighted_iou,
)
from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_b4_query_competition import IPDN as B4
from ipdn.model_paper5.adares_w2_pointmass_competition import IPDN as W2
from ipdn.utils import get_root_logger
from scipy.optimize import linear_sum_assignment


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    result = capture_predictor(self, scan_ids, object_idss, ann_ids, out,
                               superpoints, gt_pmasks, gt_spmasks, fps_seed_sp)
    sp = superpoints.detach().cpu().long()
    for row, truth in zip(result["rows"], gt_pmasks):
        row["target_mass"] = torch.bincount(
            sp, weights=truth.detach().cpu().gt(0.5).float(),
            minlength=row["mass"].numel())
    return result


def ious_for_target(binary, mass, target_mass):
    intersection = binary.float() @ target_mass
    union = binary.float() @ mass + target_mass.sum() - intersection
    return intersection / union.clamp_min(1)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scan_id"]].append(row)
    output = {"expressions": len(rows), "actual_scans": len(grouped)}
    for key in ("cross_gain", "repeat_gain", "cross_minus_repeat",
                "wrong_gain", "cross_minus_wrong"):
        eligible = [r[key] for r in rows if r[key] is not None]
        scenes = [
            statistics.fmean(r[key] for r in group if r[key] is not None)
            for group in grouped.values()
            if any(r[key] is not None for r in group)]
        output[key] = {
            "expressions": len(eligible), "scans": len(scenes),
            "expression_mean": statistics.fmean(eligible) if eligible else None,
            "scene_mean": statistics.fmean(scenes) if scenes else None,
            "positive_scans": sum(d > 0 for d in scenes),
            "negative_scans": sum(d < 0 for d in scenes),
        }
    output["material_teacher_benefits"] = sum(r["material"] for r in rows)
    output["material_repeat_flips"] = sum(r["material_repeat_flip"] for r in rows)
    output["teacher_oracle_identity_mismatches"] = sum(
        r["teacher_oracle_mismatch"] for r in rows)
    output["oracle_gap_max_repeat_change"] = max(
        r["oracle_gap_repeat_change"] for r in rows)
    output["teacher_oracle_minus_aligned_student"] = statistics.fmean(
        r["teacher_oracle_minus_aligned_student"] for r in rows)
    return output


def main():
    parser = argparse.ArgumentParser()
    for key in ("w2-config", "b4-config", "w2-checkpoint", "b4-checkpoint",
                "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.w2_config)
    data_cfg = cfg.data.train.copy()
    data_cfg.training, data_cfg.aug, data_cfg.with_elastic = False, False, False
    data_cfg.lang_num_max = 32
    dataset = build_dataset(data_cfg, get_root_logger())
    models = [load_model(W2, cfg, args.w2_checkpoint),
              load_model(B4, gorilla.Config.fromfile(args.b4_config),
                         args.b4_checkpoint)]
    for model in models:
        model.predict_by_feat = types.MethodType(predictor, model)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    start = time.monotonic()
    for scene, item in enumerate(scene_indices(dataset.scene_inputs, 64)):
        batch = dataset.collate_fn([dataset[item]])
        with torch.no_grad():
            w1 = models[0](batch, mode="predict")["rows"]
            b1 = models[1](batch, mode="predict")["rows"]
            w2 = models[0](batch, mode="predict")["rows"]
            b2 = models[1](batch, mode="predict")["rows"]
        if not len(w1) == len(b1) == len(w2) == len(b2):
            raise ValueError("Expression count mismatch")
        for index, (a, b, ar, br) in enumerate(zip(w1, b1, w2, b2)):
            identities = {(r["scan_id"], r["object_ids"], r["ann_id"])
                          for r in (a, b, ar, br)}
            if len(identities) != 1:
                raise ValueError("Expression alignment mismatch")
            ao, bo = float(a["ious"].max()), float(b["ious"].max())
            aro, bro = float(ar["ious"].max()), float(br["ious"].max())
            cross, repeat = max(ao, bo)-ao, max(ao, aro)-ao
            wrong = None
            for offset in range(1, len(b1)):
                candidate = b1[(index + offset) % len(b1)]
                if set(candidate["object_ids"]) != set(a["object_ids"]):
                    wo = float(ious_for_target(
                        candidate["binary"], a["mass"], a["target_mass"]).max())
                    wrong = max(ao, wo) - ao
                    break
            teacher_oracle = int(b["ious"].argmax())
            similarity = weighted_iou(a["binary"], b["binary"], a["mass"])
            left, right = linear_sum_assignment(-similarity.numpy())
            inverse = {int(j): int(i) for i, j in zip(left, right)}
            rows.append({
                "scene_index": scene, "scan_id": a["scan_id"],
                "ann_id": a["ann_id"], "object_ids": a["object_ids"],
                "w2_oracle": ao, "b4_oracle": bo, "cross_gain": cross,
                "repeat_gain": repeat, "cross_minus_repeat": cross-repeat,
                "wrong_gain": wrong,
                "cross_minus_wrong": cross-wrong if wrong is not None else None,
                "material": bo-ao > 0.05,
                "material_repeat_flip": (bo-ao > 0.05) != (bro-aro > 0.05),
                "teacher_oracle_mismatch": teacher_oracle != int(br["ious"].argmax()),
                "oracle_gap_repeat_change": abs((bo-ao)-(bro-aro)),
                "teacher_oracle_minus_aligned_student":
                    bo-float(a["ious"][inverse[teacher_oracle]]),
            })
        print(json.dumps({"scene": scene, "expressions": len(rows),
                          "elapsed_s": time.monotonic()-start}), flush=True)
    summary = summarize(rows)
    supported = (summary["cross_minus_repeat"]["scene_mean"] > 0
                 and summary["cross_minus_wrong"]["scene_mean"] is not None
                 and summary["cross_minus_wrong"]["scene_mean"] > 0
                 and summary["material_repeat_flips"] == 0)
    report = {
        "hypothesis": "complementary-mask-candidate-pool",
        "summary": summary, "mechanism_supported": supported,
        "rows": rows, "elapsed_s": time.monotonic()-start,
        "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
        "sources": {key: {"path": getattr(args, key),
                         "sha256": sha256(getattr(args, key))}
                    for key in ("w2_config", "b4_config",
                                "w2_checkpoint", "b4_checkpoint")},
        "scope": "train-only oracle diagnosis; not inference or trained efficacy",
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "mechanism_supported": supported}),
          flush=True)


if __name__ == "__main__":
    main()
