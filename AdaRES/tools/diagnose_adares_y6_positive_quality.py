import argparse
import json
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


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    _, expanded_pos, cached_fps, offsets, expanded_labels, _ = self.probe_expand
    if not torch.equal(cached_fps, fps_seed_sp):
        raise RuntimeError("FPS indices changed")
    rows = []
    for index, truth in enumerate(gt_pmasks):
        sp_pos = expanded_pos[int(offsets[index]):int(offsets[index + 1])]
        sp_labels = expanded_labels[index].to(sp_pos.device)
        fps = fps_seed_sp[index].long()
        sampled = out["sample_inds"][index].long()
        sampled_pos = sp_pos[fps][sampled]
        sampled_labels = sp_labels[fps][sampled]
        obj_ids = object_idss[index]
        masks = torch.stack([sp_labels == obj_id for obj_id in obj_ids])
        if not masks.any(-1).all():
            raise RuntimeError("Target instance absent")
        centroids = torch.stack([
            scatter_mean(sp_pos, mask.long(), dim=0)[1] for mask in masks
        ])
        _, _, indices, _ = self.criterion.match_sampled_ins(
            sampled_labels, obj_ids, sampled_pos, centroids
        )
        scores = out["indis"][index].softmax(-1)[:, 1]
        points = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(points.device)
        ious = (points & target).sum(-1).float() / (
            (points | target).sum(-1).clamp_min(1).float())
        positive = torch.zeros_like(scores, dtype=torch.bool)
        positive[indices[0].long()] = True

        def measure(membership):
            count = int(membership.sum())
            if count < 2:
                return None
            ids = membership.nonzero().flatten()
            best = float(ious[ids].max())
            worst = float(ious[ids].min())
            top = int(ids[scores[ids].argmax()])
            return {
                "count": count,
                "range": best - worst,
                "score_to_best_gap": best - float(ious[top]),
                "score_misordered": best - float(ious[top]) > 0.05,
            }

        rows.append({
            "scene_index": self.probe_scene,
            "expression_index": index,
            "scan_id": str(scan_ids[0]),
            "ann_id": str(ann_ids[index]),
            "selected_iou": float(ious[int(scores.argmax())]),
            "oracle_iou": float(ious.max()),
            "true_positive": measure(positive),
            "shifted_positive": measure(positive.roll(1)),
        })
    return {"rows": rows}


def summarize(rows):
    result = {}
    for name, subset in (
        ("all", rows),
        ("material_errors", [
            row for row in rows
            if row["oracle_iou"] - row["selected_iou"] > 0.05
        ]),
    ):
        entry = {"expressions": len(subset)}
        for kind in ("true_positive", "shifted_positive"):
            eligible = [row for row in subset if row[kind] is not None]
            by_scene = defaultdict(list)
            for row in eligible:
                by_scene[row["scene_index"]].append(row[kind])
            entry[kind] = {
                "eligible": len(eligible),
                "scenes": len(by_scene),
                "mean_count": statistics.fmean(row[kind]["count"] for row in eligible)
                if eligible else None,
            }
            for metric in ("range", "score_to_best_gap", "score_misordered"):
                entry[kind]["mean_" + metric] = statistics.fmean(
                    row[kind][metric] for row in eligible) if eligible else None
                entry[kind]["scene_mean_" + metric] = statistics.fmean(
                    statistics.fmean(item[metric] for item in scene)
                    for scene in by_scene.values()) if by_scene else None
        result[name] = entry
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
    loaded = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    allowed = {"text_encoder.embeddings.position_ids",
               "text_encoder.embeddings.token_type_ids"}
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.eval()
    original_expand = model.expand_and_fps

    def expand_and_capture(self, *args, **kwargs):
        result = original_expand(*args, **kwargs)
        self.probe_expand = result
        return result

    model.expand_and_fps = types.MethodType(expand_and_capture, model)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        if len(first) != len(second):
            raise RuntimeError("Repeated expression count mismatch")
        mismatch += sum(a != b for a, b in zip(first, second))
        rows.extend(first)
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "repeat_mismatch": mismatch,
        "summary": summarize(rows),
        "peak_inference_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
