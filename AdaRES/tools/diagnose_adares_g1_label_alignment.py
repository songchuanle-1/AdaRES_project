import argparse
import json
import statistics
import sys
import types
from pathlib import Path

import gorilla
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_b4_query_competition import IPDN
from ipdn.utils import get_root_logger


def predictor():
    def predict_by_feat(
        self, scan_ids, object_idss, ann_ids, out, superpoints,
        gt_pmasks, gt_spmasks, fps_seed_sp,
    ):
        rows = []
        for index, target in enumerate(gt_pmasks):
            valid = ~out["batch_mask"][index]
            coords = self.probe_coords[index, valid]
            labels = self.probe_labels[0]
            sampled = fps_seed_sp[index].long()[out["sample_inds"][index].long()]
            sampled_labels = labels[sampled]
            sampled_pos = coords[sampled]
            object_ids = object_idss[index]
            object_centers = []
            for object_id in object_ids:
                object_points = coords[labels == object_id]
                if object_points.numel() == 0:
                    raise RuntimeError(f"Missing object {object_id} in scene")
                object_centers.append(object_points.mean(0))
            nearest = torch.cdist(
                sampled_pos, torch.stack(object_centers)
            ).argmin(0)
            positives = torch.zeros_like(sampled_labels, dtype=torch.bool)
            for object_id, closest in zip(object_ids, nearest):
                positives |= sampled_labels == object_id
                positives[closest] = True

            masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            truth = target.gt(0.5).to(masks.device)
            ious = (masks & truth).sum(-1).float() / (
                (masks | truth).sum(-1).clamp_min(1).float()
            )
            scores = F.softmax(out["indis"][index], dim=-1)[:, 1]
            selected = int(scores.argmax())
            oracle = int(ious.argmax())
            positive_oracle = float(ious[positives].max()) if positives.any() else 0.0
            negative_oracle = float(ious[~positives].max()) if (~positives).any() else 0.0
            positive_selected = int(scores.masked_fill(~positives.to(scores.device), -1).argmax()) if positives.any() else selected
            rows.append({
                "scene_index": self.probe_scene_index,
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "positive_oracle_iou": positive_oracle,
                "negative_oracle_iou": negative_oracle,
                "positive_selected_iou": float(ious[positive_selected]),
                "selected_positive": bool(positives[selected]),
                "oracle_positive": bool(positives[oracle]),
                "positive_count": int(positives.sum()),
                "query_count": int(positives.numel()),
                "selected_seed_object": int(sampled_labels[selected]),
                "oracle_seed_object": int(sampled_labels[oracle]),
            })
        return {"rows": rows}

    return predict_by_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--split", choices=("train", "val"), default="val")
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
    allowed = {
        "text_encoder.embeddings.position_ids",
        "text_encoder.embeddings.token_type_ids",
    }
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.eval()

    def capture(module, inputs):
        model.probe_coords = module.get_batches(inputs[5], inputs[2])[0]

    model.dec.register_forward_pre_hook(capture)
    model.predict_by_feat = types.MethodType(predictor(), model)
    rows = []
    repeat_mismatch = 0
    for index in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[index]])
        model.probe_scene_index = index
        model.probe_labels = batch["sp_ins_labels"]
        with torch.no_grad():
            base = model(batch, mode="predict")["rows"]
            repeat = model(batch, mode="predict")["rows"]
        repeat_mismatch += sum(
            a["selected_positive"] != b["selected_positive"]
            or a["oracle_positive"] != b["oracle_positive"]
            or abs(a["selected_iou"] - b["selected_iou"]) > 1e-4
            or abs(a["oracle_iou"] - b["oracle_iou"]) > 1e-4
            for a, b in zip(base, repeat)
        )
        rows.extend(base)
    if not rows:
        raise RuntimeError("No expressions in fixed diagnostic subset")
    errors = [row for row in rows if row["oracle_iou"] - row["selected_iou"] > 0.05]
    conflicting = [
        row for row in errors if row["selected_positive"] and not row["oracle_positive"]
    ]
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "repeat_mismatch": repeat_mismatch,
        "material_errors": len(errors),
        "error_selected_positive_oracle_negative": len(conflicting),
        "error_oracle_positive": sum(row["oracle_positive"] for row in errors),
        "error_selected_positive": sum(row["selected_positive"] for row in errors),
        "mean_positive_fraction": statistics.fmean(
            row["positive_count"] / row["query_count"] for row in rows
        ),
        "selected_miou": statistics.fmean(row["selected_iou"] for row in rows),
        "oracle_miou": statistics.fmean(row["oracle_iou"] for row in rows),
        "positive_oracle_miou": statistics.fmean(row["positive_oracle_iou"] for row in rows),
        "negative_oracle_miou": statistics.fmean(row["negative_oracle_iou"] for row in rows),
        "positive_selected_miou": statistics.fmean(row["positive_selected_iou"] for row in rows),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
