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
            superpoint_masks = out["masks"][index].sigmoid().gt(0.5)
            masks = superpoint_masks[:, superpoints]
            truth = target.gt(0.5).to(masks.device)
            ious = (masks & truth).sum(-1).float() / (
                (masks | truth).sum(-1).clamp_min(1).float()
            )
            confidence = F.softmax(out["indis"][index], dim=-1)[:, 1]
            sampler = out["ref_scores"][index][out["sample_inds"][index]]
            selected = int(confidence.argmax())
            oracle = int(ious.argmax())
            rows.append({
                "scene_index": self.probe_scene_index,
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "sampler_only_iou": float(ious[sampler.argmax()]),
                "reverse_rank_iou": float(ious[sampler.argmin()]),
                "largest_mask_iou": float(ious[masks.sum(-1).argmax()]),
                "smallest_mask_iou": float(ious[masks.sum(-1).argmin()]),
                "selected_area": int(masks[selected].sum()),
                "oracle_area": int(masks[oracle].sum()),
                "target_area": int(truth.sum()),
                "selected_rank": selected + 1,
                "oracle_rank": oracle + 1,
                "selected_sampler_score": float(sampler[selected]),
                "oracle_sampler_score": float(sampler[oracle]),
                "selected_confidence": float(confidence[selected]),
                "oracle_confidence": float(confidence[oracle]),
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
    args = parser.parse_args()

    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = cfg.data.val.copy()
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
    model.predict_by_feat = types.MethodType(predictor(), model)
    rows = []
    for index in range(min(args.scenes, len(dataset))):
        model.probe_scene_index = index
        with torch.no_grad():
            rows.extend(model(dataset.collate_fn([dataset[index]]), mode="predict")["rows"])
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "sampler_only_miou": statistics.fmean(r["sampler_only_iou"] for r in rows),
        "reverse_rank_miou": statistics.fmean(r["reverse_rank_iou"] for r in rows),
        "largest_mask_miou": statistics.fmean(r["largest_mask_iou"] for r in rows),
        "smallest_mask_miou": statistics.fmean(r["smallest_mask_iou"] for r in rows),
        "error_oracle_later": sum(
            r["oracle_rank"] > r["selected_rank"] for r in errors
        ),
        "error_oracle_sampler_higher": sum(
            r["oracle_sampler_score"] > r["selected_sampler_score"]
            for r in errors
        ),
        "error_sampler_better": sum(
            r["sampler_only_iou"] > r["selected_iou"] for r in errors
        ),
        "error_sampler_worse": sum(
            r["sampler_only_iou"] < r["selected_iou"] for r in errors
        ),
        "error_oracle_larger": sum(
            r["oracle_area"] > r["selected_area"] for r in errors
        ),
        "error_largest_better": sum(
            r["largest_mask_iou"] > r["selected_iou"] for r in errors
        ),
        "error_largest_worse": sum(
            r["largest_mask_iou"] < r["selected_iou"] for r in errors
        ),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
