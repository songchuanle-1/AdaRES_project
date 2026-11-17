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
            masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            truth = target.gt(0.5).to(masks.device)
            ious = (masks & truth).sum(-1).float() / (
                (masks | truth).sum(-1).clamp_min(1).float()
            )
            indicator = F.softmax(out["indis"][index], dim=-1)[:, 1]
            quality = out["scores"][index].sigmoid().flatten()
            selected = int(indicator.argmax())
            score_selected = int(quality.argmax())
            oracle = int(ious.argmax())
            rows.append({
                "scene_index": self.probe_scene_index,
                "indicator_iou": float(ious[selected]),
                "score_iou": float(ious[score_selected]),
                "oracle_iou": float(ious[oracle]),
                "indicator_query": selected,
                "score_query": score_selected,
                "oracle_query": oracle,
                "indicator_score_quality": float(quality[selected]),
                "oracle_score_quality": float(quality[oracle]),
                "indicator_confidence": float(indicator[selected]),
                "oracle_confidence": float(indicator[oracle]),
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
    errors = [row for row in rows if row["oracle_iou"] - row["indicator_iou"] > 0.05]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "indicator_miou": statistics.fmean(r["indicator_iou"] for r in rows),
        "score_miou": statistics.fmean(r["score_iou"] for r in rows),
        "score_better_on_errors": sum(
            r["score_iou"] > r["indicator_iou"] for r in errors
        ),
        "score_worse_on_errors": sum(
            r["score_iou"] < r["indicator_iou"] for r in errors
        ),
        "oracle_quality_gt_selected_on_errors": sum(
            r["oracle_score_quality"] > r["indicator_score_quality"]
            for r in errors
        ),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
