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
from ipdn.utils import get_root_logger


def selection_predictor():
    def predict_by_feat(
        self, scan_ids, object_idss, ann_ids, out, superpoints,
        gt_pmasks, gt_spmasks, fps_seed_sp,
    ):
        rows = []
        for index, target in enumerate(gt_pmasks):
            masks = out["masks"][index].sigmoid().gt(0.5)
            scores = F.softmax(out["indis"][index], dim=-1)[:, 1]
            selected = int(scores.argmax())
            point_masks = masks[:, superpoints]
            truth = target.gt(0.5).to(point_masks.device)
            intersection = (point_masks & truth).sum(-1).float()
            union = (point_masks | truth).sum(-1).float()
            ious = intersection / union.clamp_min(1)
            oracle = int(ious.argmax())
            pair_intersection = (masks[selected] & masks[oracle]).sum().float()
            pair_union = (masks[selected] | masks[oracle]).sum().float()
            rows.append({
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "selected_oracle_overlap": float(
                    pair_intersection / pair_union.clamp_min(1)
                ),
                "oracle_score_rank": int(
                    (scores.argsort(descending=True) == oracle)
                    .nonzero(as_tuple=True)[0].item()
                ) + 1,
                "selected_query": selected,
                "oracle_query": oracle,
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
    parser.add_argument(
        "--model",
        choices=("adares_b4_query_competition", "adares_g1_metric_targets"),
        default="adares_b4_query_competition",
    )
    args = parser.parse_args()

    if args.model == "adares_g1_metric_targets":
        from ipdn.model_paper5.adares_g1_metric_targets import IPDN
    else:
        from ipdn.model_paper5.adares_b4_query_competition import IPDN

    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = cfg.data.val.copy()
    data_cfg.training = False
    data_cfg.aug = False
    data_cfg.with_elastic = False
    data_cfg.lang_num_max = 32
    logger = get_root_logger()
    dataset = build_dataset(data_cfg, logger)
    model = IPDN(**cfg.model).cuda()
    model.eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    loaded = model.load_state_dict(state, strict=False)
    allowed = {
        "text_encoder.embeddings.position_ids",
        "text_encoder.embeddings.token_type_ids",
    }
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.predict_by_feat = types.MethodType(selection_predictor(), model)

    rows = []
    for index in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[index]])
        with torch.no_grad():
            rows.extend(model(batch, mode="predict")["rows"])
    if not rows:
        raise RuntimeError("No expressions in diagnostic subset")
    errors = [row for row in rows if row["oracle_iou"] - row["selected_iou"] > 0.05]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "top1_miou": statistics.fmean(row["selected_iou"] for row in rows),
        "oracle_miou": statistics.fmean(row["oracle_iou"] for row in rows),
        "material_error_count": len(errors),
        "error_pair_overlap_mean": (
            statistics.fmean(row["selected_oracle_overlap"] for row in errors)
            if errors else None
        ),
        "error_pair_overlap_ge_half": sum(
            row["selected_oracle_overlap"] >= 0.5 for row in errors
        ),
        "error_oracle_rank_le_three": sum(
            row["oracle_score_rank"] <= 3 for row in errors
        ),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
