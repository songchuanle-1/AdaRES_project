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
sys.path.insert(0, str(Path.cwd()))

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_b4_query_competition import IPDN
from ipdn.utils import get_root_logger


def rank_correlation(x, y):
    if len(x) < 2:
        return None
    a = torch.tensor(x, dtype=torch.float64)
    b = torch.tensor(y, dtype=torch.float64)
    a = a.argsort().argsort().double()
    b = b.argsort().argsort().double()
    a -= a.mean()
    b -= b.mean()
    denom = a.norm() * b.norm()
    return float((a * b).sum() / denom) if denom > 0 else None


def predictor():
    def predict_by_feat(
        self, scan_ids, object_idss, ann_ids, out, superpoints,
        gt_pmasks, gt_spmasks, fps_seed_sp,
    ):
        rows = []
        for index, target in enumerate(gt_pmasks):
            final = out["masks"][index].sigmoid().gt(0.5)
            previous = out["aux_outputs"][-1]["masks"][index].sigmoid().gt(0.5)
            valid = ~out["batch_mask"][index]
            final = final[:, valid]
            previous = previous[:, valid]
            intersection = (final & previous).sum(-1).float()
            union = (final | previous).sum(-1).float()
            stability = intersection / union.clamp_min(1)
            point_masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            truth = target.gt(0.5).to(point_masks.device)
            iou = (point_masks & truth).sum(-1).float() / (
                (point_masks | truth).sum(-1).float().clamp_min(1)
            )
            scores = F.softmax(out["indis"][index], dim=-1)[:, 1]
            selected = int(scores.argmax())
            oracle = int(iou.argmax())
            rows.append({
                "selected_iou": float(iou[selected]),
                "oracle_iou": float(iou[oracle]),
                "selected_stability": float(stability[selected]),
                "oracle_stability": float(stability[oracle]),
                "query_stability": stability.cpu().tolist(),
                "query_iou": iou.cpu().tolist(),
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
    model = IPDN(**cfg.model)
    model.cuda()
    model.eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loaded = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    allowed = {
        "text_encoder.embeddings.position_ids",
        "text_encoder.embeddings.token_type_ids",
    }
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.predict_by_feat = types.MethodType(predictor(), model)

    rows = []
    for index in range(min(args.scenes, len(dataset))):
        with torch.no_grad():
            rows.extend(model(dataset.collate_fn([dataset[index]]), mode="predict")["rows"])
    if not rows:
        raise RuntimeError("No expressions in diagnostic subset")
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    correlations = [
        rank_correlation(r["query_stability"], r["query_iou"]) for r in rows
    ]
    correlations = [value for value in correlations if value is not None]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "mean_within_expression_rank_correlation": (
            statistics.fmean(correlations) if correlations else None
        ),
        "error_oracle_stability_above_selected": sum(
            r["oracle_stability"] > r["selected_stability"] for r in errors
        ),
        "error_selected_stability_mean": (
            statistics.fmean(r["selected_stability"] for r in errors) if errors else None
        ),
        "error_oracle_stability_mean": (
            statistics.fmean(r["oracle_stability"] for r in errors) if errors else None
        ),
        "definition": "Binary mask IoU between penultimate and final decoder predictions at the same query index, on valid superpoints only.",
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
