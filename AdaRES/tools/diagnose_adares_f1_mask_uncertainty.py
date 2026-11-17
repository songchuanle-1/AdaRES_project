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


def predictor():
    def predict_by_feat(
        self, scan_ids, object_idss, ann_ids, out, superpoints,
        gt_pmasks, gt_spmasks, fps_seed_sp,
    ):
        rows = []
        for index, target in enumerate(gt_pmasks):
            valid = ~out["batch_mask"][index]
            logits = out["masks"][index][:, valid]
            probabilities = logits.sigmoid()
            entropy = F.binary_cross_entropy_with_logits(
                logits, probabilities, reduction="none"
            ).mean(-1)
            masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            truth = target.gt(0.5).to(masks.device)
            ious = ((masks & truth).sum(-1).float() /
                    (masks | truth).sum(-1).clamp_min(1).float())
            scores = F.softmax(out["indis"][index], dim=-1)[:, 1]
            selected = int(scores.argmax())
            oracle = int(ious.argmax())
            rows.append({
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "selected_entropy": float(entropy[selected]),
                "oracle_entropy": float(entropy[oracle]),
                "entropy_argmin_iou": float(ious[entropy.argmin()]),
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
        with torch.no_grad():
            rows.extend(model(dataset.collate_fn([dataset[index]]), mode="predict")["rows"])
    if not rows:
        raise RuntimeError("No expressions in fixed subset")
    errors = [row for row in rows if row["oracle_iou"] - row["selected_iou"] > 0.05]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "error_oracle_lower_entropy": sum(
            row["oracle_entropy"] < row["selected_entropy"] for row in errors
        ),
        "error_selected_entropy_mean": statistics.fmean(
            row["selected_entropy"] for row in errors
        ),
        "error_oracle_entropy_mean": statistics.fmean(
            row["oracle_entropy"] for row in errors
        ),
        "entropy_argmin_miou": statistics.fmean(
            row["entropy_argmin_iou"] for row in rows
        ),
        "selected_miou": statistics.fmean(row["selected_iou"] for row in rows),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
