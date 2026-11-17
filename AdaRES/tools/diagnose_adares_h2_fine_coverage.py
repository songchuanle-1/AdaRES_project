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
            truth = gt_spmasks[index].gt(0.5) & valid
            coverages = []
            first_balanced_coverage = None
            first_balanced_unique = None
            for layer, attention in enumerate(self.probe_attentions):
                attention = attention[index] * valid.float()
                aggregate = attention.sum(-2)
                n = min(self.dec.num_proposal_fine, aggregate.numel())
                chosen = aggregate.topk(n).indices
                coverages.append(float(truth[chosen].sum() / truth.sum().clamp_min(1)))
                if layer == 0:
                    per_query = max(1, n // attention.shape[0])
                    reserved = attention.topk(per_query, dim=-1).indices.flatten().unique()
                    first_balanced_unique = int(reserved.numel())
                    remaining = aggregate.masked_fill(
                        torch.zeros_like(valid).scatter_(0, reserved, True),
                        float("-inf"),
                    ).topk(n - reserved.numel()).indices
                    balanced = torch.cat((reserved, remaining))
                    first_balanced_coverage = float(
                        truth[balanced].sum() / truth.sum().clamp_min(1)
                    )
            masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            point_truth = target.gt(0.5).to(masks.device)
            ious = (masks & point_truth).sum(-1).float() / (
                (masks | point_truth).sum(-1).clamp_min(1).float()
            )
            scores = F.softmax(out["indis"][index], dim=-1)[:, 1]
            selected, oracle = int(scores.argmax()), int(ious.argmax())
            rows.append({
                "scene_index": self.probe_scene_index,
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "valid_superpoints": int(valid.sum()),
                "target_superpoints": int(truth.sum()),
                "coverage_by_layer": coverages,
                "first_balanced_coverage": first_balanced_coverage,
                "first_balanced_unique": first_balanced_unique,
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
    model.probe_attentions = [None] * model.dec.num_layer
    for layer, module in enumerate(model.dec.qia_layers):
        def capture(_, inputs, output, layer=layer):
            model.probe_attentions[layer] = output[1]
        module.register_forward_hook(capture)
    model.predict_by_feat = types.MethodType(predictor(), model)
    rows = []
    for index in range(min(args.scenes, len(dataset))):
        model.probe_scene_index = index
        with torch.no_grad():
            rows.extend(model(dataset.collate_fn([dataset[index]]), mode="predict")["rows"])
    errors = [row for row in rows if row["oracle_iou"] - row["selected_iou"] > 0.05]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "mean_coverage_by_layer": [
            statistics.fmean(row["coverage_by_layer"][layer] for row in rows)
            for layer in range(model.dec.num_layer)
        ],
        "error_mean_coverage_by_layer": [
            statistics.fmean(row["coverage_by_layer"][layer] for row in errors)
            for layer in range(model.dec.num_layer)
        ],
        "error_final_coverage_below_half": sum(
            row["coverage_by_layer"][-1] < 0.5 for row in errors
        ),
        "mean_first_balanced_coverage": statistics.fmean(
            row["first_balanced_coverage"] for row in rows
        ),
        "error_first_balanced_coverage": statistics.fmean(
            row["first_balanced_coverage"] for row in errors
        ),
        "error_balanced_better": sum(
            row["first_balanced_coverage"] > row["coverage_by_layer"][0]
            for row in errors
        ),
        "error_balanced_worse": sum(
            row["first_balanced_coverage"] < row["coverage_by_layer"][0]
            for row in errors
        ),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
