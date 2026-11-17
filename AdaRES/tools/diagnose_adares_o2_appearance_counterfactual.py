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


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    rows = []
    for index, truth in enumerate(gt_pmasks):
        masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(masks.device)
        ious = (masks & target).sum(-1).float() / (
            (masks | target).sum(-1).clamp_min(1).float()
        )
        confidence = F.softmax(out["indis"][index], dim=-1)[:, 1]
        selected = int(confidence.argmax())
        rows.append({
            "selected": selected,
            "iou": float(ious[selected]),
            "oracle_iou": float(ious.max()),
            "mask": torch.where(masks[selected])[0].tolist(),
            "confidence": float(confidence[selected]),
        })
    return {"rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="train")
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
    model.eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loaded = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    allowed = {"text_encoder.embeddings.position_ids",
               "text_encoder.embeddings.token_type_ids"}
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            base = model(batch, mode="predict")["rows"]
            repeat = model(batch, mode="predict")["rows"]
            batch["feats_2d"] = torch.zeros_like(batch["feats_2d"])
            ablated = model(batch, mode="predict")["rows"]
        for original, control, changed in zip(base, repeat, ablated):
            def overlap(left, right):
                a, b = set(left["mask"]), set(right["mask"])
                return len(a & b) / max(1, len(a | b))
            rows.append({
                "scene": scene,
                "base_iou": original["iou"],
                "oracle_iou": original["oracle_iou"],
                "repeat_iou": control["iou"],
                "counterfactual_iou": changed["iou"],
                "repeat_mask_iou": overlap(original, control),
                "counterfactual_mask_iou": overlap(original, changed),
                "query_flip": original["selected"] != changed["selected"],
                "base_confidence": original["confidence"],
                "counterfactual_confidence": changed["confidence"],
            })
    errors = [r for r in rows if r["oracle_iou"] - r["base_iou"] > 0.05]
    correct = [r for r in rows if r["oracle_iou"] - r["base_iou"] <= 0.05]
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": sum(r["repeat_mask_iou"] < 1 for r in rows),
        "flips_errors": sum(r["query_flip"] for r in errors),
        "flips_correct": sum(r["query_flip"] for r in correct),
        "overlap_errors": statistics.fmean(r["counterfactual_mask_iou"] for r in errors),
        "overlap_correct": statistics.fmean(r["counterfactual_mask_iou"] for r in correct),
        "base_miou": statistics.fmean(r["base_iou"] for r in rows),
        "counterfactual_miou": statistics.fmean(r["counterfactual_iou"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
