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
    model.eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loaded = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    allowed = {"text_encoder.embeddings.position_ids",
               "text_encoder.embeddings.token_type_ids"}
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.predict_by_feat = types.MethodType(predictor, model)
    mask_id = 50264  # RoBERTa <mask>
    rows = []
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        tokens = batch["lang_tokenss"]
        valid = batch["lang_masks"].bool()
        first = tokens.clone()
        last = tokens.clone()
        changed = []
        for index in range(tokens.shape[0]):
            positions = torch.where(valid[index] & (tokens[index] > 3))[0]
            if len(positions):
                first[index, positions[0]] = mask_id
                last[index, positions[-1]] = mask_id
                changed.append(True)
            else:
                changed.append(False)
        with torch.no_grad():
            original = model(batch, mode="predict")["rows"]
            repeat = model(batch, mode="predict")["rows"]
            batch["lang_tokenss"] = first
            first_rows = model(batch, mode="predict")["rows"]
            batch["lang_tokenss"] = last
            last_rows = model(batch, mode="predict")["rows"]
        for index, (base, control, one, two) in enumerate(
                zip(original, repeat, first_rows, last_rows)):
            def overlap(left, right):
                a, b = set(left["mask"]), set(right["mask"])
                return len(a & b) / max(1, len(a | b))
            rows.append({
                "scene": scene,
                "changed": changed[index],
                "base_iou": base["iou"],
                "oracle_iou": base["oracle_iou"],
                "repeat_iou": control["iou"],
                "first_iou": one["iou"],
                "last_iou": two["iou"],
                "repeat_mask_iou": overlap(base, control),
                "first_mask_iou": overlap(base, one),
                "last_mask_iou": overlap(base, two),
                "first_flip": one["selected"] != base["selected"],
                "last_flip": two["selected"] != base["selected"],
            })
    changed_rows = [r for r in rows if r["changed"]]
    errors = [r for r in changed_rows if r["oracle_iou"] - r["base_iou"] > 0.05]
    correct = [r for r in changed_rows if r["oracle_iou"] - r["base_iou"] <= 0.05]
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "changed_expressions": len(changed_rows),
        "material_errors": len(errors),
        "repeat_mismatch": sum(r["repeat_mask_iou"] < 1 for r in rows),
        "first_flip_errors": sum(r["first_flip"] for r in errors),
        "first_flip_correct": sum(r["first_flip"] for r in correct),
        "last_flip_errors": sum(r["last_flip"] for r in errors),
        "last_flip_correct": sum(r["last_flip"] for r in correct),
        "base_miou": statistics.fmean(r["base_iou"] for r in rows),
        "first_miou": statistics.fmean(r["first_iou"] for r in rows),
        "last_miou": statistics.fmean(r["last_iou"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
