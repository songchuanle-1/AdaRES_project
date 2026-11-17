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
    for index, target in enumerate(gt_pmasks):
        valid = ~out["batch_mask"][index]
        masks = out["masks"][index, :, valid].sigmoid()
        area = masks.sum(-1)
        intersection = masks @ masks.T
        overlap = intersection / (
            area[:, None] + area[None, :] - intersection
        ).clamp_min(1e-6)
        overlap.fill_diagonal_(0)
        confidence = F.softmax(out["indis"][index], dim=-1)[:, 1]
        # Frozen control: the same peer confidence assigned to shifted query IDs.
        null_confidence = confidence.roll(1)
        support = overlap @ confidence / overlap.sum(-1).clamp_min(1e-6)
        null_support = overlap @ null_confidence / overlap.sum(-1).clamp_min(1e-6)
        plain_support = overlap.mean(-1)
        product = confidence * support
        null_product = confidence * null_support
        plain_product = confidence * plain_support
        hard_masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        truth = target.gt(0.5).to(hard_masks.device)
        ious = (hard_masks & truth).sum(-1).float() / (
            (hard_masks | truth).sum(-1).clamp_min(1).float()
        )
        selected, oracle = int(confidence.argmax()), int(ious.argmax())
        rows.append({
            "scene": self.probe_scene,
            "selected": selected,
            "oracle": oracle,
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "selected_support": float(support[selected]),
            "oracle_support": float(support[oracle]),
            "selected_null": float(null_support[selected]),
            "oracle_null": float(null_support[oracle]),
            "selected_plain": float(plain_support[selected]),
            "oracle_plain": float(plain_support[oracle]),
            "support_iou": float(ious[support.argmax()]),
            "product_iou": float(ious[product.argmax()]),
            "null_product_iou": float(ious[null_product.argmax()]),
            "plain_product_iou": float(ious[plain_product.argmax()]),
            "product_flip": int(product.argmax()) != selected,
            "null_product_flip": int(null_product.argmax()) != selected,
            "plain_product_flip": int(plain_product.argmax()) != selected,
        })
    return {"rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=64)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = cfg.data.train.copy()
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
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    repeat_mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            base = model(batch, mode="predict")["rows"]
            repeat = model(batch, mode="predict")["rows"]
        repeat_mismatch += sum(
            a["selected"] != b["selected"] or a["oracle"] != b["oracle"]
            or abs(a["selected_support"] - b["selected_support"]) > 1e-4
            for a, b in zip(base, repeat)
        )
        rows.extend(base)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": repeat_mismatch,
        "oracle_higher_support": sum(
            r["oracle_support"] > r["selected_support"] for r in errors),
        "oracle_higher_null": sum(
            r["oracle_null"] > r["selected_null"] for r in errors),
        "oracle_higher_plain": sum(
            r["oracle_plain"] > r["selected_plain"] for r in errors),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "support_only_miou": statistics.fmean(r["support_iou"] for r in rows),
        "product_miou": statistics.fmean(r["product_iou"] for r in rows),
        "null_product_miou": statistics.fmean(r["null_product_iou"] for r in rows),
        "plain_product_miou": statistics.fmean(r["plain_product_iou"] for r in rows),
        "product_flips": sum(r["product_flip"] for r in rows),
        "null_product_flips": sum(r["null_product_flip"] for r in rows),
        "plain_product_flips": sum(r["plain_product_flip"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
