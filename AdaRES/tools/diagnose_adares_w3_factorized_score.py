import argparse
import json
import statistics
import sys
import types
from pathlib import Path

import gorilla
import torch

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
        identity = out["indis"][index].softmax(-1)[:, 1]
        quality = out["scores"][index, :, 0]
        product = identity * quality.clamp(min=0)
        shifted = identity * quality.roll(1).clamp(min=0)
        selected, oracle = int(identity.argmax()), int(ious.argmax())
        rows.append({
            "scene": self.probe_scene,
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "product_iou": float(ious[product.argmax()]),
            "shifted_iou": float(ious[shifted.argmax()]),
            "product_flip": int(product.argmax()) != selected,
            "shifted_flip": int(shifted.argmax()) != selected,
            "product_oracle_advantage": float(product[oracle] - product[selected]),
            "shifted_oracle_advantage": float(shifted[oracle] - shifted[selected]),
            "original_oracle_advantage": float(identity[oracle] - identity[selected]),
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
    mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        mismatch += sum(
            abs(a["product_iou"] - b["product_iou"]) > 1e-6
            or abs(a["selected_iou"] - b["selected_iou"]) > 1e-6
            for a, b in zip(first, second))
        rows.extend(first)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": mismatch,
        "product_oracle_margin_improves": sum(
            r["product_oracle_advantage"] > r["original_oracle_advantage"]
            for r in errors),
        "shifted_oracle_margin_improves": sum(
            r["shifted_oracle_advantage"] > r["original_oracle_advantage"]
            for r in errors),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "product_miou": statistics.fmean(r["product_iou"] for r in rows),
        "shifted_miou": statistics.fmean(r["shifted_iou"] for r in rows),
        "product_flips": sum(r["product_flip"] for r in rows),
        "shifted_flips": sum(r["shifted_flip"] for r in rows),
        "peak_inference_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
