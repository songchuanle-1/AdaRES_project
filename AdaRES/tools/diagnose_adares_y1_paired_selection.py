import argparse
import importlib
import json
import statistics
import sys
import time
import types
from pathlib import Path

import gorilla
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    rows = []
    for index, truth in enumerate(gt_pmasks):
        points = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(points.device)
        ious = (points & target).sum(-1).float() / (
            (points | target).sum(-1).clamp_min(1).float())
        scores = out["indis"][index].softmax(-1)[:, 1]
        chosen = int(scores.argmax())
        oracle = int(ious.argmax())
        rows.append({
            "scene_index": self.probe_scene,
            "expression_index": index,
            "scan_id": str(scan_ids[0]),
            "ann_id": str(ann_ids[index]),
            "selected_iou": float(ious[chosen]),
            "oracle_iou": float(ious[oracle]),
            "oracle_rank": int((scores > scores[oracle]).sum()) + 1,
            "query_count": int(scores.numel()),
        })
    return {"rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--model", choices=("adares_b4_query_competition",
                                          "adares_w2_pointmass_competition"),
                        required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--scenes", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", required=True)
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
    model = importlib.import_module(
        "ipdn.model_paper5." + args.model).IPDN(**cfg.model).cuda()
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
    start = time.monotonic()
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        if len(first) != len(second):
            raise RuntimeError("Repeated expression count differs")
        repeat_mismatch += sum(
            abs(a["selected_iou"] - b["selected_iou"]) > 1e-6
            or abs(a["oracle_iou"] - b["oracle_iou"]) > 1e-6
            or a["oracle_rank"] != b["oracle_rank"]
            for a, b in zip(first, second))
        rows.extend(first)
    errors = [row for row in rows
              if row["oracle_iou"] - row["selected_iou"] > 0.05]
    result = {
        "model": args.model,
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "repeat_mismatch": repeat_mismatch,
        "selected_mean": statistics.fmean(row["selected_iou"] for row in rows),
        "oracle_mean": statistics.fmean(row["oracle_iou"] for row in rows),
        "material_errors": len(errors),
        "oracle_outside_top3_errors": sum(row["oracle_rank"] > 3 for row in errors),
        "elapsed_seconds": time.monotonic() - start,
        "peak_inference_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
