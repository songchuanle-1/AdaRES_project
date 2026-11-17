import argparse
import json
import statistics
import sys
import types
from collections import defaultdict
from pathlib import Path

import gorilla
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_w2_pointmass_competition import IPDN
from ipdn.utils import get_root_logger


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    query, masks, batch_mask, mass = self.probe_inputs
    head = self.dec.indi_embedding
    original = out["indis"].softmax(-1)[..., 1]
    recomputed = head(query, masks, batch_mask, mass).softmax(-1)[..., 1]
    if not torch.allclose(original, recomputed, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Native W2 indicator reconstruction mismatch")
    valid = ~batch_mask
    uniform = valid.to(mass.dtype)
    rotated = mass.clone()
    for index in range(mass.shape[0]):
        count = int(valid[index].sum())
        if count > 1:
            rotated[index, :count] = mass[index, :count].roll(1)
    uniform_scores = head(query, masks, batch_mask, uniform).softmax(-1)[..., 1]
    rotated_scores = head(query, masks, batch_mask, rotated).softmax(-1)[..., 1]
    rows = []
    for index, truth in enumerate(gt_pmasks):
        points = masks[index].sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(points.device)
        ious = (points & target).sum(-1).float() / (
            (points | target).sum(-1).clamp_min(1).float())
        rows.append({
            "scene_index": self.probe_scene,
            "expression_index": index,
            "scan_id": str(scan_ids[0]),
            "ann_id": str(ann_ids[index]),
            "native_iou": float(ious[original[index].argmax()]),
            "uniform_iou": float(ious[uniform_scores[index].argmax()]),
            "rotated_iou": float(ious[rotated_scores[index].argmax()]),
            "oracle_iou": float(ious.max()),
        })
    return {"rows": rows}


def summary(rows):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_index"]].append(row)
    result = {}
    for name in ("native", "uniform", "rotated", "oracle"):
        field = name + "_iou"
        result[name] = {
            "expression_mean": statistics.fmean(row[field] for row in rows),
            "scene_mean": statistics.fmean(
                statistics.fmean(row[field] for row in scene)
                for scene in by_scene.values()),
        }
    for name in ("uniform", "rotated"):
        gains = [row["native_iou"] - row[name + "_iou"] for row in rows]
        result[name]["native_better"] = sum(gain > 0.001 for gain in gains)
        result[name]["ablation_better"] = sum(gain < -0.001 for gain in gains)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
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
    model = IPDN(**cfg.model).cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loaded = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    allowed = {"text_encoder.embeddings.position_ids",
               "text_encoder.embeddings.token_type_ids"}
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.eval()

    def capture(module, inputs):
        model.probe_inputs = inputs

    model.dec.indi_embedding.register_forward_pre_hook(capture)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        if len(first) != len(second):
            raise RuntimeError("Repeat expression count mismatch")
        mismatch += sum(
            any(abs(a[key] - b[key]) > 1e-6
                for key in ("native_iou", "uniform_iou", "rotated_iou", "oracle_iou"))
            for a, b in zip(first, second))
        rows.extend(first)
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "repeat_mismatch": mismatch,
        "summary": summary(rows),
        "peak_inference_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
