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
    rows = []
    for index, truth in enumerate(gt_pmasks):
        points = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(points.device)
        ious = (points & target).sum(-1).float() / (
            (points | target).sum(-1).clamp_min(1).float())
        scores = out["indis"][index].softmax(-1)[:, 1]
        selected, oracle = int(scores.argmax()), int(ious.argmax())
        seed_support = gt_spmasks[index].to(scores.device)[
            fps_seed_sp[index].long()[out["sample_inds"][index].long()]]
        if seed_support.numel() != scores.numel():
            raise RuntimeError("Sampled seed/query alignment mismatch")
        positive = seed_support.gt(0.5)
        shifted = positive.roll(1)
        seed_labels = self.probe_labels[0][
            fps_seed_sp[index].long()[out["sample_inds"][index].long()]]
        object_ids = torch.as_tensor(object_idss[index], device=seed_labels.device)
        in_object = (seed_labels[:, None] == object_ids[None, :]).any(-1)
        if in_object.numel() != positive.numel():
            raise RuntimeError("Instance-label/query alignment mismatch")
        in_object_shifted = in_object.roll(1)
        rows.append({
            "scene_index": self.probe_scene,
            "expression_index": index,
            "scan_id": str(scan_ids[0]),
            "ann_id": str(ann_ids[index]),
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "positive_count": int(positive.sum()),
            "query_count": int(scores.numel()),
            "oracle_in_object": bool(positive[oracle]),
            "selected_in_object": bool(positive[selected]),
            "oracle_in_shifted": bool(shifted[oracle]),
            "selected_in_shifted": bool(shifted[selected]),
            "instance_positive_count": int(in_object.sum()),
            "oracle_in_instance": bool(in_object[oracle]),
            "selected_in_instance": bool(in_object[selected]),
            "oracle_in_instance_shifted": bool(in_object_shifted[oracle]),
            "selected_in_instance_shifted": bool(in_object_shifted[selected]),
        })
    return {"rows": rows}


def summarize(rows):
    result = {}
    for name, subset in (
        ("all", rows),
        ("material_errors", [
            row for row in rows
            if row["oracle_iou"] - row["selected_iou"] > 0.05
        ]),
    ):
        scenes = defaultdict(list)
        for row in subset:
            scenes[row["scene_index"]].append(row)
        result[name] = {
            "expressions": len(subset),
            "scenes": len(scenes),
            "eligible_in_object": sum(row["positive_count"] > 0 for row in subset),
            "mean_positive_count": statistics.fmean(
                row["positive_count"] for row in subset) if subset else None,
        }
        for field in ("oracle_in_object", "selected_in_object",
                      "oracle_in_shifted", "selected_in_shifted",
                      "oracle_in_instance", "selected_in_instance",
                      "oracle_in_instance_shifted",
                      "selected_in_instance_shifted"):
            result[name][field] = sum(row[field] for row in subset)
            result[name]["scene_mean_" + field] = statistics.fmean(
                statistics.fmean(row[field] for row in scene)
                for scene in scenes.values()) if scenes else None
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
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        batch = dataset.collate_fn([dataset[scene]])
        model.probe_labels = batch["sp_ins_labels"]
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        if len(first) != len(second):
            raise RuntimeError("Repeat expression count mismatch")
        mismatch += sum(a != b for a, b in zip(first, second))
        rows.extend(first)
    result = {
        "split": args.split,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "repeat_mismatch": mismatch,
        "summary": summarize(rows),
        "peak_inference_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
