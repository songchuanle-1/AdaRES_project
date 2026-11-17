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
    attention = self.probe_attention
    if attention.ndim == 4:
        attention = attention.mean(1)
    if attention.ndim != 3 or attention.shape != out["masks"].shape:
        raise RuntimeError(f"Unexpected scene attention shape: {attention.shape}")
    valid = (~out["batch_mask"]).unsqueeze(1)
    attention = attention.masked_fill(~valid, 0)
    attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
    current = out["masks"].sigmoid().gt(0.5) & valid
    previous = out["aux_outputs"][-1]["masks"].sigmoid().gt(0.5) & valid
    current_support = (attention * current.float()).sum(-1)
    previous_support = (attention * previous.float()).sum(-1)
    rows = []
    for index, target in enumerate(gt_pmasks):
        masks = current[index, :, superpoints]
        truth = target.gt(0.5).to(masks.device)
        ious = (masks & truth).sum(-1).float() / (
            (masks | truth).sum(-1).clamp_min(1).float()
        )
        confidence = F.softmax(out["indis"][index], dim=-1)[:, 1]
        selected, oracle = int(confidence.argmax()), int(ious.argmax())
        rows.append({
            "scene": self.probe_scene,
            "selected": selected,
            "oracle": oracle,
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "selected_current_support": float(current_support[index, selected]),
            "oracle_current_support": float(current_support[index, oracle]),
            "selected_previous_support": float(previous_support[index, selected]),
            "oracle_previous_support": float(previous_support[index, oracle]),
            "current_support_iou": float(ious[current_support[index].argmax()]),
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

    def capture(module, inputs, output):
        model.probe_attention = output[1]

    model.dec.qia_layers[-1].register_forward_hook(capture)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    repeat_mismatch = 0
    max_repeat_deviation = 0.0
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            base = model(batch, mode="predict")["rows"]
            repeat = model(batch, mode="predict")["rows"]
        for a, b in zip(base, repeat):
            deviation = max(
                abs(a["selected_current_support"] - b["selected_current_support"]),
                abs(a["oracle_current_support"] - b["oracle_current_support"]),
            )
            max_repeat_deviation = max(max_repeat_deviation, deviation)
            repeat_mismatch += (
                a["selected"] != b["selected"] or a["oracle"] != b["oracle"]
                or deviation > 1e-4
            )
        rows.extend(base)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": repeat_mismatch,
        "max_repeat_deviation": max_repeat_deviation,
        "oracle_higher_current": sum(
            r["oracle_current_support"] > r["selected_current_support"]
            for r in errors
        ),
        "oracle_higher_previous": sum(
            r["oracle_previous_support"] > r["selected_previous_support"]
            for r in errors
        ),
        "selected_current_mean": statistics.fmean(
            r["selected_current_support"] for r in errors
        ) if errors else None,
        "oracle_current_mean": statistics.fmean(
            r["oracle_current_support"] for r in errors
        ) if errors else None,
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "current_support_only_miou": statistics.fmean(
            r["current_support_iou"] for r in rows
        ),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
