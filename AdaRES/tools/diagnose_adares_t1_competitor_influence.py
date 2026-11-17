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


def counterfactual_scores(indicator, query, pred_masks, batch_mask):
    valid = (~batch_mask).unsqueeze(1).to(pred_masks.dtype)
    probabilities = pred_masks.sigmoid() * valid
    areas = probabilities.sum(-1)
    intersection = probabilities @ probabilities.transpose(1, 2)
    overlap = intersection / (
        areas[:, :, None] + areas[:, None, :] - intersection
    ).clamp_min(1e-6)
    diagonal = torch.eye(
        overlap.shape[-1], dtype=torch.bool, device=overlap.device
    ).unsqueeze(0)
    overlap = overlap.masked_fill(diagonal, 0)
    closest = overlap.argmax(-1)
    farthest = overlap.masked_fill(diagonal, float("inf")).argmin(-1)
    normalized = indicator.query_norm(query)

    def evaluate(dropped):
        competing = overlap.clone()
        competing.scatter_(2, dropped.unsqueeze(-1), 0)
        weights = competing / competing.sum(-1, keepdim=True).clamp_min(1e-6)
        context = weights @ normalized
        evidence = torch.cat((
            normalized, context, normalized - context,
            competing.max(-1).values.unsqueeze(-1),
        ), dim=-1)
        return indicator.head(indicator.evidence_norm(evidence)).softmax(-1)[..., 1]

    return evaluate(closest), evaluate(farthest), overlap.gather(
        2, closest.unsqueeze(-1)
    ).squeeze(-1)


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    query, masks, batch_mask = self.probe_indicator_inputs
    closest, farthest, max_overlap = counterfactual_scores(
        self.dec.indi_embedding, query, masks, batch_mask
    )
    base = out["indis"].softmax(-1)[..., 1]
    rows = []
    for index, target in enumerate(gt_pmasks):
        point_masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        truth = target.gt(0.5).to(point_masks.device)
        ious = (point_masks & truth).sum(-1).float() / (
            (point_masks | truth).sum(-1).clamp_min(1).float()
        )
        selected, oracle = int(base[index].argmax()), int(ious.argmax())
        rows.append({
            "selected": selected,
            "oracle": oracle,
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "selected_influence": float(closest[index, selected] - base[index, selected]),
            "oracle_influence": float(closest[index, oracle] - base[index, oracle]),
            "selected_control": float(farthest[index, selected] - base[index, selected]),
            "oracle_control": float(farthest[index, oracle] - base[index, oracle]),
            "selected_overlap": float(max_overlap[index, selected]),
            "oracle_overlap": float(max_overlap[index, oracle]),
            "closest_iou": float(ious[closest[index].argmax()]),
            "farthest_iou": float(ious[farthest[index].argmax()]),
            "closest_flip": int(closest[index].argmax()) != selected,
            "farthest_flip": int(farthest[index].argmax()) != selected,
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
    allowed = {
        "text_encoder.embeddings.position_ids",
        "text_encoder.embeddings.token_type_ids",
    }
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.eval()

    def capture(module, inputs, output):
        model.probe_indicator_inputs = inputs

    model.dec.indi_embedding.register_forward_hook(capture)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    repeat_mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        repeat_mismatch += sum(
            a["selected"] != b["selected"]
            or a["oracle"] != b["oracle"]
            or abs(a["selected_influence"] - b["selected_influence"]) > 1e-4
            for a, b in zip(first, second)
        )
        rows.extend(first)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": repeat_mismatch,
        "oracle_influence_higher": sum(
            r["oracle_influence"] > r["selected_influence"] for r in errors
        ),
        "oracle_control_higher": sum(
            r["oracle_control"] > r["selected_control"] for r in errors
        ),
        "oracle_overlap_higher": sum(
            r["oracle_overlap"] > r["selected_overlap"] for r in errors
        ),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "closest_miou": statistics.fmean(r["closest_iou"] for r in rows),
        "farthest_miou": statistics.fmean(r["farthest_iou"] for r in rows),
        "closest_flips": sum(r["closest_flip"] for r in rows),
        "farthest_flips": sum(r["farthest_flip"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
