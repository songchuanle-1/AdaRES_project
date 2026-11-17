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


def reweight(head, query, masks, batch_mask, weights):
    valid = (~batch_mask).to(masks.dtype)
    probabilities = masks.sigmoid() * valid[:, None]
    weighted = probabilities * weights[:, None]
    areas = weighted.sum(-1)
    intersection = torch.bmm(weighted, probabilities.transpose(1, 2))
    union = areas[:, :, None] + areas[:, None, :] - intersection
    overlap = intersection / union.clamp_min(1e-6)
    eye = torch.eye(overlap.shape[-1], dtype=torch.bool,
                    device=overlap.device)[None]
    overlap = overlap.masked_fill(eye, 0)
    competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    normalized_query = head.query_norm(query)
    context = torch.bmm(competition, normalized_query)
    evidence = torch.cat(
        [normalized_query, context, normalized_query - context,
         overlap.max(-1).values[..., None]], dim=-1
    )
    return head.head(head.evidence_norm(evidence)).softmax(-1)[..., 1]


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    counts = torch.bincount(superpoints.long(),
                            minlength=out["masks"].shape[-1]).to(
                                device=out["masks"].device, dtype=out["masks"].dtype)
    count_pad = counts[None].expand(out["masks"].shape[0], -1)
    shifted = torch.roll(counts, shifts=1)[None].expand_as(count_pad)
    head = self.dec.indi_embedding
    query = self.probe_query
    weighted = reweight(head, query, out["masks"], out["batch_mask"], count_pad)
    control = reweight(head, query, out["masks"], out["batch_mask"], shifted)
    original = out["indis"].softmax(-1)[..., 1]
    rows = []
    for index, truth in enumerate(gt_pmasks):
        point_masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(point_masks.device)
        ious = (point_masks & target).sum(-1).float() / (
            (point_masks | target).sum(-1).clamp_min(1).float()
        )
        selected, oracle = int(original[index].argmax()), int(ious.argmax())
        weighted_selected, shifted_selected = (
            int(weighted[index].argmax()), int(control[index].argmax()))
        rows.append({
            "scene": self.probe_scene,
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "weighted_iou": float(ious[weighted_selected]),
            "shifted_iou": float(ious[shifted_selected]),
            "weighted_flip": weighted_selected != selected,
            "shifted_flip": shifted_selected != selected,
            "weighted_oracle_advantage": float(
                weighted[index, oracle] - weighted[index, selected]),
            "shifted_oracle_advantage": float(
                control[index, oracle] - control[index, selected]),
            "original_oracle_advantage": float(
                original[index, oracle] - original[index, selected]),
            "count_cv": float(counts.std() / counts.float().mean().clamp_min(1)),
            "zero_count": int((counts == 0).sum()),
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

    def capture(module, inputs):
        model.probe_query = inputs[0]

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
        mismatch += sum(
            abs(a["weighted_iou"] - b["weighted_iou"]) > 1e-6
            or abs(a["selected_iou"] - b["selected_iou"]) > 1e-6
            for a, b in zip(first, second)
        )
        rows.extend(first)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": mismatch,
        "weighted_oracle_advantage_over_original": sum(
            r["weighted_oracle_advantage"] > r["original_oracle_advantage"]
            for r in errors),
        "shifted_oracle_advantage_over_original": sum(
            r["shifted_oracle_advantage"] > r["original_oracle_advantage"]
            for r in errors),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "weighted_miou": statistics.fmean(r["weighted_iou"] for r in rows),
        "shifted_miou": statistics.fmean(r["shifted_iou"] for r in rows),
        "weighted_flips": sum(r["weighted_flip"] for r in rows),
        "shifted_flips": sum(r["shifted_flip"] for r in rows),
        "mean_count_cv": statistics.fmean(r["count_cv"] for r in rows),
        "peak_inference_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
