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


def point_ious(mask_logits, superpoints, truth):
    masks = mask_logits.sigmoid().gt(0.5)[:, superpoints]
    target = truth.gt(0.5).to(masks.device)
    ious = (masks & target).sum(-1).float() / (
        (masks | target).sum(-1).clamp_min(1).float()
    )
    return masks, ious


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    previous = out["aux_outputs"][-1]
    rows = []
    for index, truth in enumerate(gt_pmasks):
        final_masks, final_iou = point_ious(out["masks"][index], superpoints, truth)
        prior_masks, prior_iou = point_ious(previous["masks"][index], superpoints, truth)
        scores = F.softmax(out["indis"][index], dim=-1)[:, 1]
        prior_scores = F.softmax(previous["indis"][index], dim=-1)[:, 1]
        selected = int(scores.argmax())
        prior_selected = int(prior_scores.argmax())
        final_oracle = int(final_iou.argmax())
        prior_oracle = int(prior_iou.argmax())
        common = (prior_masks.float() @ final_masks.float().T)
        union = prior_masks.sum(-1)[:, None] + final_masks.sum(-1)[None, :] - common
        correspondence = (common / union.clamp_min(1)).argmax(-1)
        rows.append({
            "scene": self.probe_scene,
            "selected_final_iou": float(final_iou[selected]),
            "selected_prior_iou": float(prior_iou[selected]),
            "prior_selected_iou": float(prior_iou[prior_selected]),
            "prior_selected_final_iou": float(final_iou[prior_selected]),
            "prior_selection_changed": prior_selected != selected,
            "final_oracle_iou": float(final_iou[final_oracle]),
            "prior_oracle_iou": float(prior_iou[prior_oracle]),
            "prior_oracle_final_iou_same_index": float(final_iou[prior_oracle]),
            "prior_oracle_final_iou_matched": float(final_iou[correspondence[prior_oracle]]),
            "prior_oracle_match_overlap": float(
                common[prior_oracle, correspondence[prior_oracle]] /
                union[prior_oracle, correspondence[prior_oracle]].clamp_min(1)
            ),
        })
    return {"rows": rows}


def summarize(rows):
    errors = [r for r in rows
              if r["final_oracle_iou"] - r["selected_final_iou"] > 0.05]
    recoverable = [r for r in rows
                   if r["prior_oracle_iou"] > r["final_oracle_iou"] + 0.05]
    matched_degraded = [r for r in recoverable
                        if r["prior_oracle_iou"] >
                        r["prior_oracle_final_iou_matched"] + 0.05]
    return {
        "expressions": len(rows),
        "scenes": len(set(r["scene"] for r in rows)),
        "material_selection_errors": len(errors),
        "prior_oracle_better_than_final_by_005": len(recoverable),
        "matched_prior_oracle_degraded_by_005": len(matched_degraded),
        "same_index_degraded_by_005": sum(
            r["prior_oracle_iou"] > r["prior_oracle_final_iou_same_index"] + 0.05
            for r in recoverable
        ),
        "selected_query_prior_better_by_005": sum(
            r["selected_prior_iou"] > r["selected_final_iou"] + 0.05
            for r in rows
        ),
        "mean_final_selected": statistics.fmean(r["selected_final_iou"] for r in rows),
        "mean_prior_mask_final_selection": statistics.fmean(
            r["selected_prior_iou"] for r in rows
        ),
        "mean_prior_mask_prior_selection": statistics.fmean(
            r["prior_selected_iou"] for r in rows
        ),
        "prior_selection_changed": sum(r["prior_selection_changed"] for r in rows),
        "mean_final_oracle": statistics.fmean(r["final_oracle_iou"] for r in rows),
        "mean_prior_oracle": statistics.fmean(r["prior_oracle_iou"] for r in rows),
        "mean_matched_overlap_recoverable": (
            statistics.fmean(r["prior_oracle_match_overlap"] for r in recoverable)
            if recoverable else None
        ),
    }


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
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        repeat_mismatch += sum(
            abs(a["selected_final_iou"] - b["selected_final_iou"]) > 1e-6
            or abs(a["prior_oracle_iou"] - b["prior_oracle_iou"]) > 1e-6
            or abs(a["prior_oracle_match_overlap"] -
                   b["prior_oracle_match_overlap"]) > 1e-6
            for a, b in zip(first, second)
        )
        rows.extend(first)
    result = {
        **summarize(rows),
        "repeat_mismatch": repeat_mismatch,
        "peak_inference_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
