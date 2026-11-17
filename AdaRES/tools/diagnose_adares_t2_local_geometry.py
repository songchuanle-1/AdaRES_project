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
        masks = out["masks"][index, :, valid].sigmoid().gt(0.5)
        truth = target.gt(0.5).to(masks.device)
        point_masks = masks[:, superpoints]
        ious = (point_masks & truth).sum(-1).float() / (
            (point_masks | truth).sum(-1).clamp_min(1).float()
        )
        scores = F.softmax(out["indis"][index], -1)[:, 1]
        selected = int(scores.argmax())
        top_two = scores.topk(2).indices.tolist()
        changed = []
        for variant, alternate in enumerate(self.alternates):
            alt_masks = alternate["masks"][index, :, valid].sigmoid().gt(0.5)
            overlap = masks.float() @ alt_masks.float().T
            union = (masks.sum(-1)[:, None] +
                     alt_masks.sum(-1)[None, :] - overlap).clamp_min(1)
            query = selected if variant < 3 else top_two[1]
            base_seed = out["sample_inds"][index, query]
            same_seed = (alternate["sample_inds"][index] == base_seed).nonzero(
                as_tuple=True)[0]
            match = (int(same_seed[0]) if same_seed.numel()
                     else int((overlap / union)[query].argmax()))
            alt_scores = F.softmax(alternate["indis"][index], -1)[:, 1]
            alt_point = alt_masks[:, superpoints]
            alt_ious = (alt_point & truth).sum(-1).float() / (
                (alt_point | truth).sum(-1).clamp_min(1).float()
            )
            changed.append({
                "score_drop": float(scores[query] - alt_scores[match]),
                "matched_score": float(alt_scores[match]),
                "matched_mask_iou": float((overlap / union)[query, match]),
                "selection_iou": float(alt_ious[alt_scores.argmax()]),
                "same_seed": bool(same_seed.numel()),
            })
        rows.append({
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious.max()),
            "repeat": changed[0],
            "inside": changed[1],
            "outside": changed[2],
            "runner_inside": changed[3],
            "runner_outside": changed[4],
            "top_two_inside_iou": float(ious[
                top_two[0] if changed[1]["matched_score"] >=
                changed[3]["matched_score"] else top_two[1]]),
            "top_two_outside_iou": float(ious[
                top_two[0] if changed[2]["matched_score"] >=
                changed[4]["matched_score"] else top_two[1]]),
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
    original = model.dec.forward

    def intervened_forward(self, x, fps_seed_sp, batch_offsets,
                           lang_feats=None, lang_masks=None, sp_pos=None,
                           feats_2d=None):
        base = original(x, fps_seed_sp, batch_offsets, lang_feats,
                        lang_masks, sp_pos, feats_2d)
        repeat = original(x, fps_seed_sp, batch_offsets, lang_feats,
                          lang_masks, sp_pos, feats_2d)
        lengths = (batch_offsets[1:] - batch_offsets[:-1]).tolist()
        top_two = base["indis"].softmax(-1)[..., 1].topk(2).indices
        alternatives = []
        for rank in (0, 1):
            for inside in (True, False):
                changed = x.clone()
                for index, (start, length) in enumerate(zip(
                        batch_offsets[:-1].tolist(), lengths)):
                    query = top_two[index, rank]
                    mask = base["masks"][index, query, :length].sigmoid() > 0.5
                    on = mask.nonzero(as_tuple=True)[0]
                    off = (~mask).nonzero(as_tuple=True)[0]
                    count = min(on.numel(), off.numel())
                    indices = (on if inside else off)[:count] + start
                    if count > 1:
                        changed[indices] = x[indices.roll(1)]
                alternatives.append(original(
                    changed, fps_seed_sp, batch_offsets, lang_feats, lang_masks,
                    sp_pos, feats_2d,
                ))
        model.alternates = [repeat] + alternatives
        return base

    model.dec.forward = types.MethodType(intervened_forward, model.dec)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    for scene in range(min(args.scenes, len(dataset))):
        with torch.no_grad():
            rows.extend(model(dataset.collate_fn([dataset[scene]]),
                              mode="predict")["rows"])
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": sum(
            abs(r["repeat"]["score_drop"]) > 1e-4
            or r["repeat"]["matched_mask_iou"] < 0.9999 for r in rows),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "inside_miou": statistics.fmean(r["inside"]["selection_iou"] for r in rows),
        "outside_miou": statistics.fmean(r["outside"]["selection_iou"] for r in rows),
        "top_two_inside_miou": statistics.fmean(
            r["top_two_inside_iou"] for r in rows),
        "top_two_outside_miou": statistics.fmean(
            r["top_two_outside_iou"] for r in rows),
        "error_inside_greater_drop": sum(
            r["inside"]["score_drop"] > r["outside"]["score_drop"]
            for r in errors),
        "inside_same_seed": sum(r["inside"]["same_seed"] for r in rows),
        "outside_same_seed": sum(r["outside"]["same_seed"] for r in rows),
        "inside_mask_iou": statistics.fmean(
            r["inside"]["matched_mask_iou"] for r in rows),
        "outside_mask_iou": statistics.fmean(
            r["outside"]["matched_mask_iou"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
