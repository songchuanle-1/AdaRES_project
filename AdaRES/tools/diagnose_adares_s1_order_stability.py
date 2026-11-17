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
    alternate = self.alternate_output
    rows = []
    for index, target in enumerate(gt_pmasks):
        valid = ~out["batch_mask"][index]
        base_masks = out["masks"][index, :, valid].sigmoid()
        alt_masks = alternate["masks"][index, :, valid].sigmoid()
        base_hard = base_masks.gt(0.5)
        alt_hard = alt_masks.gt(0.5)
        intersections = base_hard.float() @ alt_hard.float().T
        unions = (base_hard.sum(-1)[:, None] +
                  alt_hard.sum(-1)[None, :] - intersections).clamp_min(1)
        correspondence = (intersections / unions).argmax(-1)
        base_scores = F.softmax(out["indis"][index], -1)[:, 1]
        alt_scores = F.softmax(alternate["indis"][index], -1)[:, 1]
        point_masks = base_hard[:, superpoints]
        truth = target.gt(0.5).to(point_masks.device)
        ious = (point_masks & truth).sum(-1).float() / (
            (point_masks | truth).sum(-1).clamp_min(1).float()
        )
        selected = int(base_scores.argmax())
        oracle = int(ious.argmax())
        alt_selected = int(alt_scores.argmax())
        aligned_stability = (intersections / unions).max(-1).values
        matched_alt_scores = alt_scores[correspondence]
        averaged = (base_scores + matched_alt_scores) / 2
        rows.append({
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "selected_stability": float(aligned_stability[selected]),
            "oracle_stability": float(aligned_stability[oracle]),
            "selected_match_score": float(alt_scores[correspondence[selected]]),
            "oracle_match_score": float(alt_scores[correspondence[oracle]]),
            "selected_score": float(base_scores[selected]),
            "oracle_score": float(base_scores[oracle]),
            "alt_selection_matches": bool(correspondence[selected] == alt_selected),
            "alt_selection_mask_iou": float(
                (intersections / unions)[selected, alt_selected]
            ),
            "average_iou": float(ious[averaged.argmax()]),
            "average_flip": int(averaged.argmax()) != selected,
            "matched_alt_iou": float(ious[matched_alt_scores.argmax()]),
            "base_score_margin": float(
                base_scores.topk(2).values.diff().abs().item()
            ),
        })
    return {"rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=16)
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

    def paired_forward(self, x, fps_seed_sp, batch_offsets, lang_feats=None,
                       lang_masks=None, sp_pos=None, feats_2d=None):
        base = original(x, fps_seed_sp, batch_offsets, lang_feats, lang_masks,
                        sp_pos, feats_2d)
        lengths = (batch_offsets[1:] - batch_offsets[:-1]).tolist()
        order = torch.cat([
            torch.arange(end - 1, start - 1, -1, device=x.device)
            for start, end in zip(batch_offsets[:-1].tolist(),
                                  batch_offsets[1:].tolist())
        ])
        remapped_seeds = torch.stack([
            length - 1 - seeds for length, seeds in zip(lengths, fps_seed_sp)
        ])
        changed = original(
            x[order], remapped_seeds, batch_offsets, lang_feats, lang_masks,
            sp_pos[order], feats_2d[order],
        )
        # Reverse is self-inverse within each scene. Padding remains at the end.
        for index, length in enumerate(lengths):
            changed["masks"][index, :, :length] = changed["masks"][
                index, :, :length].flip(-1)
        model.alternate_output = changed
        return base

    model.dec.forward = types.MethodType(paired_forward, model.dec)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    repeat_mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            second = model(batch, mode="predict")["rows"]
        repeat_mismatch += sum(
            abs(a["selected_iou"] - b["selected_iou"]) > 1e-4
            or abs(a["selected_stability"] - b["selected_stability"]) > 1e-4
            for a, b in zip(first, second)
        )
        rows.extend(first)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": repeat_mismatch,
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "oracle_more_stable": sum(
            r["oracle_stability"] > r["selected_stability"] for r in errors),
        "oracle_higher_matched_score": sum(
            r["oracle_match_score"] > r["selected_match_score"] for r in errors),
        "selection_match_fraction": statistics.fmean(
            r["alt_selection_matches"] for r in rows),
        "selection_mask_iou_mean": statistics.fmean(
            r["alt_selection_mask_iou"] for r in rows),
        "selected_stability_mean": statistics.fmean(
            r["selected_stability"] for r in errors) if errors else None,
        "oracle_stability_mean": statistics.fmean(
            r["oracle_stability"] for r in errors) if errors else None,
        "average_miou": statistics.fmean(r["average_iou"] for r in rows),
        "matched_alt_miou": statistics.fmean(r["matched_alt_iou"] for r in rows),
        "average_flips": sum(r["average_flip"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
