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
from ipdn.model_paper5.adares_d1_full_query_comparison import IPDN
from ipdn.utils import get_root_logger


def install_counterfactual(indicator):
    original_forward = indicator.forward

    def forward(self, query, pred_masks, batch_mask):
        actual = original_forward(query, pred_masks, batch_mask)
        valid = (~batch_mask).unsqueeze(1).to(pred_masks.dtype)
        probabilities = pred_masks.sigmoid() * valid
        areas = probabilities.sum(-1)
        valid_count = valid.sum(-1).clamp_min(1)
        intersection = torch.bmm(probabilities, probabilities.transpose(1, 2))
        union = areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection
        overlap = intersection / union.clamp_min(1e-6)
        normalized = self.query_norm(query)
        pair_evidence = torch.cat(
            [
                normalized.unsqueeze(2) - normalized.unsqueeze(1),
                normalized.unsqueeze(2) * normalized.unsqueeze(1),
                overlap.unsqueeze(-1),
                ((areas.unsqueeze(-1) - areas.unsqueeze(1)) /
                 valid_count.unsqueeze(-1)).unsqueeze(-1),
            ],
            dim=-1,
        )
        pair_scores = self.pair_score(pair_evidence).squeeze(-1)
        count = query.shape[1]
        diagonal = torch.eye(count, device=query.device, dtype=torch.bool).unsqueeze(0)
        if count > 1:
            pair_scores = pair_scores.masked_fill(diagonal, -torch.inf)
            uniform = (~diagonal).to(pair_scores.dtype).expand_as(pair_scores) / (count - 1)
        else:
            uniform = torch.ones_like(pair_scores)
        learned = pair_scores.softmax(-1)

        def logits(weights):
            context = torch.bmm(weights, normalized)
            evidence = torch.cat(
                [
                    normalized,
                    context,
                    normalized - context,
                    (weights * overlap).sum(-1, keepdim=True),
                    (weights * (areas.unsqueeze(-1) - areas.unsqueeze(1)) /
                     valid_count.unsqueeze(-1)).sum(-1, keepdim=True),
                ],
                dim=-1,
            )
            return self.head(self.evidence_norm(evidence))

        torch.testing.assert_close(actual, logits(learned))
        self.mechanism_probe = {
            "uniform_logits": logits(uniform),
            "learned_weights": learned,
            "overlap": overlap,
        }
        return actual

    indicator.forward = types.MethodType(forward, indicator)


def selection_predictor():
    def predict_by_feat(
        self, scan_ids, object_idss, ann_ids, out, superpoints,
        gt_pmasks, gt_spmasks, fps_seed_sp,
    ):
        rows = []
        probe = self.dec.indi_embedding.mechanism_probe
        for index, target in enumerate(gt_pmasks):
            actual_logits = out["indis"][index]
            actual = F.softmax(actual_logits, dim=-1)[:, 1]
            uniform_logits = probe["uniform_logits"][index]
            uniform = F.softmax(uniform_logits, dim=-1)[:, 1]
            chosen = int(actual.argmax())
            alternative = int(uniform.argmax())
            masks = out["masks"][index].sigmoid().gt(0.5)
            point_masks = masks[:, superpoints]
            truth = target.gt(0.5).to(point_masks.device)
            ious = ((point_masks & truth).sum(-1).float() /
                    (point_masks | truth).sum(-1).clamp_min(1).float())
            weights = probe["learned_weights"][index, chosen]
            overlap = probe["overlap"][index, chosen]
            rows.append({
                "scene_index": self.mechanism_scene_index,
                "actual_iou": float(ious[chosen]),
                "uniform_iou": float(ious[alternative]),
                "oracle_iou": float(ious.max()),
                "actual_query": chosen,
                "uniform_query": alternative,
                "oracle_query": int(ious.argmax()),
                "actual_oracle_rank": int(
                    (actual.argsort(descending=True) == ious.argmax())
                    .nonzero(as_tuple=True)[0].item()
                ) + 1,
                "uniform_oracle_rank": int(
                    (uniform.argsort(descending=True) == ious.argmax())
                    .nonzero(as_tuple=True)[0].item()
                ) + 1,
                "learned_disjoint_mass": float(
                    weights.masked_fill(overlap >= 0.1, 0).sum()
                ),
                "uniform_disjoint_mass": float(
                    (overlap < 0.1).logical_and(
                        ~torch.eye(len(overlap), device=overlap.device, dtype=torch.bool)[chosen]
                    ).sum() / max(1, len(overlap) - 1)
                ),
            })
        return {"rows": rows}

    return predict_by_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=1)
    args = parser.parse_args()

    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = cfg.data.val.copy()
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
    install_counterfactual(model.dec.indi_embedding)
    model.predict_by_feat = types.MethodType(selection_predictor(), model)

    rows = []
    for index in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[index]])
        model.mechanism_scene_index = index
        with torch.no_grad():
            rows.extend(model(batch, mode="predict")["rows"])
    if not rows:
        raise RuntimeError("No expressions in diagnostic subset")
    changed = [row for row in rows if row["actual_query"] != row["uniform_query"]]
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("meta", {}).get("epoch"),
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "actual_miou": statistics.fmean(row["actual_iou"] for row in rows),
        "uniform_miou": statistics.fmean(row["uniform_iou"] for row in rows),
        "oracle_miou": statistics.fmean(row["oracle_iou"] for row in rows),
        "changed_top1": len(changed),
        "changed_better": sum(row["actual_iou"] > row["uniform_iou"] + 0.05 for row in changed),
        "changed_worse": sum(row["actual_iou"] + 0.05 < row["uniform_iou"] for row in changed),
        "learned_disjoint_mass": statistics.fmean(row["learned_disjoint_mass"] for row in rows),
        "uniform_disjoint_mass": statistics.fmean(row["uniform_disjoint_mass"] for row in rows),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
