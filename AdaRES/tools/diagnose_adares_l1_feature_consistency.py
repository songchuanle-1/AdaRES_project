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


def predictor():
    def predict_by_feat(
        self, scan_ids, object_idss, ann_ids, out, superpoints,
        gt_pmasks, gt_spmasks, fps_seed_sp,
    ):
        valid = (~out["batch_mask"]).float()
        features = F.normalize(self.probe_scene, dim=-1)
        weights = out["masks"].sigmoid() * valid[:, None, :]
        area = weights.sum(-1).clamp_min(1e-6)
        centroid = torch.bmm(weights, features) / area[..., None]
        # Unit feature vectors make the centroid norm a bounded coherence statistic.
        coherence = centroid.norm(dim=-1)
        hard = (out["masks"] > 0).float() * valid[:, None, :]
        hard_area = hard.sum(-1).clamp_min(1)
        hard_coherence = (torch.bmm(hard, features) / hard_area[..., None]).norm(dim=-1)
        boundary_scores = []
        for batch_index in range(features.shape[0]):
            count = int(valid[batch_index].sum())
            coords = self.probe_positions[batch_index, :count]
            feature = features[batch_index, :count]
            distance = torch.cdist(coords, coords)
            neighbors = distance.topk(min(9, count), largest=False).indices[:, 1:]
            contrast = 1 - (feature[:, None, :] * feature[neighbors]).sum(-1)
            membership = hard[batch_index, :, :count].bool()
            neighbor_membership = membership[:, neighbors]
            crossing = membership[:, :, None] != neighbor_membership
            across = (crossing.float() * contrast).sum((1, 2)) / crossing.sum(
                (1, 2)
            ).clamp_min(1)
            within = ((~crossing).float() * contrast).sum((1, 2)) / (
                ~crossing
            ).sum((1, 2)).clamp_min(1)
            boundary_scores.append(across - within)
        boundary = torch.stack(boundary_scores)
        confidence = F.softmax(out["indis"], dim=-1)[..., 1]
        rows = []
        for index, target in enumerate(gt_pmasks):
            masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            truth = target.gt(0.5).to(masks.device)
            ious = (masks & truth).sum(-1).float() / (
                (masks | truth).sum(-1).clamp_min(1).float()
            )
            selected = int(confidence[index].argmax())
            oracle = int(ious.argmax())
            rows.append({
                "scene_index": self.probe_scene_index,
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "selected_coherence": float(coherence[index, selected]),
                "oracle_coherence": float(coherence[index, oracle]),
                "selected_hard_coherence": float(hard_coherence[index, selected]),
                "oracle_hard_coherence": float(hard_coherence[index, oracle]),
                "selected_area": float(area[index, selected]),
                "oracle_area": float(area[index, oracle]),
                "selected_boundary": float(boundary[index, selected]),
                "oracle_boundary": float(boundary[index, oracle]),
                "coherence_only_iou": float(ious[coherence[index].argmax()]),
                "hard_coherence_only_iou": float(ious[hard_coherence[index].argmax()]),
                "boundary_only_iou": float(ious[boundary[index].argmax()]),
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

    def capture_scene(module, inputs, output):
        model.probe_scene = output

    def capture_positions(module, inputs):
        model.probe_positions, _ = module.get_batches(inputs[5], inputs[2])

    model.dec.register_forward_pre_hook(capture_positions)
    model.dec.ila_ffn_layers[-1].register_forward_hook(capture_scene)
    model.predict_by_feat = types.MethodType(predictor(), model)
    rows = []
    for index in range(min(args.scenes, len(dataset))):
        model.probe_scene_index = index
        with torch.no_grad():
            rows.extend(model(dataset.collate_fn([dataset[index]]), mode="predict")["rows"])
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "oracle_more_coherent": sum(
            r["oracle_coherence"] > r["selected_coherence"] for r in errors
        ),
        "oracle_more_hard_coherent": sum(
            r["oracle_hard_coherence"] > r["selected_hard_coherence"] for r in errors
        ),
        "selected_coherence_mean": statistics.fmean(
            r["selected_coherence"] for r in errors
        ),
        "oracle_coherence_mean": statistics.fmean(
            r["oracle_coherence"] for r in errors
        ),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "coherence_only_miou": statistics.fmean(
            r["coherence_only_iou"] for r in rows
        ),
        "hard_coherence_only_miou": statistics.fmean(
            r["hard_coherence_only_iou"] for r in rows
        ),
        "oracle_boundary_higher": sum(
            r["oracle_boundary"] > r["selected_boundary"] for r in errors
        ),
        "selected_boundary_mean": statistics.fmean(
            r["selected_boundary"] for r in errors
        ),
        "oracle_boundary_mean": statistics.fmean(
            r["oracle_boundary"] for r in errors
        ),
        "boundary_only_miou": statistics.fmean(
            r["boundary_only_iou"] for r in rows
        ),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
