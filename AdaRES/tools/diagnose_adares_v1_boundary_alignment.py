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


def boundary_scores(features, positions, masks, valid):
    rows = []
    for feat, pos, mask, keep in zip(features, positions, masks, valid):
        count = int(keep.sum())
        # The fixed stride bounds quadratic neighbor search without using labels.
        stride = max(1, (count + 511) // 512)
        indices = torch.arange(0, count, stride, device=pos.device)
        xyz = pos[indices].float()
        embedding = F.normalize(feat[indices].float(), dim=-1)
        distance = torch.cdist(xyz[None], xyz[None])[0]
        distance.fill_diagonal_(float("inf"))
        neighbors = distance.topk(min(4, len(indices) - 1), largest=False).indices
        source = torch.arange(len(indices), device=pos.device)[:, None].expand_as(neighbors)
        feature_edge = 1 - (embedding[source] * embedding[neighbors]).sum(-1)
        geometry_edge = distance[source, neighbors]
        chosen = mask[:, indices].sigmoid() > 0.5
        crossing = chosen[:, source] != chosen[:, neighbors]
        # Contrast across a boundary relative to non-crossing local edges.
        def contrast(edge):
            cross = (crossing * edge).sum((1, 2)) / crossing.sum((1, 2)).clamp_min(1)
            rest = ((~crossing) * edge).sum((1, 2)) / (~crossing).sum((1, 2)).clamp_min(1)
            return cross - rest

        rows.append((contrast(feature_edge), contrast(geometry_edge),
                     crossing.flatten(1).any(-1)))
    return rows


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    alignment = boundary_scores(*self.probe_features, out["masks"], ~out["batch_mask"])
    scores = out["indis"].softmax(-1)[..., 1]
    rows = []
    for index, target in enumerate(gt_pmasks):
        masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        truth = target.gt(0.5).to(masks.device)
        ious = (masks & truth).sum(-1).float() / (
            (masks | truth).sum(-1).clamp_min(1).float()
        )
        selected, oracle = int(scores[index].argmax()), int(ious.argmax())
        feature, geometry, has_boundary = alignment[index]
        rows.append({
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "selected_feature": float(feature[selected]),
            "oracle_feature": float(feature[oracle]),
            "selected_geometry": float(geometry[selected]),
            "oracle_geometry": float(geometry[oracle]),
            "selected_boundary": bool(has_boundary[selected]),
            "oracle_boundary": bool(has_boundary[oracle]),
            "feature_only_iou": float(ious[feature.argmax()]),
            "geometry_only_iou": float(ious[geometry.argmax()]),
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
        features = module.input_proj(inputs[0]) + module.input_proj_2d(inputs[6])
        features, _ = module.get_batches(features, inputs[2])
        positions, _ = module.get_batches(inputs[5], inputs[2])
        model.probe_features = features, positions

    model.dec.register_forward_pre_hook(capture)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            repeat = model(batch, mode="predict")["rows"]
        mismatch += sum(
            abs(a["selected_feature"] - b["selected_feature"]) > 1e-4
            or a["selected_iou"] != b["selected_iou"]
            for a, b in zip(first, repeat)
        )
        rows.extend(first)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05
              and r["selected_boundary"] and r["oracle_boundary"]]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors_with_boundaries": len(errors),
        "repeat_mismatch": mismatch,
        "oracle_feature_higher": sum(r["oracle_feature"] > r["selected_feature"] for r in errors),
        "oracle_geometry_higher": sum(r["oracle_geometry"] > r["selected_geometry"] for r in errors),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "feature_only_miou": statistics.fmean(r["feature_only_iou"] for r in rows),
        "geometry_only_miou": statistics.fmean(r["geometry_only_iou"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
