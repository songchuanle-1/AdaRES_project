import argparse
import json
import statistics
import sys
import types
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
    query, masks, batch_mask, point_mass = self.probe_inputs
    head = self.dec.indi_embedding
    probabilities = masks.sigmoid() * (~batch_mask).unsqueeze(1).to(masks.dtype)
    weighted = probabilities * point_mass.unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.bmm(weighted, probabilities.transpose(1, 2))
    union = areas[:, :, None] + areas[:, None, :] - intersection
    overlap = intersection / union.clamp_min(1e-6)
    diagonal = torch.eye(overlap.shape[-1], dtype=torch.bool,
                         device=overlap.device)[None]
    overlap = overlap.masked_fill(diagonal, 0.0)
    containment = intersection / areas[:, :, None].clamp_min(1e-6)
    containment = containment.masked_fill(diagonal, 0.0)
    normalized = head.query_norm(query)

    def scores(graph, maximum):
        graph = graph / graph.sum(-1, keepdim=True).clamp_min(1e-6)
        context = torch.bmm(graph, normalized)
        evidence = torch.cat(
            [normalized, context, normalized - context, maximum[..., None]],
            dim=-1,
        )
        return head.head(head.evidence_norm(evidence)).softmax(-1)[..., 1]

    original = out["indis"].softmax(-1)[..., 1]
    reconstructed = scores(overlap, overlap.max(-1).values)
    if not torch.allclose(original, reconstructed, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Native W2 output mismatch")
    actual = scores(containment, containment.max(-1).values)
    shifted = scores(containment.roll(1, dims=-1),
                     containment.max(-1).values.roll(1, dims=-1))
    rows = []
    for index, truth in enumerate(gt_pmasks):
        points = masks[index].sigmoid().gt(0.5)[:, superpoints]
        target = truth.gt(0.5).to(points.device)
        ious = (points & target).sum(-1).float() / (
            (points | target).sum(-1).clamp_min(1).float())
        selected = int(original[index].argmax())
        oracle = int(ious.argmax())
        included = int(actual[index].argmax())
        control = int(shifted[index].argmax())
        rows.append({
            "scene": self.probe_scene,
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "containment_iou": float(ious[included]),
            "shifted_iou": float(ious[control]),
            "containment_flip": included != selected,
            "shifted_flip": control != selected,
            "containment_margin": float(actual[index, oracle] - actual[index, selected]),
            "shifted_margin": float(shifted[index, oracle] - shifted[index, selected]),
            "original_margin": float(original[index, oracle] - original[index, selected]),
        })
    return {"rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=64)
    parser.add_argument("--gpu-id", type=int, default=1)
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
        model.probe_inputs = inputs

    model.dec.indi_embedding.register_forward_pre_hook(capture)
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
            abs(a["selected_iou"] - b["selected_iou"]) > 1e-6
            or abs(a["containment_iou"] - b["containment_iou"]) > 1e-6
            for a, b in zip(first, second))
        rows.extend(first)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": repeat_mismatch,
        "native_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "containment_miou": statistics.fmean(r["containment_iou"] for r in rows),
        "shifted_miou": statistics.fmean(r["shifted_iou"] for r in rows),
        "containment_flips": sum(r["containment_flip"] for r in rows),
        "shifted_flips": sum(r["shifted_flip"] for r in rows),
        "containment_margin_improved": sum(
            r["containment_margin"] > r["original_margin"] for r in errors),
        "shifted_margin_improved": sum(
            r["shifted_margin"] > r["original_margin"] for r in errors),
        "peak_inference_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
