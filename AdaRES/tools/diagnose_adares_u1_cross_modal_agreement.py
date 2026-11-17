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


def agreement(geometry, appearance, masks, valid):
    geometry = F.normalize(geometry, dim=-1)
    appearance = F.normalize(appearance, dim=-1)
    weights = masks.sigmoid() * valid[:, None, :]
    area = weights.sum(-1).clamp_min(1e-6)
    mean_geometry = F.normalize(torch.bmm(weights, geometry) / area[..., None], dim=-1)
    mean_appearance = F.normalize(torch.bmm(weights, appearance) / area[..., None], dim=-1)
    return (mean_geometry * mean_appearance).sum(-1)


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    valid = (~out["batch_mask"]).float()
    geometry, appearance = self.probe_features
    inside = agreement(geometry, appearance, out["masks"], valid)
    outside = agreement(geometry, appearance, -out["masks"], valid)
    shifted = agreement(geometry, appearance.roll(1, dims=1), out["masks"], valid)
    scores = out["indis"].softmax(-1)[..., 1]
    rows = []
    for index, target in enumerate(gt_pmasks):
        masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        truth = target.gt(0.5).to(masks.device)
        ious = (masks & truth).sum(-1).float() / (
            (masks | truth).sum(-1).clamp_min(1).float()
        )
        selected, oracle = int(scores[index].argmax()), int(ious.argmax())
        rows.append({
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "selected_inside": float(inside[index, selected]),
            "oracle_inside": float(inside[index, oracle]),
            "selected_outside": float(outside[index, selected]),
            "oracle_outside": float(outside[index, oracle]),
            "selected_shifted": float(shifted[index, selected]),
            "oracle_shifted": float(shifted[index, oracle]),
            "inside_only_iou": float(ious[inside[index].argmax()]),
            "outside_only_iou": float(ious[outside[index].argmax()]),
            "shifted_only_iou": float(ious[shifted[index].argmax()]),
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
        geometry = module.input_proj(inputs[0])
        appearance = module.input_proj_2d(inputs[6])
        geometry, _ = module.get_batches(geometry, inputs[2])
        appearance, _ = module.get_batches(appearance, inputs[2])
        model.probe_features = geometry, appearance

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
            abs(a["selected_inside"] - b["selected_inside"]) > 1e-4
            or a["selected_iou"] != b["selected_iou"]
            for a, b in zip(first, repeat)
        )
        rows.extend(first)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": mismatch,
        "oracle_inside_higher": sum(r["oracle_inside"] > r["selected_inside"] for r in errors),
        "oracle_outside_higher": sum(r["oracle_outside"] > r["selected_outside"] for r in errors),
        "oracle_shifted_higher": sum(r["oracle_shifted"] > r["selected_shifted"] for r in errors),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "inside_only_miou": statistics.fmean(r["inside_only_iou"] for r in rows),
        "outside_only_miou": statistics.fmean(r["outside_only_iou"] for r in rows),
        "shifted_only_miou": statistics.fmean(r["shifted_only_iou"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
