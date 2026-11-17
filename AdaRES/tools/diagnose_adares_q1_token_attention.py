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
    weights = self.probe_attention
    if weights.ndim == 4:
        weights = weights.mean(1)
    if weights.ndim != 3 or weights.shape[:2] != out["indis"].shape[:2]:
        raise RuntimeError(f"Unexpected language attention shape: {weights.shape}")
    valid = self.probe_token_mask.to(weights.device).bool()
    weights = weights.masked_fill(~valid[:, None, :], 0)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    concentration = weights.square().sum(-1)
    rows = []
    for index, target in enumerate(gt_pmasks):
        target_tokens = self.probe_dense_maps[index]["positive_map"][0, :weights.shape[-1]]
        target_tokens = target_tokens.to(weights.device).bool() & valid[index]
        token_count = int(valid[index].sum())
        rotated_tokens = torch.zeros_like(target_tokens)
        rotated_tokens[:token_count] = target_tokens[:token_count].roll(1)
        target_mass = weights[index, :, target_tokens].sum(-1)
        rotated_mass = weights[index, :, rotated_tokens].sum(-1)
        masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        truth = target.gt(0.5).to(masks.device)
        ious = (masks & truth).sum(-1).float() / (
            (masks | truth).sum(-1).clamp_min(1).float()
        )
        confidence = F.softmax(out["indis"][index], dim=-1)[:, 1]
        selected, oracle = int(confidence.argmax()), int(ious.argmax())
        rows.append({
            "scene": self.probe_scene,
            "selected": selected,
            "oracle": oracle,
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "selected_concentration": float(concentration[index, selected]),
            "oracle_concentration": float(concentration[index, oracle]),
            "selected_target_mass": float(target_mass[selected]),
            "oracle_target_mass": float(target_mass[oracle]),
            "selected_rotated_mass": float(rotated_mass[selected]),
            "oracle_rotated_mass": float(rotated_mass[oracle]),
            "target_tokens": int(target_tokens.sum()),
            "concentration_iou": float(ious[concentration[index].argmax()]),
            "valid_tokens": int(valid[index].sum()),
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

    def capture(module, inputs, output):
        model.probe_attention = output[1]

    model.dec.qla_layers[-1].register_forward_hook(capture)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    repeat_mismatch = 0
    max_repeat_deviation = 0.0
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        model.probe_scene = scene
        model.probe_token_mask = batch["lang_masks"]
        model.probe_dense_maps = batch["dense_maps"]
        with torch.no_grad():
            base = model(batch, mode="predict")["rows"]
            repeat = model(batch, mode="predict")["rows"]
        for a, b in zip(base, repeat):
            deviation = max(
                abs(a["selected_concentration"] - b["selected_concentration"]),
                abs(a["oracle_concentration"] - b["oracle_concentration"]),
            )
            max_repeat_deviation = max(max_repeat_deviation, deviation)
            repeat_mismatch += (
                a["selected"] != b["selected"] or a["oracle"] != b["oracle"]
                or deviation > 1e-5
            )
        rows.extend(base)
    errors = [row for row in rows
              if row["oracle_iou"] - row["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": repeat_mismatch,
        "max_repeat_deviation": max_repeat_deviation,
        "oracle_more_concentrated": sum(
            row["oracle_concentration"] > row["selected_concentration"]
            for row in errors
        ),
        "oracle_more_target_mass": sum(
            row["oracle_target_mass"] > row["selected_target_mass"]
            for row in errors if row["target_tokens"]
        ),
        "oracle_more_rotated_mass": sum(
            row["oracle_rotated_mass"] > row["selected_rotated_mass"]
            for row in errors if row["target_tokens"]
        ),
        "errors_with_target_tokens": sum(
            bool(row["target_tokens"]) for row in errors
        ),
        "selected_concentration_mean": statistics.fmean(
            row["selected_concentration"] for row in errors
        ) if errors else None,
        "oracle_concentration_mean": statistics.fmean(
            row["oracle_concentration"] for row in errors
        ) if errors else None,
        "indicator_miou": statistics.fmean(row["selected_iou"] for row in rows),
        "concentration_only_miou": statistics.fmean(
            row["concentration_iou"] for row in rows
        ),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
