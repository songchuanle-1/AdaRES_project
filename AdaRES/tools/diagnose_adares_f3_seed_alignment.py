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
sys.path.insert(0, str(Path.cwd()))

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_b4_query_competition import IPDN
from ipdn.utils import get_root_logger


def predictor():
    def predict_by_feat(
        self, scan_ids, object_idss, ann_ids, out, superpoints,
        gt_pmasks, gt_spmasks, fps_seed_sp,
    ):
        rows = []
        for index, target in enumerate(gt_pmasks):
            valid = ~out["batch_mask"][index]
            coords = self.probe_coords[index, valid]
            probabilities = out["masks"][index][:, valid].sigmoid()
            foreground = (probabilities > 0.5).float()
            masses = foreground.sum(-1)
            centers = foreground @ coords / masses.clamp_min(1).unsqueeze(-1)
            seed_indices = fps_seed_sp[index].long()
            sampled_indices = out["sample_inds"][index].long()
            seeds = coords[seed_indices[sampled_indices]]
            if seeds.shape != centers.shape:
                raise RuntimeError(f"Seed/query mismatch: {seeds.shape} vs {centers.shape}")
            scene_diag = (coords.max(0).values - coords.min(0).values).norm().clamp_min(1e-6)
            displacement = (centers - seeds).norm(dim=-1) / scene_diag
            point_masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            truth = target.gt(0.5).to(point_masks.device)
            ious = (point_masks & truth).sum(-1).float() / (
                (point_masks | truth).sum(-1).clamp_min(1).float()
            )
            scores = F.softmax(out["indis"][index], dim=-1)[:, 1]
            selected = int(scores.argmax())
            oracle = int(ious.argmax())
            rows.append({
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "selected_displacement": float(displacement[selected]),
                "oracle_displacement": float(displacement[oracle]),
                "selected_empty": bool(masses[selected] == 0),
                "oracle_empty": bool(masses[oracle] == 0),
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
    model = IPDN(**cfg.model)
    model.cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loaded = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    allowed = {
        "text_encoder.embeddings.position_ids",
        "text_encoder.embeddings.token_type_ids",
    }
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.eval()

    def capture_coords(module, inputs):
        model.probe_coords = module.get_batches(inputs[5], inputs[2])[0]

    model.dec.register_forward_pre_hook(capture_coords)
    model.predict_by_feat = types.MethodType(predictor(), model)
    rows = []
    for index in range(min(args.scenes, len(dataset))):
        with torch.no_grad():
            rows.extend(model(dataset.collate_fn([dataset[index]]), mode="predict")["rows"])
    if not rows:
        raise RuntimeError("No expressions in fixed subset")
    errors = [row for row in rows if row["oracle_iou"] - row["selected_iou"] > 0.05]
    nonempty = [row for row in errors if not row["selected_empty"] and not row["oracle_empty"]]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "nonempty_error_pairs": len(nonempty),
        "error_oracle_closer_to_seed": sum(
            row["oracle_displacement"] < row["selected_displacement"]
            for row in nonempty
        ),
        "selected_displacement_mean": (
            statistics.fmean(row["selected_displacement"] for row in nonempty)
            if nonempty else None
        ),
        "oracle_displacement_mean": (
            statistics.fmean(row["oracle_displacement"] for row in nonempty)
            if nonempty else None
        ),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
