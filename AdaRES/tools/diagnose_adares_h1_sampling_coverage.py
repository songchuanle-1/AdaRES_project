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
        rows = []
        for index, target in enumerate(gt_pmasks):
            labels = self.probe_labels[0]
            seed_labels = labels[fps_seed_sp[index].long()]
            sample_inds = out["sample_inds"][index].long()
            selected_labels = seed_labels[sample_inds]
            ref_scores = self.probe_ref_scores[index]
            masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            truth = target.gt(0.5).to(masks.device)
            ious = (masks & truth).sum(-1).float() / (
                (masks | truth).sum(-1).clamp_min(1).float()
            )
            scores = F.softmax(out["indis"][index], dim=-1)[:, 1]
            selected, oracle = int(scores.argmax()), int(ious.argmax())
            object_ids = object_idss[index]
            seed_target = torch.zeros_like(seed_labels, dtype=torch.bool)
            for object_id in object_ids:
                seed_target |= seed_labels == object_id
            selected_target = seed_target[sample_inds]
            rows.append({
                "scene_index": self.probe_scene_index,
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "target_seed_count": int(seed_target.sum()),
                "selected_target_seed_count": int(selected_target.sum()),
                "oracle_target_seed": bool(selected_target[oracle]),
                "selected_target_seed": bool(selected_target[selected]),
                "target_seed_max_rank": (
                    int((ref_scores > ref_scores[seed_target].max()).sum()) + 1
                    if seed_target.any() else None
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

    def capture_sampling(module, inputs, output):
        model.probe_ref_scores = output[1]

    model.dec.sampling_module.register_forward_hook(capture_sampling)
    model.predict_by_feat = types.MethodType(predictor(), model)
    rows = []
    for index in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[index]])
        model.probe_scene_index = index
        model.probe_labels = batch["sp_ins_labels"]
        with torch.no_grad():
            rows.extend(model(batch, mode="predict")["rows"])
    errors = [row for row in rows if row["oracle_iou"] - row["selected_iou"] > 0.05]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "target_absent_all_seeds": sum(r["target_seed_count"] == 0 for r in rows),
        "target_absent_selected_seeds": sum(
            r["selected_target_seed_count"] == 0 for r in rows
        ),
        "error_target_absent_selected": sum(
            r["selected_target_seed_count"] == 0 for r in errors
        ),
        "error_target_available_but_unsampled": sum(
            r["target_seed_count"] > 0 and r["selected_target_seed_count"] == 0
            for r in errors
        ),
        "mean_target_seed_count": statistics.fmean(
            r["target_seed_count"] for r in rows
        ),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
