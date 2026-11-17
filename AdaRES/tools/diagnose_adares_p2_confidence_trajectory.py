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
    previous = out["aux_outputs"][-1]["indis"]
    rows = []
    for index, target in enumerate(gt_pmasks):
        point_masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        truth = target.gt(0.5).to(point_masks.device)
        ious = (point_masks & truth).sum(-1).float() / (
            (point_masks | truth).sum(-1).clamp_min(1).float()
        )
        final = F.softmax(out["indis"][index], dim=-1)[:, 1]
        prior = F.softmax(previous[index], dim=-1)[:, 1]
        selected, previous_selected, oracle = (
            int(final.argmax()), int(prior.argmax()), int(ious.argmax())
        )
        rows.append({
            "selected": selected,
            "previous_selected": previous_selected,
            "selected_iou": float(ious[selected]),
            "previous_iou": float(ious[previous_selected]),
            "oracle_iou": float(ious[oracle]),
            "oracle_final_rank": int((final > final[oracle]).sum()) + 1,
            "oracle_previous_rank": int((prior > prior[oracle]).sum()) + 1,
            "final_margin": float(final[selected] - final[oracle]),
            "previous_margin": float(prior[selected] - prior[oracle]),
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
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    repeat_mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            base = model(batch, mode="predict")["rows"]
            repeat = model(batch, mode="predict")["rows"]
        repeat_mismatch += sum(
            a["selected"] != b["selected"]
            or a["previous_selected"] != b["previous_selected"]
            or abs(a["selected_iou"] - b["selected_iou"]) > 1e-6
            or abs(a["previous_iou"] - b["previous_iou"]) > 1e-6
            for a, b in zip(base, repeat)
        )
        rows.extend(base)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": repeat_mismatch,
        "selection_changed": sum(
            r["selected"] != r["previous_selected"] for r in rows
        ),
        "previous_recovers_error": sum(
            r["previous_iou"] > r["selected_iou"] + 0.05 for r in errors
        ),
        "previous_harms_correct": sum(
            r["previous_iou"] + 0.05 < r["selected_iou"]
            for r in rows if r not in errors
        ),
        "oracle_previous_rank_better": sum(
            r["oracle_previous_rank"] < r["oracle_final_rank"] for r in errors
        ),
        "oracle_previous_margin_lower": sum(
            r["previous_margin"] < r["final_margin"] for r in errors
        ),
        "final_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "previous_miou": statistics.fmean(r["previous_iou"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
