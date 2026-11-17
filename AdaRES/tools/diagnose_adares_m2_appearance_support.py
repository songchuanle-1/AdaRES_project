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
        valid = (~out["batch_mask"]).float()
        features = F.normalize(self.probe_2d, dim=-1)
        weights = out["masks"].sigmoid() * valid[:, None, :]
        appearance = F.normalize(
            torch.bmm(weights, features) / weights.sum(-1, keepdim=True).clamp_min(1e-6),
            dim=-1,
        )
        for index, target in enumerate(gt_pmasks):
            masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            truth = target.gt(0.5).to(masks.device)
            ious = (masks & truth).sum(-1).float() / (
                (masks | truth).sum(-1).clamp_min(1).float()
            )
            seeds = fps_seed_sp[index].long()[out["sample_inds"][index].long()]
            support = (appearance[index] * features[index, seeds]).sum(-1)
            confidence = F.softmax(out["indis"][index], dim=-1)[:, 1]
            selected = int(confidence.argmax())
            oracle = int(ious.argmax())
            rows.append({
                "scene_index": self.probe_scene_index,
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "selected_appearance_support": float(support[selected]),
                "oracle_appearance_support": float(support[oracle]),
                "appearance_only_iou": float(ious[support.argmax()]),
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

    def capture_2d(module, inputs):
        model.probe_2d, _ = module.get_batches(inputs[6], inputs[2])

    model.dec.register_forward_pre_hook(capture_2d)
    model.predict_by_feat = types.MethodType(predictor(), model)
    rows = []
    for index in range(min(args.scenes, len(dataset))):
        model.probe_scene_index = index
        with torch.no_grad():
            rows.extend(model(dataset.collate_fn([dataset[index]]), mode="predict")["rows"])
    errors = [row for row in rows if row["oracle_iou"] - row["selected_iou"] > 0.05]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "oracle_support_higher": sum(
            r["oracle_appearance_support"] > r["selected_appearance_support"]
            for r in errors
        ),
        "selected_support_mean": statistics.fmean(
            r["selected_appearance_support"] for r in errors
        ),
        "oracle_support_mean": statistics.fmean(
            r["oracle_appearance_support"] for r in errors
        ),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "appearance_only_miou": statistics.fmean(
            r["appearance_only_iou"] for r in rows
        ),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
