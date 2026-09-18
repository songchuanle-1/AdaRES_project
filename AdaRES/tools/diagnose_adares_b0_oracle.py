import argparse
import json
import os
from pathlib import Path
import sys
import time
import types

import gorilla
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare AdaRES-B0 top-1 selection with the all-query oracle "
            "on a bounded validation subset."
        )
    )
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--scenes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1999)
    return parser.parse_args()


def point_iou(superpoint_mask, superpoints, target):
    prediction = superpoint_mask.gt(0.0)[superpoints]
    target = target.gt(0.5)
    intersection = (prediction & target).sum().to(torch.float32)
    union = (prediction | target).sum().to(torch.float32)
    return float(
        (intersection / union.clamp_min(1.0)).item()
    )


def make_oracle_predictor():
    def predict_by_feat(
        self,
        scan_ids,
        object_idss,
        ann_ids,
        out,
        superpoints,
        gt_pmasks,
        gt_spmasks,
        fps_seed_sp,
    ):
        rows = []
        superpoints = superpoints.to(out["masks"].device)
        for i in range(len(object_idss)):
            if self.task_type != "res":
                raise NotImplementedError("RES only.")
            masks = out["masks"][i]
            indicators = out["indis"][i]
            score = F.softmax(indicators, dim=-1)[:, 1]
            top_index = int(score.argmax().item())
            point_target = gt_pmasks[i].to(masks.device)
            top_iou = point_iou(
                masks[top_index], superpoints, point_target
            )
            all_ious = [
                point_iou(masks[index], superpoints, point_target)
                for index in range(masks.shape[0])
            ]
            oracle_index = int(np.argmax(all_ious))
            rows.append(
                {
                    "scan_id": scan_ids[0],
                    "ann_id": int(ann_ids[i]),
                    "top1_iou": top_iou,
                    "oracle_iou": all_ious[oracle_index],
                    "oracle_index": oracle_index,
                    "top1_index": top_index,
                    "score_top1": float(score[top_index].item()),
                    "score_oracle": float(score[oracle_index].item()),
                    "oracle_rank": int(
                        (score.argsort(descending=True)
                         == oracle_index).nonzero().item()
                    ) + 1,
                }
            )
        return {
            "rows": rows,
            "scan_id": [row["scan_id"] for row in rows],
            "ann_id": [row["ann_id"] for row in rows],
            "piou": [torch.tensor(row["top1_iou"]) for row in rows],
            "spiou": [torch.tensor(row["top1_iou"]) for row in rows],
            "nt_label": [False] * len(rows),
            "pred_pmask": [],
            "gt_pmask": [],
            "sp_r_iou": [],
            "p_r_iou": [],
            "seed_ind": [],
            "sample_ind": [],
            "indi": [],
        }

    return predict_by_feat


def main():
    args = parse_args()
    root = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, root)
    from ipdn.dataset import build_dataset
    from ipdn import IPDN
    from ipdn.utils import get_root_logger

    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(args.seed)
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = cfg.data.val.copy()
    data_cfg.training = False
    data_cfg.aug = False
    data_cfg.with_elastic = False
    data_cfg.lang_num_max = 8
    logger = get_root_logger(
        log_file=os.path.splitext(args.output)[0] + ".log"
    )
    dataset = build_dataset(data_cfg, logger)
    model = IPDN(**cfg.model)
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    own = model.state_dict()
    filtered = {
        key: value
        for key, value in state.items()
        if key in own and own[key].shape == value.shape
    }
    result = model.load_state_dict(filtered, strict=False)
    logger.info(
        "Loaded=%d missing=%d unexpected=%d meta=%s",
        len(filtered),
        len(result.missing_keys),
        len(result.unexpected_keys),
        checkpoint.get("meta"),
    )
    model.predict_by_feat = types.MethodType(
        make_oracle_predictor(), model
    )

    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for dataset_index in range(min(args.scenes, len(dataset))):
            batch = dataset.collate_fn([dataset[dataset_index]])
            output = model(batch, mode="predict")
            rows.extend(output["rows"])
            del output
            del batch
            torch.cuda.empty_cache()
    top1 = np.asarray([row["top1_iou"] for row in rows])
    oracle = np.asarray([row["oracle_iou"] for row in rows])
    summary = {
        "expressions": len(rows),
        "top1_miou": float(top1.mean()),
        "oracle_miou": float(oracle.mean()),
        "oracle_gap": float((oracle - top1).mean()),
        "oracle_rank_histogram": {
            str(rank): int(
                sum(row["oracle_rank"] == rank for row in rows)
            )
            for rank in sorted(
                {row["oracle_rank"] for row in rows}
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
