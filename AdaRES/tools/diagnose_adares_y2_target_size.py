import argparse
import json
import sys
from pathlib import Path

import gorilla
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--scenes", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = getattr(cfg.data, args.split).copy()
    data_cfg.training = False
    data_cfg.aug = False
    data_cfg.with_elastic = False
    data_cfg.lang_num_max = 32
    dataset = build_dataset(data_cfg, get_root_logger())
    rows = []
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        if len(batch["gt_pmasks"]) != len(batch["ann_ids"]):
            raise RuntimeError("Expression and annotation counts differ")
        for expression, (truth, ann_id) in enumerate(
                zip(batch["gt_pmasks"], batch["ann_ids"])):
            mask = torch.as_tensor(truth).gt(0.5)
            rows.append({
                "scene_index": scene,
                "expression_index": expression,
                "scan_id": str(batch["scan_ids"][0]),
                "ann_id": str(ann_id),
                "target_points": int(mask.sum()),
                "scene_points": int(mask.numel()),
                "target_fraction": float(mask.float().mean()),
            })
    result = {"split": args.split, "scenes": min(args.scenes, len(dataset)),
              "expressions": len(rows), "rows": rows}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
