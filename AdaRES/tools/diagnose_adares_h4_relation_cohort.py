import argparse
import json
import sys
from pathlib import Path

import gorilla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=16)
    args = parser.parse_args()
    cfg = gorilla.Config.fromfile(args.config)
    data_cfg = cfg.data.val.copy()
    data_cfg.training = False
    data_cfg.aug = False
    data_cfg.with_elastic = False
    data_cfg.lang_num_max = 32
    dataset = build_dataset(data_cfg, get_root_logger())
    rows = []
    for index in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[index]])
        for dense_map in batch["dense_maps"]:
            rows.append({
                "scene_index": index,
                "relation": bool(dense_map["rel_positive_map"].any()),
                "other_entity": bool(dense_map["other_entity_map"].any()),
                "modifier": bool(dense_map["modify_positive_map"].any()),
            })
    result = {"scenes": min(args.scenes, len(dataset)), "expressions": len(rows), "rows": rows}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"scenes": result["scenes"], "expressions": len(rows)}))


if __name__ == "__main__":
    main()
