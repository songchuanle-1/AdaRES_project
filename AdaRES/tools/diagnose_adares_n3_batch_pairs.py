import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import gorilla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--indices", type=int, nargs="+", default=(0, 1, 64, 65, 128, 129, 256, 257))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = gorilla.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    rows = []
    for index in args.indices:
        batch = dataset.collate_fn([dataset[index]])
        ids = [tuple(int(v) for v in values) for values in batch["object_idss"]]
        count = Counter(ids)
        different = sum(1 for i in range(len(ids)) for j in range(i + 1, len(ids))
                        if len(ids[i]) == len(ids[j]) == 1 and ids[i] != ids[j])
        rows.append({
            "scene_index": index,
            "expressions": len(ids),
            "unique_objects": len(count),
            "single_target_different_pairs": different,
            "single_target_same_pairs": sum(
                n * (n - 1) // 2 for key, n in count.items() if len(key) == 1
            ),
        })
    result = {"rows": rows, "different_pairs": sum(row["single_target_different_pairs"] for row in rows)}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
