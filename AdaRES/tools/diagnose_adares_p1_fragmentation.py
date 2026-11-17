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


def component_fraction(mask, neighbors):
    foreground = mask.nonzero(as_tuple=True)[0].tolist()
    if not foreground:
        return 0.0
    active = set(foreground)
    seen = set()
    largest = 0
    for start in foreground:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in neighbors[current]:
                if neighbor in active and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        largest = max(largest, size)
    return largest / len(foreground)


def predictor(self, scan_ids, object_idss, ann_ids, out, superpoints,
              gt_pmasks, gt_spmasks, fps_seed_sp):
    rows = []
    for index, target in enumerate(gt_pmasks):
        valid = ~out["batch_mask"][index]
        coords = self.probe_coords[index, valid]
        count = coords.shape[0]
        nearest = torch.cdist(coords, coords).topk(
            min(9, count), largest=False
        ).indices[:, 1:].tolist()
        neighbors = [set(row) for row in nearest]
        for source, row in enumerate(nearest):
            for destination in row:
                neighbors[destination].add(source)
        masks = out["masks"][index, :, valid].sigmoid().gt(0.5)
        point_masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        truth = target.gt(0.5).to(point_masks.device)
        ious = (point_masks & truth).sum(-1).float() / (
            (point_masks | truth).sum(-1).clamp_min(1).float()
        )
        confidence = F.softmax(out["indis"][index], dim=-1)[:, 1]
        selected = int(confidence.argmax())
        oracle = int(ious.argmax())
        fractions = [
            component_fraction(mask.cpu(), neighbors) for mask in masks
        ]
        rows.append({
            "selected": selected,
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "selected_fraction": fractions[selected],
            "oracle_fraction": fractions[oracle],
            "fraction_only_iou": float(ious[max(
                range(len(fractions)), key=fractions.__getitem__
            )]),
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

    def capture_coords(module, inputs):
        model.probe_coords = module.get_batches(inputs[5], inputs[2])[0]

    model.dec.register_forward_pre_hook(capture_coords)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    repeat_mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            base = model(batch, mode="predict")["rows"]
            repeat = model(batch, mode="predict")["rows"]
        repeat_mismatch += sum(a != b for a, b in zip(base, repeat))
        rows.extend(base)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": repeat_mismatch,
        "oracle_more_connected": sum(
            r["oracle_fraction"] > r["selected_fraction"] for r in errors
        ),
        "selected_fraction_mean": statistics.fmean(
            r["selected_fraction"] for r in errors
        ) if errors else None,
        "oracle_fraction_mean": statistics.fmean(
            r["oracle_fraction"] for r in errors
        ) if errors else None,
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "fraction_only_miou": statistics.fmean(
            r["fraction_only_iou"] for r in rows
        ),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
