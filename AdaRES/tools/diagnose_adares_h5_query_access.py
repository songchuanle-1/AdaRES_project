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
            masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            truth = target.gt(0.5).to(masks.device)
            ious = (masks & truth).sum(-1).float() / (
                (masks | truth).sum(-1).clamp_min(1).float()
            )
            scores = F.softmax(out["indis"][index], dim=-1)[:, 1]
            selected, oracle = int(scores.argmax()), int(ious.argmax())
            pair_intersection = (masks[selected] & masks[oracle]).sum().float()
            pair_union = (masks[selected] | masks[oracle]).sum().float()
            rank = self.probe_neighbor_ranks[index]
            rows.append({
                "scene_index": self.probe_scene_index,
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "pair_overlap": float(pair_intersection / pair_union.clamp_min(1)),
                "selected_to_oracle_rank": int(rank[selected, oracle]),
                "oracle_to_selected_rank": int(rank[oracle, selected]),
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

    def capture(module, inputs):
        _, fps_seed, offsets, _, _, positions, _ = inputs
        positions = module.get_batches(positions, offsets)[0]
        seeds = positions.gather(1, fps_seed.long().unsqueeze(-1).expand(-1, -1, 3))
        model.probe_seed_positions = seeds

    def capture_sampling(module, inputs, output):
        chosen = output[0].long()
        query_pos = model.probe_seed_positions.gather(
            1, chosen.unsqueeze(-1).expand(-1, -1, 3)
        )
        distances = torch.cdist(query_pos, query_pos)
        model.probe_neighbor_ranks = distances.argsort(-1).argsort(-1) + 1

    model.dec.register_forward_pre_hook(capture)
    model.dec.sampling_module.register_forward_hook(capture_sampling)
    model.predict_by_feat = types.MethodType(predictor(), model)
    rows = []
    for index in range(min(args.scenes, len(dataset))):
        model.probe_scene_index = index
        with torch.no_grad():
            rows.extend(model(dataset.collate_fn([dataset[index]]), mode="predict")["rows"])
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "error_pair_disjoint": sum(r["pair_overlap"] < 0.5 for r in errors),
        "error_mutual_outside_final_four": sum(
            r["selected_to_oracle_rank"] > 4 and r["oracle_to_selected_rank"] > 4
            for r in errors
        ),
        "error_mutual_outside_initial_128": sum(
            r["selected_to_oracle_rank"] > 128 and r["oracle_to_selected_rank"] > 128
            for r in errors
        ),
        "mean_selected_to_oracle_rank": statistics.fmean(
            r["selected_to_oracle_rank"] for r in errors
        ),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
