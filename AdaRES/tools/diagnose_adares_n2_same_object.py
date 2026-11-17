import argparse
import json
import sys
import types
from collections import defaultdict
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
    def predict_by_feat(self, scan_ids, object_idss, ann_ids, out, superpoints,
                        gt_pmasks, gt_spmasks, fps_seed_sp):
        rows = []
        for index, truth in enumerate(gt_pmasks):
            masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            target = truth.gt(0.5).to(masks.device)
            ious = (masks & target).sum(-1).float() / (
                (masks | target).sum(-1).clamp_min(1).float()
            )
            scores = F.softmax(out["indis"][index], dim=-1)[:, 1]
            selected = int(scores.argmax())
            oracle = int(ious.argmax())
            rows.append({
                "scene": self.probe_scene,
                "object_ids": [int(v) for v in object_idss[index]],
                "language": self.probe_language[index].tolist(),
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "selected_mask": torch.where(masks[selected])[0].tolist(),
                "oracle_mask": torch.where(masks[oracle])[0].tolist(),
            })
        return {"rows": rows}
    return predict_by_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--output", required=True)
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
    allowed = {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.eval()
    def capture_language(module, inputs):
        language, mask = inputs[3], inputs[4].bool()
        valid = mask.unsqueeze(-1).to(language.dtype)
        pooled = (language * valid).sum(1) / valid.sum(1).clamp_min(1)
        model.probe_language = F.normalize(pooled, dim=-1)

    model.dec.register_forward_pre_hook(capture_language)
    model.predict_by_feat = types.MethodType(predictor(), model)
    rows = []
    for scene in range(min(args.scenes, len(dataset))):
        model.probe_scene = scene
        with torch.no_grad():
            rows.extend(model(dataset.collate_fn([dataset[scene]]), mode="predict")["rows"])
    groups = defaultdict(list)
    for row in rows:
        if len(row["object_ids"]) == 1:
            groups[(row["scene"], row["object_ids"][0])].append(row)
    pairs = []
    for (scene, object_id), group in groups.items():
        if len(group) < 2:
            continue
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                a, b = set(left["selected_mask"]), set(right["selected_mask"])
                c, d = set(left["oracle_mask"]), set(right["oracle_mask"])
                pairs.append({
                    "scene": scene,
                    "object_id": object_id,
                    "selected_overlap": len(a & b) / max(1, len(a | b)),
                    "oracle_overlap": len(c & d) / max(1, len(c | d)),
                    "left_iou": left["selected_iou"],
                    "right_iou": right["selected_iou"],
                    "left_oracle_iou": left["oracle_iou"],
                    "right_oracle_iou": right["oracle_iou"],
                })
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "same_object_groups": sum(len(g) >= 2 for g in groups.values()),
        "same_object_pairs": len(pairs),
        "pairs": pairs,
    }
    nearest = []
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    for group in by_scene.values():
        if len(group) < 2:
            continue
        languages = torch.tensor([row["language"] for row in group])
        similarities = languages @ languages.T
        similarities.fill_diagonal_(-float("inf"))
        for index, partner_index in enumerate(similarities.argmax(-1).tolist()):
            left, right = group[index], group[partner_index]
            a, b = set(left["selected_mask"]), set(right["selected_mask"])
            nearest.append({
                "same_object": len(left["object_ids"]) == len(right["object_ids"]) == 1
                and left["object_ids"] == right["object_ids"],
                "selected_overlap": len(a & b) / max(1, len(a | b)),
                "left_iou": left["selected_iou"],
                "right_iou": right["selected_iou"],
            })
    result["language_nearest"] = nearest
    scene_pairs = []
    for group in by_scene.values():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                a, b = set(left["selected_mask"]), set(right["selected_mask"])
                c, d = set(left["oracle_mask"]), set(right["oracle_mask"])
                scene_pairs.append({
                    "same_object": len(left["object_ids"]) == len(right["object_ids"]) == 1
                    and left["object_ids"] == right["object_ids"],
                    "selected_overlap": len(a & b) / max(1, len(a | b)),
                    "oracle_overlap": len(c & d) / max(1, len(c | d)),
                    "left_iou": left["selected_iou"],
                    "right_iou": right["selected_iou"],
                })
    result["scene_pairs"] = scene_pairs
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}))


if __name__ == "__main__":
    main()
