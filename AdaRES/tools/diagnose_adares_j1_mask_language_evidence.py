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
        masks = out["masks"].sigmoid() * valid[:, None, :]
        scene = self.probe_scene
        language = self.probe_language
        token_mask = self.probe_token_mask.to(language.device).float()
        text = (language * token_mask[:, :, None]).sum(1) / token_mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        fg = torch.bmm(masks, scene) / masks.sum(-1, keepdim=True).clamp_min(1e-6)
        bg_weights = (1 - out["masks"].sigmoid()) * valid[:, None, :]
        bg = torch.bmm(bg_weights, scene) / bg_weights.sum(
            -1, keepdim=True
        ).clamp_min(1e-6)
        contrast = F.cosine_similarity(fg - bg, text[:, None, :], dim=-1)
        scene_mean = (scene * valid[:, :, None]).sum(1) / valid.sum(
            1, keepdim=True
        ).clamp_min(1)
        scene_only = F.cosine_similarity(
            fg - bg, scene_mean[:, None, :], dim=-1
        )
        intersection = torch.bmm(masks, masks.transpose(1, 2))
        area = masks.sum(-1)
        union = area[:, :, None] + area[:, None, :] - intersection
        overlap = intersection / union.clamp_min(1e-6)
        overlap.diagonal(dim1=1, dim2=2).zero_()
        crowding = overlap.sum(-1)
        indicator = F.softmax(out["indis"], dim=-1)[..., 1]
        for index, target in enumerate(gt_pmasks):
            point_masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
            truth = target.gt(0.5).to(point_masks.device)
            ious = (point_masks & truth).sum(-1).float() / (
                (point_masks | truth).sum(-1).clamp_min(1).float()
            )
            selected = int(indicator[index].argmax())
            oracle = int(ious.argmax())
            rows.append({
                "scene_index": self.probe_scene_index,
                "selected_iou": float(ious[selected]),
                "oracle_iou": float(ious[oracle]),
                "selected_contrast": float(contrast[index, selected]),
                "oracle_contrast": float(contrast[index, oracle]),
                "selected_scene_only": float(scene_only[index, selected]),
                "oracle_scene_only": float(scene_only[index, oracle]),
                "selected_crowding": float(crowding[index, selected]),
                "oracle_crowding": float(crowding[index, oracle]),
                "contrast_iou": float(ious[contrast[index].argmax()]),
                "scene_only_iou": float(ious[scene_only[index].argmax()]),
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

    def capture_scene(module, inputs, output):
        model.probe_scene = output

    def capture_language(module, inputs, output):
        model.probe_language = output

    model.dec.ila_ffn_layers[-1].register_forward_hook(capture_scene)
    model.dec.lia_ffn_layers[-1].register_forward_hook(capture_language)
    model.predict_by_feat = types.MethodType(predictor(), model)
    rows = []
    for index in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[index]])
        model.probe_scene_index = index
        model.probe_token_mask = batch["lang_masks"]
        with torch.no_grad():
            rows.extend(model(batch, mode="predict")["rows"])
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "checkpoint": args.checkpoint,
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "oracle_contrast_higher": sum(
            r["oracle_contrast"] > r["selected_contrast"] for r in errors
        ),
        "oracle_scene_only_higher": sum(
            r["oracle_scene_only"] > r["selected_scene_only"] for r in errors
        ),
        "oracle_more_crowded": sum(
            r["oracle_crowding"] > r["selected_crowding"] for r in errors
        ),
        "error_selected_crowding_mean": statistics.fmean(
            r["selected_crowding"] for r in errors
        ),
        "error_oracle_crowding_mean": statistics.fmean(
            r["oracle_crowding"] for r in errors
        ),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "contrast_only_miou": statistics.fmean(r["contrast_iou"] for r in rows),
        "scene_only_miou": statistics.fmean(r["scene_only_iou"] for r in rows),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
