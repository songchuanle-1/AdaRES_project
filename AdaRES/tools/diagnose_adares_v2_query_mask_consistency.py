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
    query = F.normalize(self.probe_query, dim=-1)
    scene = F.normalize(self.probe_scene, dim=-1)
    valid = (~out["batch_mask"]).float()
    weights = out["masks"].sigmoid() * valid[:, None, :]
    inside = F.normalize(torch.bmm(weights, scene) /
                         weights.sum(-1, keepdim=True).clamp_min(1e-6), dim=-1)
    outside_weights = valid[:, None, :] - weights
    outside = F.normalize(torch.bmm(outside_weights, scene) /
                          outside_weights.sum(-1, keepdim=True).clamp_min(1e-6), dim=-1)
    consistency = (query * inside).sum(-1)
    control = (query * outside).sum(-1)
    scores = out["indis"].softmax(-1)[..., 1]
    rows = []
    for index, target in enumerate(gt_pmasks):
        masks = out["masks"][index].sigmoid().gt(0.5)[:, superpoints]
        truth = target.gt(0.5).to(masks.device)
        ious = (masks & truth).sum(-1).float() / (
            (masks | truth).sum(-1).clamp_min(1).float()
        )
        selected, oracle = int(scores[index].argmax()), int(ious.argmax())
        rows.append({
            "selected_iou": float(ious[selected]),
            "oracle_iou": float(ious[oracle]),
            "selected_inside": float(consistency[index, selected]),
            "oracle_inside": float(consistency[index, oracle]),
            "selected_outside": float(control[index, selected]),
            "oracle_outside": float(control[index, oracle]),
            "inside_only_iou": float(ious[consistency[index].argmax()]),
            "outside_only_iou": float(ious[control[index].argmax()]),
        })
    return {"rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=64)
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

    def capture_query(module, inputs):
        model.probe_query = inputs[0]

    def capture_scene(module, inputs, output):
        model.probe_scene = output

    model.dec.indi_embedding.register_forward_pre_hook(capture_query)
    model.dec.ila_ffn_layers[-1].register_forward_hook(capture_scene)
    model.predict_by_feat = types.MethodType(predictor, model)
    rows = []
    mismatch = 0
    for scene in range(min(args.scenes, len(dataset))):
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            first = model(batch, mode="predict")["rows"]
            repeat = model(batch, mode="predict")["rows"]
        mismatch += sum(
            abs(a["selected_inside"] - b["selected_inside"]) > 1e-4
            or a["selected_iou"] != b["selected_iou"]
            for a, b in zip(first, repeat)
        )
        rows.extend(first)
    errors = [r for r in rows if r["oracle_iou"] - r["selected_iou"] > 0.05]
    result = {
        "scenes": min(args.scenes, len(dataset)),
        "expressions": len(rows),
        "material_errors": len(errors),
        "repeat_mismatch": mismatch,
        "oracle_inside_higher": sum(r["oracle_inside"] > r["selected_inside"] for r in errors),
        "oracle_outside_higher": sum(r["oracle_outside"] > r["selected_outside"] for r in errors),
        "indicator_miou": statistics.fmean(r["selected_iou"] for r in rows),
        "inside_only_miou": statistics.fmean(r["inside_only_iou"] for r in rows),
        "outside_only_miou": statistics.fmean(r["outside_only_iou"] for r in rows),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
