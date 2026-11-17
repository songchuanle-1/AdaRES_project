import argparse
import json
import sys
from pathlib import Path

import gorilla
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_b4_query_competition import IPDN
from ipdn.utils import get_root_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--indices", type=int, nargs="+", default=(0, 1, 64, 65, 128, 129, 256, 257))
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    model = IPDN(**cfg.model).cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loaded = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    allowed = {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.eval()
    capture = {}

    def capture_decoder(module, inputs, output):
        capture["output"] = output

    model.dec.register_forward_hook(capture_decoder)
    pairs = []
    for scene in args.indices:
        batch = dataset.collate_fn([dataset[scene]])
        with torch.no_grad():
            model(batch, mode="loss")
            output = capture["output"]
            masks = output["masks"].sigmoid().gt(0.5)
            confidence = F.softmax(output["indis"], dim=-1)[..., 1]
            selected = confidence.argmax(-1)
            targets = batch["gt_spmasks"]
            ids = batch["object_idss"]
            for left in range(len(ids)):
                for right in range(left + 1, len(ids)):
                    if len(ids[left]) != 1 or len(ids[right]) != 1 or ids[left] == ids[right]:
                        continue
                    a = masks[left, selected[left]]
                    b = masks[right, selected[right]]
                    valid = ~(output["batch_mask"][left] | output["batch_mask"][right])
                    overlap = ((a & b) & valid).sum().float() / (((a | b) & valid).sum().clamp_min(1))
                    def quality(expr, pred):
                        target = targets[expr].to(pred.device).gt(0.5)
                        pred = pred[:target.numel()]
                        return float((pred & target).sum().float() / (pred | target).sum().clamp_min(1))
                    pairs.append({
                        "scene": scene,
                        "left": left,
                        "right": right,
                        "overlap": float(overlap),
                        "left_iou": quality(left, a),
                        "right_iou": quality(right, b),
                        "left_confidence": float(confidence[left, selected[left]]),
                        "right_confidence": float(confidence[right, selected[right]]),
                    })
        capture.clear()
    result = {"scenes": len(args.indices), "distinct_target_pairs": len(pairs), "pairs": pairs}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"scenes": result["scenes"], "distinct_target_pairs": len(pairs),
                      "overlap_ge_half": sum(row["overlap"] >= 0.5 for row in pairs)}))


if __name__ == "__main__":
    main()
