"""Native label counterfactual versus repeated sparse-forward noise."""

import argparse
import gc
import json
import time
from pathlib import Path

import gorilla
import torch

from diagnose_adares_y14_reliability import scene_indices
from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_y15_query_completion import IPDN
from ipdn.utils import get_root_logger
from train import load_shape_compatible_pretrain


def compare(left, right):
    return {
        "probability_max": max(float((a-b).abs().max())
                               for a, b in zip(left, right)),
        "probability_mean": sum(float((a-b).abs().mean())
                                for a, b in zip(left, right))/len(left),
        "binary_flips": sum(int((a.gt(.5) != b.gt(.5)).sum())
                            for a, b in zip(left, right)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", required=True, type=int)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.config)
    dc = cfg.data.train.copy()
    dc.training, dc.aug, dc.with_elastic = False, False, False
    dataset = build_dataset(dc, get_root_logger())
    model = IPDN(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.eval()
    rows = []
    with torch.no_grad():
        for item in scene_indices(dataset.scene_inputs, 8):
            batch = dataset.collate_fn([dataset[item]])
            changed = dict(batch)
            changed["gt_pmasks"] = [torch.zeros_like(t) for t in batch["gt_pmasks"]]
            changed["gt_spmasks"] = [torch.zeros_like(t) for t in batch["gt_spmasks"]]
            model(batch, mode="predict")
            forward_rows = []
            for counterfactual in (False, True, False, True, False, True):
                gorilla.set_random_seed(1999)
                capture = []
                hook = model.dec.register_forward_hook(
                    lambda module, inputs, output: capture.append(output))
                prediction = model(changed if counterfactual else batch,
                                   mode="predict")
                out = capture[-1]
                forward_rows.append({
                    "counterfactual": counterfactual,
                    "probabilities": [p.detach().cpu().clone()
                                      for p in prediction["pred_pmask"]],
                    "selection": out["indis"].softmax(-1)[:, :, 1]
                    .argmax(-1).detach().cpu(),
                    "seeds": out["sample_inds"].detach().cpu(),
                })
                hook.remove()
                del out, capture, prediction
            reference = forward_rows[0]
            comparisons = []
            for row in forward_rows[1:]:
                comparisons.append(dict(
                    changed_labels=row["counterfactual"],
                    **compare(reference["probabilities"], row["probabilities"]),
                    selection_flips=int((reference["selection"] != row["selection"]).sum()),
                    seed_flips=int((reference["seeds"] != row["seeds"]).sum())))
            times = {}
            for enabled in (False, True):
                model.dec.completion_enabled = enabled
                for _ in range(2):
                    model(batch, mode="predict")
                elapsed = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    model(batch, mode="predict")
                    torch.cuda.synchronize()
                    elapsed.append(time.monotonic()-start)
                times[str(enabled)] = elapsed
            model.dec.completion_enabled = True
            row = {"scan_id": batch["scan_ids"][0], "comparisons": comparisons,
                   "inference_seconds": times,
                   "teacher_absent": model._training_teacher is None}
            rows.append(row)
            Path(args.output).write_text(json.dumps({
                "scope": "released-init train8, alternating GT counterfactual",
                "rows": rows}, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
            del forward_rows, reference, batch, changed
            gc.collect()
    Path(args.output).write_text(json.dumps({
        "scope": "released-init train8, alternating GT counterfactual",
        "rows": rows, "complete": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
