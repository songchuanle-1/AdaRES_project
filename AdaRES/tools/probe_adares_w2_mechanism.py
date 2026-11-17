import argparse
import importlib
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
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--indices", type=int, nargs=2, default=[0, 1])
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    batch = dataset.collate_fn([dataset[i] for i in args.indices])
    import ipdn.model_paper5
    ipdn.model_paper5.__path__.append(str(args.package_dir.resolve()))
    module = importlib.import_module(
        "ipdn.model_paper5.adares_w2_pointmass_competition")
    model = module.IPDN(**cfg.model)
    model.cuda()
    model.train()
    original = model.point_masses
    captures = []

    def capture(_, inputs, output):
        captures.append(output.detach().clone())

    model.dec.indi_embedding.register_forward_hook(capture)

    def run(rotated):
        captures.clear()
        if rotated:
            def rotated_mass(points, offsets, scenes_len):
                masses = original(points, offsets, scenes_len)
                scene_masses = original(points, offsets,
                                         [1] * len(scenes_len))
                scene_lengths = [int(offsets[i + 1] - offsets[i])
                                 for i in range(len(scenes_len))]
                expanded = []
                start = 0
                for count, repeats in zip(scene_lengths, scenes_len):
                    expanded.append(scene_masses[start:start + count].roll(1)
                                    .repeat(int(repeats)))
                    start += count
                return torch.cat(expanded)
            model.point_masses = rotated_mass
        else:
            model.point_masses = original
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1999)
        torch.cuda.manual_seed_all(1999)
        loss, _ = model(batch, mode="loss")
        loss.backward()
        gradients = {
            key: getattr(model.dec.indi_embedding.head[0], key).grad.detach().clone()
            for key in ("weight", "bias")
        }
        return float(loss.detach()), [v.clone() for v in captures], gradients

    actual_loss, actual_logits, actual_grads = run(False)
    rotated_loss, rotated_logits, rotated_grads = run(True)
    result = {
        "indices": args.indices,
        "actual_loss": actual_loss,
        "rotated_loss": rotated_loss,
        "layer_logit_max_diff": [
            float((a - b).abs().max()) for a, b in zip(actual_logits, rotated_logits)
        ],
        "head_weight_grad_max_diff": float(
            (actual_grads["weight"] - rotated_grads["weight"]).abs().max()),
        "head_weight_grad_norm": float(actual_grads["weight"].norm()),
        "head_weight_grad_finite": bool(
            torch.isfinite(actual_grads["weight"]).all()),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
