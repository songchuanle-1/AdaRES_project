import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import gorilla
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.cwd()))

from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger


def check_anchor(decoder, device):
    positions = torch.tensor([[[0., 0., 0.], [2., 0., 0.], [99., 0., 0.]]], device=device)
    logits = torch.tensor([[[0., 8., 100.], [-100., -100., 100.]]], device=device)
    padding = torch.tensor([[False, False, True]], device=device)
    seed = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]], device=device)
    anchor = decoder.mask_anchor(logits, positions, padding, seed)
    if not (1.3 < anchor[0, 0, 0].item() < 1.4):
        raise RuntimeError(f"Mask anchor ignored valid foreground: {anchor}")
    if not torch.equal(anchor[0, 1], seed[0, 1]):
        raise RuntimeError(f"Empty predicted foreground did not revert to seed: {anchor}")
    positions[0, 2] = -999.
    changed = decoder.mask_anchor(logits, positions, padding, seed)
    if not torch.equal(anchor, changed):
        raise RuntimeError("Padding position changed anchor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--module", required=True)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--indices", type=int, nargs=2, default=[0, 1])
    args = parser.parse_args()

    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    batch = dataset.collate_fn([dataset[index] for index in args.indices])
    model = importlib.import_module(args.module).IPDN(**cfg.model).cuda()
    model.train()
    if hasattr(model.dec, "mask_anchor"):
        check_anchor(model.dec, torch.device("cuda"))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss, _ = model(batch, mode="loss")
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite native loss")
    loss.backward()
    torch.cuda.synchronize()
    gradients = {}
    for name in ("dec.indi_embedding.head.0.weight", "dec.out_norm.weight"):
        grad = dict(model.named_parameters())[name].grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
            raise RuntimeError(f"Missing or non-finite native gradient: {name}")
        gradients[name] = float(grad.norm())
    print(json.dumps({
        "module": args.module,
        "indices": args.indices,
        "loss": float(loss.detach()),
        "gradient_norms": gradients,
        "forward_backward_seconds": time.perf_counter() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }))


if __name__ == "__main__":
    main()
