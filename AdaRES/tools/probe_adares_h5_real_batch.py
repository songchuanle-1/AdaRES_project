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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--module", default="ipdn.model_paper5.adares_h5_late_global_query")
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--indices", type=int, nargs="+")
    args = parser.parse_args()

    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    if args.package_dir:
        import ipdn.model_paper5
        ipdn.model_paper5.__path__.append(str(args.package_dir.resolve()))
    dataset = build_dataset(cfg.data.train, get_root_logger())
    indices = args.indices if args.indices is not None else list(range(args.batch_size))
    if len(indices) != args.batch_size or any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError("Scene indices must match batch size and be within the dataset")
    batch = dataset.collate_fn([dataset[i] for i in indices])
    model = importlib.import_module(args.module).IPDN(**cfg.model).cuda()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss, _ = model(batch, mode="loss")
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite real-batch loss: {loss}")
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    names = [
        "dec.indi_embedding.head.0.weight",
        "dec.out_norm.weight",
        "dec.global_query_attention.attn.in_proj_weight",
    ]
    params = dict(model.named_parameters())
    gradients = {}
    for name in names:
        grad = params[name].grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
            raise RuntimeError(f"Missing, non-finite or zero gradient: {name}")
        gradients[name] = float(grad.norm())
    print(json.dumps({
        "batch_size": args.batch_size,
        "indices": indices,
        "loss": float(loss.detach()),
        "gradient_norms": gradients,
        "forward_backward_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }))


if __name__ == "__main__":
    main()
