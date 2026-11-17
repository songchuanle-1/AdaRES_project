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
    parser.add_argument("--model", choices=("adares_b4_query_competition", "adares_g1_metric_targets"), required=True)
    parser.add_argument("--indices", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--checkpoint")
    parser.add_argument("--detach-mask-evidence", action="store_true")
    parser.add_argument("--one-step", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    model_type = importlib.import_module(f"ipdn.model_paper5.{args.model}").IPDN
    model = model_type(**cfg.model).cuda()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        loaded = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        allowed = {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
        if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
            raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.train()
    if args.detach_mask_evidence:
        indicator_module = model.dec.indi_embedding
        original_forward = indicator_module.forward

        def detached_forward(query, pred_masks, batch_mask):
            return original_forward(query, pred_masks.detach(), batch_mask)

        indicator_module.forward = detached_forward
    batch = dataset.collate_fn([dataset[i] for i in args.indices])
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    loss, parts = model(batch, mode="loss")
    params = dict(model.named_parameters())
    parameter = params["dec.x_mask.0.weight"]
    indicator = parts["indi_loss"]
    mask = parts["mask_bce_loss"] + parts["mask_dice_loss"]
    grad_indicator = torch.autograd.grad(indicator, parameter, retain_graph=True, allow_unused=True)[0]
    grad_mask = torch.autograd.grad(mask, parameter, retain_graph=True, allow_unused=True)[0]
    if grad_mask is None or (grad_indicator is None and not args.detach_mask_evidence):
        raise RuntimeError("Unexpected missing gradient")
    if grad_indicator is None:
        grad_indicator = torch.zeros_like(grad_mask)
    if not torch.isfinite(grad_indicator).all() or not torch.isfinite(grad_mask).all():
        raise RuntimeError("Non-finite gradient")
    indicator_norm = grad_indicator.norm()
    mask_norm = grad_mask.norm()
    if mask_norm == 0 or (indicator_norm == 0 and not args.detach_mask_evidence):
        raise RuntimeError("Zero gradient")
    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "detach_mask_evidence": args.detach_mask_evidence,
        "indices": args.indices,
        "total_loss": float(loss.detach()),
        "indicator_loss": float(indicator.detach()),
        "mask_loss": float(mask.detach()),
        "indicator_gradient_norm": float(indicator_norm),
        "mask_gradient_norm": float(mask_norm),
        "gradient_cosine": (
            float((grad_indicator * grad_mask).sum() / (indicator_norm * mask_norm))
            if indicator_norm > 0 else None
        ),
        "opposed_element_fraction": float(((grad_indicator * grad_mask) < 0).float().mean()),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    if args.one_step:
        before = float(mask.detach())
        loss.backward()
        with torch.no_grad():
            parameter.add_(parameter.grad, alpha=-float(cfg.optimizer.lr))
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        _, after_parts = model(batch, mode="loss")
        after = float((after_parts["mask_bce_loss"] + after_parts["mask_dice_loss"]).detach())
        result["one_step_mask_before"] = before
        result["one_step_mask_after"] = after
        result["one_step_mask_delta"] = after - before
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
