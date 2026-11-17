"""Paired augmented native loss and physical-package control checks."""

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import gorilla
import torch

from diagnose_adares_y14_reliability import scene_indices
from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_y15_query_completion import IPDN
from ipdn.model_paper5.adares_y15_query_completion_gt import IPDN as GT
from ipdn.utils import get_root_logger
from train import load_shape_compatible_pretrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gt-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", required=True, type=int)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.config)
    gt_cfg = gorilla.Config.fromfile(args.gt_config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    selected = scene_indices(dataset.scene_inputs, 4)
    batches = [dataset.collate_fn([dataset[i] for i in selected[j:j+2]])
               for j in (0, 2)]
    model = IPDN(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    initial = {key: value.detach().cpu().clone()
               for key, value in model.state_dict().items()}
    results = []
    for index, batch in enumerate(batches):
        model.load_state_dict(initial)
        model.train()
        model.zero_grad(set_to_none=True)
        criterion_args = []
        criterion_kwargs = []
        original_forward = model.criterion.forward

        def captured_forward(*inputs, **kwargs):
            criterion_args.append(inputs)
            criterion_kwargs.append(kwargs)
            return original_forward(*inputs, **kwargs)

        model.criterion.forward = captured_forward
        loss, teacher_values = model(batch, mode="loss")
        model.criterion.forward = original_forward
        inputs, kwargs = criterion_args[-1], criterion_kwargs[-1]
        model.criterion.completion_target = "gt"
        gt_loss, gt_values = model.criterion(*inputs)
        model.criterion.completion_target = "teacher"
        names, params = zip(*[(name, param) for name, param in model.named_parameters()
                             if name.startswith("dec.completion_query.")])
        teacher_grads = torch.autograd.grad(loss, params, retain_graph=True)
        gt_grads = torch.autograd.grad(gt_loss, params, retain_graph=True)
        tvec = torch.cat([g.flatten() for g in teacher_grads])
        gvec = torch.cat([g.flatten() for g in gt_grads])
        mass = inputs[0]["point_mass"]
        pads = inputs[0]["batch_mask"]
        masks = kwargs["completion_targets"]
        differences = []
        for expr, target in enumerate(masks):
            gt = batch["gt_spmasks"][expr].to(target.device)
            n = mass[expr, ~pads[expr]]
            differences.append({
                "expression": expr,
                "point_mass_disagreement": float(
                    ((target.gt(.5) != gt.gt(.5)).float()*n).sum()/n.sum()),
            })
        non_bce_diffs = {key: float((value-gt_values[key]).abs().max())
                         for key, value in teacher_values.items()
                         if key != "loss" and "mask_bce_loss" not in key}
        auxiliary_bce_diffs = {key: float((value-gt_values[key]).abs().max())
                              for key, value in teacher_values.items()
                              if key.startswith("layer_") and "mask_bce" in key}
        dense_inputs = []
        handle = model.dec.register_forward_pre_hook(
            lambda module, values: dense_inputs.append(values))
        del loss, gt_loss, inputs
        criterion_args.clear()
        criterion_kwargs.clear()
        model.zero_grad(set_to_none=True)
        # Warm current provider and record the actual online forward/backward.
        gorilla.set_random_seed(1999)
        warm_loss, _ = model(batch, mode="loss")
        warm_loss.backward()
        del warm_loss
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        actual_loss, _ = model(batch, mode="loss")
        actual_loss.backward()
        torch.cuda.synchronize()
        resource = {
            "online_teacher_seconds": time.monotonic()-start,
            "online_teacher_peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "online_teacher_peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
        }
        handle.remove()
        with torch.no_grad():
            model.dec.eval()
            values = dense_inputs[-1]
            model.dec.completion_query.mode = "teacher"
            correct = model.dec(*values)
            model.dec.completion_query.mode = "language_shuffled"
            shuffled = model.dec(*values)
            model.dec.completion_query.mode = "slot_only"
            capacity = model.dec(*values)
            mechanism = {
                "language_extra_mask_max_diff": float((
                    correct["masks"][:, -1]-shuffled["masks"][:, -1]).abs().max()),
                "slot_extra_mask_max_diff": float((
                    correct["masks"][:, -1]-capacity["masks"][:, -1]).abs().max()),
                "native_mask_language_max_diff": float((
                    correct["masks"][:, :-1]-shuffled["masks"][:, :-1]).abs().max()),
            }
            model.dec.completion_query.mode = "teacher"
        result = {
            "batch": index, "scans": batch["scan_ids"],
            "teacher_vs_gt_gradient_cosine": float(
                (tvec @ gvec)/(tvec.norm()*gvec.norm())),
            "teacher_gt_gradient_max_diff": float((tvec-gvec).abs().max()),
            "all_new_teacher_gradients_finite_nonzero": all(
                torch.isfinite(g).all() and g.norm() > 0 for g in teacher_grads),
            "all_new_gt_gradients_finite_nonzero": all(
                torch.isfinite(g).all() and g.norm() > 0 for g in gt_grads),
            "non_bce_loss_max_differences": non_bce_diffs,
            "auxiliary_bce_max_differences": auxiliary_bce_diffs,
            "teacher_target_differences": differences,
            "mechanism": mechanism, "resources": resource,
        }
        results.append(result)
        del actual_loss, dense_inputs, correct, shuffled, capacity
        model.zero_grad(set_to_none=True)
        print(json.dumps(result), flush=True)
    # Separate physical package must produce finite native gradients with same init.
    gt_model = GT(**gt_cfg.model).cuda()
    incompatible = gt_model.load_state_dict(initial, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(str(incompatible))
    gt_model.train()
    gt_loss, _ = gt_model(batches[0], mode="loss")
    gt_loss.backward()
    gt_gradients = {name: bool(param.grad is not None
        and torch.isfinite(param.grad).all() and param.grad.norm() > 0)
        for name, param in gt_model.named_parameters()
        if name.startswith("dec.completion_query.")}
    output = {
        "scope": "matched same-output criterion, current augmented batches",
        "results": results, "physical_gt_package_native_gradients": gt_gradients,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True}), flush=True)


if __name__ == "__main__":
    main()
