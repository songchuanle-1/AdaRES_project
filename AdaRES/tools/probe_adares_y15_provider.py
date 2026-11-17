"""Current augmented batch teacher, released init and dense inference checks."""

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

import gorilla
import torch

from diagnose_adares_y14_reliability import scene_indices, weighted_iou
from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_y15_query_completion import IPDN
from ipdn.model_paper5.adares_y15_query_completion_gt import IPDN as GT
from ipdn.utils import get_root_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gt-config", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = gorilla.Config.fromfile(args.config)
    gt_cfg = gorilla.Config.fromfile(args.gt_config)
    trainer = importlib.import_module("train")
    dataset = build_dataset(cfg.data.train, get_root_logger())
    selected = scene_indices(dataset.scene_inputs, 4)
    batches = [dataset.collate_fn([dataset[i] for i in selected[j:j+2]])
               for j in (0, 2)]
    model = IPDN(**cfg.model).cuda()
    trainer.load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    native = []
    capture = []
    handle = model.dec.register_forward_hook(
        lambda module, inputs, output: capture.append((inputs, output)))
    names = [name for name, _ in model.named_parameters()
             if name.startswith("dec.completion_query.")]
    for index, batch in enumerate(batches):
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        loss, values = model(batch, mode="loss")
        loss.backward()
        torch.cuda.synchronize()
        grads = {name: {
            "norm": float(param.grad.norm()) if param.grad is not None else None,
            "finite_nonzero": bool(param.grad is not None
                and torch.isfinite(param.grad).all() and param.grad.norm() > 0),
        } for name, param in model.named_parameters() if name in names}
        native.append({
            "batch": index, "scans": batch["scan_ids"], "loss": float(loss),
            "gradients": grads, "teacher_targets": model.last_teacher_targets,
            "teacher_resident": model._training_teacher is not None,
            "teacher_excluded_from_state": not any(
                "training_teacher" in key for key in model.state_dict()),
            "query_count": capture[-1][1]["masks"].shape[1],
            "seconds": time.monotonic()-start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
        })
        capture.clear()
        del loss, values
        print(json.dumps(native[-1]), flush=True)
    handle.remove()
    # Teacher model is already loaded; check actual native optimizer steps.
    cfg.dist = False
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
    before = {name: param.detach().clone()
              for name, param in model.named_parameters() if name in names}

    class Batches:
        sampler = None

        def __iter__(self):
            return iter(batches)

        def __len__(self):
            return len(batches)

    class Writer:
        def add_scalar(self, *args, **kwargs):
            pass

    trainer.train(1, model, Batches(), optimizer, scheduler,
                  cfg, get_root_logger(), Writer())
    updates = {name: float((param-before[name]).abs().max())
               for name, param in model.named_parameters() if name in names}
    # Cache one complete set of decoder inputs; no sparse ops in these checks.
    model.eval()
    dc = gt_cfg.data.train.copy()
    dc.training, dc.aug, dc.with_elastic, dc.lang_num_max = False, False, False, 4
    clean = build_dataset(dc, get_root_logger())
    first = scene_indices(clean.scene_inputs, 1)[0]
    clean.scene_inputs[first] = clean.scene_inputs[first][:2]
    one = clean.collate_fn([clean[first]])
    captured = []
    handle = model.dec.register_forward_hook(
        lambda module, inputs, output: captured.append(inputs))
    with torch.no_grad():
        model(one, mode="predict")
    handle.remove()
    inputs = captured[-1]
    dense_checks = []
    with torch.no_grad():
        normal = model.dec(*inputs)
        repeated = model.dec(*inputs)
        for expression in range(normal["masks"].shape[0]):
            count = int((~normal["batch_mask"][expression]).sum())
            mass = normal["point_mass"][expression, :count].cpu()
            masks = normal["masks"][expression, :, :count].gt(0).cpu()
            overlap = weighted_iou(masks[-1:], masks[:-1], mass)
            dense_checks.append({
                "expression": expression,
                "repeat_mask_max_diff": float((normal["masks"][expression] -
                                              repeated["masks"][expression]).abs().max()),
                "repeat_indicator_max_diff": float((normal["indis"][expression] -
                                                   repeated["indis"][expression]).abs().max()),
                "extra_max_duplicate_iou": float(overlap.max()),
                "selected_query": int(normal["indis"][expression].softmax(-1)[:, 1].argmax()),
            })
        label_args = (
            one["scan_ids"], one["object_idss"], one["ann_ids"], normal,
            one["superpoints"].cuda(), [t.cuda() for t in one["gt_pmasks"]],
            [t.cuda() for t in one["gt_spmasks"]], inputs[1],
        )
        original_prediction = model.predict_by_feat(*label_args)
        changed = list(label_args)
        changed[5] = [torch.zeros_like(t) for t in changed[5]]
        changed[6] = [torch.zeros_like(t) for t in changed[6]]
        changed_prediction = model.predict_by_feat(*changed)
        exact_label_invariance = all(torch.equal(a, b) for a, b in zip(
            original_prediction["pred_pmask"], changed_prediction["pred_pmask"]))
    # A teacher-naive instance must stay teacher-free for student inference.
    fresh = IPDN(**cfg.model).cuda()
    fresh.eval()
    with torch.no_grad():
        fresh(one, mode="predict")
    teacher_free = fresh._training_teacher is None
    result = {
        "scope": "released SPFormer init, current augmented teacher integration only",
        "native": native, "new_tensor_optimizer_updates": updates,
        "dense_fixed_input_checks": dense_checks,
        "fixed_decoder_output_label_invariance": exact_label_invariance,
        "fresh_inference_does_not_construct_teacher": teacher_free,
        "teacher_checkpoint_sha256": hashlib.sha256(
            Path(cfg.model.teacher_checkpoint).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "output": args.output}), flush=True)


if __name__ == "__main__":
    main()
