"""Train-only native integration and matched controls; not full training."""

import argparse
import copy
import gc
import hashlib
import importlib
import json
import statistics
import time
from pathlib import Path

import gorilla
import torch

from diagnose_adares_y14_reliability import load_model, scene_indices
from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_b4_query_competition import IPDN as B4
from ipdn.model_paper5.adares_w2_pointmass_competition import IPDN as W2
from ipdn.model_paper5.adares_y15_query_completion import IPDN as Y15
from ipdn.utils import get_root_logger


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reset():
    gorilla.set_random_seed(1999)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def cpu_output(out):
    return {key: out[key].detach().cpu() for key in (
        "masks", "indis", "sample_inds")}


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "w2-config", "b4-config", "w2-checkpoint",
                 "b4-checkpoint", "diagnosis", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reset()
    cfg = gorilla.Config.fromfile(args.config)
    dc = cfg.data.train.copy()
    dc.training, dc.aug, dc.with_elastic = False, False, False
    dc.lang_num_max = 32
    dataset = build_dataset(dc, get_root_logger())
    diagnosis = json.loads(Path(args.diagnosis).read_text())
    material = [r for r in diagnosis["rows"] if r["material"]]
    scans = sorted({r["scan_id"] for r in material})[:4]
    selected = []
    material_keys = {(r["scan_id"], tuple(r["object_ids"]), str(r["ann_id"]))
                     for r in material}
    for scan in scans:
        index = next(i for i, group in enumerate(dataset.scene_inputs)
                     if group[0]["scene_id"] == scan)
        original = dataset.scene_inputs[index]
        keep = next(row for row in original if (
            scan, (int(row["object_id"]),), str(row["ann_id"])) in material_keys)
        other = next(row for row in original
                     if row["object_id"] != keep["object_id"])
        dataset.scene_inputs[index] = [keep, other]
        selected.append(index)
    if len(selected) != 4:
        raise ValueError("Need four independent real training scans")
    batches = [dataset.collate_fn([dataset[i] for i in selected[j:j+2]])
               for j in (0, 2)]
    teacher = load_model(B4, gorilla.Config.fromfile(args.b4_config),
                         args.b4_checkpoint)
    targets, teacher_metadata = [], []
    with torch.no_grad():
        for index in selected:
            batch = dataset.collate_fn([dataset[index]])
            rows = teacher(batch, mode="predict")["rows"]
            for row in rows:
                key = (row["scan_id"], tuple(row["object_ids"]), row["ann_id"])
                target = row["binary"][int(row["ious"].argmax())].float()
                if key not in material_keys:
                    target = batch["gt_spmasks"][len(targets) % 2].float().cpu()
                targets.append(target)
                teacher_metadata.append({
                    "identity": key, "material": key in material_keys,
                    "target_sha256": hashlib.sha256(
                        target.numpy().tobytes()).hexdigest(),
                })
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    pretrained = torch.load(args.w2_checkpoint, map_location="cpu")
    shared = pretrained.get("model", pretrained)
    reset()
    candidate = Y15(**cfg.model).cuda()
    missing = candidate.load_state_dict(shared, strict=False)
    if any(not name.startswith("dec.completion_query.")
           for name in missing.missing_keys):
        raise ValueError(str(missing))
    initial = {key: value.detach().cpu().clone()
               for key, value in candidate.state_dict().items()}
    new_names = [name for name, _ in candidate.named_parameters()
                 if name.startswith("dec.completion_query.")]
    new_params = sum(p.numel() for name, p in candidate.named_parameters()
                     if name in new_names)
    rows = []
    outputs = {}
    for arm in ("gt", "teacher", "language_shuffled", "slot_only", "disabled"):
        arm_rows = []
        for batch_index, original in enumerate(batches):
            candidate.load_state_dict(initial)
            candidate.dec.completion_enabled = arm != "disabled"
            candidate.dec.completion_query.mode = "gt" if arm == "disabled" else arm
            candidate.criterion.completion_target = (
                "gt" if arm in {"gt", "disabled"} else "teacher")
            candidate.train()
            candidate.zero_grad(set_to_none=True)
            batch = dict(original)
            if candidate.criterion.completion_target == "teacher":
                batch["completion_targets"] = [
                    t.clone() for t in targets[batch_index*4:batch_index*4+4]]
            capture = []
            hook = candidate.dec.register_forward_hook(
                lambda module, inputs, output: capture.append(output))
            reset()
            start = time.monotonic()
            loss, values = candidate(batch, mode="loss")
            loss.backward()
            torch.cuda.synchronize()
            gradients = {}
            for name, param in candidate.named_parameters():
                if name in new_names:
                    grad = param.grad
                    gradients[name] = {
                        "norm": float(grad.norm()) if grad is not None else None,
                        "finite_nonzero": bool(grad is not None
                            and torch.isfinite(grad).all() and grad.norm() > 0),
                    }
            row = {
                "arm": arm, "batch": batch_index,
                "scans": original["scan_ids"], "loss": float(loss.detach()),
                "mask_bce": float(values["mask_bce_loss"].detach()),
                "gradients": gradients,
                "finite_loss": bool(torch.isfinite(loss).all()),
                "seconds": time.monotonic()-start,
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                "query_count": capture[-1]["masks"].shape[1],
                "seed_query_count": capture[-1]["sample_inds"].shape[1],
                "aux_query_counts": [o["masks"].shape[1]
                                     for o in capture[-1]["aux_outputs"]],
            }
            outputs[(arm, batch_index)] = cpu_output(capture[-1])
            arm_rows.append(row)
            hook.remove()
            capture.clear()
            del loss, values, batch
            print(json.dumps(row), flush=True)
        rows.extend(arm_rows)
    controls = []
    for batch_index in range(2):
        gt = outputs[("gt", batch_index)]
        control = {"batch": batch_index}
        for arm in ("teacher", "language_shuffled", "slot_only", "disabled"):
            current = outputs[(arm, batch_index)]
            control[arm] = {
                "native_mask_max_diff": float((
                    gt["masks"][:, :128] - current["masks"][:, :128]).abs().max()),
                "seed_identity": bool(torch.equal(
                    gt["sample_inds"], current["sample_inds"])),
            }
            if arm != "disabled":
                control[arm]["extra_mask_max_diff"] = float((
                    gt["masks"][:, -1]-current["masks"][:, -1]).abs().max())
        controls.append(control)
    # Verify fail-closed teacher supervision without supplied labels.
    candidate.dec.completion_enabled = True
    candidate.criterion.completion_target = "teacher"
    failed_closed = False
    try:
        candidate(batches[0], mode="loss")
    except ValueError as error:
        failed_closed = "requires detached training targets" in str(error)
    candidate.load_state_dict(initial)
    candidate.dec.completion_query.mode = "gt"
    candidate.criterion.completion_target = "gt"
    candidate.eval()
    inference = []
    with torch.no_grad():
        one = dataset.collate_fn([dataset[selected[0]]])
        pred = candidate(one, mode="predict")
        repeat = candidate(one, mode="predict")
        changed = dict(one)
        changed["gt_pmasks"] = [torch.zeros_like(t) for t in one["gt_pmasks"]]
        changed["gt_spmasks"] = [torch.zeros_like(t) for t in one["gt_spmasks"]]
        counterfactual = candidate(changed, mode="predict")
        label_blind = all(torch.equal(a, b) for a, b in zip(
            pred["pred_pmask"], counterfactual["pred_pmask"]))
        label_checks = []
        for a, b, c in zip(pred["pred_pmask"], repeat["pred_pmask"],
                           counterfactual["pred_pmask"]):
            label_checks.append({
                "repeat_probability_max_diff": float((a-b).abs().max()),
                "label_probability_max_diff": float((a-c).abs().max()),
                "repeat_binary_difference": int(((a > .5) != (b > .5)).sum()),
                "label_binary_difference": int(((a > .5) != (c > .5)).sum()),
            })
        for enabled in (False, True):
            candidate.dec.completion_enabled = enabled
            for index in selected[:2]:
                one = dataset.collate_fn([dataset[index]])
                for _ in range(3):
                    candidate(one, mode="predict")
                reset()
                times = []
                for _ in range(10):
                    start = time.monotonic()
                    candidate(one, mode="predict")
                    torch.cuda.synchronize()
                    times.append(time.monotonic()-start)
                inference.append({
                    "completion": enabled,
                    "scan": dataset.scene_inputs[index][0]["scene_id"],
                    "median_seconds": statistics.median(times),
                    "times": times,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                })
    # Exercise the actual tools/train.py train loop, not a hand-written optimizer.
    candidate.load_state_dict(initial)
    candidate.dec.completion_enabled = True
    candidate.criterion.completion_target = "gt"
    trainer = importlib.import_module("train")
    cfg.dist = False
    optimizer = gorilla.build_optimizer(candidate, cfg.optimizer)
    scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
    before = {name: param.detach().clone()
              for name, param in candidate.named_parameters() if name in new_names}

    class TwoBatches:
        sampler = None

        def __len__(self):
            return len(batches)

        def __iter__(self):
            return iter(batches)

    class NullWriter:
        def add_scalar(self, *args, **kwargs):
            pass

    reset()
    trainer.train(1, candidate, TwoBatches(), optimizer, scheduler,
                  cfg, get_root_logger(), NullWriter())
    optimizer_changes = {
        name: float((param.detach()-before[name]).abs().max())
        for name, param in candidate.named_parameters() if name in new_names
    }
    result = {
        "scope": "native real-batch integration, no full training or admission",
        "local_registration": "pipeline/designs/AdaRES-Y15_native_prereg_20260930.md",
        "scans": scans, "new_parameters": new_params, "rows": rows,
        "controls": controls, "teacher_targets": teacher_metadata,
        "teacher_missing_fails_closed": failed_closed,
        "student_inference_label_blind": label_blind,
        "label_repeat_checks": label_checks,
        "warmed_student_only_inference": inference,
        "tools_train_two_steps": {
            "scope": "bounded native integration, not full-model training",
            "steps": len(batches),
            "all_new_tensor_updates_nonzero": all(
                value > 0 for value in optimizer_changes.values()),
            "new_tensor_max_updates": optimizer_changes,
        },
        "source_sha256": digest(__file__),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "output": args.output}), flush=True)


if __name__ == "__main__":
    main()
