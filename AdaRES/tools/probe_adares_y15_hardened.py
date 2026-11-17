"""Hardened-provider fail-closed/RNG checks plus native two-step regression."""

import argparse
import copy
import json
import random
from pathlib import Path

import gorilla
import numpy as np
import torch

from diagnose_adares_y14_reliability import scene_indices
from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_y15_query_completion import IPDN
from ipdn.utils import get_root_logger
from train import load_shape_compatible_pretrain, train


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
    dataset = build_dataset(cfg.data.train, get_root_logger())
    indices = scene_indices(dataset.scene_inputs, 4)
    batches = [dataset.collate_fn([dataset[i] for i in indices[j:j+2]])
               for j in (0, 2)]
    model = IPDN(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    correct_hash = model.teacher_checkpoint_sha256
    model.teacher_checkpoint_sha256 = "0"*64
    wrong_hash_rejected = False
    try:
        model.training_teacher_masks(batches[0])
    except ValueError as exc:
        wrong_hash_rejected = "SHA256 mismatch" in str(exc)
    model.teacher_checkpoint_sha256 = correct_hash
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all())
    masks = model.training_teacher_masks(batches[0])
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all())
    rng_equal = [
        states[0] == after[0],
        states[1][0] == after[1][0]
        and np.array_equal(states[1][1], after[1][1])
        and states[1][2:] == after[1][2:],
        torch.equal(states[2], after[2]),
        all(torch.equal(a, b) for a, b in zip(states[3], after[3])),
    ]
    assert wrong_hash_rejected and all(rng_equal)
    assert all(not mask.requires_grad for mask in masks)
    assert model.train() is model and model.eval() is model
    names = [name for name, _ in model.named_parameters()
             if name.startswith("dec.completion_query.")]
    results = []
    for batch in batches:
        model.train()
        model.zero_grad(set_to_none=True)
        loss, values = model(batch, mode="loss")
        loss.backward()
        gradients = {
            name: bool(p.grad is not None and torch.isfinite(p.grad).all()
                       and p.grad.norm() > 0)
            for name, p in model.named_parameters() if name in names}
        assert len(gradients) == 9 and all(gradients.values())
        assert all("object_ids" in row for row in model.last_teacher_targets)
        results.append({"scans": batch["scan_ids"], "loss": float(loss),
                        "gradients": gradients,
                        "teacher_metadata": model.last_teacher_targets})
        del loss, values
    optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
    scheduler = gorilla.build_lr_scheduler(optimizer, copy.deepcopy(cfg.lr_scheduler))
    before = {name: p.detach().clone() for name, p in model.named_parameters()
              if name in names}
    cfg.dist = False

    class Batches:
        sampler = None

        def __iter__(self):
            return iter(batches)

        def __len__(self):
            return len(batches)

    class Writer:
        def add_scalar(self, *args, **kwargs):
            pass

    train(1, model, Batches(), optimizer, scheduler, cfg,
          get_root_logger(), Writer())
    updates = {name: float((p-before[name]).abs().max())
               for name, p in model.named_parameters() if name in names}
    assert all(value > 0 for value in updates.values())
    model.eval()
    with torch.no_grad():
        model(dataset.collate_fn([dataset[indices[0]]]), mode="predict")
    output = {"complete": True, "wrong_hash_rejected": wrong_hash_rejected,
              "rng_equal_python_numpy_torch_cuda": rng_equal,
              "native_batches": results, "native_optimizer_updates": updates,
              "teacher_checkpoint_sha256": correct_hash,
              "teacher_excluded_from_state": not any(
                  "teacher" in key for key in model.state_dict()),
              "scope": "hardened teacher, released-init native two-step regression"}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
