"""Physical Y17 sampler/control/optimizer native integration."""

import argparse
import copy
import gc
import json
from collections import Counter
from pathlib import Path

import gorilla
import torch

from ipdn.dataset import build_dataset, build_dataloader
from ipdn.model_paper5.adares_y17_target_equivalent_pairing import (
    IPDN, build_train_dataset)
from ipdn.model_paper5.adares_y17_target_equivalent_pairing.pairing import identity
from train import load_shape_compatible_pretrain, train


class Batches:
    sampler = None

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "manifest", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    from ipdn.utils import get_root_logger
    logger = get_root_logger()
    cfg = gorilla.Config.fromfile(args.config)
    paired = build_train_dataset(cfg.data.train, logger)
    parent = build_dataset(cfg.data.train, logger)
    before = Counter(identity(r) for chunk in parent.scene_inputs for r in chunk)
    after = Counter(identity(r) for chunk in paired.scene_inputs for r in chunk)
    assert before == after
    assert [len(c) for c in parent.scene_inputs] == [
        len(c) for c in paired.scene_inputs]
    annotations = {identity(r): r for chunk in parent.scene_inputs for r in chunk}
    manifest = json.loads(Path(args.manifest).read_text())["scans"][:8]
    groups = [[annotations[(str(r["scan_id"]),
                            tuple(int(v) for v in r["object_ids"]),
                            str(r["ann_id"]))] for r in group] for group in manifest]
    paired.scene_inputs = groups
    loader = build_dataloader(paired, batch_size=2, num_workers=2,
                              persistent_workers=False, training=True)
    worker_batch = next(iter(loader))
    assert len(worker_batch["target_pairs"]) == 2
    del loader, worker_batch
    batches = []
    for start in range(0, 8, 2):
        gorilla.set_random_seed(1999+start//2)
        batches.append(paired.collate_fn([paired[start], paired[start+1]]))
    model = IPDN(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    result = {"scope": "physical Y17 released-init native training",
              "annotation_preserved": before == after,
              "expressions": sum(before.values()), "chunks": len(parent.scene_inputs),
              "dataloader_workers": 2, "rows": [], "complete": False}
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for batch in batches:
        model.zero_grad(set_to_none=True)
        loss, values = model(batch, mode="loss")
        pair = values["pair_js_loss"]
        assert model.last_pair_count > 0 and pair > 0
        partial = [p for n, p in params if n.startswith("dec.indi_embedding.")
                   or n.startswith("dec.input_proj.")]
        grads = torch.autograd.grad(pair, partial, retain_graph=True)
        assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
        loss.backward()
        all_grads = {n: {"present": p.grad is not None,
                        "finite": bool(torch.isfinite(p.grad).all())
                        if p.grad is not None else None,
                        "norm": float(p.grad.norm()) if p.grad is not None else None}
                     for n, p in params}
        assert all(r["finite"] for r in all_grads.values() if r["present"])
        result["rows"].append({
            "scans": batch["scan_ids"], "pairs": batch["target_pairs"],
            "pair_count": model.last_pair_count,
            "native_loss": float(values["native_loss"]),
            "pair_js": float(pair), "total_loss": float(loss),
            "partial_gradient_norms": [float(g.norm()) for g in grads],
            "all_native_gradients": all_grads})
        del loss, values, pair, grads
        model.zero_grad(set_to_none=True)
    optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
    scheduler = gorilla.build_lr_scheduler(optimizer, copy.deepcopy(cfg.lr_scheduler))
    selected = [(n, p) for n, p in params if n.startswith("dec.indi_embedding.")
                or n.startswith("dec.input_proj.")]
    before_update = {n: p.detach().clone() for n, p in selected}
    cfg.dist = False
    train(1, model, Batches(batches[:2]), optimizer, scheduler, cfg, logger, Writer())
    result["optimizer_updates"] = {
        n: float((p-before_update[n]).abs().max()) for n, p in selected}
    assert all(v > 0 for v in result["optimizer_updates"].values())
    result["parameters"] = sum(p.numel() for p in model.parameters())
    result["peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
    result["complete"] = True
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
