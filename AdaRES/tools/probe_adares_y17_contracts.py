"""Matched same-graph controls and bounded Y17 native deployment contracts."""

import argparse
import copy
import gc
import importlib
import json
import time
from pathlib import Path

import gorilla
import torch

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_y17_target_equivalent_pairing import IPDN, build_train_dataset
from ipdn.model_paper5.adares_y17_target_equivalent_pairing.paired_objective import target_pair_js
from ipdn.utils import get_root_logger
from train import load_shape_compatible_pretrain


def dump(path, result):
    Path(path).write_text(json.dumps(result, indent=2))


def grad(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                      for p, g in zip(params, gs)])


def compare(a, b):
    return {"cosine": float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30)),
            "difference_norm": float((a-b).norm()),
            "left_norm": float(a.norm()), "right_norm": float(b.norm())}


def configure(args):
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    return gorilla.Config.fromfile(args.config)


def controls(args, cfg):
    dataset = build_train_dataset(cfg.data.train, get_root_logger())
    annotations = {(str(r["scene_id"]),
                    tuple(int(v) for v in r.get("object_ids", [r.get("object_id")])),
                    str(r["ann_id"])): r
                   for chunk in dataset.scene_inputs for r in chunk}
    manifest = json.loads(Path(args.manifest).read_text())["scans"][:8]
    dataset.scene_inputs = [[annotations[(str(r["scan_id"]),
                                         tuple(int(v) for v in r["object_ids"]),
                                         str(r["ann_id"]))] for r in group]
                            for group in manifest]
    model = IPDN(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    result = {"scope": "same prediction tensors, only target pairing changes",
              "rows": [], "complete": False}
    for start in range(0, 8, 2):
        gorilla.set_random_seed(1999+start//2)
        batch = dataset.collate_fn([dataset[start], dataset[start+1]])
        captured = []
        hook = model.dec.register_forward_hook(
            lambda module, inputs, out: captured.append((inputs, out)))
        loss, values = model(batch, mode="loss")
        hook.remove()
        inputs, out = captured[0]
        # DEC receives expanded point mass and expression-specific offsets.
        padded, _ = model.dec.get_batches(inputs[-1].unsqueeze(-1), inputs[2])
        mass = padded.squeeze(-1)
        direct, n = target_pair_js(out, batch["target_pairs"],
                                  batch["object_idss"], batch["scenes_len"], mass)
        off, _ = target_pair_js(out, batch["target_pairs"],
                               batch["object_idss"], batch["scenes_len"], mass, False)
        wrong_pairs = [[(s*4, s*4+2), (s*4+1, s*4+3)] for s in range(2)]
        # Wrong pairs are permitted solely in this explicit diagnostic label control.
        control_ids = [[0] for _ in range(8)]
        wrong, _ = target_pair_js(out, wrong_pairs, control_ids,
                                 batch["scenes_len"], mass)
        params = [p for name, p in model.named_parameters()
                  if name.startswith("dec.indi_embedding.") or
                  name.startswith("dec.input_proj.")]
        native = values["native_loss"]
        same_g, wrong_g, native_g = [
            grad(v, params) for v in (direct, wrong, native)]
        assert torch.equal(direct, values["pair_js_loss"])
        assert torch.equal(native+direct, loss) and float(off) == 0
        assert all(torch.isfinite(g).all() for g in (same_g, wrong_g, native_g))
        result["rows"].append({
            "scans": batch["scan_ids"], "pair_count": n,
            "native": float(native), "same": float(direct), "wrong": float(wrong),
            "off": float(off), "total_native_plus_js_exact": True,
            "same_vs_wrong": compare(same_g, wrong_g),
            "same_vs_native": compare(same_g, native_g)})
        dump(args.output, result)
        print(json.dumps(result["rows"][-1]), flush=True)
        del captured, inputs, out, loss, values, same_g, wrong_g, native_g
        del direct, off, wrong, native
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    result["complete"] = True
    dump(args.output, result)


def stress(args, cfg):
    module = importlib.import_module(args.module)
    builder = getattr(module, "build_train_dataset", build_dataset)
    dataset = builder(cfg.data.train, get_root_logger())
    stress_manifest = json.loads(Path(args.manifest).read_text())
    selected = stress_manifest["selected"]
    indices = []
    for row in selected:
        index = next(i for i, chunk in enumerate(dataset.scene_inputs)
                     if str(chunk[0]["scene_id"]) == row["scan_id"])
        indices.append(index)
    model = module.IPDN(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
    result = {"scope": "isolated process stress; native batch2, paired sampler",
              "module": args.module, "rows": [], "complete": False}
    for start in range(0, len(indices), 2):
        gorilla.set_random_seed(1999+start)
        batch = dataset.collate_fn([dataset[i] for i in indices[start:start+2]])
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, values = model(batch, mode="loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            del loss, values
        measurements = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            loss, values = model(batch, mode="loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            torch.cuda.synchronize()
            assert torch.isfinite(loss)
            measurements.append({
                "seconds": time.monotonic()-begin,
                "reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                "allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                "pair_js": float(values.get("pair_js_loss", loss*0)),
                "pair_count": getattr(model, "last_pair_count", None)})
            del loss, values
        result["rows"].append({
            "scans": batch["scan_ids"], "actual_points": len(batch["superpoints"]),
            "actual_superpoints": int(batch["batch_offsets"][-1]),
            "expressions": len(batch["ann_ids"]), "measurements": measurements})
        dump(args.output, result)
        print(json.dumps(result["rows"][-1]), flush=True)
        del batch
    result["parameters"] = sum(p.numel() for p in model.parameters())
    result["complete"] = True
    dump(args.output, result)


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "manifest", "output", "mode"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--module", default="ipdn.model_paper5.adares_y17_target_equivalent_pairing")
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    cfg = configure(args)
    {"controls": controls, "stress": stress}[args.mode](args, cfg)


if __name__ == "__main__":
    main()
