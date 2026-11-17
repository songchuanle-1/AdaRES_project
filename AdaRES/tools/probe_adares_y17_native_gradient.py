"""Paired-description JS reachability and native gradient direction."""

import argparse
import gc
import json
from pathlib import Path

import gorilla
import torch

from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_w2_pointmass_competition import IPDN
from ipdn.utils import get_root_logger
from train import load_shape_compatible_pretrain


def js(p, q, mass):
    p, q = p.clamp(1e-6, 1-1e-6), q.clamp(1e-6, 1-1e-6)
    m = (p+q)/2
    def kl(x):
        return x*(x/m).log() + (1-x)*((1-x)/(1-m)).log()
    return (((kl(p)+kl(q))/2)*mass).sum()/mass.sum()


def gradients(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                      for p, g in zip(params, grads)])


def compare(a, b):
    return {"cosine": float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30)),
            "difference_norm": float((a-b).norm()),
            "left_norm": float(a.norm()), "right_norm": float(b.norm()),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


def identity(row):
    return (str(row["scene_id"]),
            tuple(int(v) for v in row.get("object_ids", [row.get("object_id")])),
            str(row["ann_id"]))


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "manifest", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    annotations = {identity(r): r for chunk in dataset.scene_inputs for r in chunk}
    manifest = json.loads(Path(args.manifest).read_text())["scans"][:8]
    groups = [[annotations[(str(r["scan_id"]),
                            tuple(int(v) for v in r["object_ids"]),
                            str(r["ann_id"]))] for r in group] for group in manifest]
    dataset.scene_inputs = groups
    model = IPDN(**cfg.model).cuda()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        incompatible = model.load_state_dict(checkpoint.get("model", checkpoint),
                                             strict=False)
        allowed = {"text_encoder.embeddings.position_ids",
                   "text_encoder.embeddings.token_type_ids"}
        if incompatible.missing_keys or set(incompatible.unexpected_keys)-allowed:
            raise ValueError(str(incompatible))
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    result = {"scope": "native augmented paired partial gradients",
              "initialization": args.checkpoint or cfg.train.pretrain,
              "production_loss_changed": False, "rows": [], "complete": False}
    for start in range(0, 8, 2):
        gorilla.set_random_seed(1999+start//2)
        batch = dataset.collate_fn([dataset[start], dataset[start+1]])
        captured = []
        hook = model.dec.register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        loss, values = model(batch, mode="loss")
        hook.remove()
        out = captured[0]
        if len(out["masks"]) != 8:
            raise ValueError("Expected native2scenes*4expressions")
        same, wrong, self_control = [], [], []
        for scene in range(2):
            offset = scene*4
            object_ids = batch["object_idss"][offset:offset+4]
            if object_ids[0] != object_ids[1] or object_ids[2] != object_ids[3]:
                raise ValueError("Same-target native annotation identity mismatch")
            if object_ids[0] == object_ids[2]:
                raise ValueError("Wrong-target control not distinct")
            valid = ~out["batch_mask"][offset]
            if not all(torch.equal(valid, ~out["batch_mask"][j])
                       for j in range(offset, offset+4)):
                raise ValueError("Padding mismatch")
            counts = torch.bincount(
                batch["superpoints"].long(),
                minlength=int(batch["batch_offsets"][-1])).to(loss.device).float()
            mass = counts[int(batch["batch_offsets"][scene]):
                          int(batch["batch_offsets"][scene+1])]
            if mass.numel() != int(valid.sum()):
                raise ValueError("Superpoint point-mass mismatch")
            predictions = []
            for expr in range(offset, offset+4):
                margin = out["indis"][expr, :, 1]-out["indis"][expr, :, 0]
                probability = out["masks"][expr, :, valid].sigmoid()
                predictions.append((margin.softmax(0)[:, None]*probability).sum(0))
            same.extend([js(predictions[0], predictions[1], mass),
                         js(predictions[2], predictions[3], mass)])
            wrong.extend([js(predictions[0], predictions[2], mass),
                          js(predictions[1], predictions[3], mass)])
            self_control.extend(js(p, p, mass) for p in predictions)
        native_indicator = sum(value for key, value in values.items()
                               if "indi" in key and torch.is_tensor(value))
        objectives = {"native": loss, "same": torch.stack(same).mean(),
                      "wrong": torch.stack(wrong).mean()}
        if torch.is_tensor(native_indicator) and native_indicator.requires_grad:
            objectives["native_indicator"] = native_indicator
        row = {"scan_ids": batch["scan_ids"],
               "exact_self_js": [float(v) for v in self_control],
               "object_idss": batch["object_idss"],
               "losses": {k: float(v) for k, v in objectives.items()},
               "native_loss_keys": list(values),
               "parameter_groups": {}}
        for prefix in ("dec.indi_embedding.", "dec.input_proj."):
            params = [p for n, p in model.named_parameters()
                      if n.startswith(prefix) and p.requires_grad]
            grads = {k: gradients(v, params) for k, v in objectives.items()}
            row["parameter_groups"][prefix] = {
                "tensor_count": len(params),
                "norms": {k: float(v.norm()) for k, v in grads.items()},
                "same_vs_native": compare(grads["same"], grads["native"]),
                "same_vs_wrong": compare(grads["same"], grads["wrong"])}
            if "native_indicator" in grads:
                row["parameter_groups"][prefix]["same_vs_indicator"] = compare(
                    grads["same"], grads["native_indicator"])
        result["rows"].append(row)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(row), flush=True)
        del loss, values, out, captured, objectives, grads, same, wrong, batch
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    result["complete"] = True
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
