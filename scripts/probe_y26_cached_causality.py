"""Physical Y26 cached native loss controls separate centering/scale/affine."""

import hashlib
from pathlib import Path
import statistics
import types
import gc

import gorilla
import torch
import torch.nn.functional as F
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain
from probe_adares_y15_resume import rng, restore

torch.set_num_threads(4)
config = "configs_song/model_paper5/adares_y26_qla_postnorm_res.yaml"
cfg = gorilla.Config.fromfile(config)
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
indices, seen = [], set()
for index, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]["scene_id"] not in seen:
        seen.add(chunk[0]["scene_id"])
        indices.append(index)
    if len(indices) == 8:
        break
mode, captured = "native", {}


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(clone(v) for v in value)
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    return value


def before_dec(module, args):
    captured["dec"] = clone(args)


def before_criterion(module, args):
    captured["criterion"] = clone(args[1:])


def norm_forward(self, value):
    weight = self.weight.detach() if mode == "detached_affine" else self.weight
    bias = self.bias.detach() if mode == "detached_affine" else self.bias
    return F.layer_norm(value, self.normalized_shape, weight, bias, self.eps)


def after_qla(module, args, output):
    value = output[0]
    if mode == "rms":
        return (value / (value.square().mean(-1, keepdim=True)+module.norm.eps).sqrt(), *output[1:])
    if mode == "center":
        return (value-value.mean(-1, keepdim=True), *output[1:])


rows = []
for phase in ("trained", "released_init"):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    if phase == "trained":
        ck = torch.load("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth", map_location="cpu")
        loaded = model.load_state_dict(ck["model"], strict=False)
        assert not loaded.missing_keys and not set(loaded.unexpected_keys)-{
            "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
        del ck
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    for layer in model.dec.qla_layers:
        layer.norm.forward = types.MethodType(norm_forward, layer.norm)
    qla_hooks = [layer.register_forward_hook(after_qla) for layer in model.dec.qla_layers]
    for index in indices[:8 if phase == "trained" else 4]:
        mode = "native"
        for layer in model.dec.qla_layers:
            layer.postnorm = False
        captured.clear()
        hooks = [model.dec.register_forward_pre_hook(before_dec),
                 model.criterion.register_forward_pre_hook(before_criterion)]
        batch = dataset.collate_fn([dataset[index]])
        with torch.no_grad():
            model(batch, mode="loss")
        for hook in hooks:
            hook.remove()
        assert set(captured) == {"dec", "criterion"}
        state = rng()
        buffers = {k: v.detach().clone() for k, v in model.dec.named_buffers()}
        readings, saved = {}, {}
        for mode in ("native", "connected", "rms", "center", "detached_affine", "connected_repeat"):
            restore(state)
            with torch.no_grad():
                for name, value in model.dec.named_buffers():
                    value.copy_(buffers[name])
            for layer in model.dec.qla_layers:
                layer.postnorm = mode in ("connected", "detached_affine", "connected_repeat")
            model.zero_grad(set_to_none=True)
            pred = model.dec(*captured["dec"])
            loss, terms = model.criterion(pred, *captured["criterion"])
            loss.backward()
            gradients = {k: p.grad.detach().cpu().clone() for k, p in model.dec.named_parameters()
                         if p.grad is not None}
            assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients.values())
            readings[mode] = dict(loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                                  qla_norm_grads={k: float(g.norm()) for k, g in gradients.items()
                                                  if k.startswith("qla_layers.") and ".norm." in k})
            saved[mode] = (pred["masks"].detach().cpu().clone(), gradients)
            del pred, loss, terms
        shared = set(saved["connected"][1]) & set(saved["detached_affine"][1])
        row = dict(phase=phase, scan_id=str(batch["scan_ids"][0]), dataset_index=index,
                   readings=readings,
                   detached_mask_max=float((saved["connected"][0]-saved["detached_affine"][0]).abs().max()),
                   repeat_mask_max=float((saved["connected"][0]-saved["connected_repeat"][0]).abs().max()),
                   shared_detached_gradient_max=max(float((saved["connected"][1][k]-saved["detached_affine"][1][k]).abs().max()) for k in shared),
                   shared_repeat_gradient_max=max(float((saved["connected"][1][k]-saved["connected_repeat"][1][k]).abs().max()) for k in saved["connected"][1]))
        assert not readings["native"]["qla_norm_grads"] and not readings["detached_affine"]["qla_norm_grads"]
        assert len(readings["connected"]["qla_norm_grads"]) == 12
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "readings"}), flush=True)
        captured.clear()
        del saved, gradients, buffers, batch
    for hook in qla_hooks:
        hook.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
summary = {}
for phase in ("trained", "released_init"):
    group = [r for r in rows if r["phase"] == phase]
    summary[phase] = {m: statistics.mean(r["readings"][m]["loss"]-r["readings"]["native"]["loss"] for r in group)
                      for m in ("connected", "rms", "center")}
result = dict(complete=True, id="AdaRES-Y26", rows=rows, loss_deltas=summary,
              detached_mask_max=max(r["detached_mask_max"] for r in rows),
              repeat_mask_max=max(r["repeat_mask_max"] for r in rows),
              shared_detached_gradient_max=max(r["shared_detached_gradient_max"] for r in rows),
              shared_repeat_gradient_max=max(r["shared_repeat_gradient_max"] for r in rows),
              minimum_connected_affine_gradient=min(g for r in rows for g in r["readings"]["connected"]["qla_norm_grads"].values()),
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Physical candidate native decoder/criterion with frozen actual encoded inputs.",
                           "Cached controls exclude sparse-backbone/FPS repeat variation, not global determinism.",
                           "Center-only and RMS are causal probes, not additional variants or validation tuning.",
                           "Initial total loss change retained; no trained benefit or calibration superiority claim."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
