"""Native full-loss controls for the existing qla affine gradient path."""

import gc
import hashlib
from pathlib import Path
import statistics

import gorilla
import torch
import torch.nn.functional as F
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain
from probe_adares_y15_resume import rng, restore

torch.set_num_threads(4)
assert Path(".codex_source_commit").read_text().strip() == "94918c1a008e21c72b9478c28f934e8b329e0c97"
source = Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py").read_bytes()
assert hashlib.sha256(source).hexdigest() == "6362d882662b0e51594cd4af994359b93f99497affc93600322a34ca6c349b01"
config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
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
mode, output_cache = "native", {}


def after_qla(module, args, output):
    if mode == "native":
        return None
    norm = module.norm
    if mode == "detached_affine":
        value = F.layer_norm(output[0], norm.normalized_shape,
                             norm.weight.detach(), norm.bias.detach(), norm.eps)
    else:
        value = norm(output[0])
    return (value, *output[1:])


def after_decoder(module, args, output):
    output_cache["masks"] = output["masks"].detach().cpu().clone()
    output_cache["indis"] = output["indis"].detach().cpu().clone()


rows = []
for phase in ("trained", "released_init"):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    if phase == "trained":
        checkpoint = "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth"
        ck = torch.load(checkpoint, map_location="cpu")
        loaded = model.load_state_dict(ck["model"], strict=False)
        assert not loaded.missing_keys and not set(loaded.unexpected_keys)-{
            "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
        del ck
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    hooks = [layer.register_forward_hook(after_qla) for layer in model.dec.qla_layers]
    hooks.append(model.dec.register_forward_hook(after_decoder))
    for index in indices[:8 if phase == "trained" else 4]:
        batch = dataset.collate_fn([dataset[index]])
        state = rng()
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        readings, saved = {}, {}
        for mode in ("native", "connected", "detached_affine", "connected_repeat"):
            restore(state)
            with torch.no_grad():
                for key, value in model.named_buffers():
                    value.copy_(buffers[key])
            model.zero_grad(set_to_none=True)
            output_cache.clear()
            loss, terms = model(batch, mode="loss")
            assert torch.isfinite(loss)
            loss.backward()
            gradients = {name: p.grad.detach().cpu().clone() for name, p in model.dec.named_parameters()
                         if p.grad is not None}
            assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
            norm_grads = {name: gradient for name, gradient in gradients.items()
                          if name.startswith("qla_layers.") and ".norm." in name}
            if mode in ("native", "detached_affine"):
                assert not norm_grads
            else:
                assert len(norm_grads) == 12 and all(g.abs().max() > 0 for g in norm_grads.values())
            readings[mode] = dict(loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                                  decoder_gradient_tensors=len(gradients),
                                  norm_gradients={name: float(g.norm()) for name, g in norm_grads.items()})
            saved[mode] = (dict(output_cache), gradients)
            del loss, terms
        forward_delta = lambda a, b: max(float((saved[a][0][k]-saved[b][0][k]).abs().max())
                                         for k in saved[a][0])
        shared = set(saved["connected"][1]) & set(saved["detached_affine"][1])
        row = dict(phase=phase, scan_id=str(batch["scan_ids"][0]), dataset_index=index,
                   expressions=len(batch["ann_ids"]), readings=readings,
                   connected_detached_forward_max=forward_delta("connected", "detached_affine"),
                   connected_repeat_forward_max=forward_delta("connected", "connected_repeat"),
                   connected_native_forward_max=forward_delta("connected", "native"),
                   shared_detached_gradient_max=max(float((saved["connected"][1][k]-saved["detached_affine"][1][k]).abs().max()) for k in shared),
                   shared_repeat_gradient_max=max(float((saved["connected"][1][k]-saved["connected_repeat"][1][k]).abs().max()) for k in saved["connected"][1]))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "readings"}), flush=True)
        del saved, gradients, norm_grads, batch, buffers
    parameters = sum(p.numel() for p in model.parameters())
    for hook in hooks:
        hook.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
result = dict(complete=True, id="AdaRES-Y26", trained_scans=8, released_init_scans=4, rows=rows,
              parameters=parameters, added_parameters=0, connected_norm_tensors=12,
              minimum_norm_gradient=min(g for r in rows for g in r["readings"]["connected"]["norm_gradients"].values()),
              connected_detached_forward_max=max(r["connected_detached_forward_max"] for r in rows),
              connected_repeat_forward_max=max(r["connected_repeat_forward_max"] for r in rows),
              loss_delta_scene_mean={phase: statistics.mean(r["readings"]["connected"]["loss"]-r["readings"]["native"]["loss"]
                                                           for r in rows if r["phase"] == phase)
                                     for phase in ("trained", "released_init")},
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Native full-loss TRAIN path, hooks are not an independent physical candidate.",
                           "Existing dead qla affine tensors are reconnected, not newly added parameters.",
                           "No optimizer update, validation or trained candidate efficacy claim.",
                           "All adverse initial loss shifts and native repeated-forward numerical differences retained."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
