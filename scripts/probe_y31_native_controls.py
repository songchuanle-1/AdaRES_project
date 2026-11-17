"""Cached native TRAIN8/init4 shape mechanism controls and causal gradients."""

import gc
from pathlib import Path
import statistics

import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain
from probe_adares_y15_resume import rng, restore

torch.set_num_threads(4)
config = "configs_song/model_paper5/adares_y31_mask_shape_res.yaml"
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
mode, captured = "enabled", {}


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


def shape_pre(module, args):
    captured["shape"] = clone(args)
    q, masks, pad, mass, pos = args
    if mode == "mask_detach":
        return q, masks.detach(), pad, mass, pos
    if mode == "rolled":
        return q, masks.roll(1, 1), pad, mass, pos
    if mode == "uniform":
        return q, torch.zeros_like(masks), pad, mass, pos


def shape_post(module, args, output):
    if mode == "disabled":
        return torch.zeros_like(output)


rows, controls = [], []
for phase in ("trained", "released_init"):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    new = {k: p for k, p in model.dec.named_parameters() if ".shape_readout." in "." + k}
    assert len(new) == 6
    if phase == "trained":
        ck = torch.load("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth", map_location="cpu")
        loaded = model.load_state_dict(ck["model"], strict=False)
        assert set(loaded.missing_keys) == {"dec." + k for k in new}
        assert not set(loaded.unexpected_keys) - {
            "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
        del ck
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    readout = model.dec.shape_readout
    path_hooks = [readout.register_forward_pre_hook(shape_pre),
                  readout.register_forward_hook(shape_post)]
    for index in indices[:8 if phase == "trained" else 4]:
        mode = "enabled"
        captured.clear()
        hooks = [model.dec.register_forward_pre_hook(before_dec),
                 model.criterion.register_forward_pre_hook(before_criterion)]
        batch = dataset.collate_fn([dataset[index]])
        with torch.no_grad():
            model(batch, mode="loss")
        for hook in hooks:
            hook.remove()
        state = rng()
        buffers = {k: v.detach().clone() for k, v in model.dec.named_buffers()}
        readings, saved = {}, {}
        for mode in ("disabled", "enabled", "repeat", "mask_detach", "rolled", "uniform"):
            restore(state)
            with torch.no_grad():
                for name, value in model.dec.named_buffers():
                    value.copy_(buffers[name])
            model.zero_grad(set_to_none=True)
            pred = model.dec(*captured["dec"])
            pred["masks"].retain_grad()
            loss, terms = model.criterion(pred, *captured["criterion"])
            loss.backward()
            grads = {k: float(p.grad.norm()) for k, p in new.items() if p.grad is not None}
            if mode != "disabled":
                assert len(grads) == 6 and all(v > 0 for v in grads.values()), (phase, mode, grads)
            else:
                assert not grads
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all()
                                               for p in model.dec.parameters() if p.grad is not None)
            readings[mode] = dict(loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                                  shape_gradients=grads)
            saved[mode] = {k: pred[k].detach().cpu().clone() for k in ("masks", "indis")}
            saved[mode]["mask_gradient"] = pred["masks"].grad.detach().cpu().clone()
            del pred, loss, terms
        delta = lambda a, b, k: float((saved[a][k] - saved[b][k]).abs().max())
        term_deltas = {k: readings["enabled"]["terms"][k] - readings["disabled"]["terms"][k]
                       for k in readings["enabled"]["terms"] if k != "loss"}
        row = dict(phase=phase, scan_id=batch["scan_ids"][0], dataset_index=index,
                   readings=readings, enabled_disabled_term_deltas=term_deltas,
                   masks_disabled_max=delta("enabled", "disabled", "masks"),
                   mask_detach_forward_max=max(delta("enabled", "mask_detach", k) for k in ("masks", "indis")),
                   mask_gradient_delta_norm=float((saved["enabled"]["mask_gradient"] - saved["mask_detach"]["mask_gradient"]).norm()),
                   repeat_indicator_max=delta("enabled", "repeat", "indis"),
                   wrong_shape_max=delta("enabled", "rolled", "indis"),
                   uniform_shape_max=delta("enabled", "uniform", "indis"))
        assert row["masks_disabled_max"] == row["mask_detach_forward_max"] == row["repeat_indicator_max"] == 0
        assert row["mask_gradient_delta_norm"] > 0
        assert min(row["wrong_shape_max"], row["uniform_shape_max"]) > 0
        rows.append(row)
        args = clone(captured["shape"])
        q, masks, pad, mass, pos = args
        mode = "enabled"
        with torch.no_grad():
            original = readout(*args)
            sp = torch.arange(mass.shape[-1]-1, -1, -1, device=mass.device)
            sp_permuted = readout(q, masks[:, :, sp], pad[:, sp], mass[:, sp], pos[:, sp])
            q_permuted = readout(q.roll(1, 1), masks.roll(1, 1), pad, mass, pos).roll(-1, 1)
            shifted = readout(q, masks, pad, mass, pos + 3)
            controls.append(dict(phase=phase, scan_id=batch["scan_ids"][0],
                                 sp_permutation=float((original-sp_permuted).abs().max()),
                                 query_permutation=float((original-q_permuted).abs().max()),
                                 translation=float((original-shifted).abs().max())))
        print(json.dumps(dict(phase=phase, scan_id=batch["scan_ids"][0],
                              mask_gradient_delta=row["mask_gradient_delta_norm"])), flush=True)
        captured.clear()
        del saved, grads, buffers, batch, args, q, masks, original
    for hook in path_hooks:
        hook.remove()
    parameters = sum(p.numel() for p in model.parameters())
    del model, new, readout
    gc.collect()
    torch.cuda.empty_cache()
summary = {phase: {m: statistics.mean(r["readings"][m]["loss"] - r["readings"]["disabled"]["loss"]
                                     for r in rows if r["phase"] == phase)
                   for m in ("enabled", "rolled", "uniform")}
           for phase in ("trained", "released_init")}
nonindicator = [abs(v) for r in rows for k, v in r["enabled_disabled_term_deltas"].items()
                if "indi_loss" not in k]
assert max(nonindicator) == 0
assert max(max(c["sp_permutation"], c["query_permutation"], c["translation"])
           for c in controls) < 1e-5
result = dict(complete=True, id="AdaRES-Y31", rows=rows, cached_controls=controls,
              parameters=parameters, added_parameters=parameters - 164629637, added_trainable_tensors=6,
              minimum_added_gradient=min(v for r in rows for v in r["readings"]["enabled"]["shape_gradients"].values()),
              mask_gradient_delta_min=min(r["mask_gradient_delta_norm"] for r in rows),
              mask_detach_forward_max=max(r["mask_detach_forward_max"] for r in rows),
              cached_repeat_max=max(r["repeat_indicator_max"] for r in rows),
              nonindicator_loss_delta_max=max(nonindicator), loss_deltas=summary,
              wrong_shape_min=min(r["wrong_shape_max"] for r in rows),
              uniform_shape_min=min(r["uniform_shape_max"] for r in rows),
              permutation_translation_max=max(max(c["sp_permutation"], c["query_permutation"], c["translation"]) for c in controls),
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Actual cached native TRAIN8/init4 losses, not trained efficacy.",
                           "Native tools.train full backward and optimizer steps checked separately.",
                           "Final confidence only, all adverse and wrong-shape losses retained.",
                           "Mask-to-covariance graph activity is not geometry necessity.",
                           "No validation, numeric search or global deterministic-inference claim."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
