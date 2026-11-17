"""Cached native TRAIN causal controls for the physical temporal confidence path."""

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
config = "configs_song/model_paper5/adares_y30_query_trajectory_res.yaml"
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


def trajectory_pre(module, args):
    queries, masks, padding, mass = args
    captured["trajectory"] = clone(args)
    if mode == "mask_detach":
        return queries, [v.detach() for v in masks], padding, mass
    if mode == "rolled":
        return ([v.roll(1, 1) for v in queries[:-1]]+queries[-1:],
                [v.roll(1, 1) for v in masks[:-1]]+masks[-1:], padding, mass)
    if mode == "reverse":
        return (list(reversed(queries[:-1]))+queries[-1:],
                list(reversed(masks[:-1]))+masks[-1:], padding, mass)
    if mode == "terminal":
        return [queries[-1]]*len(queries), [masks[-1]]*len(masks), padding, mass


def trajectory_post(module, args, output):
    if mode == "disabled":
        return torch.zeros_like(output)


rows, controls = [], []
for phase in ("trained", "released_init"):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    new = {k: p for k, p in model.dec.named_parameters() if ".trajectory_readout." in "."+k}
    assert len(new) == 8
    if phase == "trained":
        ck = torch.load("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth", map_location="cpu")
        loaded = model.load_state_dict(ck["model"], strict=False)
        assert set(loaded.missing_keys) == {"dec."+k for k in new}
        assert not set(loaded.unexpected_keys)-{
            "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
        del ck
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    trajectory = model.dec.trajectory_readout
    path_hooks = [trajectory.register_forward_pre_hook(trajectory_pre),
                  trajectory.register_forward_hook(trajectory_post)]
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
        for mode in ("disabled", "enabled", "repeat", "mask_detach", "rolled", "reverse", "terminal"):
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
                assert len(grads) == 8 and all(v > 0 for v in grads.values()), (phase, mode, grads)
            else:
                assert not grads
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all()
                                               for p in model.dec.parameters() if p.grad is not None)
            readings[mode] = dict(loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                                  trajectory_gradients=grads)
            saved[mode] = {k: pred[k].detach().cpu().clone() for k in ("masks", "indis")}
            saved[mode]["mask_gradient"] = pred["masks"].grad.detach().cpu().clone()
            del pred, loss, terms
        delta = lambda a, b, k: float((saved[a][k]-saved[b][k]).abs().max())
        term_deltas = {k: readings["enabled"]["terms"][k]-readings["disabled"]["terms"][k]
                       for k in readings["enabled"]["terms"] if k != "loss"}
        row = dict(phase=phase, scan_id=batch["scan_ids"][0], dataset_index=index,
                   readings=readings, enabled_disabled_term_deltas=term_deltas,
                   masks_disabled_max=delta("enabled", "disabled", "masks"),
                   masks_repeat_max=delta("enabled", "repeat", "masks"),
                   mask_detach_forward_max=max(delta("enabled", "mask_detach", k) for k in ("masks", "indis")),
                   mask_gradient_delta_norm=float((saved["enabled"]["mask_gradient"]-saved["mask_detach"]["mask_gradient"]).norm()),
                   repeat_mask_gradient_max=delta("enabled", "repeat", "mask_gradient"),
                   repeat_indicator_max=delta("enabled", "repeat", "indis"),
                   wrong_history_max=delta("enabled", "rolled", "indis"),
                   reverse_history_max=delta("enabled", "reverse", "indis"),
                   terminal_repeat_max=delta("enabled", "terminal", "indis"))
        assert row["masks_disabled_max"] == row["masks_repeat_max"] == row["mask_detach_forward_max"] == 0
        assert row["mask_gradient_delta_norm"] > 0
        assert min(row["wrong_history_max"], row["reverse_history_max"], row["terminal_repeat_max"]) > 0
        rows.append(row)
        args = clone(captured["trajectory"])
        queries, masks, pad, mass = args
        mode = "enabled"
        with torch.no_grad():
            original = trajectory(*args)
            repeat = trajectory(*args)
            sp = torch.arange(mass.shape[-1]-1, -1, -1, device=mass.device)
            sp_permuted = trajectory(queries, [v[:, :, sp] for v in masks], pad[:, sp], mass[:, sp])
            q_permuted = trajectory([v.roll(1, 1) for v in queries],
                                     [v.roll(1, 1) for v in masks], pad, mass).roll(-1, 1)
            controls.append(dict(phase=phase, scan_id=batch["scan_ids"][0],
                                 repeat=float((original-repeat).abs().max()),
                                 sp_permutation=float((original-sp_permuted).abs().max()),
                                 query_permutation=float((original-q_permuted).abs().max())))
        print(json.dumps({k: v for k, v in row.items() if k != "readings"}), flush=True)
        captured.clear()
        del saved, grads, buffers, batch, args, queries, masks, original, repeat
    for hook in path_hooks:
        hook.remove()
    parameters = sum(p.numel() for p in model.parameters())
    del model, new, trajectory
    gc.collect()
    torch.cuda.empty_cache()
summary = {phase: {m: statistics.mean(r["readings"][m]["loss"]-r["readings"]["disabled"]["loss"]
                                     for r in rows if r["phase"] == phase)
                   for m in ("enabled", "rolled", "reverse", "terminal")}
           for phase in ("trained", "released_init")}
nonindicator = [abs(v) for r in rows for k, v in r["enabled_disabled_term_deltas"].items()
                if "indi_loss" not in k]
print(json.dumps(dict(strict_nonindicator_max=max(nonindicator),
                      unequal_terms=[{"phase": r["phase"], "scan_id": r["scan_id"], "key": k, "delta": v}
                                     for r in rows for k, v in r["enabled_disabled_term_deltas"].items()
                                     if "indi_loss" not in k and v != 0])), flush=True)
assert max(nonindicator) == 0
assert max(max(c["sp_permutation"], c["query_permutation"]) for c in controls) < 1e-5
result = dict(complete=True, id="AdaRES-Y30", rows=rows, cached_controls=controls,
              parameters=parameters, added_parameters=parameters-164629637, added_trainable_tensors=8,
              minimum_added_gradient=min(v for r in rows for v in r["readings"]["enabled"]["trajectory_gradients"].values()),
              mask_gradient_delta_min=min(r["mask_gradient_delta_norm"] for r in rows),
              mask_detach_forward_max=max(r["mask_detach_forward_max"] for r in rows),
              cached_repeat_max=max(r["repeat_indicator_max"] for r in rows),
              nonindicator_loss_delta_max=max(nonindicator), loss_deltas=summary,
              wrong_history_min=min(r["wrong_history_max"] for r in rows),
              reverse_history_min=min(r["reverse_history_max"] for r in rows),
              terminal_repeat_min=min(r["terminal_repeat_max"] for r in rows),
              permutation_max=max(max(c["sp_permutation"], c["query_permutation"]) for c in controls),
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Physical cached native TRAIN8+releasedinit4; initial task loss and all adverse/wrong-history results retained.",
                           "True mask-to-trajectory gradient activity, not necessity or trained efficacy.",
                           "Final confidence only; masks, auxiliary losses and nonindicator losses unchanged on identical decoder inputs.",
                           "No validation, numerical scan or global deterministic-inference claim."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
