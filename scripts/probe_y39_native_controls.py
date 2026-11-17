"""Native TRAIN8/init4 incidence, matched intervention and decoder gradients."""
import gc
import hashlib
from pathlib import Path
import statistics
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain
from probe_adares_y15_resume import rng, restore

torch.set_num_threads(4)
assert Path(".codex_source_commit").read_text().strip() == "67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773"
running = Path("ipdn/model_paper5/adares_y37_pair_distribution")
running_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in running.glob("*.py")}
config = "configs_song/model_paper5/adares_y39_point_distribution_res.yaml"
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


def readout_pre(module, args):
    if mode == "enabled":
        captured["readout"] = clone(args)
    q, masks, pad, mass, variance = args
    if mode == "wrong_masks":
        return q, masks.roll(1, 1), pad, mass, variance
    if mode == "wrong_variance":
        changed = variance.clone()
        for b in range(variance.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = variance[b, :count].roll(1, 0)
        return q, masks, pad, mass, changed
    if mode == "zero_variance":
        return q, masks, pad, mass, torch.zeros_like(variance)


def readout_post(module, args, output):
    if mode == "disabled":
        return torch.zeros_like(output)


rows, controls = [], []
for phase in ("trained", "released_init"):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    new = {k: p for k, p in model.dec.named_parameters() if ".point_readout." in "." + k}
    assert len(new) == 8
    if phase == "trained":
        ck = torch.load("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth", map_location="cpu")
        loaded = model.load_state_dict(ck["model"], strict=False)
        assert set(loaded.missing_keys) == {"dec." + k for k in new}
        assert not set(loaded.unexpected_keys) - {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
        del ck
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    readout = model.dec.point_readout
    path_hooks = [readout.register_forward_pre_hook(readout_pre), readout.register_forward_hook(readout_post)]
    for index in indices[:8 if phase == "trained" else 4]:
        mode = "enabled"
        captured.clear()
        hooks = [model.dec.register_forward_pre_hook(before_dec), model.criterion.register_forward_pre_hook(before_criterion)]
        batch = dataset.collate_fn([dataset[index]])
        with torch.no_grad():
            model(batch, mode="loss")
        for hook in hooks:
            hook.remove()
        native_variance = captured["dec"][-1]
        assert native_variance.shape[1] == 32
        assert torch.isfinite(native_variance).all() and native_variance.min() >= 0
        variance_stats = dict(superpoints=int(native_variance.shape[0]),
                              nonzero_fraction=float((native_variance > 0).float().mean()),
                              mean=float(native_variance.mean()), maximum=float(native_variance.max()))
        assert variance_stats["nonzero_fraction"] > 0
        state = rng()
        buffers = {k: v.detach().clone() for k, v in model.dec.named_buffers()}
        readings, saved = {}, {}
        for mode in ("disabled", "enabled", "repeat", "wrong_masks", "wrong_variance", "zero_variance"):
            restore(state)
            with torch.no_grad():
                for name, value in model.dec.named_buffers():
                    value.copy_(buffers[name])
            model.zero_grad(set_to_none=True)
            pred = model.dec(*captured["dec"])
            loss, terms = model.criterion(pred, *captured["criterion"])
            loss.backward()
            grads = {k: float(p.grad.norm()) for k, p in new.items() if p.grad is not None}
            if mode not in ("disabled", "zero_variance"):
                assert len(grads) == 8 and all(v > 0 for v in grads.values()), (phase, mode, grads)
            elif mode == "disabled":
                assert not grads
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.dec.parameters() if p.grad is not None)
            readings[mode] = dict(loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                                  point_gradients=grads)
            saved[mode] = {k: pred[k].detach().cpu().clone() for k in ("masks", "indis")}
            del pred, loss, terms
        delta = lambda a, b, k: float((saved[a][k] - saved[b][k]).abs().max())
        row = dict(phase=phase, scan_id=batch["scan_ids"][0], dataset_index=index,
                   native_variance=variance_stats, readings=readings,
                   enabled_disabled_term_deltas={k: readings["enabled"]["terms"][k] - readings["disabled"]["terms"][k] for k in readings["enabled"]["terms"] if k != "loss"},
                   masks_disabled_max=delta("enabled", "disabled", "masks"),
                   repeat_indicator_max=delta("enabled", "repeat", "indis"),
                   wrong_masks_max=delta("enabled", "wrong_masks", "indis"),
                   wrong_variance_max=delta("enabled", "wrong_variance", "indis"),
                   zero_variance_max=delta("enabled", "zero_variance", "indis"))
        assert row["masks_disabled_max"] == row["repeat_indicator_max"] == 0
        assert min(row["wrong_masks_max"], row["wrong_variance_max"], row["zero_variance_max"]) > 0
        rows.append(row)
        q, masks, pad, mass, variance = clone(captured["readout"])
        mode = "direct"
        q.requires_grad_()
        masks.requires_grad_()
        variance.requires_grad_()
        direct = readout(q, masks, pad, mass, variance)
        direct.sum().backward()
        assert masks.grad is None and variance.grad is not None and torch.isfinite(variance.grad).all()
        with torch.no_grad():
            sp = torch.arange(mass.shape[-1] - 1, -1, -1, device=mass.device)
            support = readout(q, masks[:, :, sp], pad[:, sp], mass[:, sp], variance[:, sp])
            queries = readout(q.roll(1, 1), masks.roll(1, 1), pad, mass, variance).roll(-1, 1)
            poison = variance.detach().clone()
            poison[pad] = 1e5
            poisoned = readout(q, masks, pad, mass, poison)
            controls.append(dict(phase=phase, scan_id=batch["scan_ids"][0],
                                 support_permutation=float((direct - support).abs().max()),
                                 query_permutation=float((direct - queries).abs().max()),
                                 padding_poison=float((direct - poisoned).abs().max()),
                                 detached_mask_gradient=True,
                                 variance_gradient_norm=float(variance.grad.norm())))
        print(json.dumps(dict(phase=phase, scan_id=batch["scan_ids"][0],
                              variance=variance_stats, enabled_loss_delta=readings["enabled"]["loss"] - readings["disabled"]["loss"])), flush=True)
        captured.clear()
        del saved, grads, buffers, batch, q, masks, variance, direct, support, queries, poisoned
    for hook in path_hooks:
        hook.remove()
    parameters = sum(p.numel() for p in model.parameters())
    del model, new, readout
    gc.collect()
    torch.cuda.empty_cache()
summary = {phase: {m: statistics.mean(r["readings"][m]["loss"] - r["readings"]["disabled"]["loss"] for r in rows if r["phase"] == phase)
                   for m in ("enabled", "wrong_masks", "wrong_variance", "zero_variance")}
           for phase in ("trained", "released_init")}
nonindicator = [abs(v) for r in rows for k, v in r["enabled_disabled_term_deltas"].items() if "indi_loss" not in k]
assert max(nonindicator) == 0
permutation = max(max(c["support_permutation"], c["query_permutation"]) for c in controls)
assert permutation < 1e-5 and max(c["padding_poison"] for c in controls) == 0
running_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in running.glob("*.py")}
assert running_before == running_after
result = dict(complete=True, id="AdaRES-Y39", rows=rows, cached_controls=controls,
              parameters=parameters, added_parameters=parameters - 164629637,
              added_trainable_tensors=8,
              minimum_added_gradient=min(v for r in rows for v in r["readings"]["enabled"]["point_gradients"].values()),
              cached_repeat_max=max(r["repeat_indicator_max"] for r in rows),
              nonindicator_loss_delta_max=max(nonindicator), loss_deltas=summary,
              wrong_masks_min=min(r["wrong_masks_max"] for r in rows),
              wrong_variance_min=min(r["wrong_variance_max"] for r in rows),
              permutation_max=permutation, y37_source_unchanged=True,
              gpu_precheck=dict(physical_gpu=P["physical_gpu"], memory_mib=used, utilization=util),
              limitations=["TRAIN8/init4 actual interventions, not utility, necessity or trained efficacy.",
                           "All initialized adverse loss and zero-variance results retained.",
                           "Native optimizer/resume, physical W2 equality and resources still separate."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
