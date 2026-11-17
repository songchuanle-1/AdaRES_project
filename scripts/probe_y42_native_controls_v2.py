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
assert Path(".codex_source_commit").read_text().strip() == "79ce2de9582c369422facb27c3cbe85559bf227c"
running = Path("ipdn/model_paper5/adares_y41_boundary_probability")
running_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in running.glob("*.py")}
config = "configs_song/model_paper5/adares_y42_point_covariance_res.yaml"
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
    q, masks, pad, mass, covariance = args
    if mode == "wrong_masks":
        return q, masks.roll(1, 1), pad, mass, covariance
    if mode == "wrong_covariance":
        changed = covariance.clone()
        for b in range(covariance.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = covariance[b, :count].roll(1, 0)
        diagonal = torch.tensor([sum(32-j for j in range(i)) for i in range(32)], device=covariance.device)
        changed[:, :, diagonal] = covariance[:, :, diagonal]
        return q, masks, pad, mass, changed
    if mode == "diagonal_only":
        diagonal = torch.tensor([sum(32-j for j in range(i)) for i in range(32)], device=covariance.device)
        diagonal_only = torch.zeros_like(covariance)
        diagonal_only[:, :, diagonal] = covariance[:, :, diagonal]
        return q, masks, pad, mass, diagonal_only
    if mode == "reflected_covariance":
        tri = torch.triu_indices(32, 32, device=covariance.device)
        signs = 1 - 2 * (torch.arange(32, device=covariance.device) % 2)
        return q, masks, pad, mass, covariance * (signs[tri[0]] * signs[tri[1]])


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
        native_covariance = captured["dec"][-1]
        assert native_covariance.shape[1] == 528
        assert torch.isfinite(native_covariance).all()
        covariance_stats = dict(superpoints=int(native_covariance.shape[0]),
                              nonzero_fraction=float((native_covariance > 0).float().mean()),
                              mean=float(native_covariance.mean()), maximum=float(native_covariance.max()))
        assert covariance_stats["nonzero_fraction"] > 0
        state = rng()
        buffers = {k: v.detach().clone() for k, v in model.dec.named_buffers()}
        readings, saved = {}, {}
        for mode in ("disabled", "enabled", "repeat", "wrong_masks", "wrong_covariance", "diagonal_only", "reflected_covariance"):
            restore(state)
            with torch.no_grad():
                for name, value in model.dec.named_buffers():
                    value.copy_(buffers[name])
            model.zero_grad(set_to_none=True)
            pred = model.dec(*captured["dec"])
            loss, terms = model.criterion(pred, *captured["criterion"])
            loss.backward()
            grads = {k: float(p.grad.norm()) for k, p in new.items() if p.grad is not None}
            if mode not in ("disabled", "diagonal_only"):
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
                   native_covariance=covariance_stats, readings=readings,
                   enabled_disabled_term_deltas={k: readings["enabled"]["terms"][k] - readings["disabled"]["terms"][k] for k in readings["enabled"]["terms"] if k != "loss"},
                   masks_disabled_max=delta("enabled", "disabled", "masks"),
                   repeat_indicator_max=delta("enabled", "repeat", "indis"),
                   wrong_masks_max=delta("enabled", "wrong_masks", "indis"),
                   wrong_covariance_max=delta("enabled", "wrong_covariance", "indis"),
                   diagonal_only_max=delta("enabled", "diagonal_only", "indis"),
                   reflected_covariance_max=delta("enabled", "reflected_covariance", "indis"))
        assert row["masks_disabled_max"] == row["repeat_indicator_max"] == 0
        assert min(row["wrong_masks_max"], row["wrong_covariance_max"], row["diagonal_only_max"]) > 0
        assert row["reflected_covariance_max"] > 0
        rows.append(row)
        q, masks, pad, mass, covariance = clone(captured["readout"])
        mode = "direct"
        q.requires_grad_()
        masks.requires_grad_()
        covariance.requires_grad_()
        direct = readout(q, masks, pad, mass, covariance)
        direct.sum().backward()
        assert masks.grad is None and covariance.grad is not None and torch.isfinite(covariance.grad).all()
        with torch.no_grad():
            sp = torch.arange(mass.shape[-1] - 1, -1, -1, device=mass.device)
            support = readout(q, masks[:, :, sp], pad[:, sp], mass[:, sp], covariance[:, sp])
            queries = readout(q.roll(1, 1), masks.roll(1, 1), pad, mass, covariance).roll(-1, 1)
            poison = covariance.detach().clone()
            poison[pad] = 1e5
            poisoned = readout(q, masks, pad, mass, poison)
            controls.append(dict(phase=phase, scan_id=batch["scan_ids"][0],
                                 support_permutation=float((direct - support).abs().max()),
                                 query_permutation=float((direct - queries).abs().max()),
                                 padding_poison=float((direct - poisoned).abs().max()),
                                 detached_mask_gradient=True,
                                 covariance_gradient_norm=float(covariance.grad.norm())))
        print(json.dumps(dict(phase=phase, scan_id=batch["scan_ids"][0],
                              covariance=covariance_stats, enabled_loss_delta=readings["enabled"]["loss"] - readings["disabled"]["loss"])), flush=True)
        captured.clear()
        del saved, grads, buffers, batch, q, masks, covariance, direct, support, queries, poisoned
    for hook in path_hooks:
        hook.remove()
    parameters = sum(p.numel() for p in model.parameters())
    del model, new, readout
    gc.collect()
    torch.cuda.empty_cache()
summary = {phase: {m: statistics.mean(r["readings"][m]["loss"] - r["readings"]["disabled"]["loss"] for r in rows if r["phase"] == phase)
                   for m in ("enabled", "wrong_masks", "wrong_covariance", "diagonal_only")}
           for phase in ("trained", "released_init")}
nonindicator = [abs(v) for r in rows for k, v in r["enabled_disabled_term_deltas"].items() if "indi_loss" not in k]
assert max(nonindicator) == 0
permutation = max(max(c["support_permutation"], c["query_permutation"]) for c in controls)
assert permutation < 1e-5 and max(c["padding_poison"] for c in controls) == 0
running_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in running.glob("*.py")}
assert running_before == running_after
result = dict(complete=True, id="AdaRES-Y42", rows=rows, cached_controls=controls,
              parameters=parameters, added_parameters=parameters - 164629637,
              added_trainable_tensors=8,
              minimum_added_gradient=min(v for r in rows for v in r["readings"]["enabled"]["point_gradients"].values()),
              cached_repeat_max=max(r["repeat_indicator_max"] for r in rows),
              nonindicator_loss_delta_max=max(nonindicator), loss_deltas=summary,
              wrong_masks_min=min(r["wrong_masks_max"] for r in rows),
              wrong_covariance_min=min(r["wrong_covariance_max"] for r in rows),
              reflected_covariance_min=min(r["reflected_covariance_max"] for r in rows),
              permutation_max=permutation, y41_source_unchanged=True,
              gpu_precheck=dict(physical_gpu=P["physical_gpu"], memory_mib=used, utilization=util),
              limitations=["TRAIN8/init4 actual interventions, not utility, necessity or trained efficacy.",
                           "All initialized adverse loss and zero-covariance results retained.",
                           "Native optimizer/resume, physical W2 equality and resources still separate."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
