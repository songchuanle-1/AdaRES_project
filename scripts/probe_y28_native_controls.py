"""Native and cached causal controls for physical predicted-mask readback."""

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
config = "configs_song/model_paper5/adares_y28_mask_readback_res.yaml"
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
mode, cache = "enabled", {}


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(clone(v) for v in value)
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    return value


def indicator_pre(module, args):
    cache["indicator"] = args


def readback_pre(module, args):
    q, foreground, background = args
    if mode == "pool_detach":
        return q, foreground.detach(), background.detach()
    if mode in ("rolled", "uniform"):
        _, logits, padding, mass, scene = cache["indicator"]
        valid = (~padding).to(logits.dtype)
        if mode == "uniform":
            w = valid * mass
            mean = torch.bmm(w[:, None], scene) / w.sum(-1).view(-1, 1, 1).clamp_min(1e-6)
            return q, mean.expand_as(foreground), mean.expand_as(background)
        p = logits.sigmoid()
        weights = [p * valid[:, None] * mass[:, None],
                   (1-p) * valid[:, None] * mass[:, None]]
        pools = []
        for weight in weights:
            rolled = weight.clone()
            for b in range(weight.shape[0]):
                count = int(valid[b].sum())
                rolled[b, :, :count] = weight[b, :, :count].roll(1, -1)
            pools.append(torch.bmm(rolled, scene) / rolled.sum(-1, keepdim=True).clamp_min(1e-6))
        return q, pools[0], pools[1]


def readback_post(module, args, output):
    if mode == "disabled":
        return torch.zeros_like(output)
    if mode == "detached":
        return output.detach()


def dec_pre(module, args):
    cache["dec"] = clone(args)


def criterion_pre(module, args):
    cache["criterion"] = clone(args[1:])


rows, controls, causal = [], [], []
for phase in ("trained", "released_init"):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    new = {k: p for k, p in model.named_parameters() if ".mask_readback." in k}
    assert len(new) == 4
    if phase == "trained":
        ck = torch.load("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth", map_location="cpu")
        loaded = model.load_state_dict(ck["model"], strict=False)
        assert set(loaded.missing_keys) == set(new)
        assert not set(loaded.unexpected_keys) - {
            "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
        del ck
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    readback = model.dec.indi_embedding.mask_readback
    hooks = [model.dec.indi_embedding.register_forward_pre_hook(indicator_pre),
             readback.register_forward_pre_hook(readback_pre),
             readback.register_forward_hook(readback_post)]
    for index in indices[:8 if phase == "trained" else 4]:
        batch = dataset.collate_fn([dataset[index]])
        state = rng()
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        native = {}
        for mode in ("enabled", "disabled", "detached", "repeat"):
            restore(state)
            with torch.no_grad():
                for k, v in model.named_buffers():
                    v.copy_(buffers[k])
            cache.clear()
            capture_hooks = [model.dec.register_forward_pre_hook(dec_pre),
                             model.criterion.register_forward_pre_hook(criterion_pre)]
            model.zero_grad(set_to_none=True)
            loss, terms = model(batch, mode="loss")
            for hook in capture_hooks:
                hook.remove()
            loss.backward()
            grad = {k: float(p.grad.norm()) for k, p in new.items() if p.grad is not None}
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all()
                                               for p in model.parameters() if p.grad is not None)
            if mode in ("enabled", "repeat"):
                assert len(grad) == 4 and all(v > 0 for v in grad.values())
            else:
                assert not grad
            native[mode] = dict(loss=float(loss), new_gradients=grad)
            del loss, terms
        indicator_args = clone(cache["indicator"])
        with torch.no_grad():
            readings = {}
            for mode in ("enabled", "repeat", "disabled", "detached", "rolled", "uniform"):
                readings[mode] = model.dec.indi_embedding(*indicator_args).detach()
            mode = "enabled"
            q, masks, padding, mass, scene = indicator_args
            perm = torch.arange(scene.shape[1]-1, -1, -1, device=scene.device)
            sp_permuted = model.dec.indi_embedding(q, masks[:, :, perm], padding[:, perm], mass[:, perm], scene[:, perm])
            q_permuted = model.dec.indi_embedding(q.roll(1, 1), masks.roll(1, 1), padding, mass, scene).roll(-1, 1)
            control = dict(phase=phase, scan_id=batch["scan_ids"][0],
                           repeat=float((readings["enabled"]-readings["repeat"]).abs().max()),
                           detach=float((readings["enabled"]-readings["detached"]).abs().max()),
                           disabled=float((readings["enabled"]-readings["disabled"]).abs().max()),
                           rolled=float((readings["enabled"]-readings["rolled"]).abs().max()),
                           uniform=float((readings["enabled"]-readings["uniform"]).abs().max()),
                           superpoint_permutation=float((readings["enabled"]-sp_permuted).abs().max()),
                           query_permutation=float((readings["enabled"]-q_permuted).abs().max()))
            controls.append(control)
        # Same encoded decoder/criterion inputs isolate the new pool gradient.
        dec_args, crit_args = cache["dec"], cache["criterion"]
        cache.clear()
        cached, saved = {}, {}
        for mode in ("enabled", "repeat", "pool_detach", "disabled", "rolled", "uniform"):
            model.zero_grad(set_to_none=True)
            pred = model.dec(*dec_args)
            pred["masks"].retain_grad()
            loss, terms = model.criterion(pred, *crit_args)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.dec.parameters() if p.grad is not None)
            cached[mode] = dict(loss=float(loss), terms={k: float(v) for k, v in terms.items()})
            saved[mode] = {k: pred[k].detach().cpu().clone() for k in ("masks", "indis")}
            saved[mode]["mask_gradient"] = pred["masks"].grad.detach().cpu().clone()
            del pred, loss, terms
        delta = lambda a, b, k: float((saved[a][k]-saved[b][k]).abs().max())
        causal_row = dict(phase=phase, scan_id=batch["scan_ids"][0], readings=cached,
                          mask_disabled_max=delta("enabled", "disabled", "masks"),
                          detach_forward_max=max(delta("enabled", "pool_detach", k) for k in ("masks", "indis")),
                          pool_gradient_delta_norm=float((saved["enabled"]["mask_gradient"]-saved["pool_detach"]["mask_gradient"]).norm()),
                          repeat_gradient_max=delta("enabled", "repeat", "mask_gradient"),
                          nonindicator_loss_delta_max=max(abs(cached["enabled"]["terms"][k]-cached["disabled"]["terms"][k])
                                                         for k in cached["enabled"]["terms"] if k != "loss" and "indi_loss" not in k))
        assert causal_row["mask_disabled_max"] == causal_row["detach_forward_max"] == 0
        assert causal_row["pool_gradient_delta_norm"] > 0
        causal.append(causal_row)
        rows.append(dict(phase=phase, scan_id=batch["scan_ids"][0], dataset_index=index, readings=native))
        print(json.dumps(dict(phase=phase, scan_id=batch["scan_ids"][0], pool_gradient=causal_row["pool_gradient_delta_norm"])), flush=True)
        del batch, buffers, saved, cached, dec_args, crit_args, readings, indicator_args
        cache.clear()
    mode = "enabled"
    q = torch.randn(2, 8, 256, device="cuda")
    scene = torch.randn(2, 12, 256, device="cuda")
    masks = torch.randn(2, 8, 12, device="cuda")
    padding = torch.zeros(2, 12, dtype=torch.bool, device="cuda")
    padding[:, -2:] = True
    mass = torch.rand(2, 12, device="cuda") + 1
    with torch.no_grad():
        base = model.dec.indi_embedding(q, masks, padding, mass, scene)
        bad_scene, bad_masks, bad_mass = scene.clone(), masks.clone(), mass.clone()
        bad_scene[:, -2:] = 100000
        bad_masks[:, :, -2:] = 100000
        bad_mass[:, -2:] = 100000
        altered = model.dec.indi_embedding(q, bad_masks, padding, bad_mass, bad_scene)
        assert torch.equal(base, altered)
    parameters = sum(p.numel() for p in model.parameters())
    for hook in hooks:
        hook.remove()
    del model, readback, new, scene, q, masks, base, altered
    cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
assert all(c["repeat"] == c["detach"] == 0 for c in controls)
assert all(c["disabled"] > 0 and c["rolled"] > 0 and c["uniform"] > 0 for c in controls)
assert max(c["superpoint_permutation"] for c in controls) < 1e-5
assert max(c["query_permutation"] for c in controls) < 1e-5
result = dict(complete=True, id="AdaRES-Y28", rows=rows, cached_controls=controls, causal_rows=causal,
              parameters=parameters, added_parameters=parameters-164629637, added_trainable_tensors=4,
              minimum_added_gradient=min(v for r in rows for v in r["readings"]["enabled"]["new_gradients"].values()),
              cached_repeat_max=max(c["repeat"] for c in controls),
              cached_detach_max=max(c["detach"] for c in controls),
              cached_roll_min=min(c["rolled"] for c in controls),
              cached_uniform_min=min(c["uniform"] for c in controls),
              permutation_max=max(max(c["superpoint_permutation"], c["query_permutation"]) for c in controls),
              pool_gradient_delta_min=min(r["pool_gradient_delta_norm"] for r in causal),
              pool_detach_forward_max=max(r["detach_forward_max"] for r in causal),
              nonindicator_loss_delta_max=max(r["nonindicator_loss_delta_max"] for r in causal),
              cached_loss_deltas={phase: {m: statistics.mean(r["readings"][m]["loss"]-r["readings"]["disabled"]["loss"]
                                                           for r in causal if r["phase"] == phase)
                                         for m in ("enabled", "rolled", "uniform")}
                                  for phase in ("trained", "released_init")},
              padding_counterfactual_max=0,
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["NativeTRAIN8+releasedinit4 and cached matched losses, not trained efficacy.",
                           "All beneficial/adverse/control losses retained; no outcome-selected cohort.",
                           "New pool-to-mask gradient is causal graph activity, not proof that it improves masks.",
                           "No fixedcosine J1 retrial, validation, numerical tuning or global stability claim."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
