"""Physical Y27 geometry controls and native full-loss backward audit."""

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
config = "configs_song/model_paper5/adares_y27_relative_peer_res.yaml"
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


def peer_input(module, args):
    q, centroids = args
    if mode == "rolled":
        return q, centroids.roll(1, 1)
    if mode == "zero":
        return q, torch.zeros_like(centroids)
    if mode == "translated":
        return q, centroids + centroids.new_tensor([2., -3., 1.])


def peer_output(module, args, output):
    if mode == "disabled":
        return torch.zeros_like(output)
    if mode == "detached":
        return output.detach()


def indicator_input(module, args):
    cache["indicator"] = tuple(v.detach().clone() for v in args)


def decoder_output(module, args, output):
    cache["masks"] = output["masks"].detach().cpu().clone()
    cache["indis"] = output["indis"].detach().cpu().clone()


rows, controls = [], []
for phase in ("trained", "released_init"):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    new = {name: p for name, p in model.named_parameters()
           if ".relative_peer." in name}
    assert len(new) == 8
    if phase == "trained":
        ck = torch.load("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth",
                        map_location="cpu")
        loaded = model.load_state_dict(ck["model"], strict=False)
        assert set(loaded.missing_keys) == set(new)
        assert not set(loaded.unexpected_keys)-{
            "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
        del ck
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    peer = model.dec.indi_embedding.relative_peer
    hooks = [peer.register_forward_pre_hook(peer_input), peer.register_forward_hook(peer_output),
             model.dec.indi_embedding.register_forward_pre_hook(indicator_input),
             model.dec.register_forward_hook(decoder_output)]
    for index in indices[:8 if phase == "trained" else 4]:
        batch = dataset.collate_fn([dataset[index]])
        state = rng()
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        readings, saved = {}, {}
        for mode in ("enabled", "disabled", "detached", "repeat"):
            restore(state)
            with torch.no_grad():
                for k, value in model.named_buffers():
                    value.copy_(buffers[k])
            model.zero_grad(set_to_none=True)
            cache.clear()
            loss, terms = model(batch, mode="loss")
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            grad = {name: float(p.grad.norm()) for name, p in new.items() if p.grad is not None}
            if mode in ("enabled", "repeat"):
                assert len(grad) == 8 and all(v > 0 for v in grad.values())
            else:
                assert not grad
            readings[mode] = dict(loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                                  new_gradients=grad)
            saved[mode] = {k: cache[k] for k in ("masks", "indis")}
            del loss, terms
        args = cache["indicator"]
        handles = []
        with torch.no_grad():
            cached = {}
            for mode in ("enabled", "repeat", "disabled", "detached", "rolled", "zero", "translated"):
                cached[mode] = model.dec.indi_embedding(*args).detach()
            # Both queries and their masks are permuted jointly.
            mode = "enabled"
            permuted = model.dec.indi_embedding(args[0].roll(1, 1), args[1].roll(1, 1), *args[2:]).roll(-1, 1)
            controls.append(dict(phase=phase, scan_id=batch["scan_ids"][0],
                                 repeat=float((cached["enabled"]-cached["repeat"]).abs().max()),
                                 detach=float((cached["enabled"]-cached["detached"]).abs().max()),
                                 disable=float((cached["enabled"]-cached["disabled"]).abs().max()),
                                 rolled=float((cached["enabled"]-cached["rolled"]).abs().max()),
                                 zero=float((cached["enabled"]-cached["zero"]).abs().max()),
                                 translation=float((cached["enabled"]-cached["translated"]).abs().max()),
                                 joint_permutation=float((cached["enabled"]-permuted).abs().max())))
        row = dict(phase=phase, scan_id=batch["scan_ids"][0], dataset_index=index,
                   readings=readings,
                   enabled_disabled_mask_max=float((saved["enabled"]["masks"]-saved["disabled"]["masks"]).abs().max()),
                   detached_forward_max=max(float((saved["enabled"][k]-saved["detached"][k]).abs().max()) for k in saved["enabled"]),
                   repeat_forward_max=max(float((saved["enabled"][k]-saved["repeat"][k]).abs().max()) for k in saved["enabled"]))
        rows.append(row)
        print(json.dumps(dict(phase=phase, scan_id=row["scan_id"], loss_delta=readings["enabled"]["loss"]-readings["disabled"]["loss"])), flush=True)
        del buffers, saved, readings, cached, args, batch, grad
        cache.clear()
    parameters = sum(p.numel() for p in model.parameters())
    for hook in hooks:
        hook.remove()
    # Synthetic boundary contracts use the actual physical module, not a reimplementation.
    gorilla.set_random_seed(1999)
    q = torch.randn(2, 8, 256, device="cuda")
    positions = torch.randn(2, 12, 3, device="cuda")
    masks = torch.randn(2, 8, 12, device="cuda")
    padding = torch.zeros(2, 12, dtype=torch.bool, device="cuda")
    padding[:, -2:] = True
    mass = torch.rand(2, 12, device="cuda") + 1
    with torch.no_grad():
        base = model.dec.indi_embedding(q, masks, padding, mass, positions)
        bad_positions = positions.clone()
        bad_positions[:, -2:] = 100000
        bad_masks = masks.clone()
        bad_masks[:, :, -2:] = 100000
        bad_mass = mass.clone()
        bad_mass[:, -2:] = 100000
        altered = model.dec.indi_embedding(q, bad_masks, padding, bad_mass, bad_positions)
        assert torch.equal(base, altered)
        single = peer(q[:, :1], positions[:, :1])
        assert torch.equal(single, torch.zeros_like(single))
    del model, peer, new, hooks, q, masks, base, altered
    gc.collect()
    torch.cuda.empty_cache()
assert all(c["repeat"] == c["detach"] == 0 for c in controls)
assert all(c["disable"] > 0 and c["rolled"] > 0 and c["zero"] > 0 for c in controls)
assert max(c["translation"] for c in controls) < 1e-5
assert max(c["joint_permutation"] for c in controls) < 1e-5
result = dict(complete=True, id="AdaRES-Y27", rows=rows, cached_controls=controls,
              parameters=parameters, added_parameters=parameters-164629637, added_trainable_tensors=8,
              minimum_added_gradient=min(v for r in rows for v in r["readings"]["enabled"]["new_gradients"].values()),
              cached_repeat_max=max(c["repeat"] for c in controls),
              cached_detach_max=max(c["detach"] for c in controls),
              cached_roll_min=min(c["rolled"] for c in controls),
              cached_translation_max=max(c["translation"] for c in controls),
              cached_joint_permutation_max=max(c["joint_permutation"] for c in controls),
              padding_counterfactual_max=0, singleton_peer_max=0,
              native_loss_delta={phase: statistics.mean(r["readings"]["enabled"]["loss"]-r["readings"]["disabled"]["loss"]
                                                        for r in rows if r["phase"] == phase)
                                 for phase in ("trained", "released_init")},
              native_repeat_forward_max=max(r["repeat_forward_max"] for r in rows),
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Eight trained W2/4 released-init full native scenes, not efficacy.",
                           "Only confidence changes immediately; any native upstream noise retained.",
                           "Geometry permutation changes correspondence while preserving marginal values; not new candidate/tuning.",
                           "Synthetic padding/singleton contracts complement, never replace native backward."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
