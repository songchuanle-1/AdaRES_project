"""Cache native decoder/criterion inputs to isolate the centroid gradient path."""

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
mode, captured = "disabled", {}


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


def peer_pre(module, args):
    q, centroids = args
    if mode == "centroid_detach":
        return q, centroids.detach()
    if mode == "rolled":
        return q, centroids.roll(1, 1)
    if mode == "zero":
        return q, torch.zeros_like(centroids)


def peer_post(module, args, output):
    if mode == "disabled":
        return torch.zeros_like(output)


rows = []
for phase in ("trained", "released_init"):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    new = {k: p for k, p in model.dec.named_parameters() if ".relative_peer." in k}
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
    peer = model.dec.indi_embedding.relative_peer
    peer_hooks = [peer.register_forward_pre_hook(peer_pre), peer.register_forward_hook(peer_post)]
    for index in indices[:8 if phase == "trained" else 4]:
        mode = "disabled"
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
        for mode in ("disabled", "enabled", "repeat", "centroid_detach", "rolled", "zero"):
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
                assert len(grads) == 8 and all(v > 0 for v in grads.values())
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all()
                                               for p in model.dec.parameters() if p.grad is not None)
            readings[mode] = dict(loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                                  peer_gradients=grads)
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
                   centroid_detach_forward_max=max(delta("enabled", "centroid_detach", k) for k in ("masks", "indis")),
                   centroid_gradient_delta_norm=float((saved["enabled"]["mask_gradient"]-saved["centroid_detach"]["mask_gradient"]).norm()),
                   repeat_mask_gradient_max=delta("enabled", "repeat", "mask_gradient"))
        assert row["masks_disabled_max"] == row["masks_repeat_max"] == row["centroid_detach_forward_max"] == 0
        assert row["centroid_gradient_delta_norm"] > 0
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("readings", "enabled_disabled_term_deltas")}), flush=True)
        captured.clear()
        del saved, grads, buffers, batch
    for hook in peer_hooks:
        hook.remove()
    del model, new, peer
    gc.collect()
    torch.cuda.empty_cache()
summary = {}
for phase in ("trained", "released_init"):
    group = [r for r in rows if r["phase"] == phase]
    summary[phase] = {m: statistics.mean(r["readings"][m]["loss"]-r["readings"]["disabled"]["loss"] for r in group)
                      for m in ("enabled", "rolled", "zero")}
nonindicator = [abs(v) for r in rows for k, v in r["enabled_disabled_term_deltas"].items()
                if "indi_loss" not in k]
result = dict(complete=True, id="AdaRES-Y27", rows=rows, loss_deltas=summary,
              masks_disabled_max=max(r["masks_disabled_max"] for r in rows),
              centroid_detach_forward_max=max(r["centroid_detach_forward_max"] for r in rows),
              centroid_gradient_delta_min=min(r["centroid_gradient_delta_norm"] for r in rows),
              repeat_mask_gradient_max=max(r["repeat_mask_gradient_max"] for r in rows),
              nonindicator_loss_term_delta_max=max(nonindicator),
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Physical cached native decoder and criterion; all positive/adverse loss terms retained.",
                           "Gradient activity through mask centroids, not demonstrated learned geometric accuracy.",
                           "Only confidence loss terms change on identical native encoded inputs.",
                           "No new variants, hyperparameter scan, validation or efficacy/Pareto claim."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
