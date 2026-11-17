"""Actual point-coordinate capture, pairing interventions and native gradients."""
from pathlib import Path
import gc
import hashlib
import statistics
import gorilla
import torch
from torch_scatter import scatter_mean
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain
from probe_adares_y15_resume import rng, restore

assert P["physical_gpu"] == 1
assert Path(".codex_source_commit").read_text().strip() == "6fc7d4e46b6c26a11fede7f12d4e79b3471cc1cc"
terminal = json.loads(Path("/tmp/adares_y42_terminal_ff34_" + P["token"] + ".json").read_text())
protected = dict(terminal["source_hashes"], **terminal["protected_source_hashes"])
command = Path("/proc/13155/cmdline").read_bytes()
assert b"configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml" in command
assert command.split(b"\0")[command.split(b"\0").index(b"--gpu_ids") + 1] == b"0"


def protect():
    assert Path("/proc/13155/cmdline").read_bytes() == command
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in protected.items())


protect()
torch.set_num_threads(4)
config = "configs_song/model_paper5/adares_y44_coordinate_feature_res.yaml"
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
assert indices == [0, 44, 60, 101, 115, 126, 137, 171]
mode, captured = "enabled", {}


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(clone(v) for v in value)
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    return value


def cross_stat(coords, features, sp):
    return scatter_mean(
        ((coords - scatter_mean(coords, sp, dim=0)[sp]).unsqueeze(-1) *
         (features - scatter_mean(features, sp, dim=0)[sp]).unsqueeze(1)).flatten(1), sp, dim=0)


def before_dec(module, inputs):
    captured["dec"] = clone(inputs)


def before_criterion(module, inputs):
    captured["criterion"] = clone(inputs[1:])


def readout_pre(module, inputs):
    q, masks, pad, mass, cross = inputs
    if mode == "enabled":
        captured["readout"] = clone(inputs)
    if mode == "wrong_masks":
        return q, masks.roll(1, 1), pad, mass, cross
    if mode == "wrong_pairing":
        wrong, _ = model.dec.get_batches(captured["wrong_repeated"], captured["dec"][2])
        assert wrong.shape == cross.shape
        return q, masks, pad, mass, wrong
    if mode == "rolled_cross":
        changed = cross.clone()
        for b in range(cross.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = cross[b, :count].roll(1, 0)
        return q, masks, pad, mass, changed
    if mode == "zero_cross":
        return q, masks, pad, mass, torch.zeros_like(cross)


def readout_post(module, inputs, output):
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
    original_extract = model.extract_feat

    def captured_extract(x, sp, v2p, coords):
        values = {}
        module = sys.modules[type(model).__module__]
        original_scatter = module.scatter_mean
        reductions = []

        def audit_scatter(value, labels, **kwargs):
            output = original_scatter(value, labels, **kwargs)
            reductions.append(output.detach().clone())
            return output

        handle = model.output_layer.register_forward_hook(
            lambda module, inputs, output: values.update(features=output.features.detach().clone()))
        module.scatter_mean = audit_scatter
        try:
            pooled, native_cross = original_extract(x, sp, v2p, coords)
        finally:
            module.scatter_mean = original_scatter
            handle.remove()
        features = values["features"][v2p.long()]
        assert len(reductions) == 4
        assert [v.shape[1] for v in reductions] == [32, 3, 96, 32]
        assert torch.equal(reductions[2], native_cross)
        labels = sp.detach().cpu()
        c64, f64 = coords.detach().double().cpu(), features.detach().double().cpu()
        cm, fm = scatter_mean(c64, labels, dim=0), scatter_mean(f64, labels, dim=0)
        cc, ff = c64 - cm[labels], f64 - fm[labels]
        gpu_cm, gpu_fm = reductions[1].double().cpu(), reductions[0].double().cpu()
        eps = torch.finfo(features.dtype).eps
        coord_error = (gpu_cm - cm).abs()[labels] + eps * (c64.abs() + gpu_cm.abs()[labels])
        feature_error = (gpu_fm - fm).abs()[labels] + eps * (f64.abs() + gpu_fm.abs()[labels])
        bound_product = (cc.abs() + coord_error).unsqueeze(-1) * (ff.abs() + feature_error).unsqueeze(1)
        product_error = (
            cc.abs().unsqueeze(-1) * feature_error.unsqueeze(1) +
            ff.abs().unsqueeze(1) * coord_error.unsqueeze(-1) +
            coord_error.unsqueeze(-1) * feature_error.unsqueeze(1) + eps * bound_product)
        point_counts = torch.bincount(labels, minlength=len(native_cross)).double()
        assert torch.all(point_counts * eps < 1)
        gamma = point_counts * eps / (1 - point_counts * eps)
        envelope = scatter_mean(
            (product_error + (gamma[labels] + 2 * eps).view(-1, 1, 1) * bound_product).flatten(1),
            labels, dim=0)
        reference = scatter_mean((cc.unsqueeze(-1) * ff.unsqueeze(1)).flatten(1), labels, dim=0)
        envelope += torch.finfo(torch.double).eps * point_counts.unsqueeze(-1) * (
            scatter_mean(bound_product.flatten(1), labels, dim=0) + 1)
        native_error = (native_cross.detach().double().cpu() - reference).abs()
        assert torch.all(native_error <= envelope), {
            "native_error_max": float(native_error.max()), "envelope_max": float(envelope.max())}
        point_index = torch.arange(len(sp), device=sp.device)
        order = torch.argsort(sp.long() * (len(sp) + 1) + point_index)
        counts = torch.bincount(sp, minlength=native_cross.shape[0])
        starts = counts.cumsum(0) - counts
        positions = torch.arange(len(order), device=order.device)
        reversed_order = order[starts[sp[order]] + counts[sp[order]] - 1 - (positions - starts[sp[order]])]
        permutation = torch.empty_like(order)
        permutation[order] = reversed_order
        assert torch.equal(sp[permutation], sp)
        assert torch.equal(permutation.sort().values, positions)
        reconstructed = cross_stat(coords, features, sp)
        repeated = cross_stat(coords, features, sp)
        wrong = cross_stat(coords, features[permutation], sp)
        captured.update(
            native_raw=clone(native_cross), wrong_raw=clone(wrong),
            incidence={
                "points": len(features), "superpoints": len(native_cross),
                "axes": coords.shape[1], "channels": features.shape[1],
                "cross_nonzero_fraction": float((native_cross != 0).float().mean()),
                "native_reconstruction_max": float((native_cross - reconstructed).abs().max()),
                "independent_reconstruction_repeat_max": float((reconstructed - repeated).abs().max()),
                "native_cpu64_error_max": float(native_error.max()),
                "derived_float32_envelope_max": float(envelope.max()),
                "all_components_within_derived_envelope": True,
                "wrong_pairing_cross_max": float((native_cross - wrong).abs().max()),
                "paired_feature_mean_roundoff_max": float((scatter_mean(features, sp, dim=0) - scatter_mean(features[permutation], sp, dim=0)).abs().max()),
                "feature_multiset_preserved_by_within_sp_bijection": True,
            })
        return pooled, native_cross

    model.extract_feat = captured_extract
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
        for handle in hooks:
            handle.remove()
        native = captured["dec"][-1]
        assert native.shape[1] == 96 and torch.isfinite(native).all()
        raw = captured["native_raw"]
        repeats = native.shape[0] // raw.shape[0]
        assert repeats * raw.shape[0] == native.shape[0]
        assert torch.equal(raw.repeat(repeats, 1), native)
        captured["wrong_repeated"] = captured["wrong_raw"].repeat(repeats, 1)
        incidence = captured["incidence"]
        assert incidence["axes"] == 3 and incidence["channels"] == 32
        assert incidence["all_components_within_derived_envelope"]
        assert incidence["wrong_pairing_cross_max"] > max(
            incidence["native_reconstruction_max"], incidence["independent_reconstruction_repeat_max"])
        assert incidence["wrong_pairing_cross_max"] > 0 and incidence["cross_nonzero_fraction"] > 0
        state = rng()
        buffers = {k: v.detach().clone() for k, v in model.dec.named_buffers()}
        readings, saved = {}, {}
        for mode in ("disabled", "enabled", "repeat", "wrong_masks", "wrong_pairing", "rolled_cross", "zero_cross"):
            restore(state)
            with torch.no_grad():
                for name, value in model.dec.named_buffers():
                    value.copy_(buffers[name])
            model.zero_grad(set_to_none=True)
            pred = model.dec(*captured["dec"])
            loss, terms = model.criterion(pred, *captured["criterion"])
            loss.backward()
            grads = {k: float(p.grad.norm()) for k, p in new.items() if p.grad is not None}
            if mode not in ("disabled", "zero_cross"):
                assert len(grads) == 8 and all(v > 0 for v in grads.values()), (phase, mode, grads)
            elif mode == "disabled":
                assert not grads
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.dec.parameters() if p.grad is not None)
            readings[mode] = {"loss": float(loss), "terms": {k: float(v) for k, v in terms.items()}, "gradients": grads}
            saved[mode] = {k: pred[k].detach().clone() for k in ("masks", "indis")}
            del pred, loss, terms
        delta = lambda a, b, k: float((saved[a][k] - saved[b][k]).abs().max())
        row = {
            "phase": phase, "scan_id": batch["scan_ids"][0], "dataset_index": index,
            "incidence": incidence, "readings": readings,
            "nonindicator_deltas": {k: readings["enabled"]["terms"][k] - readings["disabled"]["terms"][k]
                                     for k in readings["enabled"]["terms"] if k != "loss" and "indi_loss" not in k},
            "masks_disabled_max": delta("enabled", "disabled", "masks"),
            "repeat_indicator_max": delta("enabled", "repeat", "indis"),
            **{name + "_max": delta("enabled", name, "indis") for name in ("wrong_masks", "wrong_pairing", "rolled_cross", "zero_cross")},
        }
        assert row["masks_disabled_max"] == row["repeat_indicator_max"] == 0
        assert min(row[name + "_max"] for name in ("wrong_masks", "wrong_pairing", "rolled_cross", "zero_cross")) > 0
        rows.append(row)
        q, masks, pad, mass, cross = clone(captured["readout"])
        mode = "direct"
        q.requires_grad_()
        masks.requires_grad_()
        cross.requires_grad_()
        direct = readout(q, masks, pad, mass, cross)
        direct.sum().backward()
        assert masks.grad is None and cross.grad is not None and torch.isfinite(cross.grad).all()
        with torch.no_grad():
            sp = torch.arange(mass.shape[-1] - 1, -1, -1, device=mass.device)
            support = readout(q, masks[:, :, sp], pad[:, sp], mass[:, sp], cross[:, sp])
            queries = readout(q.roll(1, 1), masks.roll(1, 1), pad, mass, cross).roll(-1, 1)
            poison = cross.detach().clone()
            poison[pad] = 100000.0
            poisoned = readout(q, masks, pad, mass, poison)
            controls.append({
                "phase": phase, "scan_id": batch["scan_ids"][0],
                "support_permutation": float((direct - support).abs().max()),
                "query_permutation": float((direct - queries).abs().max()),
                "padding_poison": float((direct - poisoned).abs().max()),
                "detached_mask_gradient": True, "cross_gradient_norm": float(cross.grad.norm()),
            })
        print(json.dumps({"phase": phase, "scan_id": row["scan_id"], "incidence": incidence,
                          "initial_loss_delta": readings["enabled"]["loss"] - readings["disabled"]["loss"]}), flush=True)
        captured.clear()
        del saved, buffers, batch, q, masks, cross, direct, support, queries, poisoned
    for handle in path_hooks:
        handle.remove()
    parameters = sum(p.numel() for p in model.parameters())
    del model, new, readout, original_extract
    gc.collect()
    torch.cuda.empty_cache()
nonindicator = max(abs(v) for r in rows for v in r["nonindicator_deltas"].values())
permutation = max(max(c["support_permutation"], c["query_permutation"]) for c in controls)
assert nonindicator == 0 and permutation < 1e-5
assert max(c["padding_poison"] for c in controls) == 0
protect()
result = {
    "complete": True, "id": "AdaRES-Y44", "rows": rows, "cached_controls": controls,
    "parameters": parameters, "added_parameters": parameters - 164629637, "added_trainable_tensors": 8,
    "minimum_added_gradient": min(v for r in rows for v in r["readings"]["enabled"]["gradients"].values()),
    "cached_repeat_max": max(r["repeat_indicator_max"] for r in rows),
    "nonindicator_loss_delta_max": nonindicator, "permutation_max": permutation,
    "loss_deltas": {phase: {mode: statistics.mean(r["readings"][mode]["loss"] - r["readings"]["disabled"]["loss"]
                                                  for r in rows if r["phase"] == phase)
                             for mode in ("enabled", "wrong_pairing", "rolled_cross", "zero_cross")}
                    for phase in ("trained", "released_init")},
    "protected_y42_y43_unchanged": True,
    "limitations": ["Native fixed TRAIN8/init4 mechanism/backward only, not trained efficacy or necessity.",
                    "Wrong feature pairing preserves native geometry and all feature multisets; upstream backbone is held fixed.",
                    "Every adverse initialized loss and zero-input outcome retained; physical optimizer/resume/resources separate."],
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
