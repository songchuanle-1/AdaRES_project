"""Native matched-label/categorical gradient attribution on the frozen cohort."""

import gc
import hashlib
import json
from pathlib import Path
import types
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain
from probe_adares_y15_resume import rng, restore

assert P["physical_gpu"] == 1
torch.set_num_threads(4)
config = "configs_song/model_paper5/adares_y45_metric_decomposition_res.yaml"
cfg = gorilla.Config.fromfile(config)
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
indices = [0, 44, 60, 101, 115, 126, 137, 171]
assert len({dataset.scene_inputs[i][0]["scene_id"] for i in indices}) == 8
sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
           for name in ("adares_w2_pointmass_competition", "adares_y45_metric_decomposition",
                        "adares_y43_point_asymmetry", "adares_y44_coordinate_feature")
           for p in (Path("ipdn/model_paper5") / name).glob("*.py")}
sources[config] = hashlib.sha256(Path(config).read_bytes()).hexdigest()


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(clone(v) for v in value)
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    return value


def vector(values):
    return torch.cat([v.detach().reshape(-1).double().cpu() for v in values])


def relationship(left, right):
    left, right = left.double(), right.double()
    ln, rn = float(left.norm()), float(right.norm())
    dot = float((left * right).sum())
    return dict(dot=dot, left_norm=ln, right_norm=rn,
                cosine=dot / (ln * rn) if ln and rn else None,
                opposite=dot < 0)


def targets(pred):
    with torch.no_grad():
        valid = (~pred["batch_mask"]).float()
        masks = pred["masks"].detach().gt(0).float()
        positive = pred["point_positive"] * valid
        negative = (pred["point_mass"] - positive) * valid
        tp = torch.einsum("bqn,bn->bq", masks, positive)
        fp = torch.einsum("bqn,bn->bq", masks, negative)
        fn = positive.sum(-1, keepdim=True) - tp
        triple = torch.stack((tp, fp, fn), -1)
        return triple / triple.sum(-1, keepdim=True).clamp_min(1)


rows = []
for phase in ("trained", "released_init"):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    new = {name: p for name, p in model.named_parameters() if ".metric_readout." in name}
    assert len(new) == 4
    if phase == "trained":
        checkpoint = torch.load("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth", map_location="cpu")
        loaded = model.load_state_dict(checkpoint["model"], strict=False)
        assert set(loaded.missing_keys) == set(new)
        del checkpoint
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in new)
    model.train()
    captured = {}

    def before_decoder(module, inputs):
        captured["dec"] = clone(inputs)
        captured["rng"] = rng()

    def before_criterion(module, inputs):
        captured["args"] = clone(inputs[1:])
        captured["positive"] = inputs[0]["point_positive"].detach().clone()

    original_indi = model.criterion.loss_indi

    def capture_indi(self, logits, gt, indices, boxes):
        captured.setdefault("matched", []).append(dict(
            indices=[(a.detach().clone(), b.detach().clone()) for a, b in indices],
            boxes=boxes.detach().clone()))
        return original_indi(logits, gt, indices, boxes)

    model.criterion.loss_indi = types.MethodType(capture_indi, model.criterion)
    for index in indices[:8 if phase == "trained" else 4]:
        gorilla.set_random_seed(1999)
        batch = dataset.collate_fn([dataset[index]])
        captured.clear()
        hooks = [model.dec.register_forward_pre_hook(before_decoder),
                 model.criterion.register_forward_pre_hook(before_criterion)]
        with torch.no_grad():
            model(batch, mode="loss")
        for handle in hooks:
            handle.remove()
        inputs, state = captured["dec"], captured["rng"]
        positive, args = captured["positive"], captured["args"]
        controls = {}
        first_vectors = None
        for mode in ("correct", "wrong_truth", "collapsed", "repeat"):
            restore(state)
            pred = model.dec(*inputs)
            pred["point_positive"] = positive.roll(1, 0) if mode == "wrong_truth" else positive
            captured["matched"] = []
            total, losses = model.criterion(pred, *args)
            native_new = losses["decomposition_loss"]
            legacy = total - native_new
            q = targets(pred)
            logits = pred["error_logits"]
            collapsed_logits = torch.stack((torch.logsumexp(logits[..., 1:], -1), logits[..., 0]), -1)
            binary_loss = -(torch.stack((1 - q[..., 0], q[..., 0]), -1) *
                            collapsed_logits.log_softmax(-1)).sum(-1).mean()
            conditional_targets = q[..., 1:] / q[..., 1:].sum(-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
            conditional = -((1 - q[..., 0]) *
                            (conditional_targets * logits[..., 1:].log_softmax(-1)).sum(-1)).mean()
            factorized = binary_loss + conditional
            gp_native = torch.autograd.grad(native_new, tuple(new.values()), retain_graph=True)
            gp_legacy = torch.autograd.grad(legacy, tuple(new.values()), retain_graph=True)
            gp_binary = torch.autograd.grad(binary_loss, tuple(new.values()), retain_graph=True)
            gz_native = torch.autograd.grad(native_new, logits, retain_graph=True)[0]
            gz_legacy = torch.autograd.grad(legacy, logits, retain_graph=True)[0]
            gz_binary = torch.autograd.grad(binary_loss, logits, retain_graph=True)[0]
            gz_factorized = torch.autograd.grad(factorized, logits, retain_graph=True)[0]
            assert all(torch.isfinite(v).all() for v in (*gp_native, *gp_legacy, gz_native, gz_legacy))
            info = captured["matched"][0]
            matched = torch.zeros(logits.shape[:2], device=logits.device)
            for b, (queries, _) in enumerate(info["indices"]):
                matched[b, queries] = 1
            p = logits.softmax(-1)
            analytic_new = (p - q) / matched.numel()
            indicator_probability = pred["indis"].log_softmax(-1).exp()[..., 1]
            weights = torch.where(matched.bool(), torch.ones_like(matched),
                                  torch.full_like(matched, model.criterion.eos_coef))
            du = ((indicator_probability - matched) * weights / info["boxes"] *
                  model.criterion.loss_weight[4])
            ratio = logits[..., 1:].softmax(-1)
            analytic_legacy = du.unsqueeze(-1) * torch.cat((torch.ones_like(du).unsqueeze(-1), -ratio), -1)
            eps = 8 * torch.finfo(logits.dtype).eps
            errors = dict(categorical=float((gz_native - analytic_new).abs().max()),
                          matched=float((gz_legacy - analytic_legacy).abs().max()),
                          factorization_value=abs(float(native_new - factorized)),
                          factorization_gradient=float((gz_native - gz_factorized).abs().max()))
            scales = dict(categorical=max(1., float(gz_native.abs().max())),
                          matched=max(1., float(gz_legacy.abs().max())),
                          factorization_value=max(1., abs(float(native_new)), abs(float(factorized))),
                          factorization_gradient=max(1., float(gz_native.abs().max())))
            assert all(errors[k] <= eps * scales[k] for k in errors), errors
            selected_parameter_gradients = gp_binary if mode == "collapsed" else gp_native
            selected_logit_gradients = gz_binary if mode == "collapsed" else gz_native
            left, right = vector(gp_legacy), vector(selected_parameter_gradients)
            per_query_dot = (gz_legacy.double() * selected_logit_gradients.double()).sum(-1)
            groups = {}
            for key, mask in (("matched", matched.bool()), ("unmatched", ~matched.bool())):
                groups[key] = dict(
                    queries=int(mask.sum()), opposing=int((per_query_dot[mask] < 0).sum()),
                    dot_sum=float(per_query_dot[mask].sum()),
                    target_iou_mean=float(q[..., 0][mask].mean()) if mask.any() else None,
                    indicator_probability_mean=float(indicator_probability[mask].mean()) if mask.any() else None)
            repeat_error = None
            if mode == "correct":
                first_vectors = (left.clone(), right.clone(), gz_native.detach().clone())
            elif mode == "repeat":
                repeat_error = max(float((left - first_vectors[0]).abs().max()),
                                   float((right - first_vectors[1]).abs().max()),
                                   float((gz_native - first_vectors[2]).abs().max()))
                assert repeat_error == 0
            controls[mode] = dict(
                legacy_loss=float(legacy), native_categorical_loss=float(native_new),
                collapsed_loss=float(binary_loss), conditional_FP_FN_loss=float(conditional),
                parameter_gradients=relationship(left, right),
                per_tensor={name: relationship(a.detach().double().cpu(), b.detach().double().cpu())
                            for name, a, b in zip(new, gp_legacy, selected_parameter_gradients)},
                query_groups=groups, analytic_errors=errors, operation_count8_eps=eps,
                repeat_gradient_max=repeat_error)
            del total, losses, pred, native_new, legacy, logits, factorized, binary_loss, conditional
            del gp_native, gp_legacy, gp_binary, gz_native, gz_legacy, gz_binary, gz_factorized, left, right
        rows.append(dict(phase=phase, dataset_index=index, scan_id=batch["scan_ids"][0], controls=controls))
        del inputs, positive, args, batch, first_vectors
        captured.clear()
        gc.collect()
        torch.cuda.empty_cache()
    model.criterion.loss_indi = original_indi
    del model, new
    gc.collect()
    torch.cuda.empty_cache()
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in sources.items())
summary = {
    phase: {mode: dict(
        scenes=sum(r["phase"] == phase for r in rows),
        opposing_parameter_scenes=sum(r["controls"][mode]["parameter_gradients"]["opposite"]
                                     for r in rows if r["phase"] == phase),
        parameter_cosines=[r["controls"][mode]["parameter_gradients"]["cosine"]
                           for r in rows if r["phase"] == phase],
        matched_opposing_queries=sum(r["controls"][mode]["query_groups"]["matched"]["opposing"]
                                     for r in rows if r["phase"] == phase),
        unmatched_opposing_queries=sum(r["controls"][mode]["query_groups"]["unmatched"]["opposing"]
                                       for r in rows if r["phase"] == phase))
            for mode in ("correct", "wrong_truth", "collapsed")}
    for phase in ("trained", "released_init")}
result = dict(
    complete=True, id="AdaRES-Y45", rows=rows, summary=summary,
    trained_scenes=8, released_init_scenes=4, source_hashes=sources,
    scientific_status="local_native_gradient_attribution_not_efficacy",
    limitations=["Fixed cached native TRAIN inputs, not population or training-seed uncertainty.",
                 "Opposition between distinct supervised objectives is local tradeoff, not automatic efficacy/harm/PASS.",
                 "Exact factorization tests add no weights, tune no thresholds and modify no source.",
                 "No optimizer updates, checkpoint writes, new training or global bitwise claim."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
