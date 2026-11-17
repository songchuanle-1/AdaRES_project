"""Fixed two new-only native updates; retain adverse and wrong-control results."""

import copy
import gc
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


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(clone(v) for v in value)
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    return value


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
    initial = {name: p.detach().clone() for name, p in new.items()}
    model.train()
    capture = {}
    mode = "enabled"

    def dec_pre(module, inputs):
        capture["dec"] = clone(inputs)
        capture["rng"] = rng()

    def criterion_pre(module, inputs):
        capture["args"] = clone(inputs[1:])
        capture["positive"] = inputs[0]["point_positive"].detach().clone()

    def readout_pre(module, inputs):
        if mode == "wrong_masks":
            q, masks, pad, mass, features = inputs
            return q, masks.roll(1, 1), pad, mass, features

    original = model.criterion.loss_decomposition

    def loss_decomposition(self, pred):
        if mode == "collapsed":
            with torch.no_grad():
                valid = (~pred["batch_mask"]).float()
                positive = pred["point_positive"] * valid
                binary = pred["masks"].detach().gt(0).float()
                tp = torch.einsum("bqn,bn->bq", binary, positive)
                union = torch.einsum("bqn,bn->bq", binary, pred["point_mass"] * valid) + positive.sum(-1, keepdim=True) - tp
                quality = tp / union.clamp_min(1)
            logits = pred["error_logits"]
            collapsed = torch.stack((torch.logsumexp(logits[..., 1:], -1), logits[..., 0]), -1)
            return -(torch.stack((1 - quality, quality), -1) * collapsed.log_softmax(-1)).sum(-1).mean()
        return original(pred)

    model.criterion.loss_decomposition = types.MethodType(loss_decomposition, model.criterion)
    path_hook = model.dec.metric_readout.register_forward_pre_hook(readout_pre)
    for index in indices[:8 if phase == "trained" else 4]:
        gorilla.set_random_seed(1999)
        batch = dataset.collate_fn([dataset[index]])
        mode = "enabled"
        hooks = [model.dec.register_forward_pre_hook(dec_pre),
                 model.criterion.register_forward_pre_hook(criterion_pre)]
        with torch.no_grad():
            model(batch, mode="loss")
        for handle in hooks:
            handle.remove()
        inputs, state, positive, args = capture["dec"], capture["rng"], capture["positive"], capture["args"]

        def evaluate():
            restore(state)
            pred = model.dec(*inputs)
            pred["point_positive"] = positive.roll(1, 0) if mode == "wrong_truth" else positive
            return model.criterion(pred, *args)

        controls = {}
        for mode in ("enabled", "wrong_truth", "collapsed", "wrong_masks"):
            with torch.no_grad():
                for name, parameter in new.items():
                    parameter.copy_(initial[name])
                before, loss_dict = evaluate()
                before_total, before_new = float(before), float(loss_dict["decomposition_loss"])
            optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
            gradients = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = evaluate()
                loss.backward()
                assert torch.isfinite(loss)
                norms = {name: float(p.grad.norm()) for name, p in new.items()}
                assert all(v > 0 for v in norms.values())
                assert all(torch.isfinite(p.grad).all() for p in new.values())
                gradients.append(norms)
                torch.nn.utils.clip_grad_norm_(new.values(), cfg.train.grad_clip)
                optimizer.step()
            with torch.no_grad():
                after, loss_dict = evaluate()
            controls[mode] = dict(before_total=before_total, after_total=float(after),
                                  before_new_likelihood=before_new,
                                  after_new_likelihood=float(loss_dict["decomposition_loss"]),
                                  legacy_loss_change=(float(after) - float(loss_dict["decomposition_loss"])) - (before_total - before_new),
                                  gradient_records=gradients)
            del optimizer, loss, before, after, loss_dict
        rows.append(dict(phase=phase, dataset_index=index, scan_id=batch["scan_ids"][0], controls=controls))
        del inputs, args, positive, batch
        capture.clear()
        gc.collect()
        torch.cuda.empty_cache()
    path_hook.remove()
    model.criterion.loss_decomposition = original
    del model, initial, new
    gc.collect()
    torch.cuda.empty_cache()
result = dict(
    complete=True, id="AdaRES-Y45", rows=rows, trained_scenes=8, released_init_scenes=4,
    fixed_new_only_steps_per_control=2, controls=["enabled", "wrong_truth", "collapsed", "wrong_masks"],
    new_trainable_tensors=4, native_queries=128, diagnostic_weights_deployed=False,
    scientific_status="initial_adverse_loss_investigation_only",
    limitations=["Cached actual native TRAIN decoder/criterion inputs with captured RNG; no threshold or loss-weight search.",
                 "Wrong controls may learn similarly; these tiny within-scene updates do not prove mechanism necessity or benefit.",
                 "The categorical likelihood adds a different objective; total-loss comparisons alone are not efficacy.",
                 "Only new tensors optimized; no original checkpoint/source/config changed.",
                 "Full-input bitwise repeat remains unproved and its original failure is retained."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
