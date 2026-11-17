"""Frozen next GPU test: physical released-init training and exact state restore."""

import copy
import gc
import hashlib
import json
from pathlib import Path

import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import load_shape_compatible_pretrain, resolve_model_class, train
from probe_adares_y15_resume import cpu_state, differences

assert P["physical_gpu"] in (0, 1) and torch.cuda.device_count() == 1
torch.set_num_threads(4)
root = Path(P["root"])
config = "configs_song/model_paper5/adares_y46_component_coherence_res.yaml"
decoder_path = root / "ipdn/model_paper5/adares_y46_component_coherence/dec.py"
assert hashlib.sha256(decoder_path.read_bytes()).hexdigest() == "7dd952f18632392f5e5a050b4aabcfd10de5bcfdad9c81f494de353dfa56cf8f"
protected = {path: hashlib.sha256(path.read_bytes()).hexdigest()
             for name in ("adares_w2_pointmass_competition", "adares_y45_metric_decomposition",
                          "adares_y48_mask_equivalence", "adares_y46_component_coherence")
             for path in (root / "ipdn/model_paper5" / name).glob("*.py")}
cfg = gorilla.Config.fromfile(config)
assert cfg.train.epochs == 30 and cfg.train.seed == 1999
gorilla.set_random_seed(1999)
logger = get_root_logger()
dataset = build_dataset(cfg.data.train.copy(), logger)
indices = [0, 44, 60, 101]
assert len({dataset.scene_inputs[index][0]["scene_id"] for index in indices}) == 4
batches = [dataset.collate_fn([dataset[index] for index in indices[offset:offset + 2]])
           for offset in (0, 2)]
model = resolve_model_class(config)(**cfg.model).cuda()
load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
assert sum(value.numel() for value in model.parameters()) == 164629637 + 328961
optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
scheduler = gorilla.build_lr_scheduler(optimizer, copy.deepcopy(cfg.lr_scheduler))
head = model.dec.mask_component_readout
before = {name: value.detach().cpu().clone() for name, value in head.named_parameters()}
assert len(before) == 8
records = []
original_step = optimizer.step


def audited_step(*args, **kwargs):
    gradients = {name: float(value.grad.norm()) for name, value in head.named_parameters()
                 if value.grad is not None}
    assert len(gradients) == 8 and all(value > 0 for value in gradients.values())
    assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)
    attachment = model.dec._component_attachment
    assert attachment.enabled and attachment.capture["calls"] == model.dec.num_layer + 1
    records.append(dict(gradients=gradients, final_indicator_calls=attachment.capture["calls"]))
    return original_step(*args, **kwargs)


class Batches:
    sampler = None

    def __len__(self):
        return len(batches)

    def __iter__(self):
        return iter(batches)


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


optimizer.step = audited_step
cfg.dist = False
train(1, model, Batches(), optimizer, scheduler, cfg, logger, Writer())
assert len(records) == 2
updates = {name: float((value.detach().cpu() - before[name]).abs().max())
           for name, value in head.named_parameters()}
assert len(updates) == 8 and all(value > 0 for value in updates.values())
optimizer.step = original_step
checkpoint = Path(P["output"]).with_suffix(".pth")
assert not checkpoint.exists()
gorilla.save_checkpoint(model, str(checkpoint), optimizer, scheduler, dict(epoch=1, best_miou=0.0))
saved = torch.load(str(checkpoint), map_location="cpu")
assert all(key in saved for key in ("model", "optimizer", "scheduler", "meta"))
del model, optimizer, scheduler, head, original_step
gc.collect()
torch.cuda.empty_cache()
restored = resolve_model_class(config)(**cfg.model).cuda()
loaded = restored.load_state_dict(saved["model"], strict=False)
allowed = {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
assert not loaded.missing_keys and not set(loaded.unexpected_keys) - allowed
state = cpu_state(restored)
for key in saved["model"].keys() - state.keys():
    assert key in allowed
    buffer = restored
    for part in key.split("."):
        buffer = getattr(buffer, part)
    assert torch.equal(buffer.detach().cpu(), saved["model"][key].cpu())
    state[key] = buffer.detach().cpu().clone()
assert differences(saved["model"], state)["equal"]
optimizer = gorilla.build_optimizer(restored, copy.deepcopy(cfg.optimizer))
scheduler = gorilla.build_lr_scheduler(optimizer, copy.deepcopy(cfg.lr_scheduler))
optimizer.load_state_dict(saved["optimizer"])
scheduler.load_state_dict(saved["scheduler"])
assert differences(saved["optimizer"], optimizer.state_dict())["equal"]
assert differences(saved["scheduler"], scheduler.state_dict())["equal"]
assert restored.dec._component_attachment.enabled
assert restored.dec._component_attachment.prototype is restored.dec.mask_component_readout
assert all(hashlib.sha256(path.read_bytes()).hexdigest() == sha for path, sha in protected.items())
result = dict(
    complete=True, id="AdaRES-Y46", actual_tools_train_steps=2,
    released_backbone=cfg.train.pretrain, manifest_indices=indices,
    added_parameters=328961, new_parameter_tensors=8, step_records=records,
    new_parameter_updates=updates, model_optimizer_scheduler_restore_exact=True,
    production_hook_registration_after_restore=True, protected_source_hashes_unchanged=True,
    implementation_revision=1,
    scientific_status="production_native_optimizer_restore_test_not_resource_admission",
    limitations=[
        "Temporary diagnostic model/optimizer/checkpoint only; never deploy these weights.",
        "Four frozen TRAIN scenes and two actual tools.train steps do not establish trained efficacy.",
        "Checkpoint values restored exactly; no full-loader bitwise continuation claim.",
        "Matched TRAIN8 controls, serial inference, largest-pair stress and six-gate review remain required.",
        "Original and cached runtime costs remain unresolved; no resource/Pareto PASS."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
