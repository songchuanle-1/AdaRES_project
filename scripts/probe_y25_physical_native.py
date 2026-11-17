"""Actual physical Y25 native training entry, resume and cached eval identity."""

import copy
import gc
import hashlib
from pathlib import Path
import types

import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain, train
from probe_adares_y15_resume import rng, restore, differences, cpu_state

torch.set_num_threads(4)
gorilla.set_random_seed(1999)
config = "configs_song/model_paper5/adares_y25_mask_feedback_res.yaml"
parent = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(config)
logger = get_root_logger()
dataset = build_dataset(cfg.data.train.copy(), logger)
indices, seen = [], set()
for i, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]["scene_id"] not in seen:
        seen.add(chunk[0]["scene_id"]); indices.append(i)
    if len(indices) == 4:
        break
batches = [dataset.collate_fn([dataset[i] for i in indices[j:j+2]]) for j in (0, 2)]
model = resolve_model_class(config)(**cfg.model).cuda()
load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
opt = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
sched = gorilla.build_lr_scheduler(opt, copy.deepcopy(cfg.lr_scheduler))
parameters = sum(p.numel() for p in model.parameters())
gates, records = [], []
original_mask = model.dec.get_mask
original_step = opt.step


def audit_mask(self, *args):
    masks, gate = original_mask(*args)
    if self.training:
        assert gate.is_floating_point() and gate.requires_grad
        gate.retain_grad(); gates.append(gate)
    return masks, gate


def audit_step(*args, **kwargs):
    path = [dict(norm=float(g.grad.norm()), finite=bool(torch.isfinite(g.grad).all()))
            for g in gates[:-1]]
    assert len(path) == 6 and all(v["finite"] and v["norm"] > 0 for v in path)
    gradients = [p.grad for p in model.dec.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    records.append(dict(path=path, decoder_gradient_tensors=len(gradients),
                        decoder_nonzero=sum(bool(g.abs().max()) for g in gradients)))
    gates.clear()
    return original_step(*args, **kwargs)


model.dec.get_mask = types.MethodType(audit_mask, model.dec)
opt.step = audit_step


class Batches:
    sampler = None
    def __len__(self): return len(batches)
    def __iter__(self): return iter(batches)


class Writer:
    def add_scalar(self, *args, **kwargs): pass


cfg.dist = False
before = cpu_state(model)
train(1, model, Batches(), opt, sched, cfg, logger, Writer())
assert len(records) == 2
change = differences(before, cpu_state(model))
assert not change["equal"] and change["max"] > 0
del before
model.dec.get_mask = original_mask
opt.step = original_step
checkpoint = str(Path(P["output"]).with_suffix(".pth"))
assert not Path(checkpoint).exists()
gorilla.save_checkpoint(model, checkpoint, opt, sched, dict(epoch=1, best_miou=0.0))
saved = torch.load(checkpoint, map_location="cpu")
state = rng()


def step():
    model.train(); opt.zero_grad(set_to_none=True)
    loss, _ = model(batches[0], mode="loss")
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
    opt.step()
    return float(loss)


continued = step(); final = cpu_state(model); final_opt = copy.deepcopy(opt.state_dict())
del model, opt, sched
gc.collect(); torch.cuda.empty_cache()
model = resolve_model_class(config)(**cfg.model).cuda()
opt = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
sched = gorilla.build_lr_scheduler(opt, copy.deepcopy(cfg.lr_scheduler))
meta = gorilla.resume(model=model, filename=checkpoint, optimizer=opt, scheduler=sched,
                      strict=False, map_location="cpu")
allowed = {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
resume = dict(model=differences({k: v for k, v in saved["model"].items() if k not in allowed}, cpu_state(model)),
              optimizer=differences(saved["optimizer"], opt.state_dict()),
              scheduler=differences(saved["scheduler"], sched.state_dict()), epoch=int(meta["epoch"]))
assert all(resume[k]["equal"] for k in ("model", "optimizer", "scheduler"))
restore(state); resumed = step()
continuation = dict(loss_difference=resumed-continued, model=differences(final, cpu_state(model)),
                    optimizer=differences(final_opt, opt.state_dict()))
del final, final_opt, saved, model, opt, sched, original_step, original_mask
gc.collect(); torch.cuda.empty_cache()

# Cache native decoder inputs once; both physical decoders use identical values.
dc = cfg.data.train.copy(); dc.training=False; dc.aug=False; dc.with_elastic=False
evaluation = build_dataset(dc, logger)
model = resolve_model_class(config)(**cfg.model).cuda()
ck = torch.load("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth", map_location="cpu")
loaded = model.load_state_dict(ck["model"], strict=False)
assert not loaded.missing_keys and not set(loaded.unexpected_keys)-allowed
model.eval()
capture, cached, inference_rows = {}, [], []


def prehook(module, args):
    capture["args"] = tuple(v.detach().clone() if torch.is_tensor(v) else v for v in args)


hook = model.dec.register_forward_pre_hook(prehook)
with torch.no_grad():
    for index in indices:
        batch = evaluation.collate_fn([evaluation[index]])
        capture.clear(); gorilla.set_random_seed(1999)
        model(batch, mode="predict")
        args = capture["args"]
        hook.remove()
        a = model.dec(*args); b = model.dec(*args)
        cached.append((args, {k: a[k].detach().clone() for k in ("masks", "indis", "sample_inds")}))
        inference_rows.append(dict(scan_id=batch["scan_ids"][0],
                                   cached_repeat_mask_max=float((a["masks"]-b["masks"]).abs().max())))
        hook = model.dec.register_forward_pre_hook(prehook)
hook.remove()
del model
gc.collect(); torch.cuda.empty_cache()
parent_model = resolve_model_class(parent)(**cfg.model).cuda()
loaded = parent_model.load_state_dict(ck["model"], strict=False)
assert not loaded.missing_keys and not set(loaded.unexpected_keys)-allowed
parent_model.eval()
with torch.no_grad():
    for row, (args, candidate) in zip(inference_rows, cached):
        value = parent_model.dec(*args)
        row.update(parent_mask_max=float((candidate["masks"]-value["masks"]).abs().max()),
                   parent_indicator_max=float((candidate["indis"]-value["indis"]).abs().max()),
                   parent_sample_mismatches=int((candidate["sample_inds"]!=value["sample_inds"]).sum()))
assert all(r["cached_repeat_mask_max"] == r["parent_mask_max"] == r["parent_indicator_max"] ==
           r["parent_sample_mismatches"] == 0 for r in inference_rows)
result = dict(complete=True, id="AdaRES-Y25", parameters=parameters, added_parameters=0,
              native_optimizer_steps=2, native_gradient_records=records, model_change=change,
              resume_before_continuation=resume, continuation=continuation,
              cached_inference_rows=inference_rows, cached_eval_identity=True,
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Native two-step smoke test, not production admission/trained efficacy.",
                           "Exact model/AdamW/PolyLR restoration before continuation, next-step numeric differences retained.",
                           "No whole-dataloader/RNG trajectory or global inference determinism claim.",
                           "Cached four-scan native decoder identity separates source identity from upstream sample noise."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
