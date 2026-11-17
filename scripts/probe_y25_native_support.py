"""Frozen native TRAIN cohort: hard-attention exclusion of matched GT mass."""

import hashlib
from pathlib import Path
import random
import statistics
import types

import gorilla
import numpy as np
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class

torch.set_num_threads(4)
assert Path(".codex_source_commit").read_text().strip() == "4a24410344b6ab6fe7f56d597013a4c13a9fa317"
source = Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py").read_bytes()
assert hashlib.sha256(source).hexdigest() == "6362d882662b0e51594cd4af994359b93f99497affc93600322a34ca6c349b01"
gorilla.set_random_seed(1999)
config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(config)
assert cfg.model.dec.attn_mask and cfg.model.criterion.match_last_layer
dc = cfg.data.train.copy()
dc.lang_num_max = 4
dataset = build_dataset(dc, get_root_logger())
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint = "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth"
ck = torch.load(checkpoint, map_location="cpu")
loaded = model.load_state_dict(ck["model"], strict=False)
assert not loaded.missing_keys and not set(loaded.unexpected_keys) - {
    "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
del ck
model.train()
cache = {}
old_mask = model.dec.get_mask
old_loss = model.criterion.loss_masks


def get_mask(self, *args):
    masks, gate = old_mask(*args)
    assert gate is not None and gate.dtype == torch.bool
    cache["stages"].append((masks.detach(), gate.detach()))
    return masks, gate


def loss_masks(self, masks, scores, targets, indices, count):
    if "target" not in cache:
        cache["target"] = (targets, indices)
    return old_loss(masks, scores, targets, indices, count)


model.dec.get_mask = types.MethodType(get_mask, model.dec)
model.criterion.loss_masks = types.MethodType(loss_masks, model.criterion)
indices = []
seen = set()
for i, chunk in enumerate(dataset.scene_inputs):
    scan = chunk[0]["scene_id"]
    if scan not in seen:
        seen.add(scan)
        indices.append(i)
    if len(indices) == 64:
        break
assert len(indices) == 64
rows = []
for index in indices:
    batch = dataset.collate_fn([dataset[index]])
    mass = torch.bincount(batch["superpoints"].long()).cuda()
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    captures = []
    for repeat in range(2):
        random.setstate(state[0]); np.random.set_state(state[1])
        torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])
        with torch.no_grad():
            for key, buf in model.named_buffers():
                buf.copy_(buffers[key])
            cache.clear(); cache["stages"] = []
            loss, _ = model(batch, mode="loss")
        assert len(cache["stages"]) == model.dec.num_layer + 1
        targets, matched = cache["target"]
        stage_rows = []
        for stage, (masks, gate) in enumerate(cache["stages"][:-1]):
            excluded, fallback = [], 0
            for b, (target, (iq, ig)) in enumerate(zip(targets, matched)):
                if not iq.numel():
                    continue
                gt = target["masks"][ig].gt(.5)
                assert gt.shape[1] == mass.numel() == gate.shape[-1]
                weighted = gt * mass.unsqueeze(0)
                fraction = (weighted * gate[b, iq]).sum(-1) / weighted.sum(-1).clamp_min(1)
                excluded.extend(fraction.cpu().tolist())
                raw_blocked = masks[b, iq].sigmoid().lt(.5).all(-1)
                fallback += int(raw_blocked.sum())
            assert excluded
            stage_rows.append(dict(stage=stage, positive_queries=len(excluded),
                                   mean_excluded_target_mass=statistics.mean(excluded),
                                   fully_locked_queries=sum(v == 1 for v in excluded),
                                   partly_locked_queries=sum(0 < v < 1 for v in excluded),
                                   fallback_queries=fallback))
        captures.append(stage_rows)
        del loss
    rows.append(dict(scan_id=str(batch["scan_ids"][0]), dataset_index=index,
                     expressions=len(batch["ann_ids"]), native=captures[0], repeat=captures[1],
                     repeat_max_exclusion_difference=max(abs(a["mean_excluded_target_mass"]-b["mean_excluded_target_mass"])
                                                        for a, b in zip(*captures))))
    cache.clear()
    del batch, buffers, mass
    if len(rows) % 8 == 0:
        print(json.dumps(dict(scans=len(rows))), flush=True)
result = dict(complete=True, id="AdaRES-Y25", config=config, checkpoint=checkpoint, seed=1999,
              source_sha256=hashlib.sha256(source).hexdigest(), rows=rows,
              scans=64, expressions=sum(r["expressions"] for r in rows),
              stage_scene_mean_excluded=[statistics.mean(r["native"][i]["mean_excluded_target_mass"] for r in rows)
                                         for i in range(model.dec.num_layer)],
              fully_locked_queries=sum(s["fully_locked_queries"] for r in rows for s in r["native"]),
              partly_locked_queries=sum(s["partly_locked_queries"] for r in rows for s in r["native"]),
              fallback_queries=sum(s["fallback_queries"] for r in rows for s in r["native"]),
              maximum_repeat_exclusion_difference=max(r["repeat_max_exclusion_difference"] for r in rows),
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Frozen TRAIN64/4 expressions per scene, not full population or validation efficacy.",
                           "Geometric final matching reused exactly as native match_last_layer; stages exclude unused final gate.",
                           "Exclusion alone does not prove continuous intervention is useful or admissible."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
