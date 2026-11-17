"""Paired native TRAIN controls for mask-to-attention feedback."""

import copy
import hashlib
from pathlib import Path
import random
import statistics
import types

import gorilla
import numpy as np
import torch
import torch.nn.functional as F
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain

torch.set_num_threads(4)
source = Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py").read_bytes()
assert hashlib.sha256(source).hexdigest() == "6362d882662b0e51594cd4af994359b93f99497affc93600322a34ca6c349b01"
config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(config)
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint = "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth"
assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == "e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52"
ck = torch.load(checkpoint, map_location="cpu")
loaded = model.load_state_dict(ck["model"], strict=False)
assert not loaded.missing_keys and not set(loaded.unexpected_keys) - {
    "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
del ck
model.train()
old_mask = model.dec.get_mask
cache = {}
mode = "hard"


def get_mask(self, *args):
    masks, gate = old_mask(*args)
    if mode != "hard":
        gate = F.logsigmoid(masks).masked_fill(args[2].unsqueeze(1), float("-inf"))
        if mode == "detach":
            gate = gate.detach()
        elif mode == "full":
            gate = torch.zeros_like(masks).masked_fill(args[2].unsqueeze(1), float("-inf"))
    cache["masks"].append(masks)
    cache["gates"].append(gate)
    return masks, gate


model.dec.get_mask = types.MethodType(get_mask, model.dec)
indices, seen = [], set()
for i, chunk in enumerate(dataset.scene_inputs):
    scan = chunk[0]["scene_id"]
    if scan not in seen:
        seen.add(scan); indices.append(i)
    if len(indices) == 8:
        break
assert len(indices) == 8
rows = []
for phase in ("trained", "released_init"):
    if phase == "released_init":
        del model
        gorilla.set_random_seed(1999)
        model = resolve_model_class(config)(**cfg.model).cuda()
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
        model.train()
        old_mask = model.dec.get_mask
        model.dec.get_mask = types.MethodType(get_mask, model.dec)
    for index in indices[:8 if phase == "trained" else 4]:
        batch = dataset.collate_fn([dataset[index]])
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        readings, saved = {}, {}
        for mode in ("hard", "soft", "detach", "full", "soft_repeat"):
            random.setstate(state[0]); np.random.set_state(state[1])
            torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])
            with torch.no_grad():
                for name, buf in model.named_buffers():
                    buf.copy_(buffers[name])
            model.zero_grad(set_to_none=True)
            cache.clear(); cache.update(masks=[], gates=[])
            loss, terms = model(batch, mode="loss")
            assert torch.isfinite(loss)
            output = cache["masks"][-1].detach().cpu()
            if mode in ("soft", "soft_repeat"):
                grads = torch.autograd.grad(loss, cache["gates"][:-1], retain_graph=True)
                path = [dict(norm=float(g.norm()), finite=bool(torch.isfinite(g).all()),
                             nonzero=int(torch.count_nonzero(g))) for g in grads]
                assert all(p["finite"] and p["norm"] > 0 for p in path)
            else:
                path = []
            loss.backward()
            decoder = {name: p.grad.detach().cpu().clone() for name, p in model.dec.named_parameters()
                       if p.grad is not None}
            assert decoder and all(torch.isfinite(g).all() for g in decoder.values())
            readings[mode] = dict(loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                                  gate_gradients=path, decoder_gradients=len(decoder),
                                  decoder_gradients_nonzero=sum(bool(g.abs().max()) for g in decoder.values()))
            saved[mode] = (output, decoder)
            del loss, terms
            cache.clear()
        gradient_delta = lambda a, b: max(float((saved[a][1][k]-saved[b][1][k]).abs().max())
                                          for k in saved[a][1])
        row = dict(phase=phase, scan_id=str(batch["scan_ids"][0]), dataset_index=index,
                   expressions=len(batch["ann_ids"]), readings=readings,
                   soft_detach_forward_max=float((saved["soft"][0]-saved["detach"][0]).abs().max()),
                   soft_repeat_forward_max=float((saved["soft"][0]-saved["soft_repeat"][0]).abs().max()),
                   soft_detach_gradient_max=gradient_delta("soft", "detach"),
                   soft_repeat_gradient_max=gradient_delta("soft", "soft_repeat"),
                   soft_full_forward_max=float((saved["soft"][0]-saved["full"][0]).abs().max()))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "readings"}), flush=True)
        del saved, batch, buffers
result = dict(complete=True, id="AdaRES-Y25", rows=rows, trained_scans=8, released_init_scans=4,
              source_sha256=hashlib.sha256(source).hexdigest(),
              soft_detach_forward_max=max(r["soft_detach_forward_max"] for r in rows),
              soft_repeat_forward_max=max(r["soft_repeat_forward_max"] for r in rows),
              soft_detach_gradient_min=min(r["soft_detach_gradient_max"] for r in rows),
              soft_repeat_gradient_max=max(r["soft_repeat_gradient_max"] for r in rows),
              parameters=sum(p.numel() for p in model.parameters()), added_parameters=0,
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Prototype changes only the native TRAIN gate, not yet a physical package/admission.",
                           "Finite nonzero native gate gradients do not prove their direction is useful.",
                           "Loss shifts, full-valid differences and repeat noise retained without task-benefit threshold.",
                           "Frozen first8 distinct TRAIN scans, first4 released initialization; no optimization or validation tuning."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
