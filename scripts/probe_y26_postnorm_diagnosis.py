"""Native TRAIN64 audit and cached qla postnorm mechanism controls."""

import hashlib
from pathlib import Path
import statistics

import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class

torch.set_num_threads(4)
assert Path(".codex_source_commit").read_text().strip() == "94918c1a008e21c72b9478c28f934e8b329e0c97"
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
indices, seen = [], set()
for index, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]["scene_id"] not in seen:
        seen.add(chunk[0]["scene_id"])
        indices.append(index)
    if len(indices) == 64:
        break
assert len(indices) == 64
cache, stages, mode = {}, [], "native"


def before_decoder(module, args):
    cache["args"] = tuple(v.detach().clone() if torch.is_tensor(v) else v for v in args)


def after_qla(module, args, output):
    value = output[0]
    normalized = module.norm(value)
    stages.append(dict(mean_abs=float(value.mean(-1).abs().mean()),
                       std_mean=float(value.std(-1, unbiased=False).mean()),
                       std_min=float(value.std(-1, unbiased=False).min()),
                       std_max=float(value.std(-1, unbiased=False).max()),
                       norm_difference=float((value-normalized).abs().max())))
    if mode == "connected":
        return (normalized, *output[1:])
    if mode == "rms":
        rms = (value.square().mean(-1, keepdim=True) + module.norm.eps).sqrt()
        return (value / rms, *output[1:])
    if mode == "identity":
        return (value, *output[1:])


def summarize(pred, batch):
    masks = pred["masks"].sigmoid().ge(.5)
    choice = pred["indis"].softmax(-1)[..., 1].argmax(-1)
    superpoints = batch["superpoints"].long().cuda()
    mass = torch.bincount(superpoints)
    positives = torch.stack([torch.bincount(superpoints[gt.cuda().bool()], minlength=mass.numel())
                             for gt in batch["gt_pmasks"]])
    assert positives.shape == (masks.shape[0], masks.shape[-1])
    intersection = (masks * positives.unsqueeze(1)).sum(-1)
    union = ((masks * mass).sum(-1) + positives.sum(-1).unsqueeze(1) - intersection).clamp_min(1)
    iou = intersection / union
    selected = iou.gather(1, choice.unsqueeze(1)).squeeze(1)
    return dict(selected_iou=selected.cpu().tolist(), oracle_iou=iou.max(-1).values.cpu().tolist(),
                selected_query=choice.cpu().tolist(), queries=masks.shape[1],
                finite=bool(torch.isfinite(pred["masks"]).all() and torch.isfinite(pred["indis"]).all()))


capture_hook = model.dec.register_forward_pre_hook(before_decoder)
qla_hooks = [layer.register_forward_hook(after_qla) for layer in model.dec.qla_layers]
rows = []
with torch.no_grad():
    for index in indices:
        batch = dataset.collate_fn([dataset[index]])
        cache.clear()
        stages.clear()
        mode = "native"
        model(batch, mode="loss")
        args = cache["args"]
        capture_hook.remove()
        readings, tensors = {}, {}
        state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        buffers = {name: value.detach().clone() for name, value in model.dec.named_buffers()}
        for mode in ("native", "repeat", "identity", "connected", "rms"):
            torch.set_rng_state(state[0])
            torch.cuda.set_rng_state_all(state[1])
            for name, value in model.dec.named_buffers():
                value.copy_(buffers[name])
            stages.clear()
            pred = model.dec(*args)
            assert len(stages) == 6
            readings[mode] = summarize(pred, batch)
            readings[mode]["qla_stages"] = list(stages)
            tensors[mode] = (pred["masks"].detach().clone(), pred["indis"].detach().clone())
            assert readings[mode]["finite"]
        row = dict(scan_id=str(batch["scan_ids"][0]), dataset_index=index,
                   expressions=len(batch["ann_ids"]), readings=readings,
                   repeat_mask_max=float((tensors["native"][0]-tensors["repeat"][0]).abs().max()),
                   identity_mask_max=float((tensors["native"][0]-tensors["identity"][0]).abs().max()),
                   connected_mask_max=float((tensors["native"][0]-tensors["connected"][0]).abs().max()),
                   connected_rms_mask_max=float((tensors["connected"][0]-tensors["rms"][0]).abs().max()))
        rows.append(row)
        del pred, tensors, args, batch, buffers
        cache.clear()
        capture_hook = model.dec.register_forward_pre_hook(before_decoder)
        if len(rows) % 8 == 0:
            print(json.dumps(dict(scans=len(rows))), flush=True)
capture_hook.remove()
for hook in qla_hooks:
    hook.remove()
scene_mean = lambda m: statistics.mean(statistics.mean(r["readings"][m]["selected_iou"]) for r in rows)
result = dict(complete=True, id="AdaRES-Y26", scans=64, expressions=sum(r["expressions"] for r in rows),
              rows=rows, source_sha256=hashlib.sha256(source).hexdigest(), checkpoint=checkpoint,
              scene_mean_point_iou={m: scene_mean(m) for m in readings},
              repeat_mask_max=max(r["repeat_mask_max"] for r in rows),
              identity_mask_max=max(r["identity_mask_max"] for r in rows),
              connected_mask_min=min(r["connected_mask_max"] for r in rows),
              connected_rms_mask_min=min(r["connected_rms_mask_max"] for r in rows),
              native_stage_mean_abs=[statistics.mean(r["readings"]["native"]["qla_stages"][i]["mean_abs"] for r in rows) for i in range(6)],
              native_stage_std=[statistics.mean(r["readings"]["native"]["qla_stages"][i]["std_mean"] for r in rows) for i in range(6)],
              norm_parameters=[dict(name=name, weight_min=float(layer.norm.weight.min()),
                                    weight_max=float(layer.norm.weight.max()), bias_max=float(layer.norm.bias.abs().max()))
                               for name, layer in enumerate(model.dec.qla_layers)],
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Frozen native augmented TRAIN64, not validation or trained candidate efficacy.",
                           "Cached decoder controls remove upstream variation, not proof of global determinism.",
                           "Exact point IoU from current augmented GT and superpoint point counts, not a new training target.",
                           "No parent file/weights/config mutation; qla hooks only; all other cross-attention unchanged."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
