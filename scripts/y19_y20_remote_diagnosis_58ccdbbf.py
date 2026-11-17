import os
import sys

CODE = "/bak_home/songcl/Project/AdaRES_project/AdaRES"
sys.path[:0] = [
    CODE, CODE + "/tools", CODE + "/ipdn/lib",
    "/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages",
    "/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg",
]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(CODE)

import ast
import fcntl
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import types
import numpy as np
import torch
import torch.nn.functional as F

assert subprocess.check_output(["hostname"]).decode().strip() == "dl01-MS-7C60"
lease = open("/bak_home/songcl/Project/AdaRES_project/.y15_gpu0_diagnostic.lock", "a")
fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
used, util = [int(v.strip()) for v in subprocess.check_output(
    ["nvidia-smi", "-i", "0", "--query-gpu=memory.used,utilization.gpu",
     "--format=csv,noheader,nounits"]).decode().strip().split(",")]
assert used < 1024 and util < 10
torch.set_num_threads(4)

# Execute the actual old objective, not a textual resemblance claim.
g1_path = Path("ipdn/model_paper5/adares_g1_metric_targets/loss.py")
tree = ast.parse(g1_path.read_text())
old = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "loss_indi")
module = ast.Module(body=[old], type_ignores=[])
namespace = {"torch": torch}
exec(compile(ast.fix_missing_locations(module), str(g1_path), "exec"), namespace)
criterion = types.SimpleNamespace(eos_coef=0.1)
masks = torch.tensor([[[5., -5., -5.], [5., 5., -5.], [-5., -5., 5.]]])
truth = torch.tensor([[1., 0., 0.]])
logits = torch.tensor([[[0.2, -0.1], [-0.3, 0.7], [0.8, -0.4]]], requires_grad=True)
indices = [(torch.tensor([0]), torch.tensor([0]))]
old_loss = namespace["loss_indi"](criterion, logits, masks, [{"masks": truth}], indices, torch.tensor([1.]))
old_grad = torch.autograd.grad(old_loss, logits)[0]
old_comparison = []
for masses in ([1, 1, 1], [1, 4, 2]):
    expanded = torch.repeat_interleave(torch.arange(3), torch.tensor(masses))
    binary = masks[0].sigmoid().ge(.5)[:, expanded]
    target = truth[0, expanded].bool()
    q = (binary & target).sum(-1).float() / (binary | target).sum(-1).clamp_min(1).float()
    soft = torch.stack([1 - q, q], -1).unsqueeze(0)
    loss = (-(soft * logits.log_softmax(-1)).sum(-1)
            + (soft * soft.clamp_min(1e-6).log()).sum(-1))
    loss = ((.1 + .9 * q) * loss).sum()
    grad = torch.autograd.grad(loss, logits)[0]
    old_comparison.append(dict(masses=masses, quality=q.tolist(),
        loss_difference=float(loss - old_loss), gradient_max_difference=float((grad - old_grad).abs().max())))

import gorilla
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class

gorilla.set_random_seed(1999)
config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(config)
data_cfg = cfg.data.train.copy()
data_cfg.lang_num_max = 4
dataset = build_dataset(data_cfg, get_root_logger())
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint_path = "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
loaded = model.load_state_dict(checkpoint["model"], strict=False)
assert not loaded.missing_keys
assert not set(loaded.unexpected_keys) - {
    "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
del checkpoint
model.train()
capture = {}
original_indicator = model.dec.indi_embedding.forward
original_indi_loss = model.criterion.loss_indi
original_mask_loss = model.criterion.loss_masks


def indicator(self, query, masks, pad, mass):
    capture["mask"] = masks
    capture["logits"] = original_indicator(query, masks.detach() if capture.get("detach") else masks, pad, mass)
    return capture["logits"]


def indi_loss(self, *args):
    result = original_indi_loss(*args)
    if "indi_loss" not in capture:
        capture["indi_loss"] = result
    return result


def mask_loss(self, *args):
    result = original_mask_loss(*args)
    if "mask_loss" not in capture:
        capture["mask_loss"] = result[0] + result[1]
    return result


model.dec.indi_embedding.forward = types.MethodType(indicator, model.dec.indi_embedding)
model.criterion.loss_indi = types.MethodType(indi_loss, model.criterion)
model.criterion.loss_masks = types.MethodType(mask_loss, model.criterion)
rows = []
seen = set()
for index in range(len(dataset)):
    batch = dataset.collate_fn([dataset[index]])
    scan = str(batch["scan_ids"][0])
    if scan in seen:
        continue
    seen.add(scan)
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    captures = []
    for mode in ("native", "repeat", "detached"):
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
        with torch.no_grad():
            for key, buf in model.named_buffers():
                buf.copy_(buffers[key])
        capture.clear()
        capture["detach"] = mode == "detached"
        model.zero_grad(set_to_none=True)
        loss, _ = model(batch, mode="loss")
        masks = capture["mask"]
        gi = torch.autograd.grad(capture["indi_loss"], masks, retain_graph=True, allow_unused=True)[0]
        gm = torch.autograd.grad(capture["mask_loss"], masks, retain_graph=True, allow_unused=True)[0]
        assert gm is not None and torch.isfinite(gm).all()
        normi = float(gi.norm()) if gi is not None else 0.
        normm = float(gm.norm())
        cosine = float((gi * gm).sum() / (gi.norm() * gm.norm()).clamp_min(1e-20)) if gi is not None else None
        active = (gi != 0) & (gm != 0) if gi is not None else torch.zeros_like(gm, dtype=torch.bool)
        opposite = float(((gi * gm)[active] < 0).float().mean()) if active.any() else None
        head = model.dec.indi_embedding.head[0].weight
        gh = torch.autograd.grad(capture["indi_loss"], head, retain_graph=True)[0]
        item = dict(loss=float(loss.detach()), indicator_loss=float(capture["indi_loss"].detach()),
            mask_loss=float(capture["mask_loss"].detach()), indicator_mask_norm=normi,
            segmentation_mask_norm=normm, cosine=cosine, opposite_fraction=opposite,
            finite=bool(torch.isfinite(gh).all()) and (gi is None or bool(torch.isfinite(gi).all())),
            indicator_head_norm=float(gh.norm()), logits=capture["logits"].detach().cpu())
        captures.append(item)
        del loss, gi, gm, gh, masks
        capture.clear()
    native, repeat, detached = captures
    assert detached["indicator_mask_norm"] == 0
    native_logits = native.pop("logits")
    logit_repeat = float((native_logits - repeat.pop("logits")).abs().max())
    logit_detached = float((native_logits - detached.pop("logits")).abs().max())
    rows.append(dict(scan_id=scan, dataset_index=index,
        expressions=len(batch["ann_ids"]), native=native, repeat=repeat, detached=detached,
        repeat_logit_max_difference=logit_repeat,
        detached_logit_max_difference=logit_detached,
        native_detached_loss_difference=abs(native["loss"] - detached["loss"])))
    del captures, batch, buffers
    if len(rows) % 8 == 0:
        print(json.dumps(dict(scans=len(rows))), flush=True)
    if len(rows) == 64:
        break

assert len(rows) == 64
cosines = [r["native"]["cosine"] for r in rows]
summary = dict(scans=len(rows), expressions=sum(r["expressions"] for r in rows),
    mean_gradient_cosine=statistics.mean(cosines), negative_scenes=sum(v < 0 for v in cosines),
    positive_scenes=sum(v > 0 for v in cosines),
    mean_opposite_fraction=statistics.mean(r["native"]["opposite_fraction"] for r in rows),
    maximum_repeat_logit_difference=max(r["repeat_logit_max_difference"] for r in rows),
    maximum_detached_logit_difference=max(r["detached_logit_max_difference"] for r in rows),
    maximum_native_detached_loss_difference=max(r["native_detached_loss_difference"] for r in rows),
    all_finite=all(r[m]["finite"] for r in rows for m in ("native", "repeat", "detached")),
    all_detached_mask_grad_zero=all(r["detached"]["indicator_mask_norm"] == 0 for r in rows),
    all_detached_head_grad_nonzero=all(r["detached"]["indicator_head_norm"] > 0 for r in rows),
    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
result = dict(candidate_ids=["AdaRES-Y19", "AdaRES-Y20"], manifest_sha256="__MANIFEST_DIGEST__",
    y19=dict(g1_source_sha256=hashlib.sha256(g1_path.read_bytes()).hexdigest(),
             executed_comparison=old_comparison,
             interpretation="Point-expanded objective equals actual G1 at equal point mass; unequal mass changes its target/weights, not the all-query soft-target mechanism."),
    y20=summary, rows=rows, checkpoint=checkpoint_path, complete=True,
    limitations=["TRAIN diagnostic only, one frozen checkpoint; not trained efficacy.",
                 "Gradients target final indicator mask route, not whole-parameter gradient interactions.",
                 "Repeated native training captures may have numeric noise; no tuning."])
Path("/tmp/adares_y19_y20_diagnosis_58ccdbbf.json").write_text(json.dumps(result))
print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
