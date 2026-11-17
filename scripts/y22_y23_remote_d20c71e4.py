import os
import sys

ROOT = "/bak_home/songcl/Project/AdaRES_project/AdaRES"
sys.path[:0] = [ROOT, ROOT + "/tools", ROOT + "/ipdn/lib",
    "/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages",
    "/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg"]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(ROOT)

import ast
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import textwrap
import types
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
import gorilla
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class

gorilla.set_random_seed(1999)
config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(config)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
model = resolve_model_class(config)(**cfg.model).cuda()
ck_path = "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth"
ck = torch.load(ck_path, map_location="cpu")
loaded = model.load_state_dict(ck["model"], strict=False)
assert not loaded.missing_keys and not set(loaded.unexpected_keys) - {
    "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
del ck
model.train()
criterion = model.criterion
original_sem = criterion.loss_sem_align
sem_source = textwrap.dedent(inspect.getsource(original_sem))
assert "negative_logits = logits" in sem_source
masked_source = sem_source.replace(
    "negative_logits = logits",
    "negative_logits = logits.masked_fill(~lang_masks.bool().unsqueeze(1), -1e4)")
masked_source = masked_source.replace(
    "token_to_box_loss = (token_to_box_loss * tmask).sum()",
    "token_to_box_loss = (token_to_box_loss * tmask * lang_masks.bool()).sum()")
assert masked_source != sem_source
ns = {"torch": torch}
exec(compile(masked_source, "native_sem_support_masked", "exec"), ns)
masked_sem = types.MethodType(ns["loss_sem_align"], criterion)
capture = {}
original_forward = criterion.forward

def intercept(self, *args, **kwargs):
    capture["args"] = args
    capture["kwargs"] = kwargs
    return original_forward(*args, **kwargs)

criterion.forward = types.MethodType(intercept, criterion)
original_indi = criterion.loss_indi

def indi_capture(self, *args):
    capture.setdefault("targets", args[1])
    capture.setdefault("indices", args[2])
    capture.setdefault("normalizer", args[3])
    return original_indi(*args)

criterion.loss_indi = types.MethodType(indi_capture, criterion)

def clone(value, grad=False):
    if torch.is_tensor(value):
        value = value.detach().clone()
        if grad and value.is_floating_point():
            value.requires_grad_(True)
        return value
    if isinstance(value, dict):
        return {k: clone(v, grad) for k, v in value.items()}
    if isinstance(value, list):
        return [clone(v, grad) for v in value]
    if isinstance(value, tuple):
        return tuple(clone(v, grad) for v in value)
    return value

def permute(pred, permutation):
    result = dict(pred)
    for key in ("masks", "scores", "indis", "proj_queries", "sample_inds"):
        if key in result and torch.is_tensor(result[key]):
            result[key] = result[key][:, permutation]
        elif key in result and isinstance(result[key], list):
            result[key] = [v[permutation] for v in result[key]]
    if "aux_outputs" in result:
        result["aux_outputs"] = [permute(v, permutation) for v in result["aux_outputs"]]
    return result

rows = []
seen = set()
for index, chunk in enumerate(dataset.scene_inputs):
    scan = chunk[0]["scene_id"]
    if scan in seen:
        continue
    seen.add(scan)
    batch = dataset.collate_fn([dataset[index]])
    capture.clear()
    with torch.no_grad():
        model(batch, mode="loss")
    args = clone(capture["args"])
    target = clone(capture["targets"])
    indices = clone(capture["indices"])
    normalizer = clone(capture["normalizer"])
    native = clone(args[0], True)
    p = torch.roll(torch.arange(native["indis"].shape[1], device="cuda"), 1)
    inv = torch.argsort(p)
    nloss, nterms = original_forward(native, *args[1:])
    repeat = clone(args[0], True)
    rloss, rterms = original_forward(repeat, *args[1:])
    perm_leaf = clone(args[0], True)
    perm = permute(perm_leaf, p)
    ploss, pterms = original_forward(perm, *args[1:])
    keys = ("masks", "indis", "proj_queries")
    ng = torch.autograd.grad(nloss, [native[k] for k in keys])
    pg = torch.autograd.grad(ploss, [perm_leaf[k] for k in keys])
    # perm_leaf is in original order; autograd already maps through indexing.
    grad_diff = max(float((a-b).abs().max()) for a, b in zip(ng, pg))
    token = args[0]["proj_tokens"].detach().clone().requires_grad_(True)
    query = args[0]["proj_queries"].detach().clone().requires_grad_(True)
    lang = args[6]
    valid = lang.bool()
    replaced = token.detach().clone()
    replacement = torch.zeros_like(replaced)
    replacement[..., 0] = 1
    replaced = torch.where(valid.unsqueeze(-1), replaced, replacement)
    sem = original_sem(token, query, lang, target, indices, normalizer)
    sem_repeat = original_sem(token, query, lang, target, indices, normalizer)
    sem_cf = original_sem(replaced, query, lang, target, indices, normalizer)
    sem_mask = masked_sem(token, query, lang, target, indices, normalizer)
    sem_mask_cf = masked_sem(replaced, query, lang, target, indices, normalizer)
    tg, qg = torch.autograd.grad(sem, (token, query), retain_graph=True)
    mtg, mqg = torch.autograd.grad(sem_mask, (token, query))
    padding = ~valid
    rows.append(dict(scan_id=scan, expressions=len(batch["ann_ids"]),
        query_loss_difference=float((nloss-ploss).abs()),
        query_repeat_loss_difference=float((nloss-rloss).abs()),
        query_term_max_difference=max(float((nterms[k]-pterms[k]).abs().max()) for k in nterms),
        query_gradient_max_difference=grad_diff,
        padding_tokens=int(padding.sum()), native_sem=float(sem), masked_sem=float(sem_mask),
        native_padding_replacement_difference=float((sem-sem_cf).abs()),
        masked_padding_replacement_difference=float((sem_mask-sem_mask_cf).abs()),
        semantic_repeat_difference=float((sem-sem_repeat).abs()),
        native_padding_gradient_norm=float(tg[padding].norm()),
        masked_padding_gradient_norm=float(mtg[padding].norm()),
        native_valid_gradient_norm=float(tg[valid].norm()),
        masked_valid_gradient_norm=float(mtg[valid].norm()),
        masked_query_gradient_norm=float(mqg.norm()),
        finite=all(bool(torch.isfinite(v).all()) for v in (tg,qg,mtg,mqg))))
    print(json.dumps(rows[-1]), flush=True)
    del native, repeat, perm_leaf, perm, nloss, rloss, ploss, nterms, rterms, pterms
    del token, query, replaced, replacement, sem, sem_cf, sem_repeat, sem_mask, sem_mask_cf
    del ng, pg, tg, qg, mtg, mqg, args, target, indices, normalizer, batch
    capture.clear()
    if len(rows) == 8:
        break

assert len(rows) == 8
summary = dict(scans=8, expressions=sum(r["expressions"] for r in rows),
    query_loss_max_difference=max(r["query_loss_difference"] for r in rows),
    query_term_max_difference=max(r["query_term_max_difference"] for r in rows),
    query_gradient_max_difference=max(r["query_gradient_max_difference"] for r in rows),
    query_repeat_max_difference=max(r["query_repeat_loss_difference"] for r in rows),
    padding_tokens=sum(r["padding_tokens"] for r in rows),
    native_padding_cf_mean=statistics.mean(r["native_padding_replacement_difference"] for r in rows),
    native_padding_grad_all_nonzero=all(r["native_padding_gradient_norm"]>0 for r in rows),
    masked_padding_cf_max=max(r["masked_padding_replacement_difference"] for r in rows),
    masked_padding_grad_max=max(r["masked_padding_gradient_norm"] for r in rows),
    masked_valid_and_query_all_nonzero=all(r["masked_valid_gradient_norm"]>0 and r["masked_query_gradient_norm"]>0 for r in rows),
    all_finite=all(r["finite"] for r in rows),
    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
result = dict(summary=summary, rows=rows, complete=True, checkpoint=ck_path,
    source_sha256=hashlib.sha256(Path("ipdn/model_paper5/adares_w2_pointmass_competition/loss.py").read_bytes()).hexdigest(),
    source_correction="Actual matcher is geometric seed-instance correspondence, not Hungarian.",
    limitations=["Train-only cached criterion counterfactual; not trained efficacy or whole-model repeat proof."])
Path("/tmp/adares_y22_y23_d20c71e4.json").write_text(json.dumps(result))
print(json.dumps(summary), flush=True)
