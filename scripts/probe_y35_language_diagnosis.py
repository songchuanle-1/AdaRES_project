"""Prediction-blind TRAIN64 mask-evidence to word-token incidence."""

import ast
import hashlib
from pathlib import Path
import statistics
import gorilla
import torch
from torch import nn
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class

torch.set_num_threads(4)
config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(config)
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint = "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth"
assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == "e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52"
loaded = model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=False)
assert not loaded.missing_keys and not set(loaded.unexpected_keys)-{
    "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
source = Path("ipdn/model_paper5/adares_y35_language_evidence/dec.py").read_bytes()
node = next(n for n in ast.parse(source.decode()).body
            if isinstance(n, ast.ClassDef) and n.name == "MaskLanguageEvidenceReadout")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[node], type_ignores=[]), "y35_actual_class", "exec"), namespace)
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(1999)
    readout = namespace["MaskLanguageEvidenceReadout"](256).cuda()
indices, seen = [], set()
for index, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]["scene_id"] not in seen:
        seen.add(chunk[0]["scene_id"])
        indices.append(index)
    if len(indices) == 64:
        break
assert len(indices) == 64
model.train()
cache = {}

def visual_post(module, args, output):
    cache["visual"] = output.detach()

def words_post(module, args, output):
    cache["words"] = output.detach()

def indicator_pre(module, args):
    cache["indicator"] = tuple(v.detach() for v in args)

hooks = [model.dec.ila_ffn_layers[-1].register_forward_hook(visual_post),
         model.dec.lia_ffn_layers[-1].register_forward_hook(words_post),
         model.dec.indi_embedding.register_forward_pre_hook(indicator_pre)]
rows = []
with torch.no_grad():
    for index in indices:
        cache.clear()
        batch = dataset.collate_fn([dataset[index]])
        model(batch, mode="loss")
        q, masks, pad, mass = cache["indicator"]
        visual, words = cache["visual"], cache["words"]
        wp = ~batch["lang_masks"].cuda().bool()
        assert wp.shape == words.shape[:2] and (~wp).any(-1).all()
        direct = readout(q, masks, pad, mass, visual, words, wp)
        repeat = readout(q, masks, pad, mass, visual, words, wp)
        wrong_masks = readout(q, masks.roll(1, 1), pad, mass, visual, words, wp)

        def wrong_values(module, args, output):
            changed = output.clone()
            for b in range(words.shape[0]):
                count = int((~wp[b]).sum())
                changed[b, :count] = output[b, :count].roll(1, 0)
            return changed

        handle = readout.v_proj.register_forward_hook(wrong_values)
        wrong_words = readout(q, masks, pad, mass, visual, words, wp)
        handle.remove()
        lp = torch.arange(words.shape[1]-1, -1, -1, device=words.device)
        joint_words = readout(q, masks, pad, mass, visual, words[:, lp], wp[:, lp])
        valid = (~pad).to(masks.dtype)
        fg = masks.sigmoid()*mass[:, None]*valid[:, None]
        bg = (-masks).sigmoid()*mass[:, None]*valid[:, None]
        fg_mean = fg @ visual / fg.sum(-1, keepdim=True).clamp_min(1e-6)
        bg_mean = bg @ visual / bg.sum(-1, keepdim=True).clamp_min(1e-6)
        original = model.dec.indi_embedding(q, masks, pad, mass)
        binary = masks.sigmoid().ge(.5)
        sp = batch["superpoints"].long().cuda()
        counts = torch.bincount(sp)
        positive = torch.stack([torch.bincount(sp[gt.cuda().bool()], minlength=counts.numel())
                                for gt in batch["gt_pmasks"]])
        intersection = (binary*positive[:, None]).sum(-1)
        union = ((binary*counts).sum(-1)+positive.sum(-1)[:, None]-intersection).clamp_min(1)
        quality = intersection/union
        choice = original.softmax(-1)[..., 1].argmax(-1)
        gap = quality.max(-1).values-quality.gather(1, choice[:, None]).squeeze(1)
        row = dict(scan_id=batch["scan_ids"][0], dataset_index=index,
                   expressions=len(batch["ann_ids"]), oracle_gap=gap.cpu().tolist(),
                   fg_bg_contrast=float((fg_mean-bg_mean).norm(dim=-1).mean()),
                   word_variance=float(words[~wp].var(0).mean()),
                   wrong_word_alignment_max=float((direct-wrong_words).abs().max()),
                   wrong_masks_max=float((direct-wrong_masks).abs().max()),
                   repeat_max=float((direct-repeat).abs().max()),
                   word_permutation_max=float((direct-joint_words).abs().max()),
                   finite=bool(torch.isfinite(direct).all()))
        assert row["finite"] and row["repeat_max"] == 0
        assert row["word_permutation_max"] < 1e-5
        rows.append(row)
        if len(rows) % 8 == 0:
            print(json.dumps(dict(scans=len(rows))), flush=True)
        cache.clear()
for hook in hooks:
    hook.remove()
result = dict(
    complete=True, id="AdaRES-Y35", rows=rows, scans=64,
    expressions=sum(r["expressions"] for r in rows),
    expressions_with_oracle_gap=sum(g > 0 for r in rows for g in r["oracle_gap"]),
    scene_mean_oracle_gap=statistics.mean(statistics.mean(r["oracle_gap"]) for r in rows),
    positive_fg_bg_scans=sum(r["fg_bg_contrast"] > 0 for r in rows),
    positive_word_variance_scans=sum(r["word_variance"] > 0 for r in rows),
    wrong_words_nonzero_scans=sum(r["wrong_word_alignment_max"] > 0 for r in rows),
    wrong_masks_nonzero_scans=sum(r["wrong_masks_max"] > 0 for r in rows),
    repeat_max=max(r["repeat_max"] for r in rows),
    permutation_max=max(r["word_permutation_max"] for r in rows),
    minimum_wrong_word_delta=min(r["wrong_word_alignment_max"] for r in rows),
    source_sha256=hashlib.sha256(source).hexdigest(),
    physical_gpu=P["physical_gpu"], gpu_precheck=dict(memory_mib=used, utilization=util),
    limitations=["TRAIN incidence and random head, not semantic necessity or efficacy.",
                 "GT used for diagnosis only; all native language tokens retained.",
                 "Wrong value alignment preserves word marginals, not a trained ablation.",
                 "Y33/GPU1 and all production weights unchanged; no validation search."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
