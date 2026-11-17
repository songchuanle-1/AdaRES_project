"""Prediction-blind first64 TRAIN scene incidence for the fixed pair readout."""

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
marker = Path(".codex_source_commit").read_text().strip()
assert marker == "6725f9bd6a8f9b47addae6a4b8336d004f73f1ae"
config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(config)
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
model = resolve_model_class(config)(**cfg.model).cuda()
checkpoint = "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth"
assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == "e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52"
loaded = model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=False)
assert not loaded.missing_keys
assert not set(loaded.unexpected_keys) - {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
source = Path("ipdn/model_paper5/adares_y37_pair_distribution/dec.py").read_bytes()
node = next(n for n in ast.parse(source.decode("utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "MaskPairDistributionReadout")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[node], type_ignores=[]), "y37_actual_class", "exec"), namespace)
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(1999)
    readout = namespace["MaskPairDistributionReadout"](256).cuda()
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


def coordinate_pre(module, args):
    cache["coordinates"] = module.get_batches(args[5], args[2])[0].detach()


def indicator_pre(module, args):
    cache["indicator"] = tuple(v.detach() for v in args)


hooks = [model.dec.register_forward_pre_hook(coordinate_pre),
         model.dec.indi_embedding.register_forward_pre_hook(indicator_pre)]
rows = []
with torch.no_grad():
    for index in indices:
        cache.clear()
        batch = dataset.collate_fn([dataset[index]])
        model(batch, mode="loss")
        query, masks, padding, mass = cache["indicator"]
        coords = cache["coordinates"]
        tokens = query.new_zeros((query.shape[0], mass.shape[1], 256))
        direct = readout(query, masks, padding, mass, tokens, coords)
        repeat = readout(query, masks, padding, mass, tokens, coords)
        wrong_masks = readout(query, masks.roll(1, 1), padding, mass, tokens, coords)
        changed = coords.clone()
        for b in range(coords.shape[0]):
            count = int((~padding[b]).sum())
            changed[b, :count] = coords[b, :count].roll(1, 0)
        wrong_pairs = readout(query, masks, padding, mass, tokens, changed)
        perm = torch.arange(mass.shape[1]-1, -1, -1, device=mass.device)
        joint = readout(query, masks[:, :, perm], padding[:, perm], mass[:, perm],
                        tokens[:, perm], coords[:, perm])
        descriptor = readout.descriptor(masks, padding, mass, coords)
        original = model.dec.indi_embedding(query, masks, padding, mass)
        binary = masks.sigmoid().ge(.5)
        sp = batch["superpoints"].long().cuda()
        counts = torch.bincount(sp)
        positives = torch.stack([torch.bincount(sp[gt.cuda().bool()], minlength=counts.numel())
                                 for gt in batch["gt_pmasks"]])
        intersection = (binary * positives[:, None]).sum(-1)
        union = ((binary*counts).sum(-1)+positives.sum(-1)[:, None]-intersection).clamp_min(1)
        quality = intersection/union
        choice = original.softmax(-1)[..., 1].argmax(-1)
        gap = quality.max(-1).values-quality.gather(1, choice[:, None]).squeeze(1)
        row = dict(scan_id=batch["scan_ids"][0], dataset_index=index,
                   expressions=len(batch["ann_ids"]), oracle_gap=gap.cpu().tolist(),
                   global_pair_contrast=float((descriptor[..., 0, :]-descriptor[..., 1, :]).abs().mean()),
                   wrong_masks_max=float((direct-wrong_masks).abs().max()),
                   wrong_pairs_max=float((direct-wrong_pairs).abs().max()),
                   repeat_max=float((direct-repeat).abs().max()),
                   permutation_max=float((direct-joint).abs().max()),
                   finite=bool(torch.isfinite(direct).all()))
        assert row["finite"] and row["repeat_max"] == 0
        assert row["permutation_max"] < 1e-5
        rows.append(row)
        if len(rows) % 8 == 0:
            print(json.dumps(dict(scans=len(rows))), flush=True)
for hook in hooks:
    hook.remove()
result = dict(complete=True, id="AdaRES-Y37", rows=rows, scans=64,
              expressions=sum(r["expressions"] for r in rows),
              expressions_with_oracle_gap=sum(g > 0 for r in rows for g in r["oracle_gap"]),
              positive_pair_contrast_scans=sum(r["global_pair_contrast"] > 0 for r in rows),
              wrong_masks_nonzero_scans=sum(r["wrong_masks_max"] > 0 for r in rows),
              wrong_pairs_nonzero_scans=sum(r["wrong_pairs_max"] > 0 for r in rows),
              repeat_max=max(r["repeat_max"] for r in rows),
              permutation_max=max(r["permutation_max"] for r in rows),
              source_sha256=hashlib.sha256(source).hexdigest(), source_marker=marker,
              checkpoint=checkpoint, gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Native TRAIN incidence/random head, not efficacy or semantic necessity.",
                           "All rows retained; no selection of responsive scenes.",
                           "GT quality only used for diagnosis, never inference.",
                           "Superpoint pair summary can reflect scene shape as well as support; trained tests required."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
