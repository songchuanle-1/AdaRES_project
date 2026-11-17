"""Prediction-blind native TRAIN64 relational evidence-gap audit."""

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
assert not loaded.missing_keys and not set(loaded.unexpected_keys)-{
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
cache = {}


def decoder_pre(module, args):
    cache["coords"], _ = module.get_batches(args[5], args[2])


def indicator_pre(module, args):
    cache["indicator"] = tuple(v.detach() for v in args)


def decoder_post(module, args, output):
    cache["pred"] = output


hooks = [model.dec.register_forward_pre_hook(decoder_pre),
         model.dec.indi_embedding.register_forward_pre_hook(indicator_pre),
         model.dec.register_forward_hook(decoder_post)]
rows = []
with torch.no_grad():
    for index in indices:
        batch = dataset.collate_fn([dataset[index]])
        cache.clear()
        model(batch, mode="loss")
        query, logits, padding, mass = cache["indicator"]
        positions = cache["coords"]
        probabilities = logits.sigmoid() * (~padding).unsqueeze(1)
        weighted = probabilities * mass.unsqueeze(1)
        area = weighted.sum(-1)
        centroid = torch.einsum("bqm,bmd->bqd", weighted, positions) / area.clamp_min(1e-6).unsqueeze(-1)
        delta = centroid.unsqueeze(1) - centroid.unsqueeze(2)
        # W2 has no position argument: the matched confidence input is identical.
        original = model.dec.indi_embedding(query, logits, padding, mass)
        repeat = model.dec.indi_embedding(query, logits, padding, mass)
        binary = logits.sigmoid().ge(.5)
        sp = batch["superpoints"].long().cuda()
        counts = torch.bincount(sp)
        positive = torch.stack([torch.bincount(sp[gt.cuda().bool()], minlength=counts.numel())
                                for gt in batch["gt_pmasks"]])
        intersection = (binary * positive.unsqueeze(1)).sum(-1)
        union = ((binary * counts).sum(-1) + positive.sum(-1).unsqueeze(1)-intersection).clamp_min(1)
        iou = intersection / union
        choice = original.softmax(-1)[..., 1].argmax(-1)
        chosen = iou.gather(1, choice.unsqueeze(1)).squeeze(1)
        oracle = iou.max(-1).values
        relation = [bool((m["rel_positive_map"] > 0).any()) for m in batch["dense_maps"]]
        other = [bool((m["other_entity_map"] > 0).any()) for m in batch["dense_maps"]]
        assert len(relation) == len(batch["ann_ids"]) == len(other)
        row = dict(scan_id=batch["scan_ids"][0], dataset_index=index,
                   expressions=len(batch["ann_ids"]), relation=relation, other_entity=other,
                   selected_point_iou=chosen.cpu().tolist(), oracle_point_iou=oracle.cpu().tolist(),
                   oracle_gap=(oracle-chosen).cpu().tolist(),
                   centroid_pair_distance_mean=float(delta.norm(dim=-1).mean()),
                   centroid_cyclic_mismatch_max=float((centroid-centroid.roll(1, 1)).abs().max()),
                   native_repeat_indicator_max=float((original-repeat).abs().max()),
                   native_geometry_counterfactual_max=0.0,
                   finite=bool(torch.isfinite(centroid).all()))
        assert row["finite"] and row["native_repeat_indicator_max"] == 0
        rows.append(row)
        cache.clear()
        del batch, query, logits, padding, mass, positions, probabilities, weighted, binary, positive
        if len(rows) % 8 == 0:
            print(json.dumps(dict(scans=len(rows))), flush=True)
for hook in hooks:
    hook.remove()
all_gaps = [g for r in rows for g in r["oracle_gap"]]
result = dict(complete=True, id="AdaRES-Y27", rows=rows, scans=64,
              expressions=sum(r["expressions"] for r in rows),
              relation_expressions=sum(sum(r["relation"]) for r in rows),
              other_entity_expressions=sum(sum(r["other_entity"]) for r in rows),
              scene_mean_oracle_gap=statistics.mean(statistics.mean(r["oracle_gap"]) for r in rows),
              expressions_with_oracle_gap=sum(g > 0 for g in all_gaps),
              centroid_spread_mean=statistics.mean(r["centroid_pair_distance_mean"] for r in rows),
              minimum_centroid_cyclic_difference=min(r["centroid_cyclic_mismatch_max"] for r in rows),
              native_repeat_max=max(r["native_repeat_indicator_max"] for r in rows),
              geometry_input_absent_in_confidence=True, checkpoint=checkpoint,
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Upstream query already contains spatial information; no whole-model geometry-blind claim.",
                           "W2 has no position input; counterfactual insensitivity is an input-contract fact, not a simulated efficacy test.",
                           "TRAIN64 oracle gap and relation incidence do not prove new-path necessity or trained benefit.",
                           "No validation, GT inference input, hyperparameter scan or parent/source/weight modification."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
