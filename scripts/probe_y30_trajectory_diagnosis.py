"""Prediction-blind native TRAIN64 query/mask trajectory diagnosis."""

import ast
import hashlib
from pathlib import Path
import statistics

import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class

torch.set_num_threads(4)
marker = Path(".codex_source_commit").read_text().strip()
assert marker == "ce2dc75ee9e706da68c5768d9b3c855f33bda7fb"
source_path = Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "6362d882662b0e51594cd4af994359b93f99497affc93600322a34ca6c349b01"
tree = ast.parse(source.decode("utf-8"))
head = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QueryCompetitionIndicator")
forward = next(n for n in head.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
assert [n.arg for n in forward.args.args] == ["self", "query", "pred_masks", "batch_mask", "point_mass"]
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
cache = []
hook = model.dec.indi_embedding.register_forward_pre_hook(
    lambda module, args: cache.append(tuple(v.detach() for v in args)))
rows = []
with torch.no_grad():
    for index in indices:
        cache.clear()
        batch = dataset.collate_fn([dataset[index]])
        model(batch, mode="loss")
        assert len(cache) == 7
        final, masks, pad, mass = cache[-1]
        assert all(v[0].shape == final.shape and v[1].shape == masks.shape
                   and torch.equal(v[2], pad) and torch.equal(v[3], mass) for v in cache)
        probabilities = [v[1].sigmoid() * (~pad).unsqueeze(1) for v in cache]
        descriptor = []
        for layer, current in enumerate(probabilities):
            previous = probabilities[max(0, layer-1)]
            intersection = (current * previous * mass.unsqueeze(1)).sum(-1)
            a = (current * mass.unsqueeze(1)).sum(-1)
            b = (previous * mass.unsqueeze(1)).sum(-1)
            descriptor.append(torch.stack([intersection/(a+b-intersection).clamp_min(1e-6),
                                           (a.clamp_min(1e-6)/b.clamp_min(1e-6)).log()], -1))
        desc = torch.stack(descriptor)
        history = torch.stack([v[0] for v in cache])
        q_dist = (history[:-1]-final.unsqueeze(0)).square().mean(-1).sqrt()
        scores = model.dec.indi_embedding(final, masks, pad, mass)
        repeat = model.dec.indi_embedding(final, masks, pad, mass)
        binary = masks.sigmoid().ge(.5)
        sp = batch["superpoints"].long().cuda()
        counts = torch.bincount(sp)
        positive = torch.stack([torch.bincount(sp[gt.cuda().bool()], minlength=counts.numel())
                                for gt in batch["gt_pmasks"]])
        inter = (binary * positive.unsqueeze(1)).sum(-1)
        union = ((binary*counts).sum(-1)+positive.sum(-1).unsqueeze(1)-inter).clamp_min(1)
        iou = inter/union
        choice = scores.softmax(-1)[..., 1].argmax(-1)
        oracle_choice = iou.argmax(-1)
        chosen = iou.gather(1, choice[:, None]).squeeze(1)
        oracle = iou.max(-1).values
        chosen_traj = history.gather(2, choice[None, :, None, None].expand(7, -1, 1, 256)).squeeze(2)
        oracle_traj = history.gather(2, oracle_choice[None, :, None, None].expand(7, -1, 1, 256)).squeeze(2)
        rows.append(dict(scan_id=batch["scan_ids"][0], dataset_index=index,
                         expressions=len(batch["ann_ids"]), stages=7, queries=final.shape[1],
                         oracle_gap=(oracle-chosen).cpu().tolist(),
                         selected_oracle_trajectory_distance=(chosen_traj-oracle_traj).square().mean((0, 2)).sqrt().cpu().tolist(),
                         history_final_rms_mean=float(q_dist.mean()),
                         transition_iou_mean=float(desc[1:, ..., 0].mean()),
                         log_area_ratio_abs_mean=float(desc[1:, ..., 1].abs().mean()),
                         temporal_descriptor_variance=float(desc.var(0, unbiased=False).mean()),
                         native_repeat_max=float((scores-repeat).abs().max())))
        assert rows[-1]["native_repeat_max"] == 0 and torch.isfinite(desc).all()
        if len(rows) % 8 == 0:
            print(json.dumps(dict(scans=len(rows))), flush=True)
        cache.clear()
hook.remove()
missed = [(g, d) for r in rows for g, d in
          zip(r["oracle_gap"], r["selected_oracle_trajectory_distance"]) if g > 0]
result = dict(complete=True, id="AdaRES-Y30", rows=rows, scans=64,
              expressions=sum(r["expressions"] for r in rows),
              expressions_with_oracle_gap=len(missed),
              missed_with_distinct_trajectory=sum(d > 0 for _, d in missed),
              scene_mean_oracle_gap=statistics.mean(statistics.mean(r["oracle_gap"]) for r in rows),
              history_final_rms_mean=statistics.mean(r["history_final_rms_mean"] for r in rows),
              transition_iou_mean=statistics.mean(r["transition_iou_mean"] for r in rows),
              log_area_ratio_abs_mean=statistics.mean(r["log_area_ratio_abs_mean"] for r in rows),
              temporal_descriptor_variance=statistics.mean(r["temporal_descriptor_variance"] for r in rows),
              native_repeat_max=max(r["native_repeat_max"] for r in rows),
              source=dict(path=str(source_path), sha256=hashlib.sha256(source).hexdigest(),
                          explicit_temporal_input_absent=True, aligned_stage_count=7),
              checkpoint=checkpoint, source_marker=marker,
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Implicit upstream history already exists; necessity and efficacy are not established.",
                           "Distinct trajectories plus regret support an executable mechanism hypothesis, not a trained gain.",
                           "Y9 fixed layer average and Y28 mask content readback negatives remain closed.",
                           "No validation, GT candidate inputs, weights/source mutation or numerical search."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
