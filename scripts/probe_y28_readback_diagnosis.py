"""Native TRAIN64 audit of final-mask visual content missing from confidence."""

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
assert marker == "359a89d59dcf3990f4e45b1202d11d3257233e76"
source_path = Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py")
source = source_path.read_bytes()
source_hash = hashlib.sha256(source).hexdigest()
assert source_hash == "6362d882662b0e51594cd4af994359b93f99497affc93600322a34ca6c349b01"
tree = ast.parse(source.decode("utf-8"))
indicator = next(n for n in tree.body
                 if isinstance(n, ast.ClassDef) and n.name == "QueryCompetitionIndicator")
forward = next(n for n in indicator.body
               if isinstance(n, ast.FunctionDef) and n.name == "forward")
arguments = [n.arg for n in forward.args.args]
assert arguments == ["self", "query", "pred_masks", "batch_mask", "point_mass"]
assert not any(n.id in {"inst_feats", "mask_feats", "lang_feats"}
               for n in ast.walk(forward) if isinstance(n, ast.Name))
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
cache = {}


def scene_post(module, args, output):
    cache["scene"] = output.detach()


def indicator_pre(module, args):
    cache["indicator"] = tuple(v.detach() for v in args)


hooks = [model.dec.ila_ffn_layers[-1].register_forward_hook(scene_post),
         model.dec.indi_embedding.register_forward_pre_hook(indicator_pre)]
rows = []
with torch.no_grad():
    for index in indices:
        cache.clear()
        batch = dataset.collate_fn([dataset[index]])
        model(batch, mode="loss")
        query, logits, padding, mass = cache["indicator"]
        scene = cache["scene"]
        assert scene.shape[:2] == padding.shape
        valid = (~padding).to(logits.dtype)
        weights = logits.sigmoid() * valid.unsqueeze(1) * mass.unsqueeze(1)
        foreground = torch.bmm(weights, scene) / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        background_weights = (1-logits.sigmoid()) * valid.unsqueeze(1) * mass.unsqueeze(1)
        background = torch.bmm(background_weights, scene) / background_weights.sum(-1, keepdim=True).clamp_min(1e-6)
        uniform_weights = valid * mass
        uniform = torch.bmm(uniform_weights.unsqueeze(1), scene) / uniform_weights.sum(-1).view(-1, 1, 1).clamp_min(1e-6)
        rolled_weights = weights.clone()
        for b in range(weights.shape[0]):
            count = int(valid[b].sum())
            rolled_weights[b, :, :count] = weights[b, :, :count].roll(1, -1)
        rolled = torch.bmm(rolled_weights, scene) / rolled_weights.sum(-1, keepdim=True).clamp_min(1e-6)
        original = model.dec.indi_embedding(query, logits, padding, mass)
        repeat = model.dec.indi_embedding(query, logits, padding, mass)
        # Permute all superpoint-indexed inputs together, not query identities.
        perm = torch.arange(scene.shape[1]-1, -1, -1, device=scene.device)
        perm_pool = torch.bmm(weights[:, :, perm].double(), scene[:, perm].double()) / weights[:, :, perm].double().sum(-1, keepdim=True).clamp_min(1e-6)
        exact_pool = torch.bmm(weights.double(), scene.double()) / weights.double().sum(-1, keepdim=True).clamp_min(1e-6)
        binary = logits.sigmoid().ge(.5)
        sp = batch["superpoints"].long().cuda()
        counts = torch.bincount(sp)
        positive = torch.stack([torch.bincount(sp[gt.cuda().bool()], minlength=counts.numel())
                                for gt in batch["gt_pmasks"]])
        intersection = (binary * positive.unsqueeze(1)).sum(-1)
        union = ((binary * counts).sum(-1) + positive.sum(-1).unsqueeze(1)-intersection).clamp_min(1)
        iou = intersection / union
        choice = original.softmax(-1)[..., 1].argmax(-1)
        oracle_choice = iou.argmax(-1)
        chosen = iou.gather(1, choice.unsqueeze(1)).squeeze(1)
        oracle = iou.max(-1).values
        chosen_pool = foreground.gather(1, choice[:, None, None].expand(-1, 1, foreground.shape[-1])).squeeze(1)
        oracle_pool = foreground.gather(1, oracle_choice[:, None, None].expand(-1, 1, foreground.shape[-1])).squeeze(1)
        row = dict(scan_id=batch["scan_ids"][0], dataset_index=index,
                   expressions=len(batch["ann_ids"]),
                   selected_point_iou=chosen.cpu().tolist(), oracle_point_iou=oracle.cpu().tolist(),
                   oracle_gap=(oracle-chosen).cpu().tolist(),
                   selected_oracle_pool_distance=(chosen_pool-oracle_pool).norm(dim=-1).cpu().tolist(),
                   pool_vs_rolled_mean=float((foreground-rolled).norm(dim=-1).mean()),
                   pool_vs_uniform_mean=float((foreground-uniform).norm(dim=-1).mean()),
                   foreground_background_mean=float((foreground-background).norm(dim=-1).mean()),
                   native_indicator_repeat_max=float((original-repeat).abs().max()),
                   joint_superpoint_permutation_max=float((exact_pool-perm_pool).abs().max()),
                   finite=bool(torch.isfinite(foreground).all() and torch.isfinite(background).all()))
        assert row["finite"] and row["native_indicator_repeat_max"] == 0
        assert row["joint_superpoint_permutation_max"] < 1e-10
        rows.append(row)
        if len(rows) % 8 == 0:
            print(json.dumps(dict(scans=len(rows))), flush=True)
        del batch, query, logits, scene, weights, foreground, background, original, repeat
        cache.clear()
for hook in hooks:
    hook.remove()
missed = [(g, d) for r in rows for g, d in
          zip(r["oracle_gap"], r["selected_oracle_pool_distance"]) if g > 0]
result = dict(complete=True, id="AdaRES-Y28", rows=rows, scans=64,
              expressions=sum(r["expressions"] for r in rows),
              expressions_with_oracle_gap=len(missed),
              missed_with_distinct_pooled_content=sum(d > 0 for _, d in missed),
              scene_mean_oracle_gap=statistics.mean(statistics.mean(r["oracle_gap"]) for r in rows),
              pool_vs_rolled_mean=statistics.mean(r["pool_vs_rolled_mean"] for r in rows),
              pool_vs_uniform_mean=statistics.mean(r["pool_vs_uniform_mean"] for r in rows),
              foreground_background_mean=statistics.mean(r["foreground_background_mean"] for r in rows),
              native_repeat_max=max(r["native_indicator_repeat_max"] for r in rows),
              permutation_max=max(r["joint_superpoint_permutation_max"] for r in rows),
              source=dict(path=str(source_path), sha256=source_hash,
                          indicator_arguments=arguments, direct_visual_readback_absent=True),
              checkpoint=checkpoint, source_marker=marker,
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Upstream queries already attend to semantic content; no global blindness/necessity claim.",
                           "Distinct content and oracle gaps establish a falsifiable information-path hypothesis, not trained efficacy.",
                           "Old J1 fixed pooled-language cosine rejection remains; this does not retry that ranking rule.",
                           "No validation, GT inference input, source/weights modification or numerical search."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
