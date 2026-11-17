"""Prediction-blind W2 TRAIN64 mask-shape diagnosis, no weight updates."""

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
assert marker == "4f5c31f57994b8f9fb640d652c913863dc310a95"
source_path = Path("ipdn/model_paper5/adares_y31_mask_shape/dec.py")
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == "41e612e5eba8773c0dab108360ca4da9bcb3b756f3d526627a269a8255b57ca3"
tree = ast.parse(source_path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MaskShapeReadout")
namespace = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source_path), "exec"), namespace)
descriptor = namespace["MaskShapeReadout"].descriptor
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
heads, dec_args = [], []
head_hook = model.dec.indi_embedding.register_forward_pre_hook(
    lambda module, args: heads.append(tuple(v.detach() for v in args)))
dec_hook = model.dec.register_forward_pre_hook(
    lambda module, args: dec_args.append(tuple(v.detach() if torch.is_tensor(v) else v for v in args)))
rows = []
with torch.no_grad():
    for index in indices:
        heads.clear()
        dec_args.clear()
        batch = dataset.collate_fn([dataset[index]])
        model(batch, mode="loss")
        assert len(heads) == 7 and len(dec_args) == 1
        query, masks, pad, mass = heads[-1]
        args = dec_args[0]
        pos, position_pad = model.dec.get_batches(args[5], args[2])
        assert torch.equal(pad, position_pad)
        shape = descriptor(masks, pad, mass, pos)
        repeated = descriptor(masks, pad, mass, pos)
        shifted = descriptor(masks, pad, mass, pos + 3)
        scores = model.dec.indi_embedding(query, masks, pad, mass)
        binary = masks.sigmoid().ge(.5)
        sp = batch["superpoints"].long().cuda()
        counts = torch.bincount(sp)
        positive = torch.stack([torch.bincount(sp[gt.cuda().bool()], minlength=counts.numel())
                                for gt in batch["gt_pmasks"]])
        inter = (binary * positive.unsqueeze(1)).sum(-1)
        union = ((binary*counts).sum(-1)+positive.sum(-1).unsqueeze(1)-inter).clamp_min(1)
        iou = inter / union
        choice, oracle_choice = scores.softmax(-1)[..., 1].argmax(-1), iou.argmax(-1)
        chosen, oracle = iou.gather(1, choice[:, None]).squeeze(1), iou.max(-1).values
        selected_shape = shape.gather(1, choice[:, None, None].expand(-1, 1, 6)).squeeze(1)
        oracle_shape = shape.gather(1, oracle_choice[:, None, None].expand(-1, 1, 6)).squeeze(1)
        rows.append(dict(scan_id=batch["scan_ids"][0], dataset_index=index,
                         expressions=len(batch["ann_ids"]), queries=query.shape[1],
                         oracle_gap=(oracle-chosen).cpu().tolist(),
                         selected_oracle_shape_distance=(selected_shape-oracle_shape).square().mean(-1).sqrt().cpu().tolist(),
                         shape_query_variance=float(shape.var(1, unbiased=False).mean()),
                         native_repeat_max=float((shape-repeated).abs().max()),
                         translation_max=float((shape-shifted).abs().max())))
        assert torch.isfinite(shape).all() and rows[-1]["native_repeat_max"] == 0
        assert rows[-1]["translation_max"] < 1e-5
        if len(rows) % 8 == 0:
            print(json.dumps(dict(scans=len(rows))), flush=True)
        heads.clear()
        dec_args.clear()
head_hook.remove()
dec_hook.remove()
missed = [(g, d) for r in rows for g, d in
          zip(r["oracle_gap"], r["selected_oracle_shape_distance"]) if g > 0]
result = dict(complete=True, id="AdaRES-Y31", rows=rows, scans=64,
              expressions=sum(r["expressions"] for r in rows),
              expressions_with_oracle_gap=len(missed),
              missed_with_distinct_shape=sum(d > 0 for _, d in missed),
              scene_mean_oracle_gap=statistics.mean(statistics.mean(r["oracle_gap"]) for r in rows),
              shape_query_variance=statistics.mean(r["shape_query_variance"] for r in rows),
              native_repeat_max=max(r["native_repeat_max"] for r in rows),
              translation_max=max(r["translation_max"] for r in rows),
              checkpoint=checkpoint, source_marker=marker,
              gpu_precheck=dict(memory_mib=used, utilization=util),
              limitations=["Prediction-blind native TRAIN support, no validation or selection by outcome.",
                           "Covariance is point-mass weighted superpoint-centroid approximation, not exact raw-point covariance.",
                           "Upstream implicit geometry exists; diversity and regret do not establish geometry necessity.",
                           "GT only supplies diagnostic oracle gap, never candidate input.",
                           "Repeat is cached local identity, not global model determinism or a noise floor."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
