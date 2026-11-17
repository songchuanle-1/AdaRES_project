"""Measure whole-superpoint mask representability on first64 distinct TRAIN scenes."""

import hashlib
from pathlib import Path
import statistics
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

torch.set_num_threads(4)
assert not torch.cuda.is_available()
marker = Path(".codex_source_commit").read_text().strip()
assert marker == "67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773"
config = "configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
cfg = gorilla.Config.fromfile(config)
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
indices, seen = [], set()
for index, chunk in enumerate(dataset.scene_inputs):
    scan = chunk[0]["scene_id"]
    if scan not in seen:
        seen.add(scan)
        indices.append(index)
    if len(indices) == 64:
        break
assert len(indices) == 64
rows = []
for index in indices:
    batch = dataset.collate_fn([dataset[index]])
    sp = batch["superpoints"].long()
    counts = torch.bincount(sp)
    assert bool((counts > 0).all())
    generator = torch.Generator().manual_seed(1999 + index)
    permutation = torch.randperm(sp.numel(), generator=generator)
    ceilings, mixed_mass, shuffled_ceilings = [], [], []
    for ground_truth in batch["gt_pmasks"]:
        target = ground_truth.bool()
        positives = torch.bincount(sp[target], minlength=counts.numel())
        order = torch.argsort(positives.double() / counts.double(), descending=True)
        intersection = positives[order].double().cumsum(0)
        selected = counts[order].double().cumsum(0)
        total = int(positives.sum())
        assert total > 0
        ceilings.append(float((intersection / (selected + total - intersection)).max()))
        mixed = (positives > 0) & (positives < counts)
        mixed_mass.append(float(positives[mixed].sum()) / total)
        shuffled = torch.bincount(sp[target[permutation]], minlength=counts.numel())
        order_wrong = torch.argsort(shuffled.double() / counts.double(), descending=True)
        inter_wrong = shuffled[order_wrong].double().cumsum(0)
        selected_wrong = counts[order_wrong].double().cumsum(0)
        shuffled_ceilings.append(float(
            (inter_wrong / (selected_wrong + total - inter_wrong)).max()))
    row = dict(dataset_index=index, scan_id=batch["scan_ids"][0],
               expressions=len(ceilings), points=sp.numel(), superpoints=counts.numel(),
               exact_union_sp_iou_ceiling=ceilings, target_mass_in_mixed_sp=mixed_mass,
               shuffled_label_ceiling=shuffled_ceilings)
    assert len(ceilings) == len(batch["ann_ids"])
    rows.append(row)
    if len(rows) % 8 == 0:
        print(json.dumps(dict(scans=len(rows))), flush=True)
values = [value for row in rows for value in row["exact_union_sp_iou_ceiling"]]
mixed = [value for row in rows for value in row["target_mass_in_mixed_sp"]]
wrong = [value for row in rows for value in row["shuffled_label_ceiling"]]
result = dict(
    complete=True, id="AdaRES-Y39", scans=len(rows), rows=rows,
    expressions=len(values), expressions_with_nonzero_quantization_gap=sum(v < 1 for v in values),
    median_ceiling=statistics.median(values), minimum_ceiling=min(values),
    mean_ceiling=statistics.mean(values), median_target_mass_in_mixed_sp=statistics.median(mixed),
    median_shuffled_label_ceiling=statistics.median(wrong),
    config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
    source_marker=marker,
    limitations=["Native TRAIN partition/GT incidence only; no variance model utility conclusion.",
                 "GT is diagnosis only and is never an inference input.",
                 "All first64 distinct scenes/expressions retained; no selected responsive cohort.",
                 "A confidence head cannot remove the mask representability constraint.",
                 "Shuffled target is an annotation control, not a negative inference feature."],
)
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
