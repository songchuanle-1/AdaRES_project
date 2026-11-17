"""Frozen train8/largest-pair serial cost using each trained best checkpoint."""

import gc
import hashlib
from pathlib import Path
import statistics
import time

import gorilla
import numpy as np
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class

root = Path(P["root"])
assert (root / ".codex_source_commit").read_text().strip() == "94918c1a008e21c72b9478c28f934e8b329e0c97"
captures = {
    name: json.loads(Path("/tmp/adares_y25_" + mode + "_" + P["token"] + ".json").read_text())
    for name, mode in (("candidate", "y25"), ("parent", "w2"))
}
assert all(c["complete"] and c["completed_passes"] == 2 for c in captures.values())
assert all(c["checkpoint_meta"]["epoch"] <= 30 for c in captures.values())
cfg = gorilla.Config.fromfile(captures["candidate"]["config"])
data_cfg = cfg.data.train.copy()
data_cfg.training = False
data_cfg.aug = False
data_cfg.with_elastic = False
torch.set_num_threads(4)
gorilla.set_random_seed(1999)
dataset = build_dataset(data_cfg, get_root_logger())
indices, seen = [], set()
for i, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]["scene_id"] not in seen:
        seen.add(chunk[0]["scene_id"])
        indices.append(i)
    if len(indices) == 8:
        break
indices += [728, 6780]
batches = [dataset.collate_fn([dataset[i]]) for i in indices]
assert [batch["scan_ids"] for batch in batches[-2:]] == [["scene0054_00"], ["scene0497_00"]]
rows, parameters = {}, {}
for name in ("candidate", "parent"):
    c = captures[name]
    h = hashlib.sha256()
    with open(c["checkpoint"], "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == c["checkpoint_sha256"]
    gorilla.set_random_seed(1999)
    model = resolve_model_class(c["config"])(**cfg.model).cuda()
    checkpoint = torch.load(c["checkpoint"], map_location="cpu")
    loaded = model.load_state_dict(checkpoint["model"], strict=False)
    assert not loaded.missing_keys and not set(loaded.unexpected_keys) - {
        "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
    del checkpoint
    model.eval()
    parameters[name] = sum(p.numel() for p in model.parameters())
    readings = []
    with torch.no_grad():
        for index, batch in zip(indices, batches):
            for _ in range(2):
                gorilla.set_random_seed(1999)
                model(batch, mode="predict")
            samples = []
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                gorilla.set_random_seed(1999)
                start = time.monotonic()
                pred = model(batch, mode="predict")
                torch.cuda.synchronize()
                samples.append(dict(seconds=time.monotonic() - start,
                                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
                assert all(v.shape[0] == 128 for v in pred["indi"])
                assert all(v.numel() == len(batch["superpoints"]) for v in pred["pred_pmask"])
            readings.append(dict(scan_ids=batch["scan_ids"], dataset_index=index,
                                 points=len(batch["superpoints"]), readings=samples))
    rows[name] = readings
    del model, pred
    gc.collect()
    torch.cuda.empty_cache()
summary = {}
for name, value in rows.items():
    summary[name] = {}
    for group, chosen in (("train8", value[:8]), ("largest_pair", value[-2:])):
        samples = [s for row in chosen for s in row["readings"]]
        seconds = [s["seconds"] for s in samples]
        summary[name][group] = dict(
            median_seconds=statistics.median(seconds),
            descriptive_p95_seconds=float(np.percentile(seconds, 95)),
            peak_allocated_gib=max(s["peak_allocated_gib"] for s in samples),
            peak_reserved_gib=max(s["peak_reserved_gib"] for s in samples))
result = dict(complete=True, id="AdaRES-Y25", manifest_indices=indices,
              parameters=parameters, available_queries=128, selected_queries=1,
              checkpoints={k: dict(path=c["checkpoint"], sha256=c["checkpoint_sha256"])
                           for k, c in captures.items()},
              rows=rows, summary=summary,
              limitations=["Frozen TRAIN8 plus both preregistered largest-pair scenes, serial native predict batch1.",
                           "Two warmups and three readings per scene; descriptive p95, not reliable tail precision.",
                           "Candidate then parent serial order does not isolate transient hardware effects.",
                           "Cost readings are not causal efficacy, equivalence or Pareto dominance proof."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
