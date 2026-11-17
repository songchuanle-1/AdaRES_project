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
assert (root / ".codex_source_commit").read_text().strip() == "ce2dc75ee9e706da68c5768d9b3c855f33bda7fb"
assert not Path("/proc/19745").exists()
source_manifest = [
    ("ipdn/model_paper5/adares_y28_mask_readback/__init__.py", "8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722"),
    ("ipdn/model_paper5/adares_y28_mask_readback/backbone.py", "57db8747d3c38136f568837d70e75ceea23e101900679db4931724ecb57665ca"),
    ("ipdn/model_paper5/adares_y28_mask_readback/dec.py", "d4be08939ee151b5d8d92647bdd5594a609876ed9f290a3b64b3f7c0cf6b5e8f"),
    ("ipdn/model_paper5/adares_y28_mask_readback/ipdn.py", "55160dfb062353054f6d2d5a09d971a6dc779be2e8fe34e6906187576a623ad9"),
    ("ipdn/model_paper5/adares_y28_mask_readback/loss.py", "9e6e5925523114caba6213d181b11181abd2820b292730999986b068816046d1"),
    ("ipdn/model_paper5/adares_y28_mask_readback/sample_model.py", "674e7ef698c6e81ed1c35f14680e1ec748f5c783ed879a6ca72f099b6450bc36"),
    ("configs_song/model_paper5/adares_y28_mask_readback_res.yaml", "1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03"),
]
assert all(hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
           for path, expected in source_manifest)
captures = {
    name: json.loads(Path("/tmp/adares_y28_" + mode + "_" + P["token"] + ".json").read_text())
    for name, mode in (("candidate", "y28"), ("parent", "w2"))
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
assert parameters == {"candidate": 164892293, "parent": 164629637}
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
result = dict(complete=True, id="AdaRES-Y28", manifest_indices=indices,
              source_files=[dict(path=path, sha256=sha) for path, sha in source_manifest],
              parameters=parameters, available_queries=128, selected_queries=1,
              checkpoints={k: dict(path=c["checkpoint"], sha256=c["checkpoint_sha256"])
                           for k, c in captures.items()},
              rows=rows, summary=summary,
              limitations=["Frozen TRAIN8 plus both preregistered largest-pair scenes, serial native predict batch1.",
                           "Two warmups and three readings per scene; descriptive p95, not reliable tail precision.",
                           "Candidate then parent serial order does not isolate transient hardware effects.",
                           "Cost readings are not causal efficacy, equivalence or Pareto dominance proof."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
