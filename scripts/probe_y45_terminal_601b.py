"""Verify original Y45 terminal artifacts without modifying training state."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import torch

root = Path(P["root"])
config = "configs_song/model_paper5/adares_y45_metric_decomposition_res.yaml"
log = root / "logs/train_adares_y45_metric_decomposition_res_20261014_142939.log"
expected = {
    "__init__.py": "8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722",
    "backbone.py": "57db8747d3c38136f568837d70e75ceea23e101900679db4931724ecb57665ca",
    "dec.py": "38415a5d107b3c626cf7f972ea5121ecf940e64874bd68d408b57d192ede9480",
    "ipdn.py": "a7ffbf5dc2625331b7d69224d6f76bf63f615f5a4d6419333b0a7a407979dfb9",
    "loss.py": "f0219ccc4f46af3586baef9b211716a69f5649d681c571313053018619e4905c",
    "sample_model.py": "674e7ef698c6e81ed1c35f14680e1ec748f5c783ed879a6ca72f099b6450bc36",
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            value.update(block)
    return value.hexdigest()


def processes():
    matches = []
    for item in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = item.read_bytes().split(b"\0")
        except OSError:
            continue
        if b"tools/train.py" in argv and config.encode() in argv:
            matches.append(int(item.parent.name))
    assert not matches and not Path("/proc/78001").exists()
    return matches


before = processes()
assert (root / ".codex_source_commit").read_text().strip() == (
    "338a3d7a5b21fa463d116e566b482f0733fde9c2")
sources = {}
for name, sha in expected.items():
    path = root / "ipdn/model_paper5/adares_y45_metric_decomposition" / name
    assert digest(path) == sha
    sources[str(path.relative_to(root))] = sha
sources[config] = digest(root / config)
assert sources[config] == "1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03"
log_stat = log.stat()
with log.open("rb") as stream:
    stream.seek(max(0, log_stat.st_size - 64000))
    raw = stream.read(64000)
tail = raw.decode("utf-8", errors="replace")
assert "Val[30/30]" in tail
assert not any(marker in tail for marker in (
    "Traceback", "CUDA out of memory", "RuntimeError"))
paths = re.findall(r"save last pth to ([^\r\n]+/last\.pth)", tail)
assert paths and len(set(paths)) == 1
last_path = Path(paths[-1]).resolve()
checkpoint_root = (root.parent / "AdaRES_checkpoints/adares_y45_metric_decomposition_res").resolve()
assert checkpoint_root in last_path.parents
rows = []
for name in ("last.pth", "best.pth"):
    path = last_path.parent / name
    stat = path.stat()
    checkpoint = torch.load(str(path), map_location="cpu")
    assert all(key in checkpoint for key in ("model", "optimizer", "scheduler", "meta"))
    rows.append(dict(
        path=str(path), sha256=digest(path), bytes=stat.st_size,
        epoch=int(checkpoint["meta"]["epoch"]),
        best_miou=float(checkpoint["meta"]["best_miou"]),
        scheduler_epoch=int(checkpoint["scheduler"]["last_epoch"])))
    assert (stat.st_size, stat.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
    del checkpoint
assert rows[0]["epoch"] == rows[0]["scheduler_epoch"] == 30
assert (log_stat.st_size, log_stat.st_mtime_ns) == (log.stat().st_size, log.stat().st_mtime_ns)
gpus = []
for gpu in (0, 1):
    used, total, utilization = map(int, subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"]).decode().strip().split(","))
    pids = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"]).decode().strip()
    gpus.append(dict(
        gpu=gpu, memory_MiB=used, total_MiB=total, utilization=utilization,
        pids=[int(pid) for pid in pids.splitlines() if pid.strip()]))
result = dict(
    complete=True, id="AdaRES-Y45", checkpoints=rows, source_hashes=sources,
    matching_pids_before=before, matching_pids_after=processes(), log=str(log),
    bounded_tail_sha256=hashlib.sha256(raw).hexdigest(), tail_last2400=tail[-2400:],
    GPUs=gpus, scientific_status="fixed30_complete_frozen_evaluation_pending",
    limitations=["Bounded log tail and stable checkpoint metadata, not trained efficacy.",
                 "No checkpoint, source, configuration or training state modified."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
