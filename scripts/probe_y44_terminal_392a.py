"""Read SAME Y44 terminal state without restarting or modifying checkpoints."""

import hashlib
import json
from pathlib import Path
import re
import torch

root = Path(P["root"])
config = "configs_song/model_paper5/adares_y44_coordinate_feature_res.yaml"
work = root.parent / "AdaRES_checkpoints/adares_y44_coordinate_feature_res/20261013_072931"
log = root / "logs/resume_adares_y44_coordinate_feature_res_364e34f425ad445c958c450ac8df9691.log"
expected = {
    "__init__.py": "8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722",
    "backbone.py": "57db8747d3c38136f568837d70e75ceea23e101900679db4931724ecb57665ca",
    "dec.py": "88d5c203c2bcbf57ad713b7488d34c03ddada1d3ee25825f4c5378d4a98015fe",
    "ipdn.py": "98bef2d27e2a3008e291f48633ddce7556c6f085473564f6a9f71af023265819",
    "loss.py": "9e6e5925523114caba6213d181b11181abd2820b292730999986b068816046d1",
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
    assert not matches and not Path("/proc/15520").exists()
    return matches


before = processes()
sources = {}
for name, sha in expected.items():
    path = root / "ipdn/model_paper5/adares_y44_coordinate_feature" / name
    assert digest(path) == sha
    sources[str(path.relative_to(root))] = sha
sources[config] = digest(root / config)
assert sources[config] == "1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03"
rows = []
for name in ("last.pth", "best.pth"):
    path = work / name
    stat = path.stat()
    checkpoint = torch.load(str(path), map_location="cpu")
    assert all(key in checkpoint for key in ("model", "optimizer", "scheduler", "meta"))
    rows.append(dict(path=str(path), sha256=digest(path), bytes=stat.st_size,
                     epoch=int(checkpoint["meta"]["epoch"]),
                     best_miou=float(checkpoint["meta"]["best_miou"]),
                     scheduler_epoch=int(checkpoint["scheduler"]["last_epoch"])))
    assert (stat.st_size, stat.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
    del checkpoint
assert rows[0]["epoch"] == rows[0]["scheduler_epoch"] == 30
with log.open("rb") as stream:
    stream.seek(max(0, log.stat().st_size - 64000))
    raw = stream.read(64000)
tail = raw.decode("utf-8", errors="replace")
assert "Val[30/30]" in tail
assert "save last pth to " + str(work / "last.pth") in tail
assert not any(marker in tail for marker in ("Traceback", "CUDA out of memory", "RuntimeError"))
last_metrics = re.findall(r"mIoU[:=]\s*([0-9.]+)", tail)
result = dict(complete=True, id="AdaRES-Y44", checkpoints=rows, source_hashes=sources,
              matching_pids_before=before, matching_pids_after=processes(), log=str(log),
              bounded_tail_sha256=hashlib.sha256(raw).hexdigest(), tail_last2400=tail[-2400:],
              bounded_miou_strings=last_metrics[-4:],
              gpu_occupation_csv=subprocess.check_output(
                  ["nvidia-smi", "-i", "0,1", "--query-gpu=index,memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits"]).decode().strip(),
              compute_processes_csv=subprocess.check_output(
                  ["nvidia-smi", "-i", "0,1", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                   "--format=csv,noheader,nounits"]).decode().strip(),
              scientific_status="fixed30_complete_frozen_evaluation_pending")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
