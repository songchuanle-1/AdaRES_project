"""Capture original Y41 terminal metadata and preserve live original Y42."""

import hashlib
import re
from pathlib import Path
import torch

root = Path(P["root"])
config = "configs_song/model_paper5/adares_y41_boundary_probability_res.yaml"
log = root / "logs/train_adares_y41_boundary_probability_res_20261011_162911.log"


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def processes():
    matches = []
    for item in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            cmd = item.read_bytes()
        except OSError:
            continue
        if b"tools/train.py" in cmd and config.encode() in cmd:
            matches.append(int(item.parent.name))
    original = Path("/proc/116906/cmdline")
    preserved = Path("/proc/91215/cmdline").read_bytes().split(b"\0")
    assert b"tools/train.py" in preserved
    assert b"configs_song/model_paper5/adares_y42_point_covariance_res.yaml" in preserved
    assert preserved[preserved.index(b"--gpu_ids") + 1] == b"1"
    return {
        "original_cmd": original.read_bytes().replace(b"\0", b" ").decode()
        if original.exists() else None,
        "matching_y41_pids": matches,
        "preserved_y42_pid": 91215,
        "preserved_y42_gpu": 1,
    }


before = processes()
assert before["original_cmd"] is None and not before["matching_y41_pids"]
with log.open("rb") as stream:
    stream.seek(max(0, log.stat().st_size - 24000))
    raw = stream.read(24000)
tail = raw.decode("utf-8", errors="replace")
saved = re.findall(r"save last pth to ([^\s]+/last\.pth)", tail)
assert saved and "Val[30/30]" in tail
base = Path(saved[-1]).resolve().parent
base.relative_to(Path("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints"))
rows = []
for name in ("last.pth", "best.pth"):
    path = base / name
    stat = path.stat()
    hashed = digest(path)
    value = torch.load(str(path), map_location="cpu")
    meta = value["meta"]
    after_stat = path.stat()
    row = {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": hashed,
        "epoch": int(meta["epoch"]),
        "best_miou": float(meta["best_miou"]),
        "state_keys": {key: key in value for key in ("model", "optimizer", "scheduler")},
        "unchanged_during_read": (stat.st_size, stat.st_mtime_ns)
        == (after_stat.st_size, after_stat.st_mtime_ns),
    }
    assert all(row["state_keys"].values()) and row["unchanged_during_read"]
    rows.append(row)
    del value
assert rows[0]["epoch"] == 30
sources = {
    str(path.relative_to(root)): digest(path)
    for path in sorted((root / "ipdn/model_paper5/adares_y41_boundary_probability").glob("*.py"))
}
assert sources
sources[config] = digest(root / config)
after = processes()
assert after["original_cmd"] is None and not after["matching_y41_pids"]
result = {
    "complete": True,
    "id": "AdaRES-Y41",
    "log": str(log),
    "processes_before": before,
    "processes_after": after,
    "checkpoints": rows,
    "source_hashes": sources,
    "tail_sha256": hashlib.sha256(raw).hexdigest(),
    "tail_last2400": tail[-2400:],
    "bounded_traceback": "Traceback (most recent call last)" in tail,
    "bounded_oom": "CUDA out of memory" in tail,
    "scope": "Original terminal metadata, not efficacy; all checkpoints read only.",
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
