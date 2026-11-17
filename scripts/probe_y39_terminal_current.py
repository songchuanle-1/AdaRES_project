"""Capture original Y39 terminal checkpoint and preserve live Y41."""

import hashlib
import re
from pathlib import Path
import torch

root = Path(P["root"])
config = "configs_song/model_paper5/adares_y39_point_distribution_res.yaml"
log = root / "logs/train_adares_y39_point_distribution_res_20261010_202034.log"


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def processes():
    matching = []
    original = Path("/proc/102267/cmdline")
    for item in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            cmd = item.read_bytes()
        except OSError:
            continue
        if b"tools/train.py" in cmd and config.encode() in cmd:
            matching.append(int(item.parent.name))
    preserved = Path("/proc/116906/cmdline").read_bytes().split(b"\0")
    assert b"tools/train.py" in preserved
    assert b"configs_song/model_paper5/adares_y41_boundary_probability_res.yaml" in preserved
    assert preserved[preserved.index(b"--gpu_ids") + 1] == b"0"
    return {
        "original_cmd": original.read_bytes().replace(b"\0", b" ").decode()
                        if original.exists() else None,
        "matching_y39_pids": matching, "preserved_y41_pid": 116906,
        "preserved_y41_gpu": 0,
    }


before = processes()
assert before["original_cmd"] is None and not before["matching_y39_pids"]
with log.open("rb") as stream:
    stream.seek(max(0, log.stat().st_size - 24000))
    raw = stream.read(24000)
tail = raw.decode("utf-8", errors="replace")
saved = re.findall(r"save last pth to ([^\s]+/last\.pth)", tail)
assert saved, "Original checkpoint path missing; no inferred completion"
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
    rows.append({
        "path": str(path), "bytes": stat.st_size, "sha256": hashed,
        "epoch": int(meta["epoch"]), "best_miou": float(meta["best_miou"]),
        "state_keys": {key: key in value for key in ("model", "optimizer", "scheduler")},
        "unchanged_during_read": (stat.st_size, stat.st_mtime_ns)
                                == (after_stat.st_size, after_stat.st_mtime_ns),
    })
    del value
assert rows[0]["epoch"] == 30 and all(row["unchanged_during_read"] for row in rows)
sources = {
    str(path.relative_to(root)): digest(path)
    for path in sorted((root / "ipdn/model_paper5/adares_y39_point_distribution").glob("*.py"))
}
sources[config] = digest(root / config)
after = processes()
assert after["original_cmd"] is None and not after["matching_y39_pids"]
result = {
    "complete": True, "id": "AdaRES-Y39", "log": str(log),
    "processes_before": before, "processes_after": after, "checkpoints": rows,
    "source_hashes": sources,
    "tail_sha256": hashlib.sha256(raw).hexdigest(), "tail_last2400": tail[-2400:],
    "bounded_traceback": "Traceback (most recent call last)" in tail,
    "bounded_oom": "CUDA out of memory" in tail,
    "gpu1_csv": subprocess.check_output([
        "nvidia-smi", "-i", "1", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]).decode(),
    "gpu1_compute_pids": subprocess.check_output([
        "nvidia-smi", "-i", "1", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]).decode(),
    "scope": "Original terminal and bounded read only, not efficacy.",
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
