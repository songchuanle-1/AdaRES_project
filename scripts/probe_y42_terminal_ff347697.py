"""Read original Y42 terminal states without touching Y43 GPU0 training."""

import hashlib
import json
from pathlib import Path
import re
import torch

root = Path(P["root"])
config = "configs_song/model_paper5/adares_y42_point_covariance_res.yaml"
log = root / "logs/train_adares_y42_point_covariance_res_20261011_223156.log"


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
    protected = Path("/proc/13155/cmdline").read_bytes().split(b"\0")
    assert b"tools/train.py" in protected
    assert b"configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml" in protected
    assert protected[protected.index(b"--gpu_ids") + 1] == b"0"
    return {"matching_y42_pids": matches, "original_y42_pid_absent": not Path("/proc/91215").exists(),
            "preserved_y43_pid": 13155, "preserved_y43_gpu": 0,
            "preserved_y43_command": protected}


before = processes()
assert not before["matching_y42_pids"] and before["original_y42_pid_absent"]
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
    row = {"path": str(path), "bytes": stat.st_size, "sha256": hashed,
           "epoch": int(meta["epoch"]), "best_miou": float(meta["best_miou"]),
           "state_keys": {key: key in value for key in ("model", "optimizer", "scheduler")},
           "unchanged_during_read": (stat.st_size, stat.st_mtime_ns) == (after_stat.st_size, after_stat.st_mtime_ns)}
    assert all(row["state_keys"].values()) and row["unchanged_during_read"]
    rows.append(row)
    del value
assert rows[0]["epoch"] == 30
sources = {str(path.relative_to(root)): digest(path)
           for path in sorted((root / "ipdn/model_paper5/adares_y42_point_covariance").glob("*.py"))}
sources[config] = digest(root / config)
protected_sources = {str(path.relative_to(root)): digest(path)
                     for path in sorted((root / "ipdn/model_paper5/adares_y43_point_asymmetry").glob("*.py"))}
protected_sources["configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml"] = digest(
    root / "configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml")
assert protected_sources["ipdn/model_paper5/adares_y43_point_asymmetry/ipdn.py"] == "5eb62b420362cac6003cdf58e5a1fce2be1c8019867e45cbc17f0ae29e355d29"
assert protected_sources["ipdn/model_paper5/adares_y43_point_asymmetry/dec.py"] == "ad764621524428bf92e70e40506d73a575e949947ae58414a7214a3bb983c27f"
after = processes()
assert before == after
for value in (before, after):
    value["preserved_y43_command"] = [v.decode() for v in value["preserved_y43_command"]]
result = {"complete": True, "id": "AdaRES-Y42", "log": str(log),
          "processes_before": before, "processes_after": after, "checkpoints": rows,
          "source_hashes": sources, "protected_source_hashes": protected_sources,
          "tail_sha256": hashlib.sha256(raw).hexdigest(), "tail_last2400": tail[-2400:],
          "bounded_traceback": "Traceback (most recent call last)" in tail,
          "bounded_oom": "CUDA out of memory" in tail,
          "scope": "Original terminal metadata, not efficacy; checkpoints read-only."}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
