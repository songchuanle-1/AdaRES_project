"""Read-only checkpoint/process/GPU1 audit after all frozen measurements."""

import hashlib
from pathlib import Path

root = Path(P["root"])
assert (root / ".codex_source_commit").read_text().strip() == "ce2dc75ee9e706da68c5768d9b3c855f33bda7fb"
original_present = Path("/proc/19745").exists()
matching = []
for path in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        command = path.read_bytes()
    except OSError:
        continue
    if b"tools/train.py" in command and b"adares_y28_mask_readback_res.yaml" in command:
        matching.append(int(path.parent.name))
assert not original_present and not matching
base = Path("/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y28_mask_readback_res/20261005_075024")
checkpoints = []
for name, expected in (
    ("last.pth", "ca85d412d0d821aaa59995e00b2d5a8f3edc35bb8f2cd814e039c44595a9a556"),
    ("best.pth", "415422d63c9d2f27735ce99c9882804674f39f2bf58dd93a0bd7eb3ef2b8b0d0"),
):
    path = base / name
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    assert value.hexdigest() == expected
    assert (before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
    checkpoints.append(dict(path=str(path), sha256=expected, unchanged_during_read=True))
gpu_csv = subprocess.check_output([
    "nvidia-smi", "-i", "1", "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits"]).decode().strip()
index, used, utilization = (int(value.strip()) for value in gpu_csv.split(","))
compute = subprocess.check_output([
    "nvidia-smi", "-i", "1", "--query-compute-apps=pid",
    "--format=csv,noheader,nounits"]).decode().strip()
result = dict(complete=True, id="AdaRES-Y28", original_pid_present=original_present,
              matching_config_pids=matching, checkpoints=checkpoints,
              gpu1_csv=gpu_csv, gpu1_compute_pids=compute,
              gpu1_idle=index == 1 and used < 1024 and utilization < 10 and not compute,
              limitations=["Read-only GPU1/checkpoint audit; GPU0 not queried.",
                           "Idle observation is not a training admission or future reservation."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
