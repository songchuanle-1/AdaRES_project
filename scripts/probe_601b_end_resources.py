"""Bounded end-of-tick observation; never allocate or modify a GPU process."""

import json
from pathlib import Path
import re
import subprocess

assert P["physical_gpu"] is None
root = Path(P["root"])
assert (root / ".codex_source_commit").read_text().strip() == (
    "d18c228fd3796b5a87dbe313075b6ee748b2d51b")
pid = 8877
config = "configs_song/model_paper5/adares_y48_mask_equivalence_res.yaml"
argv = Path("/proc/" + str(pid) + "/cmdline").read_bytes().split(b"\0")
assert b"tools/train.py" in argv and config.encode() in argv
assert argv[argv.index(b"--gpu_ids") + 1] == b"1"
assert Path("/proc/" + str(pid) + "/cwd").resolve() == root
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
        pids=[int(value) for value in pids.splitlines() if value.strip()]))
assert pid in gpus[1]["pids"] and pid not in gpus[0]["pids"]
log = root / "logs/train_adares_y48_mask_equivalence_res_20261015_113625.log"
with log.open("rb") as stream:
    stream.seek(0, 2)
    stream.seek(max(0, stream.tell() - 24000))
    tail = stream.read().decode("utf-8", errors="replace")
assert not any(marker in tail for marker in ("Traceback", "CUDA out of memory", "RuntimeError"))
steps = re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)", tail)
assert steps
epoch, epochs, step, total_steps = map(int, steps[-1])
assert epochs == 30 and (epoch, step) > (1, 284)
result = dict(
    complete=True, id="AdaRES-Y48", pid=pid, config=config,
    GPUs=gpus, native_progress=dict(epoch=epoch, epochs=epochs, step=step, total_steps=total_steps),
    current_PID_config_GPU_commit_verified=True, foreign_jobs_unchanged=True,
    limitations="Bounded native progress and current occupation, not efficacy or whole-log crash absence.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
