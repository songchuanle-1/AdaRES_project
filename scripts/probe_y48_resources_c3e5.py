"""Read live authorized cards without assuming that a queue vacancy is idle."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

assert P["physical_gpu"] is None
root = Path(P["root"])
pid = 8877
assert (root / ".codex_source_commit").read_text().strip() == "d18c228fd3796b5a87dbe313075b6ee748b2d51b"
assert Path("/proc/8877/cwd").resolve() == root
argv = Path("/proc/8877/cmdline").read_bytes().split(b"\0")
assert b"tools/train.py" in argv
assert b"configs_song/model_paper5/adares_y48_mask_equivalence_res.yaml" in argv
assert argv[argv.index(b"--gpu_ids") + 1] == b"1"
log = root / "logs/train_adares_y48_mask_equivalence_res_20261015_113625.log"


def snapshot():
    rows = []
    for gpu in (0, 1):
        raw = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"]).decode().strip()
        util, used, total = map(int, raw.split(","))
        pids = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"]).decode().strip()
        rows.append(dict(gpu=gpu, utilization=util, memory_MiB=used, total_MiB=total,
                         pids=[int(value) for value in pids.splitlines() if value.strip()]))
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 65536))
        tail = stream.read().decode(errors="replace")
    assert not any(marker in tail for marker in ("Traceback", "CUDA out of memory", "RuntimeError"))
    steps = re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)", tail)
    assert steps and pid in rows[1]["pids"] and pid not in rows[0]["pids"]
    return dict(GPUs=rows, native_step=list(map(int, steps[-1])))


first = snapshot()
time.sleep(5)
second = snapshot()
assert (second["native_step"][0], second["native_step"][2]) > (
    first["native_step"][0], first["native_step"][2])
free = [row["gpu"] for row in second["GPUs"]
        if row["memory_MiB"] < 1024 and row["utilization"] < 10 and not row["pids"]]
sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
           for path in (root / "ipdn/model_paper5/adares_y48_mask_equivalence").glob("*.py")}
sources["configs_song/model_paper5/adares_y48_mask_equivalence_res.yaml"] = hashlib.sha256(
    (root / "configs_song/model_paper5/adares_y48_mask_equivalence_res.yaml").read_bytes()).hexdigest()
result = dict(
    complete=True, id="AdaRES-Y48", original_pid=pid, GPU=1, native_steps_increased=True,
    first=first, second=second, free_authorized_gpus=free, source_hashes=sources,
    scientific_status="training_not_efficacy",
    limitations=["Local runner timestamps are authoritative; server clock is not used.",
                 "No GPU allocated, no foreign/current job stopped or restarted.",
                 "No admission, completion, efficacy or full-log crash-absence claim.",
                 "Any later diagnostic must independently reacquire a live-idle exclusive GPU lease."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
