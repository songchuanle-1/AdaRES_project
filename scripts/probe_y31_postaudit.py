"""Verify original checkpoints and GPU1 release after frozen measurements."""

import hashlib
from pathlib import Path

root = Path(P["root"])
assert (root / ".codex_source_commit").read_text().strip() == "5f2489de78196e8d89c004e452a665c0946f2a94"
capture = json.loads(Path("/tmp/adares_y31_y31_" + P["token"] + ".json").read_text())
assert capture["complete"] and capture["completed_passes"] == 2
config = capture["config"].encode()
matching = []
for path in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        command = path.read_bytes()
    except OSError:
        continue
    if b"tools/train.py" in command and config in command:
        matching.append(int(path.parent.name))
assert not Path("/proc/66677").exists() and not matching
checkpoints = []
for name, expected in (
    ("last.pth", "67acbc2e92941bce855c111bc76257fcc2bbb4c3f92236289a0017af3103e0b6"),
    ("best.pth", capture["checkpoint_sha256"]),
):
    path = Path(capture["checkpoint"]).with_name(name)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    assert digest.hexdigest() == expected
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    checkpoints.append(dict(path=str(path), sha256=expected, unchanged_during_read=True))
gpu_csv = subprocess.check_output([
    "nvidia-smi", "-i", "1", "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits"
]).decode().strip()
compute = subprocess.check_output([
    "nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
]).decode().strip()
result = dict(complete=True, id="AdaRES-Y31", checkpoints=checkpoints,
              matching_train_pids=matching, original_pid_present=False,
              gpu1_csv=gpu_csv, gpu1_compute_pids=compute,
              scope="Read-only post-measurement audit; GPU0 not queried; idleness is not a reservation.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
