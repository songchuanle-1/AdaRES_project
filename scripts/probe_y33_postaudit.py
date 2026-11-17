"""Read-only Y33 checkpoint and GPU release verification after measurements."""

import hashlib
from pathlib import Path

root = Path(P["root"])
assert (root / ".codex_source_commit").read_text().strip() == "b49b744dc2943a38ab4c3aeb788acc6a2cb90011"
capture = json.loads(Path("/tmp/adares_y33_y33_" + P["token"] + ".json").read_text())
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
assert not Path("/proc/64797").exists() and not matching
checkpoints = []
for name, expected in (
    ("last.pth", "ec6c4dd70133b5b55866d50b62a8e95a3d4fdd60b4664eb1d1829a8f637dce37"),
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
result = dict(complete=True, id="AdaRES-Y33", checkpoints=checkpoints,
              matching_train_pids=matching, original_pid_present=False,
              gpu1_csv=gpu_csv, gpu1_compute_pids=compute,
              scope="Read-only post-measurement audit; idleness is not a reservation.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
