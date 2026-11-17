"""Read-only trained Y36 checkpoint/source audit while preserving Y37."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

root = Path(P["root"])
assert (root / ".codex_source_commit").read_text().strip() == "67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773"
capture = json.loads(Path("/tmp/adares_y36_y36_" + P["token"] + ".json").read_text())
assert capture["complete"] and capture["completed_passes"] == 2
checkpoints = []
for name, expected in (
    ("last.pth", "9d83e12a3d8a2574c81c5b64f1338fbd1dc4d5db10f0f60b6735921fcd3ba66e"),
    ("best.pth", "f504f6ed548affa54eb7292cf5a7d930158be5dfaac291417f4852ad0f0698d1"),
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
assert not Path("/proc/100018").exists()
old = Path("/proc/20348/cmdline").read_bytes().split(b"\0")
assert b"tools/train.py" in old
assert b"configs_song/model_paper5/adares_y37_pair_distribution_res.yaml" in old
assert old[old.index(b"--gpu_ids") + 1] == b"0"
frozen_sources = json.loads(
    Path("/tmp/adares_y36_trained_cost_" + P["token"] + ".json").read_text()
)["source_files"]
assert all(hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() == row["sha256"]
           for row in frozen_sources)
result = dict(
    complete=True, id="AdaRES-Y36", checkpoints=checkpoints,
    original_pid_absent=True, preserved_y37_pid=20348, preserved_y37_gpu=0,
    source_files=frozen_sources, source_bytes_preserved=True,
    remote_clock_at_utc=datetime.now(timezone.utc).isoformat(),
    remote_clock_relation_to_local="unknown; use runner local observation window",
    gpu1_csv=subprocess.check_output([
        "nvidia-smi", "-i", "1", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]).decode().strip(),
    scope="Read-only post-measurement source/checkpoint/job audit; no canonical lifecycle mutation.",
)
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
