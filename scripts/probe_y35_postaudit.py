"""Read-only frozen Y35 checkpoint/source and original job preservation audit."""

import hashlib
from pathlib import Path

root = Path(P["root"])
assert (root / ".codex_source_commit").read_text().strip() == "6725f9bd6a8f9b47addae6a4b8336d004f73f1ae"
capture = json.loads(Path("/tmp/adares_y35_y35_" + P["token"] + ".json").read_text())
assert capture["complete"] and capture["completed_passes"] == 2
checkpoints = []
for name, sha in (
    ("last.pth", "daf9cd828bef52c7fb09ad7e36a41f6463eb66beb4850669fa047513901bc412"),
    ("best.pth", capture["checkpoint_sha256"]),
):
    path = Path(capture["checkpoint"]).with_name(name)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    after = path.stat()
    assert digest.hexdigest() == sha
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    checkpoints.append(dict(path=str(path), sha256=sha, unchanged_during_read=True))
assert not Path("/proc/108331").exists()
old = Path("/proc/100018/cmdline").read_bytes().split(b"\0")
assert b"tools/train.py" in old and b"configs_song/model_paper5/adares_y36_local_adjacency_res.yaml" in old
assert old[old.index(b"--gpu_ids")+1] == b"1"
result = dict(complete=True, id="AdaRES-Y35", checkpoints=checkpoints,
              original_pid_absent=True, preserved_y36_pid=100018, preserved_y36_gpu=1,
              gpu0_csv=subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=index,memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits"]).decode().strip(),
              scope="Read-only post-measurement checkpoint/job audit; idle is not a reservation.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
