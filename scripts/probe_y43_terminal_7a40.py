"""Read the original resumed Y43 terminal checkpoint; preserve Y44."""

import hashlib
import json
from pathlib import Path
import torch

root = Path(P["root"])
config = "configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml"
work = root.parent / "AdaRES_checkpoints/adares_y43_point_asymmetry_res/20261013_072921"
log = root / "logs/resume_adares_y43_point_asymmetry_res_364e34f425ad445c958c450ac8df9691.log"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def processes():
    found = []
    for item in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = item.read_bytes().split(bytes([0]))
        except OSError:
            continue
        if b"tools/train.py" in argv and config.encode() in argv:
            found.append(int(item.parent.name))
    assert not found and not Path("/proc/15333").exists()
    return found


expected = {
    "__init__.py": "8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722",
    "backbone.py": "1eb4ac3483961e708c23b0abad396c181dbf38c56fcca559d88b9622af46bafd",
    "dec.py": "ad764621524428bf92e70e40506d73a575e949947ae58414a7214a3bb983c27f",
    "ipdn.py": "5eb62b420362cac6003cdf58e5a1fce2be1c8019867e45cbc17f0ae29e355d29",
    "loss.py": "4aa2818948ebfd9fe416b3b310a053384508ef92c3b67831833fece560523084",
    "sample_model.py": "03fd683acc6d49d9b53e634b42bd150a3259a1a15c7dc8fe2badf115870a345b",
}
sources = {}
for name, sha in expected.items():
    path = root / "ipdn/model_paper5/adares_y43_point_asymmetry" / name
    assert digest(path) == sha
    sources[str(path.relative_to(root))] = sha
sources[config] = digest(root / config)
assert sources[config] == "1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03"
before = processes()
rows = []
for name in ("last.pth", "best.pth"):
    path = work / name
    stat = path.stat()
    checkpoint = torch.load(str(path), map_location="cpu")
    assert all(key in checkpoint for key in ("model", "optimizer", "scheduler", "meta"))
    row = dict(path=str(path), sha256=digest(path), bytes=stat.st_size,
               epoch=int(checkpoint["meta"]["epoch"]),
               best_miou=float(checkpoint["meta"]["best_miou"]),
               scheduler_epoch=int(checkpoint["scheduler"]["last_epoch"]))
    assert stat.st_mtime_ns == path.stat().st_mtime_ns
    rows.append(row)
    del checkpoint
assert rows[0]["epoch"] == rows[0]["scheduler_epoch"] == 30
assert rows[0]["sha256"] == "aad8d00e54b30579fb407862c9054f80c9cc04b3ff67d438967ebcf9ea25efb0"
assert rows[1]["sha256"] == "5738e63c1180d3cedb7c467d6be901e3becd95cabb07815293580d8c81dd7c28"
with log.open("rb") as stream:
    stream.seek(max(0, log.stat().st_size - 64000))
    tail = stream.read(64000).decode("utf-8", errors="replace")
assert "save last pth to " + str(work / "last.pth") in tail
assert not any(v in tail for v in ("Traceback", "CUDA out of memory", "RuntimeError"))
result = dict(complete=True, id="AdaRES-Y43", checkpoints=rows, source_hashes=sources,
              matching_pids_before=before, matching_pids_after=processes(),
              log=str(log), bounded_tail_sha256=hashlib.sha256(tail.encode()).hexdigest(),
              scientific_status="fixed30_complete_frozen_evaluation_pending")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
