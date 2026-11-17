"""Read-only Y27 observation with separate training and validation progress."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def latest_progress(text):
    matches = re.findall(
        r"(Train|Val)\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)", text
    )
    if not matches:
        raise ValueError("No native training or validation progress in bounded tail")
    phase, epoch, epochs, step, batches = matches[-1]
    values = tuple(map(int, (epoch, epochs, step, batches)))
    if not (1 <= values[0] <= values[1] == 30 and 0 <= values[2] <= values[3]):
        raise ValueError("Invalid native progress")
    return dict(phase="training" if phase == "Train" else "validation",
                epoch=values[0], epochs=values[1], step=values[2],
                total_steps=values[3])


def progress_key(capture):
    return capture["epoch"], int(capture["phase"] == "validation"), capture["step"]


if "--self-test" in sys.argv:
    train = latest_progress("Train[6/30]: 100%|xxxxxxxx| 4686/4686\r")
    val = latest_progress("Train[6/30]: 100%|x| 4686/4686\rVal[6/30]: 50%|x| 182/363\r")
    val_next = latest_progress("Val[6/30]: 51%|x| 183/363\r")
    next_train = latest_progress("Train[7/30]: 0%|x| 1/4686\r")
    assert train["phase"] == "training" and val["phase"] == "validation"
    assert progress_key(train) < progress_key(val) < progress_key(val_next)
    assert progress_key(val_next) < progress_key(next_train)
    assert not progress_key(val) < progress_key(val)
    for invalid in ("nothing", "Val[6/30]: 101%|x| 364/363", "Train[31/30]: 0%|x| 1/4686"):
        try:
            latest_progress(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid progress accepted")
    print(json.dumps(dict(parser_tests="PASS", training_validation_separated=True,
                          controls=["same-phase", "train-to-val", "val-to-next-epoch",
                                    "unchanged-not-progress", "missing-or-invalid"])))
    raise SystemExit(0)

pid = 9901
config = "configs_song/model_paper5/adares_y27_relative_peer_res.yaml"
log = "/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y27_relative_peer_res_20261004_193016.log"
root = Path(P["root"]).resolve()
captures = []
for observation in range(2):
    assert Path("/proc/" + str(pid) + "/cwd").resolve() == root
    assert (root / ".codex_source_commit").read_text().strip() == "359a89d59dcf3990f4e45b1202d11d3257233e76"
    argv = Path("/proc/" + str(pid) + "/cmdline").read_bytes().split(b"\0")
    assert b"tools/train.py" in argv and config.encode() in argv
    assert argv[argv.index(b"--gpu_ids") + 1] == b"1"
    with open(log, "rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 24000))
        raw = stream.read(24000)
    text = raw.decode("utf-8", errors="replace")
    assert not any(marker in text for marker in ("Traceback", "CUDA out of memory", "RuntimeError"))
    capture = latest_progress(text)
    capture.update(observed_utc=datetime.now(timezone.utc).isoformat(),
                   file_size=size, tail_bytes=len(raw),
                   tail_sha256=hashlib.sha256(raw).hexdigest(),
                   no_crash_in_bounded_tail=True)
    captures.append(capture)
    if observation == 0:
        time.sleep(5)

increased = progress_key(captures[1]) > progress_key(captures[0])
training_increased = increased and all(c["phase"] == "training" for c in captures)
used, util = [int(value.strip()) for value in subprocess.check_output(
    ["nvidia-smi", "-i", "1", "--query-gpu=memory.used,utilization.gpu",
     "--format=csv,noheader,nounits"]).decode().strip().split(",")]
result = dict(
    complete=True, id="AdaRES-Y27", pid=pid, config=config, log=log,
    captures=captures, native_progress_increased=increased,
    native_training_steps_increased=training_increased,
    progress_check="PASS" if increased else "NO_BOUNDED_PROGRESS",
    actual_gpu=1, gpu1_device_memory_mib=used, gpu1_device_utilization=util,
    limitations=["Validation progress is not training-step launch verification.",
                 "No progress in this window is not alone proof of a stalled process.",
                 "Two bounded24KB tails are not whole-log crash proof.",
                 "Device occupation is not isolated peak; efficacy unknown; GPU0 not queried."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
