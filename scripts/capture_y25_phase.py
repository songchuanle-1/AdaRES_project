"""Read-only phase-aware Y25 observations; unchanged steps are not a crash."""

from pathlib import Path
import re
import time

pid = 98611
config = "configs_song/model_paper5/adares_y25_mask_feedback_res.yaml"
log = "/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y25_mask_feedback_res_20261003_212806.log"
root = Path(P["root"]).resolve()
assert (root / ".codex_source_commit").read_text().strip() == "94918c1a008e21c72b9478c28f934e8b329e0c97"
captures = []
for _ in range(2):
    assert Path("/proc/" + str(pid) + "/cwd").resolve() == root
    argv = Path("/proc/" + str(pid) + "/cmdline").read_bytes().split(b"\0")
    assert b"tools/train.py" in argv and config.encode() in argv
    assert argv[argv.index(b"--gpu_ids") + 1] == b"1"
    with open(log, "rb") as stream:
        stream.seek(0, 2)
        length = stream.tell()
        stream.seek(max(0, length - 24000))
        text = stream.read().decode("utf-8", errors="replace")
    matches = re.findall(r"(Train|Val)\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)", text)
    assert matches, "No native phase progress in bounded tail"
    phase, epoch, epochs, step, total = matches[-1]
    captures.append(dict(phase=phase, epoch=int(epoch), epochs=int(epochs),
                         step=int(step), total_steps=int(total), log_bytes=length,
                         log_mtime_ns=Path(log).stat().st_mtime_ns,
                         crash_markers=[v for v in ("Traceback", "CUDA out of memory", "RuntimeError") if v in text],
                         tail_bytes=min(length, 24000)))
    if len(captures) == 1:
        time.sleep(5)
first, last = captures
progress = (last["epoch"] > first["epoch"] or
            (last["epoch"] == first["epoch"] and
             ((last["phase"] == first["phase"] and last["step"] > first["step"]) or
              (first["phase"] == "Train" and last["phase"] == "Val"))))
used, util = [int(v.strip()) for v in subprocess.check_output(
    ["nvidia-smi", "-i", "1", "--query-gpu=memory.used,utilization.gpu",
     "--format=csv,noheader,nounits"]).decode().strip().split(",")]
result = dict(complete=True, id="AdaRES-Y25", pid=pid, config=config, log=log,
              captures=captures, native_phase_progress_observed=progress,
              gpu1_device_memory_mib=used, gpu1_device_utilization=util,
              limitations=["Observation completion does not require or imply training progress.",
                           "Crash markers cover two bounded tails only.",
                           "Device memory is not isolated peak; final efficacy unknown; GPU0 not queried."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
