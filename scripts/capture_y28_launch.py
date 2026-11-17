"""Read-only bounded verification of the exact admitted Y28 process."""

from pathlib import Path
import re
import time

pid = 19745
config = "configs_song/model_paper5/adares_y28_mask_readback_res.yaml"
log = "/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y28_mask_readback_res_20261005_202229.log"
root = Path(P["root"]).resolve()
assert Path("/proc/"+str(pid)+"/cwd").resolve() == root
assert (root/".codex_source_commit").read_text().strip() == "ce2dc75ee9e706da68c5768d9b3c855f33bda7fb"
argv = Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(b"\0")
assert b"tools/train.py" in argv and config.encode() in argv
assert argv[argv.index(b"--gpu_ids")+1] == b"1"
captures = []
for _ in range(2):
    assert Path("/proc/"+str(pid)).exists()
    with open(log, "rb") as f:
        f.seek(0, 2); f.seek(max(0, f.tell()-24000))
        text = f.read().decode("utf-8", errors="replace")
    steps = [(int(a), int(b), int(c), int(d)) for a, b, c, d in
             re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)", text)]
    assert steps and not any(v in text for v in ("Traceback", "CUDA out of memory", "RuntimeError"))
    captures.append(dict(epoch=steps[-1][0], epochs=steps[-1][1], step=steps[-1][2],
                         total_steps=steps[-1][3], tail_bytes=24000,
                         no_crash_in_bounded_tail=True))
    if len(captures) == 1:
        time.sleep(5)
assert (captures[-1]["epoch"], captures[-1]["step"]) > (captures[0]["epoch"], captures[0]["step"])
used, util = [int(v.strip()) for v in subprocess.check_output(
    ["nvidia-smi", "-i", "1", "--query-gpu=memory.used,utilization.gpu",
     "--format=csv,noheader,nounits"]).decode().strip().split(",")]
result = dict(complete=True, id="AdaRES-Y28", pid=pid, config=config, log=log,
              captures=captures, native_steps_increased=True, actual_gpu=1,
              gpu1_device_memory_mib=used, gpu1_device_utilization=util,
              limitations=["Only two bounded24KB tails; crash absence not whole-log proof.",
                           "Device occupation is not isolated candidate peak memory.",
                           "Training progress is not trained efficacy. GPU0 not queried."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
