"""Bounded read-only progress check of the exact canonically admitted Y33."""

from datetime import datetime
from pathlib import Path
import re
import time

pid = 64797
config = "configs_song/model_paper5/adares_y33_token_attention_res.yaml"
log = "/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y33_token_attention_res_20261008_182513.log"
commit = "eaf330166aed361ffc2532a52e352b70cba4e2fd"
root = Path(P["root"]).resolve()
assert Path("/proc/" + str(pid) + "/cwd").resolve() == root
assert (root / ".codex_source_commit").read_text().strip() == commit
argv = Path("/proc/" + str(pid) + "/cmdline").read_bytes().split(b"\0")
assert b"tools/train.py" in argv and config.encode() in argv
assert argv[argv.index(b"--gpu_ids") + 1] == b"1"
captures = []
for _ in range(2):
    assert Path("/proc/" + str(pid)).exists()
    with open(log, "rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-24000))
        text = stream.read().decode("utf-8", errors="replace")
    steps = [(int(a), int(b), int(c), int(d)) for a, b, c, d in
             re.findall(r"Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)", text)]
    assert steps and not any(v in text for v in ("Traceback", "CUDA out of memory", "RuntimeError"))
    captures.append(dict(observed_at_remote=datetime.now().astimezone().isoformat(),
                         epoch=steps[-1][0], epochs=steps[-1][1], step=steps[-1][2],
                         total_steps=steps[-1][3], maximum_tail_bytes=24000,
                         no_crash_in_bounded_tail=True))
    if len(captures) == 1:
        time.sleep(5)
assert (captures[-1]["epoch"], captures[-1]["step"]) > (captures[0]["epoch"], captures[0]["step"])
used, util = [int(v.strip()) for v in subprocess.check_output(
    ["nvidia-smi", "-i", "1", "--query-gpu=memory.used,utilization.gpu",
     "--format=csv,noheader,nounits"]).decode().strip().split(",")]
result = dict(complete=True, id="AdaRES-Y33", pid=pid, config=config, log=log,
              code_commit=commit, captures=captures, native_steps_increased=True,
              actual_gpu=1, gpu1_device_memory_mib=used, gpu1_device_utilization=util,
              limitations=["Two bounded24KB tails, not a whole-log crash-absence claim.",
                           "Device occupation is not isolated candidate peak memory.",
                           "Verified training progress is not efficacy, necessity or Pareto.",
                           "GPU0 not queried/used, prior models/checkpoints untouched."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
