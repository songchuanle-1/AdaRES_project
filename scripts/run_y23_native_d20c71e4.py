"""Execute bounded native tests with immutable input and preserved failure logs."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "d20c71e418354a6b818d9f37ab72fc7b"
assert json.loads((ROOT/"pipeline/headless/.foreground.lockdir/owner.json").read_text())["token"]==TOKEN
mode = sys.argv[1]
assert mode in ("population", "integration", "inference", "contracts", "stress", "parent_stress", "backward", "repeat_audit", "scene_repeat", "inference_profile")
out=ROOT/("pipeline/analyses/y23_"+mode+"_d20c71e4.json")
log=out.with_suffix(".log")
assert not out.exists() and not log.exists()
source=ROOT/"scripts/y23_native_remote_d20c71e4.py"
program=source.read_text().replace("__MODE__", mode)
compile(program,str(source),"exec")
r=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15","songcl@192.168.1.163",
    "timeout --signal=TERM --kill-after=10s 600 /home/songcl/.conda/envs/adares_project/bin/python -S -"],
    input=program.encode(),capture_output=True,timeout=630)
log.write_bytes(r.stdout+b"\nSTDERR:\n"+r.stderr)
if r.returncode:
    raise RuntimeError("Native test failed; retained "+str(log))
subprocess.run(["scp","-q","songcl@192.168.1.163:/tmp/y23_"+mode+"_d20c71e4.json",str(out)],check=True,timeout=60)
result=json.loads(out.read_text())
print(json.dumps({k:v for k,v in result.items() if k not in ("rows","decoder_gradients")},indent=2))
