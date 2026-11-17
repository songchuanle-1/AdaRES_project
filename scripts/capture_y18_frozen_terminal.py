"""Foreground, GPU1-only frozen full-validation captures for completed Y18."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILL = "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline"
sys.path.insert(0, SKILL)
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("y18", "w2", "b1"))
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner_path = ROOT / "pipeline/headless/.foreground.lockdir/owner.json"


def check_lease():
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    assert owner["token"] == args.token
    assert owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 620


check_lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None, "Pipeline writer busy"
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["user"] == "songcl"
assert server["allowed_gpus"] == [1]
terminal = json.loads((ROOT / ("pipeline/analyses/y18_terminal_" + args.token + ".json")).read_text())
assert terminal["processes_after"]["original_cmd"] is None
last = next(c for c in terminal["checkpoints"] if c["path"].endswith("/last.pth"))
assert last["epoch"] == 30 and last["unchanged_during_read"]
best = next(c for c in terminal["checkpoints"] if c["path"].endswith("/best.pth"))
config = "configs_song/model_paper5/adares_y18_point_measure_res.yaml"
checkpoint, checkpoint_hash = best["path"], best["sha256"]
if args.mode != "y18":
    prior = json.loads((ROOT / ("pipeline/analyses/" + args.mode + "_full_eval_3f09f39c.json")).read_text())
    config, checkpoint, checkpoint_hash = prior["config"], prior["checkpoint"], prior["checkpoint_sha256"]
source = ROOT / "AdaRES/tools/evaluate_adares_frozen_full.py"
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
output = ROOT / ("pipeline/analyses/y18_" + args.mode + "_full_eval_" + args.token + ".json")
assert not output.exists() and not output.with_suffix(".log").exists()
remote_output = "/tmp/adares_y18_" + args.mode + "_" + args.token + ".json"
payload = dict(server=server, config=config, checkpoint=checkpoint,
               checkpoint_hash=checkpoint_hash, source_hash=source_hash,
               output=remote_output, last=last)
program = r'''
import hashlib,os,pathlib,subprocess,sys
p=__PAYLOAD__
s=p['server'];root=pathlib.Path(s['remote_code_root'])
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
assert not pathlib.Path('/proc/7449').exists()
assert not pathlib.Path(p['output']).exists()
sys.path[:0]=[str(root),str(root/'tools'),str(root/'ipdn/lib'),s['site_packages'],s['pointnet_egg']]
os.chdir(str(root));os.environ['PYTHONDONTWRITEBYTECODE']='1'
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert digest(p['last']['path'])==p['last']['sha256']
assert digest(p['checkpoint'])==p['checkpoint_hash']
source=(root/'tools/evaluate_adares_frozen_full.py').read_bytes()
assert hashlib.sha256(source).hexdigest()==p['source_hash']
sys.argv=['evaluate_adares_frozen_full.py','--config',p['config'],'--checkpoint',p['checkpoint'],'--output',p['output'],'--gpu-id','1']
exec(compile(source,'tools/evaluate_adares_frozen_full.py','exec'),{'__name__':'__main__'})
'''.replace("__PAYLOAD__", repr(payload))
start = datetime.now().astimezone().isoformat()
try:
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
         "timeout --signal=TERM --kill-after=10s 580 " + server["python"] + " -S -"],
        input=program.encode("utf-8"), capture_output=True, timeout=600)
    check_lease()
    output.with_suffix(".log").write_bytes(response.stdout + b"\nSTDERR:\n" + response.stderr)
    if response.returncode:
        raise RuntimeError("Evaluation failed; preserve raw log and do not claim a PASS")
    subprocess.run(["scp", "-q", "songcl@192.168.1.163:" + remote_output, str(output)], check=True, timeout=30)
    capture = json.loads(output.read_text())
    assert capture["complete"] and capture["completed_passes"] == 2
    assert capture["checkpoint_sha256"] == checkpoint_hash
    print(json.dumps(dict(mode=args.mode, path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        observed_start_local=start, observed_end_local=datetime.now().astimezone().isoformat(),
        summaries=[c['summary'] for c in capture['passes']], repeat_max=capture['repeat_max_iou_difference'],
        repeat_mean=capture['repeat_miou_difference'], elapsed=capture['elapsed_seconds']), indent=2))
finally:
    lock.close()
