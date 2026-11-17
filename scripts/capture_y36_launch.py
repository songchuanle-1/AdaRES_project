"""Independently verify native Y36 growth and original Y35 preservation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 100
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["allowed_gpus"] == [0, 1] and server["host"] == "192.168.1.163"
queue = monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
task = next(t for t in queue if t["id"] == "AdaRES-Y36")
assert task["status"] == "running" and str(task["gpu"]) == "1"
old_receipt = json.loads((ROOT / "pipeline/admissions/AdaRES-Y35.json").read_text())
old_bundle_path = old_receipt["gates"]["independence"]["evidence"][0]["path"]
old_bundle = json.loads((ROOT / old_bundle_path).read_text())
new_bundle = json.loads((ROOT / ("pipeline/analyses/y36_source_bundle_lf_" + args.token + ".json")).read_text())
payload = dict(task=task, y35_files=old_bundle["files"], y36_files=new_bundle["files"])
program = """
import hashlib,json,pathlib,re,subprocess,time
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
code=root/'AdaRES'
t=P['task']
assert (code/'.codex_source_commit').read_text().strip()==t['code_commit']
for item in P['y35_files']+P['y36_files']:
    assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256']
pid=t['pid']
assert pathlib.Path('/proc/'+str(pid)+'/cwd').resolve()==code
argv=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\\0')
assert b'tools/train.py' in argv and t['config'].encode() in argv
assert argv[argv.index(b'--gpu_ids')+1]==b'1'
old_argv=pathlib.Path('/proc/108331/cmdline').read_bytes().split(b'\\0')
assert pathlib.Path('/proc/108331/cwd').resolve()==code
assert b'tools/train.py' in old_argv and b'configs_song/model_paper5/adares_y35_language_evidence_res.yaml' in old_argv
assert old_argv[old_argv.index(b'--gpu_ids')+1]==b'0'
captures=[]
for i in range(2):
    assert pathlib.Path('/proc/'+str(pid)).exists()
    with open(t['log'],'rb') as stream:
        stream.seek(0,2)
        stream.seek(max(0,stream.tell()-24000))
        text=stream.read().decode('utf-8',errors='replace')
    steps=[tuple(map(int,m)) for m in re.findall(r'Train\\[(\\d+)/(\\d+)\\]:[^\\r\\n]*?\\|\\s*(\\d+)/(\\d+)',text)]
    assert steps and not any(v in text for v in ('Traceback','CUDA out of memory','RuntimeError'))
    captures.append(dict(epoch=steps[-1][0],epochs=steps[-1][1],step=steps[-1][2],total_steps=steps[-1][3],maximum_tail_bytes=24000))
    if i==0: time.sleep(5)
assert (captures[-1]['epoch'],captures[-1]['step'])>(captures[0]['epoch'],captures[0]['step'])
assert all(c['epochs']==30 for c in captures)
used,util=map(int,subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(','))
print(json.dumps(dict(complete=True,id='AdaRES-Y36',pid=pid,config=t['config'],log=t['log'],code_commit=t['code_commit'],actual_gpu=1,native_steps_increased=True,captures=captures,gpu1_device_memory_mib=used,gpu1_device_utilization=util,preserved_y35_pid=108331,preserved_y35_gpu=0,old_new_source_hashes_verified=True,limitations=['Two24KB tails only; not whole-log crash absence.','Device occupation is not isolated peak or efficacy.','Original Y35 source/PID/config/GPU0 preserved, not restarted.','Global marker advances additively; old code hashes unchanged.'])))
"""
out = ROOT / ("pipeline/analyses/y36_launch_" + args.token + ".json")
assert not out.exists() and not out.with_suffix(".log").exists()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
started = datetime.now().astimezone().isoformat()
try:
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
         server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)],
        capture_output=True, timeout=60)
    out.with_suffix(".log").write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0, "Capture failed; preserve raw log, never blindly restart"
    result = json.loads(response.stdout)
    result.update(observed_start_local=started, observed_end_local=datetime.now().astimezone().isoformat())
    assert result["complete"] and result["native_steps_increased"]
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(path=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                         summary=result), indent=2))
finally:
    lock.close()
