"""Independent increasing native steps, then same-ID training disposition."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "5e172bd47ee3450dbee928bdecaede7f"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 160
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["allowed_gpus"] == [0, 1]
queue = monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"]
task = next(t for t in queue if t["id"] == "AdaRES-Y41")
assert task["status"] == "running" and str(task["gpu"]) == "0"
old_receipt = json.loads((ROOT / "pipeline/admissions/AdaRES-Y39.json").read_text())
old_bundle = json.loads((ROOT / old_receipt["gates"]["independence"]["evidence"][0]["path"]).read_text())
new_bundle = json.loads((ROOT / ("pipeline/analyses/y41_source_bundle_" + TOKEN + ".json")).read_text())
y37 = json.loads((ROOT / ("pipeline/analyses/y37_terminal_current_" + TOKEN + ".json")).read_text())
payload = {"task": task, "files": old_bundle["files"] + new_bundle["files"],
           "y37_sources": y37["source_hashes"]}
program = r'''
import hashlib,json,pathlib,re,subprocess,time
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
code=root/'AdaRES'
t=P['task']
assert (code/'.codex_source_commit').read_text().strip()==t['code_commit']
for row in P['files']:
    assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
for path,sha in P['y37_sources'].items():
    assert hashlib.sha256((code/path).read_bytes()).hexdigest()==sha
pid=t['pid']
assert pathlib.Path('/proc/'+str(pid)+'/cwd').resolve()==code
argv=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\0')
assert b'tools/train.py' in argv and t['config'].encode() in argv
assert argv[argv.index(b'--gpu_ids')+1]==b'0'
old=pathlib.Path('/proc/102267/cmdline').read_bytes().split(b'\0')
assert b'configs_song/model_paper5/adares_y39_point_distribution_res.yaml' in old
assert old[old.index(b'--gpu_ids')+1]==b'1'
captures=[]
limit=time.monotonic()+90
while len(captures)<2:
    assert pathlib.Path('/proc/'+str(pid)).exists()
    with open(t['log'],'rb') as stream:
        stream.seek(0,2)
        stream.seek(max(0,stream.tell()-24000))
        text=stream.read().decode('utf-8',errors='replace')
    assert not any(v in text for v in ('Traceback','CUDA out of memory','RuntimeError'))
    steps=[tuple(map(int,m)) for m in re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)',text)]
    if steps:
        epoch,total,step,totalsteps=steps[-1]
        if not captures or (epoch,step)>(captures[0]['epoch'],captures[0]['step']):
            captures.append(dict(epoch=epoch,epochs=total,step=step,total_steps=totalsteps,
                                 maximum_tail_bytes=24000))
    if len(captures)<2:
        assert time.monotonic()<limit,'No increasing native progress; never restart blindly'
        time.sleep(5)
assert all(c['epochs']==30 for c in captures)
assert pathlib.Path('/proc/102267/cmdline').read_bytes().split(b'\0')==old
for row in P['files']:
    assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
used,util=map(int,subprocess.check_output([
    'nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu',
    '--format=csv,noheader,nounits']).decode().strip().split(','))
print(json.dumps(dict(complete=True,id='AdaRES-Y41',pid=pid,config=t['config'],
    log=t['log'],code_commit=t['code_commit'],actual_gpu=0,native_steps_increased=True,
    captures=captures,gpu0_device_memory_mib=used,gpu0_device_utilization=util,
    preserved_y39_pid=102267,preserved_y39_gpu=1,source_hashes_verified=True,
    limitations=['Bounded original log tails, not whole-log crash absence.',
                 'Device occupation is not isolated peak or efficacy.',
                 'Y39 original job and Y37 source bytes unchanged.'])))
'''
output = ROOT / ("pipeline/analyses/y41_launch_" + TOKEN + ".json")
assert not output.exists() and not output.with_suffix(".log").exists()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
started = datetime.now().astimezone().isoformat()
try:
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
         server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)],
        capture_output=True, timeout=120)
    output.with_suffix(".log").write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0, "Preserve raw evidence; do not restart"
    proof = json.loads(response.stdout)
    proof.update(observed_start_local=started, observed_end_local=datetime.now().astimezone().isoformat())
    assert proof["complete"] and proof["native_steps_increased"]
    output.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    queue = monitor.load_queue(str(ROOT))
    current = next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y41")
    assert current["pid"] == proof["pid"] and current["code_commit"] == proof["code_commit"]
    assert current["launch_verified"] is True
    assert next(t for t in queue["gpu_queues"]["server-2"] if t["id"] == "AdaRES-Y39")["pid"] == 102267
    path = ROOT / "pipeline/research_work.json"
    work = json.loads(path.read_text(encoding="utf-8"))
    candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y41")
    assert candidate["stage"] == "validation"
    refs = [
        "pipeline/admissions/AdaRES-Y41.json",
        "pipeline/decisions/20261011-162854_next_batch.md",
        "pipeline/analyses/y41_launch_" + TOKEN + ".json",
        "pipeline/analyses/y41_dispatch_" + TOKEN + ".json",
        "pipeline/analyses/y41_parent_semantics_" + TOKEN + ".json",
        "scripts/admit_dispatch_y41.py", "scripts/capture_finalize_y41_launch.py",
    ]
    for ref in refs:
        entry = {"path": ref, "sha256": hashlib.sha256((ROOT / ref).read_bytes()).hexdigest()}
        prior = next((e for e in candidate["evidence"] if e["path"] == ref), None)
        assert prior is None or prior == entry
        if prior is None:
            candidate["evidence"].append(entry)
    candidate.update(
        stage="training", code_commit=proof["code_commit"], scientific_status="training_not_efficacy",
        next_action="Preserve original Y41 GPU0 and Y39 GPU1 fixed30/seed1999. "
                    "Canonical terminal/crash events only; never blindly restart. "
                    "On Y41 terminal run frozen full9508/141 candidate/W2/B1 twice "
                    "each, FIRST primary, scene bootstrap10000seed1999, TRAINcuts "
                    "1487/3325 subgroup harm versus either, repeat noise and trained "
                    "paired cost. Retain all failed controls and adverse initial losses.",
    )
    path.write_text(json.dumps(work, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": output.relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "summary": proof, "stage": candidate["stage"],
                      "active": sum(t["status"] in ("running", "waiting", "claiming")
                                    for t in queue["gpu_queues"]["server-2"])}, indent=2))
finally:
    lock.close()
