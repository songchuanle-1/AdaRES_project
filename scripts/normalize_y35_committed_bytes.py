"""Align only Y35's tested bytes to committed LF, under exact before hashes."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--commit", required=True)
args = parser.parse_args()
owner = json.loads((ROOT/"pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"])-datetime.now(timezone.utc)).total_seconds() > 120
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["allowed_gpus"] == [0, 1] and server["host"] == "192.168.1.163"
old = json.loads((ROOT/("pipeline/analyses/y35_source_bundle_"+args.token+".json")).read_text())
assert not (ROOT/"pipeline/admissions/AdaRES-Y35.json").exists()
commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
assert commit == args.commit
new, data, differences = [], {}, []
for item in old["files"]:
    name = item["path"]
    assert name.startswith("AdaRES/ipdn/model_paper5/adares_y35_language_evidence/") or name == "AdaRES/configs_song/model_paper5/adares_y35_language_evidence_res.yaml"
    path = (ROOT/name).resolve()
    assert ROOT.resolve() in path.parents
    before = path.read_bytes()
    assert hashlib.sha256(before).hexdigest() == item["sha256"]
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", commit+":"+name])
    assert before.replace(b"\r\n", b"\n") == blob
    if name.endswith(".py"):
        assert ast.dump(ast.parse(before.decode())) == ast.dump(ast.parse(blob.decode()))
    data[name] = blob
    new.append(dict(path=name, sha256=hashlib.sha256(blob).hexdigest()))
    if before != blob:
        differences.append(name)
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as archive:
    for name, content in data.items():
        info = tarfile.TarInfo(name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
payload = dict(old=old["files"], new=new, marker="eaf330166aed361ffc2532a52e352b70cba4e2fd")
program = """
import ast,fcntl,hashlib,io,json,pathlib,subprocess,sys,tarfile
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
assert (root/'AdaRES/.codex_source_commit').read_text().strip()==P['marker']
lease=open(root/'.foreground_gpu0_diagnostic.lock','a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
used,util=[int(v.strip()) for v in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')]
assert used<1024 and util<10
assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:')
members=archive.getmembers()
assert sorted(m.name for m in members)==sorted(r['path'] for r in P['old'])
for m in members:
    assert m.isfile()
    path=(root/m.name).resolve()
    assert root.resolve() in path.parents
    before=path.read_bytes()
    assert hashlib.sha256(before).hexdigest()==next(r['sha256'] for r in P['old'] if r['path']==m.name)
    after=archive.extractfile(m).read()
    assert before.replace(b'\\r\\n',b'\\n')==after
    assert hashlib.sha256(after).hexdigest()==next(r['sha256'] for r in P['new'] if r['path']==m.name)
    if m.name.endswith('.py'):
        assert ast.dump(ast.parse(before.decode()))==ast.dump(ast.parse(after.decode()))
for m in members:
    (root/m.name).write_bytes(archive.extractfile(m).read())
actual=[dict(path=r['path'],sha256=hashlib.sha256((root/r['path']).read_bytes()).hexdigest()) for r in P['new']]
assert actual==P['new']
print(json.dumps(dict(files=actual,gpu_precheck=dict(memory_mib=used,utilization=util))))
"""
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue = monitor.load_queue(str(ROOT))
    assert not any(t["id"] == "AdaRES-Y35" for tasks in queue["gpu_queues"].values() for t in tasks)
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
         server["python"]+" -c "+shlex.quote("P="+repr(payload)+"\n"+program)],
        input=buf.getvalue(), capture_output=True, timeout=60)
    assert response.returncode == 0, response.stderr.decode(errors="replace")
    remote = json.loads(response.stdout)
    assert remote["files"] == new
    for name, content in data.items():
        (ROOT/name).write_bytes(content)
    out = ROOT/("pipeline/analyses/y35_source_bundle_lf_"+args.token+".json")
    assert not out.exists()
    out.write_text(json.dumps(dict(
        complete=True, files=new, original_files=old["files"],
        normalized_only_files=differences, all_normalized_bytes_match_git=True,
        ast_equivalent=True, remote_byte_identity=True, physical_independence=True,
        config_exact_parent_normalized=True, code_commit=commit,
        physical_gpu=0, gpu_precheck=remote["gpu_precheck"],
        original_evidence_unchanged=True,
        limitations=["Only new7 Y35 files; Y33 and original evidence unchanged.",
                     "Original native probes AST-equivalent, fresh CPU test follows.",
                     "Not an admission or production launch."]), indent=2), encoding="utf-8")
    print(json.dumps(dict(path=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                         normalized_only_files=differences)))
finally:
    lock.close()
