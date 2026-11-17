"""Hash-restricted CRLF-to-LF alignment for only the seven new Y31 files."""

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
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 120
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["allowed_gpus"] == [1] and server["host"] == "192.168.1.163"
old_path = ROOT / ("pipeline/analyses/y31_source_bundle_" + args.token + ".json")
old = json.loads(old_path.read_text())
assert not (ROOT / "pipeline/admissions/AdaRES-Y31.json").exists()
commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
assert commit == "5f2489de78196e8d89c004e452a665c0946f2a94"
new, data, differences = [], {}, []
for item in old["files"]:
    name = item["path"]
    assert (name.startswith("AdaRES/ipdn/model_paper5/adares_y31_mask_shape/")
            or name == "AdaRES/configs_song/model_paper5/adares_y31_mask_shape_res.yaml")
    path = (ROOT / name).resolve()
    assert ROOT.resolve() in path.parents
    before = path.read_bytes()
    assert hashlib.sha256(before).hexdigest() == item["sha256"]
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + name])
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
payload = dict(old=old["files"], new=new, marker="4f5c31f57994b8f9fb640d652c913863dc310a95")
program = """
import ast,fcntl,hashlib,io,json,pathlib,subprocess,sys,tarfile
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
assert (root/'AdaRES/.codex_source_commit').read_text().strip()==P['marker']
lease=open(root/'.y15_gpu1_diagnostic.lock','a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
used,util=[int(v.strip()) for v in subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')]
assert used<1024 and util<10
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
    assert not any(t["id"] == "AdaRES-Y31" for tasks in queue["gpu_queues"].values() for t in tasks)
    command = server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)
    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                               "songcl@192.168.1.163", command],
                              input=buf.getvalue(), capture_output=True, timeout=60)
    assert response.returncode == 0, response.stderr.decode(errors="replace")
    remote = json.loads(response.stdout)
    assert remote["files"] == new
    for name, content in data.items():
        (ROOT / name).write_bytes(content)
    result = dict(complete=True, files=new, original_files=old["files"],
                  normalized_only_files=differences, all_normalized_bytes_match_git=True,
                  ast_equivalent=True, remote_byte_identity=True, physical_independence=True,
                  config_exact_parent_normalized=True, code_commit=commit,
                  gpu_precheck=remote["gpu_precheck"],
                  failed_admission_precheck="Strict git/blob/local/deployed raw-byte mismatch before any admission/queue write; only CRLF/LF.",
                  admission_exists_before=False, original_evidence_unchanged=True,
                  limitations=["Only Y31 files aligned to committed LF; parent/old evidence/goal/gates unchanged.",
                               "Prior native tests executed original CRLF-equivalent AST; no native gate was fabricated or retrospectively rerun.",
                               "No production launch; fresh CPU source controls follow."])
    output = ROOT / ("pipeline/analyses/y31_source_bundle_lf_" + args.token + ".json")
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                         normalized_only_files=differences, gpu_precheck=remote["gpu_precheck"])))
finally:
    lock.close()
