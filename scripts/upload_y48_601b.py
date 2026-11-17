"""Upload only seven new hash-bound Y48 diagnostic files."""

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
bundle_path = ROOT / ("pipeline/analyses/y48_local_bundle_601b_" + args.token + ".json")
bundle = json.loads(bundle_path.read_text())
records = bundle["files"]
assert bundle["complete"] and len(records) == 7
for row in records:
    source = ROOT / row["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]
    if source.suffix == ".py":
        tree = ast.parse(source.read_text(encoding="utf-8"))
        compile(tree, str(source), "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "model_paper5" not in (node.module or "")
            if isinstance(node, ast.Import):
                assert not any("model_paper5" in alias.name for alias in node.names)
protected = dict(bundle["parent_source_hashes"])
terminal = json.loads((ROOT / ("pipeline/analyses/y45_terminal_601b_" + args.token + ".json")).read_text())
protected.update(terminal["source_hashes"])
data = io.BytesIO()
with tarfile.open(fileobj=data, mode="w") as archive:
    for row in records:
        archive.add(ROOT / row["path"], arcname=row["path"])
payload = dict(files=records, protected=protected)
program = r'''
import ast,hashlib,io,json,pathlib,subprocess,sys,tarfile
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
assert (root/"AdaRES/.codex_source_commit").read_text().strip()=="338a3d7a5b21fa463d116e566b482f0733fde9c2"
def protect():
    assert all(hashlib.sha256((root/"AdaRES"/p).read_bytes()).hexdigest()==h for p,h in P["protected"].items())
protect()
archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode="r:")
members=archive.getmembers()
assert sorted(m.name for m in members)==sorted(f["path"] for f in P["files"])
for m in members:
    assert m.isfile()
    assert m.name.startswith("AdaRES/ipdn/model_paper5/adares_y48_mask_equivalence/") or m.name=="AdaRES/configs_song/model_paper5/adares_y48_mask_equivalence_res.yaml"
    path=(root/m.name).resolve()
    assert root.resolve() in path.parents and not path.exists()
    content=archive.extractfile(m).read()
    assert hashlib.sha256(content).hexdigest()==next(f["sha256"] for f in P["files"] if f["path"]==m.name)
    if m.name.endswith(".py"):compile(content.decode(),"new-y48","exec")
for m in members:
    path=root/m.name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:stream.write(archive.extractfile(m).read())
protect()
print(json.dumps(dict(files=[dict(path=m.name,sha256=hashlib.sha256((root/m.name).read_bytes()).hexdigest()) for m in members])))
'''
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    server = monitor.load_servers(str(ROOT))["server-2"]
    assert server["host"] == "192.168.1.163" and server["allowed_gpus"] == [0, 1]
    command = server["python"] + " -S -c " + shlex.quote("P=" + repr(payload) + "\n" + program)
    response = subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "songcl@192.168.1.163", command],
        input=data.getvalue(), capture_output=True, timeout=60)
    assert response.returncode == 0, response.stderr.decode(errors="replace")
    result = json.loads(response.stdout)
    assert result["files"] == records
    result.update(
        complete=True, id="AdaRES-Y48", remote_byte_identity=True,
        physical_independence=True, protected_original_sources_unchanged=True,
        source_marker="338a3d7a5b21fa463d116e566b482f0733fde9c2",
        scope="Seven new candidate files only; no admission or training launch.")
    output = ROOT / ("pipeline/analyses/y48_source_bundle_601b_" + args.token + ".json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
finally:
    writer.close()
