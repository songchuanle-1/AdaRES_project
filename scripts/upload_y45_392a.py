"""Add only hashed independent Y45 diagnostic files on authorized server-2."""

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
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 100
bundle = json.loads((ROOT / ("pipeline/analyses/y45_local_bundle_392a_" + args.token + ".json")).read_text())
records = bundle["files"]
assert bundle["complete"] and len(records) == 7
for row in records:
    path = ROOT / row["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    if path.suffix == ".py":
        tree = ast.parse(path.read_text(encoding="utf-8"))
        compile(tree, str(path), "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "model_paper5" not in (node.module or "")
            if isinstance(node, ast.Import):
                assert not any("model_paper5" in alias.name for alias in node.names)
    else:
        assert path.read_bytes() == (path.parent / "adares_w2_pointmass_competition_res.yaml").read_bytes()
protected = {}
for stem in ("y43_terminal_7a40", "y44_terminal_392a"):
    proof = json.loads((ROOT / ("pipeline/analyses/" + stem + "_" + args.token + ".json")).read_text(encoding="utf-8"))
    protected.update(proof["source_hashes"])
native = json.loads((ROOT / ("pipeline/analyses/y45_native_targets_392a_" + args.token + ".json")).read_text(encoding="utf-8"))
protected.update(native["actual_parent_source_hashes"])
marker = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "AdaRES").decode().strip()
assert marker == "c65ad2fcba4ef10b71ea6ae58ed01109568bd0cd"
data = io.BytesIO()
with tarfile.open(fileobj=data, mode="w") as archive:
    for row in records:
        archive.add(ROOT / row["path"], arcname=row["path"])
payload = dict(files=records, protected=protected, marker=marker)
program = r'''
import ast,hashlib,io,json,pathlib,subprocess,sys,tarfile
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
assert (root/"AdaRES/.codex_source_commit").read_text().strip()==P["marker"]
def protect():
    assert all(hashlib.sha256((root/"AdaRES"/p).read_bytes()).hexdigest()==h for p,h in P["protected"].items())
protect()
archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode="r:")
members=archive.getmembers()
assert sorted(m.name for m in members)==sorted(f["path"] for f in P["files"])
for m in members:
    assert m.isfile()
    assert m.name.startswith("AdaRES/ipdn/model_paper5/adares_y45_metric_decomposition/") or m.name=="AdaRES/configs_song/model_paper5/adares_y45_metric_decomposition_res.yaml"
    path=(root/m.name).resolve()
    assert root.resolve() in path.parents and not path.exists()
    content=archive.extractfile(m).read()
    assert hashlib.sha256(content).hexdigest()==next(f["sha256"] for f in P["files"] if f["path"]==m.name)
    if m.name.endswith(".py"):compile(content.decode(),"new-y45","exec")
for m in members:
    path=root/m.name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(archive.extractfile(m).read())
protect()
print(json.dumps(dict(files=[dict(path=m.name,sha256=hashlib.sha256((root/m.name).read_bytes()).hexdigest()) for m in members])))
'''
output = ROOT / ("pipeline/analyses/y45_source_bundle_392a_" + args.token + ".json")
assert not output.exists()
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    command = "/home/songcl/.conda/envs/adares_project/bin/python -S -c " + shlex.quote("P=" + repr(payload) + "\n" + program)
    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                               "songcl@192.168.1.163", command],
                              input=data.getvalue(), capture_output=True, timeout=60)
    assert response.returncode == 0, response.stderr.decode(errors="replace")
    result = json.loads(response.stdout)
    assert result["files"] == records
    result.update(complete=True, id="AdaRES-Y45", remote_byte_identity=True,
                  physical_independence=True, source_marker=marker,
                  protected_original_sources_unchanged=True,
                  scope="Additive seven diagnostic files only; no admission or launch.")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
finally:
    writer.close()
