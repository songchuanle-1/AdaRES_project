"""Add only new candidate files; protect original terminal/training sources."""
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
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["allowed_gpus"] == [0, 1] and server["host"] == "192.168.1.163"
bundle = json.loads((ROOT / ("pipeline/analyses/y44_local_bundle_" + args.token + ".json")).read_text())
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
records = bundle["files"]
assert len(records) == 7
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
        if path.name not in ("dec.py", "ipdn.py"):
            assert ast.dump(tree) == ast.dump(ast.parse((parent / path.name).read_text(encoding="utf-8")))
    else:
        assert path.read_bytes().replace(b"\r\n", b"\n") == (path.parent / "adares_w2_pointmass_competition_res.yaml").read_bytes().replace(b"\r\n", b"\n")
data = io.BytesIO()
with tarfile.open(fileobj=data, mode="w") as archive:
    for row in records:
        archive.add(ROOT / row["path"], arcname=row["path"])
terminal = json.loads((ROOT / ("pipeline/analyses/y42_terminal_ff34_" + args.token + ".json")).read_text())
payload = {"files": records, "protected": dict(terminal["source_hashes"], **terminal["protected_source_hashes"]),
           "marker": "6fc7d4e46b6c26a11fede7f12d4e79b3471cc1cc"}
program = r'''
import ast,hashlib,io,json,pathlib,subprocess,sys,tarfile
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
assert (root/"AdaRES/.codex_source_commit").read_text().strip()==P["marker"]
command=pathlib.Path("/proc/13155/cmdline").read_bytes()
assert b"configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml" in command
assert command.split(b"\0")[command.split(b"\0").index(b"--gpu_ids")+1]==b"0"
def protect():
    assert pathlib.Path("/proc/13155/cmdline").read_bytes()==command
    assert all(hashlib.sha256((root/"AdaRES"/p).read_bytes()).hexdigest()==h for p,h in P["protected"].items())
protect()
archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode="r:")
members=archive.getmembers()
assert sorted(m.name for m in members)==sorted(f["path"] for f in P["files"])
for m in members:
    assert m.isfile()
    assert m.name.startswith("AdaRES/ipdn/model_paper5/adares_y44_coordinate_feature/") or m.name=="AdaRES/configs_song/model_paper5/adares_y44_coordinate_feature_res.yaml"
    path=(root/m.name).resolve()
    assert root.resolve() in path.parents and not path.exists()
    content=archive.extractfile(m).read()
    assert hashlib.sha256(content).hexdigest()==next(f["sha256"] for f in P["files"] if f["path"]==m.name)
    if m.name.endswith(".py") and not m.name.endswith(("/dec.py","/ipdn.py")):
        assert ast.dump(ast.parse(content.decode()))==ast.dump(ast.parse((root/"AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"/path.name).read_text()))
for m in members:
    path=root/m.name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(archive.extractfile(m).read())
protect()
print(json.dumps({"files":[{"path":m.name,"sha256":hashlib.sha256((root/m.name).read_bytes()).hexdigest()} for m in members]}))
'''
output = ROOT / ("pipeline/analyses/y44_source_bundle_" + args.token + ".json")
assert not output.exists()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    command = server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)
    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163", command],
                              input=data.getvalue(), capture_output=True, timeout=60)
    assert response.returncode == 0, response.stderr.decode(errors="replace")
    result = json.loads(response.stdout)
    assert result["files"] == records
    result.update(complete=True, id="AdaRES-Y44", remote_byte_identity=True,
                  physical_independence=True, config_exact_parent_normalized=True,
                  source_marker=payload["marker"], protected_y42_y43_unchanged=True,
                  scope="Additive diagnostic files only; no admission or launch.")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
finally:
    lock.close()
