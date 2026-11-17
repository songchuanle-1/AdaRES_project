"""Add only the frozen Y46 bundle while protecting existing experiment bytes."""

import argparse
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
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 150
old = "601bcda714184133920945f8f067b382"
bundle_path = ROOT / ("pipeline/analyses/y46_local_reserve_601b_" + old + ".json")
bundle = json.loads(bundle_path.read_text())
records = bundle["files"]
assert bundle["complete"] and bundle["local_only"] and len(records) == 7
for row in records:
    source = ROOT / row["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]
old_bundle = json.loads((ROOT / ("pipeline/analyses/y48_local_bundle_601b_" + old + ".json")).read_text())
protected = dict(old_bundle["parent_source_hashes"])
protected.update({row["path"].removeprefix("AdaRES/"): row["sha256"] for row in old_bundle["files"]})
terminal = json.loads((ROOT / ("pipeline/analyses/y45_terminal_601b_" + old + ".json")).read_text())
protected.update(terminal["source_hashes"])
protected["configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"] = records[-1]["sha256"]
archive_bytes = io.BytesIO()
with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
    for row in records:
        archive.add(ROOT / row["path"], arcname=row["path"])
payload = dict(files=records, protected=protected)
program = r'''
import hashlib,io,json,pathlib,re,subprocess,sys,tarfile
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
root=pathlib.Path("/bak_home/songcl/Project/AdaRES_project")
marker=root/"AdaRES/.codex_source_commit"
assert marker.read_text().strip()=="d18c228fd3796b5a87dbe313075b6ee748b2d51b"
def protect():
    assert all(hashlib.sha256((root/"AdaRES"/p).read_bytes()).hexdigest()==h for p,h in P["protected"].items())
def live():
    rows=[]
    for gpu in (0,1):
        raw=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"]).decode().strip()
        util,used,total=map(int,raw.split(","))
        pids=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader,nounits"]).decode().strip()
        rows.append(dict(gpu=gpu,utilization=util,memory_MiB=used,total_MiB=total,pids=[int(p) for p in pids.splitlines() if p.strip()]))
    return rows
protect()
before=live()
assert 8877 in before[1]["pids"] and 8877 not in before[0]["pids"]
process=pathlib.Path("/proc/8877")
assert process.joinpath("cwd").resolve()==root/"AdaRES"
cmd=process.joinpath("cmdline").read_bytes().replace(b"\0",b" ").decode()
assert "adares_y48_mask_equivalence_res.yaml" in cmd
environment=dict(line.split(b"=",1) for line in process.joinpath("environ").read_bytes().split(b"\0") if b"=" in line)
assert environment[b"CUDA_VISIBLE_DEVICES"]==b"1"
log=root/"AdaRES/logs/train_adares_y48_mask_equivalence_res_20261015_113625.log"
with log.open("rb") as stream:
    stream.seek(max(0,log.stat().st_size-65536))
    tail=stream.read().decode(errors="replace")
steps=re.findall(r"Train\[(\d+)/(\d+)\].*?\|\s*(\d+)/(\d+)",tail)
assert steps
archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode="r:")
members=archive.getmembers()
assert sorted(m.name for m in members)==sorted(f["path"] for f in P["files"])
for member in members:
    assert member.isfile()
    assert member.name.startswith("AdaRES/ipdn/model_paper5/adares_y46_component_coherence/") or member.name=="AdaRES/configs_song/model_paper5/adares_y46_component_coherence_res.yaml"
    path=(root/member.name).resolve()
    assert root.resolve() in path.parents and not path.exists()
    content=archive.extractfile(member).read()
    assert hashlib.sha256(content).hexdigest()==next(f["sha256"] for f in P["files"] if f["path"]==member.name)
    if member.name.endswith(".py"):
        compile(content.decode(),"frozen-y46","exec")
for member in members:
    path=root/member.name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        stream.write(archive.extractfile(member).read())
protect()
assert marker.read_text().strip()=="d18c228fd3796b5a87dbe313075b6ee748b2d51b"
print(json.dumps(dict(files=[dict(path=m.name,sha256=hashlib.sha256((root/m.name).read_bytes()).hexdigest()) for m in members],
    GPUs_before=before,GPUs_after=live(),Y48_pid=8877,Y48_config_verified=True,Y48_GPU1_environment_verified=True,
    Y48_native_step=list(map(int,steps[-1])),protected_files=len(P["protected"]),protected_sources_unchanged=True)))
'''
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
output = ROOT / ("pipeline/analyses/y46_production_upload_131e_" + args.token + ".json")
assert not output.exists()
try:
    server = monitor.load_servers(str(ROOT))["server-2"]
    assert server["host"] == "192.168.1.163" and server["allowed_gpus"] == [0, 1]
    started = datetime.now().astimezone().isoformat()
    command = server["python"] + " -S -c " + shlex.quote("P=" + repr(payload) + "\n" + program)
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         "songcl@192.168.1.163", command],
        input=archive_bytes.getvalue(), capture_output=True, timeout=60)
    with output.with_suffix(".log").open("xb") as stream:
        stream.write(response.stdout + response.stderr)
    assert response.returncode == 0, "Upload failed; preserve raw log and inspect without replay"
    result = json.loads(response.stdout)
    assert result["files"] == records
    result.update(
        complete=True, id="AdaRES-Y46", remote_byte_identity=True,
        observed_start_local=started, observed_end_local=datetime.now().astimezone().isoformat(),
        local_bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        source_marker="d18c228fd3796b5a87dbe313075b6ee748b2d51b",
        scientific_status="production_sources_uploaded_not_native_verified",
        scope="Exactly seven new Y46 files; no marker change, admission, diagnostic GPU or training launch.")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                         summary={k: v for k, v in result.items() if k != "files"}), indent=2))
finally:
    writer.close()
