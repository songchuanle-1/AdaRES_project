"""Capture Y15 checkpoint metadata or frozen evaluation under the owned tick."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "963720b211ae49b1a59707610cfbda4a"
REMOTE = "/bak_home/songcl/Project/AdaRES_project"
CODE = REMOTE + "/AdaRES"
Y15 = REMOTE + "/AdaRES_checkpoints/adares_y15_query_completion_res/20260930_071753"
PYTHON = "/home/songcl/.conda/envs/adares_project/bin/python"
SITE = "/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages"
EGG = "/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["metadata", "y15", "w2", "b1"])
    parser.add_argument("--attempt", default="v2")
    args = parser.parse_args()
    owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
    if owner["token"] != TOKEN:
        raise RuntimeError("Owned foreground lease missing or changed")
    suffix = args.mode + "_" + args.attempt + "_963720b2"
    output = ROOT / ("pipeline/analyses/y15_" + suffix + ".json")
    if output.exists():
        raise RuntimeError("Refusing to overwrite evidence")
    common = (
        "import sys,os,json,pathlib,hashlib,subprocess\n"
        "sys.path[:0]=" + repr([CODE, CODE + "/tools", CODE + "/ipdn/lib", SITE, EGG]) + "\n"
        "os.environ['PYTHONDONTWRITEBYTECODE']='1'\n"
        "os.chdir(" + repr(CODE) + ")\n"
        "assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'\n"
    )
    if args.mode == "metadata":
        program = common + """
import torch
base=pathlib.Path(BASE)
rows=[]
for name in ['last.pth','best.pth']:
    path=base/name
    value=torch.load(str(path),map_location='cpu')
    meta=value.get('meta',{})
    rows.append(dict(path=str(path),bytes=path.stat().st_size,
        epoch=int(meta['epoch']),best_miou=float(meta['best_miou']),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    del value
print(json.dumps(dict(checkpoints=rows,
    original_pid_present=pathlib.Path('/proc/18796').exists(),
    gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode())))
""".replace("BASE", repr(Y15))
        timeout = 90
    else:
        config = "configs_song/model_paper5/adares_y15_query_completion_res.yaml"
        checkpoint = Y15 + "/best.pth"
        if args.mode != "y15":
            reference = json.loads((ROOT / (
                "pipeline/analyses/" + args.mode + "_full_eval_3f09f39c.json")).read_text())
            config, checkpoint = reference["config"], reference["checkpoint"]
        source = ROOT / "AdaRES/tools/evaluate_adares_frozen_full.py"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        remote_output = "/tmp/adares_y15_" + suffix + ".json"
        program = common + """
source=pathlib.Path('tools/evaluate_adares_frozen_full.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()==DIGEST
old='/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock'
assert source.count(old)==1
# The original evaluator has a fixed GPU1 lease; bind this GPU0 run explicitly.
source=source.replace(old,'/bak_home/songcl/Project/AdaRES_project/.y15_gpu0_diagnostic.lock')
sys.argv=['evaluate_adares_frozen_full.py','--config',CONFIG,'--checkpoint',CHECKPOINT,'--output',OUTPUT,'--gpu-id','0']
exec(compile(source,'tools/evaluate_adares_frozen_full.py','exec'),{'__name__':'__main__'})
""".replace("DIGEST", repr(digest)).replace("CONFIG", repr(config)).replace(
            "CHECKPOINT", repr(checkpoint)).replace("OUTPUT", repr(remote_output))
        timeout = 600
    started = datetime.now().astimezone().isoformat()
    evidence_log = ROOT / ("pipeline/analyses/y15_" + suffix + ".log")
    if evidence_log.exists():
        raise RuntimeError("Refusing to overwrite diagnostic log")
    response = subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "songcl@192.168.1.163", "timeout --signal=TERM --kill-after=10s " +
        str(timeout - 20) + " " + PYTHON + " -S -"
    ], input=program.encode(), capture_output=True, timeout=timeout)
    evidence_log.write_bytes(response.stdout + b"\nSTDERR:\n" + response.stderr)
    if response.returncode:
        raise RuntimeError("Remote probe failed rc=" + str(response.returncode) +
                           "; preserved " + str(evidence_log))
    if args.mode == "metadata":
        result = json.loads(response.stdout)
        result.update(observed_at_local=datetime.now().astimezone().isoformat(),
                      started_at_local=started)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    else:
        subprocess.run(["scp", "-q", "songcl@192.168.1.163:" + remote_output,
                        str(output)], check=True, timeout=60)
        result = json.loads(output.read_text())
        print(json.dumps(dict(mode=args.mode, output=str(output),
                              complete=result["complete"],
                              summary=result["passes"][0]["summary"],
                              checkpoint_meta=result["checkpoint_meta"],
                              seconds=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
