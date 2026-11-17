"""Run frozen train-only diagnosis via the authorized server's native environment."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "58ccdbbf48104660911c2a0c0fc97d3e"
CODE = "/bak_home/songcl/Project/AdaRES_project/AdaRES"
PYTHON = "/home/songcl/.conda/envs/adares_project/bin/python"


def main():
    owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
    assert owner["token"] == TOKEN
    output = ROOT / "pipeline/analyses/y19_y20_diagnosis_58ccdbbf.json"
    log = output.with_suffix(".log")
    assert not output.exists() and not log.exists()
    source = ROOT / "scripts/y19_y20_remote_diagnosis_58ccdbbf.py"
    manifest = ROOT / "pipeline/designs/Y19_Y20_native_diagnosis_58ccdbbf.md"
    program = source.read_text().replace(
        "__MANIFEST_DIGEST__", hashlib.sha256(manifest.read_bytes()).hexdigest())
    compile(program, str(source), "exec")
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         "songcl@192.168.1.163",
         "timeout --signal=TERM --kill-after=10s 720 " + PYTHON + " -S -"],
        input=program.encode(), capture_output=True, timeout=750)
    log.write_bytes(response.stdout + b"\nSTDERR:\n" + response.stderr)
    if response.returncode:
        raise RuntimeError("Diagnosis failed; retained " + str(log))
    subprocess.run(["scp", "-q",
                    "songcl@192.168.1.163:/tmp/adares_y19_y20_diagnosis_58ccdbbf.json",
                    str(output)], check=True, timeout=60)
    result = json.loads(output.read_text())
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
