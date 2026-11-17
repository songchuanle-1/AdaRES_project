"""Lease/lock-bound invocation of the canonical project decision writer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor
checks = json.loads((ROOT / ("pipeline/analyses/y44_gate_checks_" + args.token + ".json")).read_text())
commit = json.loads((ROOT / ("pipeline/analyses/y44_commit_ff34_" + args.token + ".json")).read_text())
assert checks["complete"] and all(checks["gates"].values()) and commit["complete"]
environment = os.environ.copy()
environment["PROJECT_ROOT"] = "D:/project"
environment["PATH"] = "C:/Program Files (x86)/Git/bin;C:/Program Files (x86)/Git/usr/bin;" + environment["PATH"]
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    response = subprocess.run([
        "C:/Program Files (x86)/Git/bin/bash.exe",
        "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/log_decision.sh",
        "AdaRES_project", "next_batch",
        "Y42冻结FIRST为0.504983且对W2场景区间跨零；Y44真实六门槛通过后是否开展固定训练？",
        "同一Y42科学收尾为complete_inconclusive并保留W2；仅准入同一Y44至server2 GPU1固定30轮，保护Y43 GPU0。",
        "Y44实际配对缺口、原生错配与精度审计、8张量梯度/更新、恢复/禁用父模型一致性及资源测试通过；只证明可证伪和可执行，不证明收益。无模型/准入门槛放宽或调参。",
        "pipeline/analyses/AdaRES-Y42_frozen_review_ff347697.md; pipeline/analyses/Y44_six_gate_review_ff347697.md; pipeline/designs/exp_design_Y44_ff347697.md; exact code_commit=" + commit["code_commit"],
    ], env=environment, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert response.returncode == 0, response.stderr
    match = re.search(r"-> ([^\r\n]+)", response.stdout)
    assert match
    path = Path(match.group(1).strip()).resolve()
    assert ROOT / "pipeline/decisions" == path.parent and path.is_file()
    print(json.dumps({"decision": path.relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "canonical_writer": True}, ensure_ascii=False))
finally:
    lock.close()
