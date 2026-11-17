"""Bind source-control support and freeze the next native-input CPU test."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
source_path = "pipeline/analyses/y49_source_controls_c3e5_" + args.token + ".json"
source = json.loads((ROOT / source_path).read_text())
assert source["complete"] and source["CPU_cases"] == 32
program = "scripts/probe_y49_native_capture_cpu_c3e5.py"
reference = dict(path=source_path, sha256=hashlib.sha256((ROOT / source_path).read_bytes()).hexdigest())
output = ROOT / ("pipeline/analyses/y49_native_CPU_prereg_c3e5_" + args.token + ".json")
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    prereg = dict(
        complete=True, id="AdaRES-Y49", preregistered_at=datetime.now().astimezone().isoformat(),
        source_controls=reference,
        actual_native_input_manifest="Existing immutable TRAIN8 indices0,44,60,101,115,126,137,171; all32 expressions/all128 queries, captureSHA5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89.",
        descriptor="For hard FG/BG induced weak-nearest graph, compute each inverse diagonal of I+D-A using SciPy splu and exact identity solves, then original point-mass-weighted mean. Empty side0; no component filtering or numerical sweep. Normalized unweighted first witness is a special case.",
        prototype="LayerNorm(native query) concatenated with FG/BG scalar readouts, Linear(258,256)/GELU/Linear(256,1); inherited width256, four tensors. Final-foreground residual is a prospective native connection, not executed by CPU prototype.",
        tests="All32 native captured expressions: finite descriptors, repeat, cyclic wrong-mask and query-only controls, four finite/nonzero syntheticCE gradients. Fixed first4 expressions: joint reverse permutation and exact block1vs64 solver checks. Record equal component-mass groups with distinct global values without requiring an invented effect size.",
        fixed_precision="8*float64eps*max(1,nodes)*max(1,absolute-scale), frozen before the test. No whole-input native bitwise determinism claim.",
        program=dict(path=program, sha256=hashlib.sha256((ROOT / program).read_bytes()).hexdigest()),
        limitations="Task utility, native backward, resources and physical independence remain pending. ExistingY8/Y41 are not reopened; no validation or GPU use.")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(prereg, stream, indent=2)
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y49")
    candidate["evidence"].extend([
        reference, dict(path=output.relative_to(ROOT).as_posix(),
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest()),
        prereg["program"]])
    candidate.update(stage="diagnosis",
                     next_action="Execute SAME frozen TRAIN8/all32/all128 global-resolvent CPU prototype and matched controls; native loss/GPU/resources/independence remain pending.",
                     scientific_status="actual_source_witness_supported_native_input_test_pending")
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(complete=True, id="AdaRES-Y49",
                         preregistered_at=prereg["preregistered_at"], source_controls=reference)))
finally:
    writer.close()
