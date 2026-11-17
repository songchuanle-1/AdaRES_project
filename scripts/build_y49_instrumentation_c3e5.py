"""Compose the next harness with AST-identical frozen component functions."""

import argparse
import ast
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
source = ROOT / "scripts/probe_y49_native_capture_cpu_c3e5.py"
assert hashlib.sha256(source.read_bytes()).hexdigest() == "b24ed8dbeed377e7f2c22bd4bd9647105254411275d9ce68d23d6a496692d2a0"
helpers = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
           if isinstance(node, (ast.FunctionDef, ast.ClassDef))
           and node.name in ("inverse_diagonal", "descriptors", "GlobalResolventReadout")]
assert len(helpers) == 3
harness = ROOT / "scripts/y49_instrumentation_harness_c3e5.py"
imports = ast.parse(
    "import json\nfrom pathlib import Path\nimport numpy as np\n"
    "from scipy.sparse import csr_matrix,diags,eye\n"
    "from scipy.sparse.csgraph import connected_components\n"
    "from scipy.sparse.linalg import splu\nimport torch\nfrom torch import nn\n").body
tree = ast.Module(body=imports + helpers + ast.parse(harness.read_text(encoding="utf-8")).body,
                  type_ignores=[])
text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
compile(text, "<y49-composed-harness>", "exec")
actual = [node for node in ast.parse(text).body
          if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in (
              "inverse_diagonal", "descriptors", "GlobalResolventReadout")]
assert [ast.dump(node) for node in actual] == [ast.dump(node) for node in helpers]
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    output = ROOT / ("pipeline/analyses/y49_instrumentation_program_c3e5_" + args.token + ".py")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    path = ROOT / ("pipeline/analyses/y49_native_capture_cpu_c3e5_" + args.token + ".json")
    result = json.loads(path.read_text())
    assert result["complete"] and result["native_captured_expressions"] == 32
    reference = dict(path=path.relative_to(ROOT).as_posix(),
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    program_ref = dict(path=output.relative_to(ROOT).as_posix(),
                       sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    freeze_path = ROOT / ("pipeline/analyses/y49_hook_prereg_c3e5_" + args.token + ".json")
    with freeze_path.open("x", encoding="utf-8") as stream:
        json.dump(dict(
            complete=True, id="AdaRES-Y49", preregistered_at=datetime.now().astimezone().isoformat(),
            exact_source_helpers_AST=True, descriptor_CPU_result=reference, program=program_ref,
            native_component_mass_collision_witness_expressions=0,
            next_CPU_test="Actual parent indicator source plus declared HookHarness32: final foreground only; six auxiliaries and disabled path exact; four finite/nonzero CE gradients; cyclic wrong-mask and exact repeat controls.",
            next_GPU_test="Only after live-idle authorized exclusive lease: actual W2 TRAIN8 full loss/backward, all4 new tensor gradients, cached indicator controls and paired costs. Prototype only; physical production/released-init/tools.train/restore/complete resources still pending.",
            limitation="Native capture had zero equal-component-partition/different-global-value witnesses. Retain this limitation; neither task blindness nor trained benefit is established.",
            no_model_protocol_revision=True), stream, indent=2)
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y49")
    candidate.update(stage="implementation",
                     scientific_status="native_input_CPU_supported_information_limitations_retained",
                     next_action="Execute frozen CPU final-indicator instrumentation32; then only on truly idle authorized GPU execute same full native TRAIN8 loss/backward/resources. No production/admission from CPU.")
    candidate["evidence"].extend([
        reference, program_ref, dict(path=freeze_path.relative_to(ROOT).as_posix(),
                                    sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest())])
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(complete=True, id="AdaRES-Y49", program=str(output), evidence=program_ref,
                         native_collision_witness_expressions=0)))
finally:
    writer.close()
