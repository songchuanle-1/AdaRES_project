"""Prepare independent SAME Y46 production sources, with no upload or launch."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 150
parent = ROOT / ("pipeline/analyses/w2_native_source_601b_" + args.token)
target = ROOT / "AdaRES/ipdn/model_paper5/adares_y46_component_coherence"
config = ROOT / "AdaRES/configs_song/model_paper5/adares_y46_component_coherence_res.yaml"
assert not target.exists() and not config.exists()
source = ROOT / "scripts/probe_y46_cached_native_601b.py"
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "e253e1edcc9e54d6c33c0abc7559b12de09c9a5eb29cc734e689d4fe8cbc2772")
prototype = ast.parse(source.read_text())
helpers = [node for node in prototype.body
           if isinstance(node, (ast.FunctionDef, ast.ClassDef))
           and node.name in ("component_inputs", "AllComponentReadout", "FinalIndicatorAttachment")]
assert len(helpers) == 3
component = helpers[0]
component.body = [
    node for node in component.body
    if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("started", "cached_seconds"))
    and not (isinstance(node, ast.If) and any(
        isinstance(child, ast.Name) and child.id == "P" for child in ast.walk(node.test)))
]
tree = ast.parse((parent / "dec.py").read_text())
original = ast.dump(tree)
imports = ast.parse(
    "import numpy as np\nfrom scipy.sparse import csr_matrix\n"
    "from scipy.sparse.csgraph import connected_components\n").body
tree.body[0:0] = imports
decoder = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "DEC")
tree.body[tree.body.index(decoder):tree.body.index(decoder)] = helpers
constructor = next(node for node in decoder.body
                   if isinstance(node, ast.FunctionDef) and node.name == "__init__")
constructor.body.extend(ast.parse(
    "with torch.random.fork_rng(devices=[]):\n"
    "    self.mask_component_readout = AllComponentReadout(d_model)\n"
    "self._component_attachment = FinalIndicatorAttachment(self, self.mask_component_readout)\n"
    "self._component_attachment.enabled = True\n").body)
ast.fix_missing_locations(tree)
text = ast.unparse(tree) + "\n"
compile(text, "<same-y46-reserve>", "exec")
assert not any(isinstance(node, ast.Name) and node.id == "P" for node in ast.walk(tree))
audit = ast.parse(text)
audit.body = audit.body[len(imports):]
audit.body = [node for node in audit.body
              if not (isinstance(node, (ast.FunctionDef, ast.ClassDef))
                      and node.name in ("component_inputs", "AllComponentReadout", "FinalIndicatorAttachment"))]
cls = next(node for node in audit.body if isinstance(node, ast.ClassDef) and node.name == "DEC")
init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
init.body = init.body[:-3]
assert ast.dump(audit) == original
target.mkdir()
for path in parent.glob("*.py"):
    shutil.copyfile(path, target / path.name)
(target / "dec.py").write_text(text, encoding="utf-8", newline="\n")
shutil.copyfile(parent / "config.yaml", config)
files = sorted(target.glob("*.py")) + [config]
assert len(files) == 7
result = dict(
    complete=True, id="AdaRES-Y46", local_only=True, uploaded=False, launched=False,
    files=[dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files],
    stripped_parent_AST_exact=True, same_tested_head_and_hook=True,
    pure_cached_component_no_runtime_payload_dependency=True,
    limitations="Independent source/compile/AST preparation only. Actual production native/released-init/tools.train/resources remain pending; prototype gains do not establish acceptable cost or efficacy.")
output = ROOT / ("pipeline/analyses/y46_local_reserve_601b_" + args.token + ".json")
with output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y46")
    candidate["evidence"].append(dict(
        path=output.relative_to(ROOT).as_posix(),
        sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    candidate.update(
        stage="implementation", scientific_status="cache_revision_passed_local_production_prepared",
        next_action="Guarded additive upload/test SAME independent local Y46 package, then actual released-init/tools.train updates, complete controls/resources/independence review; no queue/admission from prototype. Preserve original and cached cost risks.")
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(
        complete=True, id="AdaRES-Y46", files=len(files),
        stripped_parent_AST_exact=True, uploaded=False, admission=False)))
finally:
    writer.close()
