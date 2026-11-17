"""Repair only the inherited predict batch-size assumption in the harness."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
token = sys.argv[sys.argv.index("--token") + 1]
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 600
source = ROOT / "scripts/probe_y48_production_601b.py"
log = ROOT / ("pipeline/analyses/y48_production_601b_" + token + ".log")
assert "The size of tensor a (710) must match the size of tensor b (667)" in log.read_text(encoding="utf-8")
tree = ast.parse(source.read_text())
position = next(i for i, node in enumerate(tree.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "handle" for target in node.targets))
tree.body[position:position] = ast.parse(
    "prediction_cfg = cfg.data.train.copy()\n"
    "prediction_cfg.training = prediction_cfg.aug = prediction_cfg.with_elastic = False\n"
    "prediction_dataset = build_dataset(prediction_cfg, logger)\n"
    "prediction_batches = []\n"
    "for index in indices:\n"
    "    gorilla.set_random_seed(1999)\n"
    "    prediction_batches.append(prediction_dataset.collate_fn([prediction_dataset[index]]))\n"
).body
changes = dict(iterator=0, manifest=0, batch_size=0)
for node in ast.walk(tree):
    if isinstance(node, ast.For) and ast.dump(node.iter) == ast.dump(ast.parse("enumerate(batches)", mode="eval").body):
        node.iter = ast.parse("enumerate(prediction_batches)", mode="eval").body
        changes["iterator"] += 1
    if isinstance(node, ast.keyword) and node.arg == "manifest_indices":
        node.value = ast.parse("indices[batch_index:batch_index + 1]", mode="eval").body
        changes["manifest"] += 1
    if isinstance(node, ast.keyword) and node.arg == "scene_batch_size":
        node.value = ast.Constant(value=1)
        changes["batch_size"] += 1
assert changes == dict(iterator=1, manifest=1, batch_size=1)
destination = ROOT / "scripts/probe_y48_production_601b_v2.py"
text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
compile(text, str(destination), "exec")
with destination.open("x", encoding="utf-8") as stream:
    stream.write(text)
manifest = dict(
    complete=True, id="AdaRES-Y48", failure_scope="implementation",
    preregistered_at=datetime.now().astimezone().isoformat(),
    failed_source=dict(path=source.relative_to(ROOT).as_posix(),
                       sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
    failed_log=dict(path=log.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(log.read_bytes()).hexdigest()),
    source=dict(path=destination.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),
    defect="Inherited full-model predict does not handle this two-scene TRAIN fixture; actual tools.train already performed two steps before the failing diagnostic.",
    change="Only diagnostic predict uses SAME first4 TRAIN scenes, serial batch1/no augmentation, matching existing frozen evaluator convention. Training remains two native batch2 steps; model and scientific intervention unchanged.",
    prediction="Serial predict/physical-parent equality succeeds; all source/CPU/train/update/restore/stress checks rerun from released initialization.",
    model_changes=0, result="pending")
output = ROOT / ("pipeline/analyses/y48_production_harness_repair_601b_" + token + ".json")
with output.open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
print(json.dumps(manifest, indent=2))
