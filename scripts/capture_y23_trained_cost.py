"""Run the frozen train8/largest-pair profile with trained Y23/W2 weights."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()


def check_lease():
    owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
    assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 380


check_lease()
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["user"] == "songcl"
assert server["allowed_gpus"] == [1]
terminal = json.loads((ROOT / ("pipeline/analyses/y23_terminal_" + args.token + ".json")).read_text())
assert terminal["processes_after"]["original_cmd"] is None
assert next(c for c in terminal["checkpoints"] if c["path"].endswith("/last.pth"))["epoch"] == 30
best = next(c for c in terminal["checkpoints"] if c["path"].endswith("/best.pth"))
w2 = json.loads((ROOT / ("pipeline/analyses/y23_w2_full_eval_" + args.token + ".json")).read_text())
assert w2["complete"] and w2["completed_passes"] == 2
source_path = ROOT / "scripts/y23_native_remote_d20c71e4.py"
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "c40a816b881af162e1dfc138bd044e41f1347b0262aaccda9ca357538f22f810"
output = ROOT / ("pipeline/analyses/y23_trained_cost_serial_" + args.token + ".json")
remote_output = "/tmp/y23_trained_cost_serial_inference_profile_" + args.token + ".json"
assert not output.exists() and not output.with_suffix(".log").exists()


class TrainedProfile(ast.NodeTransformer):
    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return node
        value = node.value
        replacements = {
            "__MODE__": "inference_profile",
            "0": "1",
            "/bak_home/songcl/Project/AdaRES_project/.y15_gpu0_diagnostic.lock":
                "/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock",
            "/tmp/y23_": "/tmp/y23_trained_cost_serial_",
            "_d20c71e4.json": "_" + args.token + ".json",
            "TRAIN diagnostics and bounded execution tests, not trained efficacy or full-dataloader replay.":
                "Trained-checkpoint frozen cost profile; not a causal or novelty claim.",
            "Same frozen checkpoint serial single-model inference timings, not fully trained accuracy-cost Pareto proof; 3 readings are not a stable p95 estimate.":
                "Each method uses its own frozen trained best checkpoint; train8 plus both frozen largest-pair scenes serially at native predict batch=1, two warmups/three readings per scene. Three readings cannot establish reliable tail latency or population Pareto dominance. The earlier combined batch=2 shape failure remains preserved.",
        }
        return ast.copy_location(ast.Constant(value=replacements.get(value, value)), node)

    def visit_FunctionDef(self, node):
        if node.name == "load_model":
            node = self.generic_visit(node)
            for item in ast.walk(node):
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
                    if isinstance(item.func.value, ast.Name) and item.func.value.id == "torch" and item.func.attr == "load":
                        item.args[0] = ast.IfExp(
                            test=ast.Compare(left=ast.Name(id="config", ctx=ast.Load()),
                                             ops=[ast.Eq()], comparators=[ast.Name(id="CONFIG", ctx=ast.Load())]),
                            body=ast.Constant(value=best["path"]),
                            orelse=ast.Constant(value=w2["checkpoint"]))
            return node
        return self.generic_visit(node)

    def visit_If(self, node):
        is_profile = (isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                      and node.test.left.id == "MODE" and len(node.test.comparators) == 1
                      and isinstance(node.test.comparators[0], ast.Constant)
                      and node.test.comparators[0].value == "inference_profile")
        if not is_profile:
            return self.generic_visit(node)
        node = self.generic_visit(node)
        # Keep the original measurements; replace only the frozen batch manifest.
        manifest = ast.parse(
            "indices=unique_indices(dataset,8)\n"
            "batches=[dataset.collate_fn([dataset[i]]) for i in indices]\n"
            "batches.extend([dataset.collate_fn([dataset[i]]) for i in [728,6780]])\n"
            "result['manifest_indices']=indices+[728,6780]\n"
            "result['checkpoint_sources']={'candidate':" + repr(best) + ",'parent':"
            + repr(dict(path=w2["checkpoint"], sha256=w2["checkpoint_sha256"])) + "}\n").body
        node.body[:2] = manifest
        return node

    def visit_Dict(self, node):
        node = self.generic_visit(node)
        if any(isinstance(k, ast.Constant) and k.value == "scan_id" for k in node.keys):
            node.keys.append(ast.Constant(value="scan_ids"))
            node.values.append(ast.Subscript(value=ast.Name(id="batch", ctx=ast.Load()),
                                             slice=ast.Constant(value="scan_ids"), ctx=ast.Load()))
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "dict":
            if any(k.arg == "scan_id" for k in node.keywords):
                node.keywords.append(ast.keyword(
                    arg="scan_ids", value=ast.Subscript(value=ast.Name(id="batch", ctx=ast.Load()),
                                                       slice=ast.Constant(value="scan_ids"), ctx=ast.Load())))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "update":
            for keyword in node.keywords:
                if keyword.arg == "scans" and isinstance(keyword.value, ast.Constant) and keyword.value.value == 9:
                    keyword.value = ast.Constant(value=10)
                    node.keywords.append(ast.keyword(arg="batches", value=ast.Constant(value=10)))
                    break
        return node


tree = ast.fix_missing_locations(TrainedProfile().visit(ast.parse(source)))
program = ast.unparse(tree)
prefix = (
    "import hashlib,pathlib,subprocess\n"
    "assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'\n"
    "assert not pathlib.Path('/proc/129584').exists()\n"
    "assert not pathlib.Path(" + repr(remote_output) + ").exists()\n"
    "def frozen_hash(p):\n"
    " h=hashlib.sha256()\n"
    " with open(p,'rb') as f:\n"
    "  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)\n"
    " return h.hexdigest()\n"
    "assert frozen_hash(" + repr(best["path"]) + ")==" + repr(best["sha256"]) + "\n"
    "assert frozen_hash(" + repr(w2["checkpoint"]) + ")==" + repr(w2["checkpoint_sha256"]) + "\n"
)
compile(prefix + program, str(source_path), "exec")
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None, "Pipeline writer busy"
start = datetime.now().astimezone().isoformat()
try:
    with output.with_suffix(".log").open("xb") as log:
        response = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "songcl@192.168.1.163",
             "timeout --signal=TERM --kill-after=10s 300 " + server["python"] + " -S -"],
            input=(prefix + program).encode("utf-8"), stdout=log, stderr=subprocess.STDOUT, timeout=330)
    check_lease()
    assert response.returncode == 0, "Cost profile failed; preserve log"
    subprocess.run(["scp", "-q", "songcl@192.168.1.163:" + remote_output, str(output)], check=True, timeout=30)
    result = json.loads(output.read_text())
    assert result["complete"] and len(result["rows"]["candidate"]) == 10
    assert [r["scan_ids"] for r in result["rows"]["candidate"]] == [r["scan_ids"] for r in result["rows"]["parent"]]
    for rows in result["rows"].values():
        assert [r["scan_ids"] for r in rows[-2:]] == [["scene0054_00"], ["scene0497_00"]]
        assert all(len(r["scan_ids"]) == 1 for r in rows)
        assert all(len(r["readings"]) == 3 for r in rows)
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                         observed_start_local=start, observed_end_local=datetime.now().astimezone().isoformat(),
                         manifest=result["manifest_indices"], limitation=result["limitation"]), indent=2))
finally:
    lock.close()
