"""Prepare fail-closed Y99/server1 admission and original Y98 handoff tools."""
import argparse
import ast
import json
import re
from prepare_y60_stage_heads import ROOT, lease, ref, tick
from y98_deployment import Y96, Y98, git

Y99 = "53f5b028bdf24e8b41dd934931d1fedc99843d8d"


class Rewrite(ast.NodeTransformer):
    def __init__(self, changes):
        self.changes = changes
        self.pattern = re.compile("|".join(re.escape(k) for k in sorted(changes, key=len, reverse=True)))

    def change(self, value):
        return self.pattern.sub(lambda m: self.changes[m.group()], value)

    def visit_Name(self, node):
        node.id = self.change(node.id)
        return node

    def visit_alias(self, node):
        node.name = self.change(node.name)
        if node.asname:
            node.asname = self.change(node.asname)
        return node

    def visit_ImportFrom(self, node):
        node.module = self.change(node.module)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.name = self.change(node.name)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.change(node.value)
        return node


def transformed(source, changes, selected=None):
    tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
    if selected is not None:
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
        assert len(tree.body) == len(selected)
    tree = Rewrite(changes).visit(tree)
    body = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    ast.parse(body, feature_version=(3, 8))
    return body


def create(token):
    lease(token, 600)
    assert git("rev-parse", "HEAD").decode().strip() == Y98
    assert git("rev-parse", Y99 + "^").decode().strip() == Y98
    before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/research_work.json",
                                  "pipeline/goal.json", "SERVERS.yaml", "pipeline/admissions/AdaRES-Y98.json")}
    index = git("ls-files", "--stage", "-z")
    work = tick.load(ROOT / "pipeline/research_work.json")
    c = next(r for r in work["candidates"] if r["id"] == "AdaRES-Y99")
    assert c["stage"] == "blocked" and c["code_commit"] == Y99 and not c.get("native_smoke_result")
    sources = ["scripts/advance_y96.py", "scripts/y98_deployment.py", "scripts/prepare_y95_y96_terminal_binding.py",
               "scripts/y95_y96_terminal_contract.py", "scripts/test_y95_y96_terminal_binding.py", "scripts/test_y98_admission.py"]
    old = [ref(p) for p in sources]
    changes = {"y94": "y96", "Y94": "Y96", "y95": "y98", "Y95": "Y98", "y96": "y99", "Y96": "Y99",
        "7e40371552d170a1d764f9dab160a8dcb12b0b6fe512e5c978313cb7b642241b":
            "50213e2f1fda98e5e9477eaa5c030952d1d913d3d1aeff86d1fc6aa984dfc154",
        "sampled-seed bypass of the original five-layer query generator":
            "full32 sparse-input bypass beside the complete original static-mask MLP"}
    outputs = {"scripts/advance_y99.py": transformed("scripts/advance_y96.py", changes)}
    header = '''"""Select only actual-native Y99 additions and preserve original Y98."""
import argparse, ast, hashlib, inspect, json, subprocess
from datetime import datetime
from pathlib import Path
import yaml
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from observe_y69_server1_phase import parse_phase,phase_key,progressed,REMOTE
import closure
import y98_deployment as previous
Y96,Y98=previous.Y96,previous.Y98
Y99="53f5b028bdf24e8b41dd934931d1fedc99843d8d"
VERSIONS=previous.VERSIONS+(Y99,)
ADDITIONS=dict(previous.ADDITIONS,**{Y99:"AdaRES-Y99"})
git,adapt=previous.git,previous.adapt
def paths(cid):
    if cid!="AdaRES-Y99":
        return previous.paths(cid)
    from prepare_y99_spatial_mask_skip import files
    return [p.relative_to(ROOT).as_posix() for p in files()]
def require_running_parent_handoff(work,candidate):
    original=next(r for r in work["candidates"] if r["id"]=="AdaRES-Y98")
    assert original["stage"]=="training"
    plan=tick.load(closure.artifact(ROOT,original["terminal_handoff_plan"]))
    assert plan["complete"] is True and plan["id"]==original["id"]
    assert plan["tested_additive_native_evidence"]==candidate["native_smoke_result"]
    assert plan["payload"]["allowed_markers"]==[Y98,Y99]
    assert plan["CPU_guard_tests_passed"]==29 and plan["additive_guard_tests_passed"]==8
    for entry in plan["dependencies"]:
        closure.artifact(ROOT,entry)
'''
    header += transformed("scripts/y98_deployment.py", {"Y98": "Y99", "y98": "y99"}, {"validate_delta"})
    header += transformed("scripts/y98_deployment.py", {"UNUSED_REPLACEMENT": "UNUSED_REPLACEMENT"}, {"contract"})
    header += transformed("scripts/y98_deployment.py", {"Y96": "Y98", "y96": "y98", "Y98": "Y99", "y98": "y99",
        "advance_y98": "advance_y99"}, {"select"})
    header += transformed("scripts/y98_deployment.py", {"Y96": "Y99", "y96": "y99"}, {"observe"})
    header += '''
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("operation",choices=("select","observe"))
    p.add_argument("--id",choices=("AdaRES-Y98","AdaRES-Y99"),default="AdaRES-Y99")
    p.add_argument("--token",required=True)
    p.add_argument("--label",default="progress")
    a=p.parse_args()
    lease(a.token,240)
    writer=monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        select(a.token) if a.operation=="select" else observe(a.token,a.id,a.label)
    finally:
        writer.close()
'''
    outputs["scripts/y99_deployment.py"] = header
    pair = {"y95": "y98", "Y95": "Y98", "y96": "y99", "Y96": "Y99"}
    for source, target in (("prepare_y95_y96_terminal_binding.py", "prepare_y98_y99_terminal_binding.py"),
                           ("y95_y96_terminal_contract.py", "y98_y99_terminal_contract.py")):
        outputs["scripts/" + target] = transformed("scripts/" + source, pair)
    outputs["scripts/test_y98_y99_terminal_binding.py"] = transformed("scripts/test_y95_y96_terminal_binding.py",
        dict(pair, Y94="Y96", **{"34484": "25962", "20261109_081912": "20261110_080627",
        "de4e2620341bcd651a6d2ad3d8bfa98dc7e2e8523a7da9f1ffaea8c11d7597c5":
            "dc1a8a6dd3afde694d255cceca30b1b16009e8249390aa706632e2a567136a62"}))
    # The preserved PID assertion is an integer AST node rather than text.
    body = outputs["scripts/test_y98_y99_terminal_binding.py"]
    assert body.count("34484") == 1
    outputs["scripts/test_y98_y99_terminal_binding.py"] = body.replace("34484", "25962")
    test_changes = {"Y95": "Y96", "y95": "y96", "Y96": "Y98", "y96": "y98", "Y98": "Y99", "y98": "y99",
        "advance_y98_v2": "advance_y99", "server-2": "server-1", "server-1": "server-2",
        "appearance_mask": "spatial_mask", "base_adapted_and_original_appearance_gradients_finite":
            "base_adapted_spatial_and_appearance_gradients_finite",
        "raw_appearance_requires_grad": "raw_spatial_gradients_finite",
        "original_full1024_appearance_identity": "original_full32_spatial_identity",
        "original_appearance_projection_executed": "original_spatial_and_appearance_projections_executed",
        "float64_full_appearance_projection_reference": "float64_full_spatial_projection_reference"}
    tests = transformed("scripts/test_y98_admission.py", test_changes)
    tests = tests.replace("280.0", "350.0").replace("280.", "350.")
    # In the rejected device examples, the wrong 350W observation must become 280W.
    tests = tests.replace("'nan,0,280'", "'nan,0,350'").replace("'0,0,350'", "'0,0,280'")
    tests = tests.replace("'0,2000,280'", "'0,2000,350'").replace("'0,0,280', '99'", "'0,0,350', '99'")
    tests = tests.replace("'0,1,280'", "'0,1,350'")
    tests = tests.replace("('raw_spatial_gradients_finite', True)", "('raw_spatial_gradients_finite', False)")
    # raw-spatial gradient status is verified per-step, not in path identity rows.
    tests = tests.replace(", ('raw_spatial_gradients_finite', False)", "")
    tests = tests.replace("y98_y99_terminal_contract_v2.py", "y98_y99_terminal_contract.py")
    tests = tests.replace("prepare_y98_y99_terminal_binding_v2.py", "prepare_y98_y99_terminal_binding.py")
    outputs["scripts/test_y99_admission.py"] = tests
    for path, body in outputs.items():
        assert not (ROOT / path).exists(), path
        ast.parse(body, feature_version=(3, 8))
    for path, body in outputs.items():
        (ROOT / path).write_text(body, encoding="utf-8", newline="\n")
    assert all(ref(p) == entry for p, entry in before.items())
    assert git("ls-files", "--stage", "-z") == index
    assert [ref(entry["path"]) for entry in old] == old
    output = ROOT / ("pipeline/checks/y99_launch_helpers_" + token + ".json")
    assert not output.exists()
    tick.save(output, dict(complete=True, token=token, original_inputs=old,
        generated=[ref(p) for p in outputs], production_HEAD_preserved=Y98,
        original_helpers_queue_running_and_scientific_files_unchanged=True,
        parsed_python38=True, native_executed=False, admitted=False,
        guard_tests_pending_actual_native=True, evidence=old + [ref(__file__)] + [ref(p) for p in outputs]))
    print(json.dumps(dict(manifest=ref(output), generated=len(outputs), native_executed=False, guard_tests_pending=True)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True)
    create(p.parse_args().token)
