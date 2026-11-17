"""Specialize existing server2 launch and server1 terminal tools for actual Y100."""
import argparse
import ast
import json
import re
from prepare_y60_stage_heads import ROOT, lease, ref, tick
from y99_deployment_v2 import Y98, Y99, git

Y100 = "294de23f52e4ff93ebbf370c17c13d9a7d02e10b"


class Rewrite(ast.NodeTransformer):
    def __init__(self, changes):
        self.changes = changes
        self.pattern = re.compile("|".join(re.escape(k) for k in sorted(changes, key=len, reverse=True)))
    def text(self, v):
        return self.pattern.sub(lambda m: self.changes[m.group()], v)
    def visit_Constant(self, n):
        if isinstance(n.value, str):
            n.value = self.text(n.value)
        elif type(n.value) is int and n.value == 3893460:
            n.value = 2980836
        return n
    def visit_Name(self, n):
        n.id = self.text(n.id)
        return n
    def visit_Attribute(self, n):
        n.attr = self.text(n.attr)
        return self.generic_visit(n)
    def visit_ImportFrom(self, n):
        n.module = self.text(n.module)
        return self.generic_visit(n)
    def visit_alias(self, n):
        n.name = self.text(n.name)
        if n.asname:
            n.asname = self.text(n.asname)
        return n


def transformed(path, changes, functions=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    if functions is not None:
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
        assert len(tree.body) == len(functions)
    tree = Rewrite(changes).visit(tree)
    text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    ast.parse(text, feature_version=(3, 8))
    return text


def create(token):
    lease(token, 600)
    assert git("rev-parse", "HEAD").decode().strip() == Y99
    assert git("rev-parse", Y100 + "^").decode().strip() == Y99
    before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/research_work.json", "pipeline/goal.json", "SERVERS.yaml", "pipeline/admissions/AdaRES-Y99.json")}
    index = git("ls-files", "--stage", "-z")
    work = tick.load(ROOT / "pipeline/research_work.json")
    c = next(r for r in work["candidates"] if r["id"] == "AdaRES-Y100")
    assert c["stage"] == "validation" and c["code_commit"] == Y100 and c.get("native_smoke_result")
    sources = ["scripts/advance_y98_v2.py", "scripts/y98_deployment.py", "scripts/prepare_y96_y98_terminal_binding_v2.py", "scripts/y96_y98_terminal_contract_v2.py", "scripts/test_y96_y98_terminal_binding_v3.py", "scripts/test_y98_admission.py"]
    originals = [ref(p) for p in sources]
    changes = {
        "run_y98_bound_native_y96": "run_y100_native_v2", "advance_y98_v2": "advance_y100", "advance_y98": "advance_y100",
        "y98_deployment": "y100_deployment", "y98_contract": "y100_admission_contract",
        "y96_y98_terminal_contract_v2": "y99_y100_terminal_contract",
        "test_y96_y98_terminal_binding_v2": "test_y99_y100_terminal_binding",
        "test_y96_y98_terminal_binding_v3": "test_y99_y100_terminal_binding",
        "prepare_y96_y98_terminal_binding_v2": "prepare_y99_y100_terminal_binding",
        "pipeline/admissions/AdaRES-Y96.json": "pipeline/admissions/AdaRES-Y99.json",
        "2248202035aaa859a4cab6d5b1e4f2ffd947f4775725390562f36a944ceed6c9": "bcd19eedca805c8491568b3f81cf9cefa8dc4b0def8f6a09dd87c58f5e29d51c",
        "20261110_165702": "20261111_180638",
        "5dcd15c8da9efca80c530a5c61767fc68f9976a2e0c70aedcef3f5807ea42b1c": "4d87ad8b3a6efe78be8dc9e8214bba9226160a22b649f64f8e1dd26d73988ede",
        "50213e2f1fda98e5e9477eaa5c030952d1d913d3d1aeff86d1fc6aa984dfc154": "fd07c0536891d905a7cd89e2b4211b32ed31a532b7b70ac9d71a8d7a66e97dbc",
        "Y95": "Y98", "y95": "y98", "Y96": "Y99", "y96": "y99", "Y98": "Y100", "y98": "y100",
    }
    outputs = {"scripts/advance_y100.py": transformed(sources[0], changes)}
    # Correct descriptive template text without changing its executable resource/data guards.
    outputs["scripts/advance_y100.py"] = outputs["scripts/advance_y100.py"].replace(
        "independent full-existing-appearance bypass of the original static mask MLP", "independent full1024 appearance bypass beside the original visual-memory projection MLP").replace(
        "Full original server1 frozen evaluation/cost, exact data, seed1999/fixed30 and350W preserved. Efficacy unknown. Original Y98 and GPU1/Y57 pauses preserved.",
        "Full original server2 GPU0/280W frozen evaluation/cost, exact data and seed1999/fixed30 preserved. Efficacy unknown. Original Y99 and GPU1/Y57 pauses preserved.")
    header = '''"""Select exact actual-native Y100 additions; preserve the original Y99 run."""
import argparse, ast, hashlib, inspect, json, subprocess
from datetime import datetime
from pathlib import Path
import yaml
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from observe_y69_server1_phase import parse_phase,phase_key,progressed,REMOTE
import closure
import y99_deployment_v2 as previous
Y98,Y99=previous.Y98,previous.Y99
Y100="294de23f52e4ff93ebbf370c17c13d9a7d02e10b"
VERSIONS=previous.VERSIONS+(Y100,)
ADDITIONS=dict(previous.ADDITIONS,**{Y100:"AdaRES-Y100"})
git,adapt=previous.git,previous.adapt
def paths(cid):
    if cid!="AdaRES-Y100":
        return previous.paths(cid)
    from prepare_y100_appearance_input_skip import files
    return [p.relative_to(ROOT).as_posix() for p in files()]
def require_running_parent_handoff(work,candidate):
    original=next(r for r in work["candidates"] if r["id"]=="AdaRES-Y99")
    assert original["stage"]=="training"
    plan=tick.load(closure.artifact(ROOT,original["terminal_handoff_plan"]))
    assert plan["complete"] is True and plan["id"]==original["id"]
    assert plan["tested_additive_native_evidence"]==candidate["native_smoke_result"]
    assert plan["payload"]["allowed_markers"]==[Y99,Y100]
    assert plan["CPU_guard_tests_passed"]==30 and plan["additive_guard_tests_passed"]==8
    for entry in plan["dependencies"]:
        closure.artifact(ROOT,entry)
'''
    body = header + transformed(sources[1], changes, {"validate_delta", "contract", "select", "observe"})
    body += '''
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("operation",choices=("select","observe"))
    p.add_argument("--id",choices=("AdaRES-Y99","AdaRES-Y100"),default="AdaRES-Y100")
    p.add_argument("--token",required=True)
    p.add_argument("--label",default="progress")
    a=p.parse_args()
    lease(a.token,240)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        select(a.token) if a.operation=="select" else observe(a.token,a.id,a.label)
    finally:
        writer.close()
'''
    outputs["scripts/y100_deployment.py"] = body
    prep = transformed(sources[2], changes)
    prep = prep.replace("test_y99_y100_terminal_binding_v2.py", "test_y99_y100_terminal_binding.py")
    outputs["scripts/prepare_y99_y100_terminal_binding.py"] = prep
    binding = transformed(sources[3], changes)
    # The original terminal module provides both original-only delta and strict server1 mapping checks.
    binding = binding.replace("from y100_deployment import Y99, Y100, adapt, contract, git", "from y100_deployment import Y99, Y100, adapt, contract, git")
    outputs["scripts/y99_y100_terminal_contract.py"] = binding
    outputs["scripts/test_y99_y100_terminal_binding.py"] = transformed(sources[4], changes)
    admission_changes = dict(changes, **{
        "appearance_mask_calls": "appearance_input_calls",
        "appearance_mask_paths": "appearance_input_paths",
        "original_base_mask_and_static_reuse_preserved": "static_mask_features_preserved",
        "base_adapted_and_original_appearance_gradients_finite": "base_adapted_fusion_and_mask_gradients_finite",
        "original_base_mask_identity": "original_appearance_projection_identity",
        "original_appearance_projection_executed": "original_sum_norm_consumes_skip",
        "original_batching_consumes_skip": "original_language_memory_consumes_skip",
        "all_seven_predictions_reuse_same_features": "all_seven_original_static_masks_preserved",
        "mask_feature_delta_max": "appearance_feature_delta_max",
    })
    tests = transformed(sources[5], admission_changes)
    tests = tests.replace('"test_y100_admission.py"', '"test_y100_admission.py"').replace("'y99_y100_terminal_contract_v2.py'", "'y99_y100_terminal_contract.py'").replace("'prepare_y99_y100_terminal_binding_v2.py'", "'prepare_y99_y100_terminal_binding.py'")
    outputs["scripts/test_y100_admission.py"] = tests
    for path, text in outputs.items():
        assert not (ROOT / path).exists()
        ast.parse(text, feature_version=(3, 8))
    for path, text in outputs.items():
        (ROOT / path).write_text(text, encoding="utf-8", newline="\n")
    assert [ref(p) for p in sources] == originals
    assert all(ref(p) == r for p, r in before.items()) and git("ls-files", "--stage", "-z") == index
    out = ROOT / ("pipeline/checks/y100_launch_helpers_" + token + ".json")
    assert not out.exists()
    tick.save(out, dict(complete=True, token=token, id=c["id"], HEAD_preserved=Y99, actual_native=c["native_smoke_result"],
        exact_originals_preserved=True, source_specialization_only=True, native_replayed=False, admitted=False,
        files=[ref(p) for p in outputs], originals=originals, evidence=[ref(__file__), ref("scripts/y100_admission_contract.py")]))
    print(json.dumps(dict(preparation=ref(out), files=list(outputs), actual_native=c["native_smoke_result"], admitted=False)))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--token",required=True)
    create(p.parse_args().token)
