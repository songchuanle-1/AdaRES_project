"""Prepare exact current Y100 lifecycle tools from the preserved server2 generator."""
import argparse
import ast
import json
import re
from prepare_y60_stage_heads import ROOT, lease, ref, tick, monitor
from y100_deployment import Y100, git
from advance_y100 import prepared, validate_result
import closure

ADMISSION = "7091f206c21429c02349c73e68d5e2b06ca3adc9f585465450870ac2a0e019f2"
FREEZE = "cb9b9a80fb6f9f9b116f9e9a35a515b3c4fb542eb6f4128cbee0ed5d389eff01"


class CurrentRun(ast.NodeTransformer):
    def __init__(self, log):
        self.changes = {
            "adares_y91_hidden_confidence_gate": "adares_y100_appearance_input_skip",
            "/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y91_hidden_confidence_gate_res_20261107_232228.log": log,
            "4d59825df7e4d0cfe428fc47c3ff6102749f2b1e5f12d84b71aaf5238aa96eaf": ADMISSION,
            "e5e6eef716ac892e827fe9131907ec44add8fdad1f619da8a568dacfc6935ca1": FREEZE,
            "Y91": "Y100", "y91": "y100", "83827": "19305"}
        self.pattern = re.compile("|".join(re.escape(k) for k in sorted(self.changes, key=len, reverse=True)))
    def text(self, value):
        return self.pattern.sub(lambda m: self.changes[m.group()], value)
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
        return n
    def visit_Constant(self, n):
        if isinstance(n.value, str):
            n.value = self.text(n.value)
        elif type(n.value) is int and n.value == 83827:
            n.value = 19305
        return n


def create(token):
    lease(token, 300)
    assert git("rev-parse", "HEAD").decode().strip() == Y100
    assert ref("pipeline/admissions/AdaRES-Y100.json")["sha256"] == ADMISSION
    work = tick.load(ROOT / "pipeline/research_work.json")
    c, _, _, frozen = prepared(work)
    assert c["stage"] == "training" and not c.get("terminal_handoff_plan")
    assert c["post_training_freeze"]["sha256"] == FREEZE
    validate_result(tick.load(closure.artifact(ROOT, c["native_smoke_result"])), c, frozen)
    task = next(r for r in monitor.load_queue(str(ROOT))["gpu_queues"]["server-2"] if r["id"] == "AdaRES-Y100")
    assert task["pid"] == 19305 and task["code_commit"] == Y100
    source = ROOT / "scripts/create_y91_terminal_helpers.py"
    original = ref(source)
    tree = CurrentRun(task["log"]).visit(ast.parse(source.read_text(encoding="utf-8")))
    body = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    ast.parse(body, feature_version=(3, 8))
    out = ROOT / ("pipeline/checks/y100_terminal_wrapper_" + token + ".py")
    assert not out.exists()
    out.write_text(body, encoding="utf-8", newline="\n")
    ns = {"__name__": "y100_terminal_generation", "__file__": __file__}
    exec(compile(body, str(out), "exec"), ns)
    spec = ns["SPEC"]
    assert ns["CID"] == "AdaRES-Y100" and spec["code_commit"] == Y100
    assert spec["pid"] == 19305 and spec["log"] == task["log"]
    assert spec["admission_sha256"] == ADMISSION and spec["freeze_sha256"] == FREEZE
    ns["create"](token)
    assert ref(source) == original
    print(json.dumps(dict(wrapper=ref(out), immutable_generator=original, checkpoint_loaded=False, archived=False)))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--token",required=True)
    create(p.parse_args().token)
