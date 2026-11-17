"""Specialize preserved server2 lifecycle tools for the exact original Y95 run."""
import argparse
import ast
import json

from prepare_y60_stage_heads import ROOT, lease, ref, tick
from y95_deployment import Y95, git
from advance_y95 import prepared
from y95_contract import validate_result
import closure


ADMISSION = "de4e2620341bcd651a6d2ad3d8bfa98dc7e2e8523a7da9f1ffaea8c11d7597c5"
FREEZE = "2248202035aaa859a4cab6d5b1e4f2ffd947f4775725390562f36a944ceed6c9"
LOG = ("/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/"
       "train_adares_y95_query_set_mean_res_20261109_212023.log")


class CurrentRun(ast.NodeTransformer):
    @staticmethod
    def text(value):
        for old, new in (
            ("adares_y91_hidden_confidence_gate", "adares_y95_query_set_mean"),
            ("20261107_232228", "20261109_212023"),
            ("4d59825df7e4d0cfe428fc47c3ff6102749f2b1e5f12d84b71aaf5238aa96eaf", ADMISSION),
            ("e5e6eef716ac892e827fe9131907ec44add8fdad1f619da8a568dacfc6935ca1", FREEZE),
            ("83827", "34484"),
            ("Y91", "Y95"), ("y91", "y95"),
        ):
            value = value.replace(old, new)
        return value

    def visit_Name(self, node):
        node.id = self.text(node.id)
        return node

    def visit_ImportFrom(self, node):
        node.module = self.text(node.module)
        for alias in node.names:
            alias.name = self.text(alias.name)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.text(node.value)
        elif type(node.value) is int and node.value == 83827:
            node.value = 34484
        return node


def create(token):
    lease(token, 300)
    assert git("rev-parse", "HEAD").decode().strip() == Y95
    assert ref("pipeline/admissions/AdaRES-Y95.json")["sha256"] == ADMISSION
    work = tick.load(ROOT / "pipeline/research_work.json")
    candidate, _, _, frozen = prepared(work)
    assert candidate["stage"] == "training" and not candidate.get("terminal_handoff_plan")
    assert candidate["post_training_freeze"]["sha256"] == FREEZE
    closure.artifact(ROOT, candidate["post_training_freeze"])
    validate_result(tick.load(closure.artifact(ROOT, candidate["native_smoke_result"])), candidate, frozen)
    source = ROOT / "scripts/create_y91_terminal_helpers.py"
    original = ref(source)
    tree = CurrentRun().visit(ast.parse(source.read_text(encoding="utf-8")))
    body = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    ast.parse(body, feature_version=(3, 8))
    output = ROOT / ("pipeline/checks/y95_terminal_wrapper_" + token + ".py")
    assert not output.exists()
    output.write_text(body, encoding="utf-8", newline="\n")
    namespace = {"__name__": "y95_terminal_generation", "__file__": __file__}
    exec(compile(body, str(output), "exec"), namespace)
    spec = namespace["SPEC"]
    assert namespace["CID"] == "AdaRES-Y95"
    assert spec["pid"] == 34484 and spec["code_commit"] == Y95 and spec["log"] == LOG
    assert spec["admission_sha256"] == ADMISSION and spec["freeze_sha256"] == FREEZE
    namespace["create"](token)
    assert ref(source) == original
    print(json.dumps(dict(wrapper=ref(output), immutable_generator=original,
                          lifecycle_guards_prepared=29, checkpoint_loaded=False, archived=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
