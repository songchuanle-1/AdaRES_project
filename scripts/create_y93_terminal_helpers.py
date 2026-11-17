"""Specialize the preserved server-2 lifecycle generator for the exact Y93 run."""
import argparse
import ast
from pathlib import Path

from prepare_y60_stage_heads import ROOT, lease, ref


class CurrentRun(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = node.id.replace("Y91", "Y93").replace("y91", "y93")
        return node

    def visit_ImportFrom(self, node):
        node.module = node.module.replace("y91", "y93")
        for alias in node.names:
            alias.name = alias.name.replace("Y91", "Y93").replace("y91", "y93")
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for old, new in (
                ("adares_y91_hidden_confidence_gate", "adares_y93_evidence_hidden_skip"),
                ("20261107_232228", "20261108_212628"),
                ("4d59825df7e4d0cfe428fc47c3ff6102749f2b1e5f12d84b71aaf5238aa96eaf",
                 "f6b87f2813f1e8e429772bc0f63963fe1a43e0582c6c65e21771092dcbf69832"),
                ("e5e6eef716ac892e827fe9131907ec44add8fdad1f619da8a568dacfc6935ca1",
                 "8563a67a55b91f7c72b2899d03917b28d322e2f5ac14130c016511c72b0c7ef8"),
                ("83827", "47380"),
                ("Y91", "Y93"),
                ("y91", "y93"),
            ):
                node.value = node.value.replace(old, new)
        elif type(node.value) is int and node.value == 83827:
            node.value = 47380
        return node


def create(token):
    lease(token, 300)
    source = ROOT / "scripts/create_y91_terminal_helpers.py"
    # Keep the exact original generator and all signed lifecycle tools immutable.
    original = ref(source)
    tree = CurrentRun().visit(ast.parse(source.read_text(encoding="utf-8")))
    tree = ast.fix_missing_locations(tree)
    ast.parse(ast.unparse(tree), feature_version=(3, 8))
    namespace = {"__name__": "y93_terminal_generation", "__file__": str(Path(__file__))}
    exec(compile(tree, str(Path(__file__)), "exec"), namespace)
    assert namespace["CID"] == "AdaRES-Y93"
    assert namespace["SPEC"]["pid"] == 47380
    assert namespace["SPEC"]["code_commit"] == "daa93b4375f144f0e79da92f85fe1fe27e877e18"
    namespace["create"](token)
    assert ref(source) == original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
