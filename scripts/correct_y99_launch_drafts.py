"""Keep untested first drafts and specialize explicit attribute references."""
import argparse
import ast
import json
import re
from prepare_y60_stage_heads import ROOT, lease, ref, tick
from y98_deployment import Y98, git

FILES = ("advance_y99", "y99_deployment", "prepare_y98_y99_terminal_binding",
         "y98_y99_terminal_contract", "test_y98_y99_terminal_binding", "test_y99_admission")
RENAMES = {name: name + "_v2" for name in FILES}
RENAMES.update(finalize_y98_fixed30_y99="finalize_y98_fixed30_y99_v2",
               test_y98_terminal_handoff_y99="test_y98_terminal_handoff_y99_v2")
PATTERN = re.compile("|".join(re.escape(k) for k in sorted(RENAMES, key=len, reverse=True)))


def rename(value):
    return PATTERN.sub(lambda m: RENAMES[m.group()], value)


class Correct(ast.NodeTransformer):
    def __init__(self, attributes):
        self.attributes = attributes

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "deploy":
            node.attr = self.attributes.get(node.attr, node.attr)
        return node

    def visit_Name(self, node):
        node.id = rename(node.id)
        return node

    def visit_alias(self, node):
        node.name = rename(node.name)
        return node

    def visit_ImportFrom(self, node):
        node.module = rename(node.module)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = rename(node.value)
        return node


def correct(token):
    lease(token, 600)
    assert git("rev-parse", "HEAD").decode().strip() == Y98
    before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/research_work.json",
                                  "pipeline/goal.json", "SERVERS.yaml")}
    original = [ref("scripts/" + name + ".py") for name in FILES]
    outputs = []
    for name in FILES:
        attrs = ({"Y95": "Y98", "Y96": "Y99", "Y94": "Y96"}
                 if name == "test_y98_y99_terminal_binding" else
                 {"Y96": "Y98", "Y98": "Y99"} if name == "test_y99_admission" else {})
        tree = Correct(attrs).visit(ast.parse((ROOT / "scripts" / (name + ".py")).read_text(encoding="utf-8")))
        body = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        if name == "test_y99_admission":
            body = body.replace("test_full1024_appearance_and_seven_static_consumers", "test_full32_spatial_and_seven_static_consumers")
            body = body.replace("test_full_original_server2_protocol_exact", "test_full_original_server1_protocol_exact")
            body = body.replace("('original_visual_parameter_tensors', 0)",
                "('original_visual_parameter_tensors', 0), ('raw_spatial_gradients_finite', False), ('original_spatial_projection_calls', 0)")
        output = ROOT / "scripts" / (name + "_v2.py")
        assert not output.exists()
        ast.parse(body, feature_version=(3, 8))
        output.write_text(body, encoding="utf-8", newline="\n")
        outputs.append(ref(output))
    assert [ref(row["path"]) for row in original] == original
    assert all(ref(p) == entry for p, entry in before.items())
    manifest = ROOT / ("pipeline/checks/y99_launch_helpers_" + token + "_v2.json")
    assert not manifest.exists()
    tick.save(manifest, dict(complete=True, token=token, preserved_untested_drafts=original,
        correction="Explicit deploy version attributes follow the same role mapping as identifiers; names and added raw-spatial gradient guards match actual Y99 contract. No scientific code or control changed.",
        generated=outputs, actual_native_and_guard_tests_pending=True, production_HEAD_preserved=Y98,
        original_inputs=ref("pipeline/checks/y99_launch_helpers_" + token + ".json"),
        evidence=original + outputs + [ref(__file__)]))
    print(json.dumps(dict(manifest=ref(manifest), generated=len(outputs), tests_pending=True)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True)
    correct(p.parse_args().token)
