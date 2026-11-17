"""Specialize immutable server1 lifecycle tools for the exact live Y94 run."""
import argparse
import ast
import json

from prepare_y60_stage_heads import ROOT, lease, ref
from y94_deployment import Y94, git


ADMISSION = "b9deb51efa3c274283f8a23ae678d42a4f0c11a29e758d19ae696e80df13cb84"
FREEZE = "7e40371552d170a1d764f9dab160a8dcb12b0b6fe512e5c978313cb7b642241b"


class CurrentRun(ast.NodeTransformer):
    @staticmethod
    def text(value):
        for old, new in (
            ("27d6f67e7121a5168bbfeca9db745377e0bd49e5", Y94),
            ("56c0a076cc22dd82200f934781dd50b7a5972ab2adb3c383acb42cb6cb0822dc", ADMISSION),
            ("b866bf8a784b9b44ff63cb1dd7de74fd6288b0b84a1f2280a45c9eaa85a76f95", FREEZE),
            ("mask_feature_quadratic", "query_geometry_bias"),
            ("Y92", "Y94"), ("y92", "y94"),
            ("1305468", "218513"), ("2442962", "1305468"),
        ):
            value = value.replace(old, new)
        return value

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.text(node.value)
        elif type(node.value) is int:
            node.value = {1305468: 218513, 2442962: 1305468}.get(node.value, node.value)
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(target, ast.Name) and target.id == "EXTRA_TESTS" for target in node.targets):
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            node.value.value = node.value.value.replace("AdaRES-Y90", "AdaRES-Y92")
        return node

    def visit_ImportFrom(self, node):
        node.module = self.text(node.module)
        for alias in node.names:
            alias.name = self.text(alias.name)
        return node

    def visit_alias(self, node):
        node.name = self.text(node.name)
        return node

    def visit_Name(self, node):
        node.id = self.text(node.id)
        return node


def create(token):
    lease(token, 300)
    assert git("rev-parse", "HEAD").decode().strip() == Y94
    assert ref("pipeline/admissions/AdaRES-Y94.json")["sha256"] == ADMISSION
    source = ROOT / "scripts/create_y92_terminal_helpers.py"
    original = ref(source)
    tree = CurrentRun().visit(ast.parse(source.read_text(encoding="utf-8")))
    ast.fix_missing_locations(tree)
    body = ast.unparse(tree) + "\n"
    ast.parse(body, feature_version=(3, 8))
    output = ROOT / ("pipeline/checks/y94_terminal_wrapper_" + token + ".py")
    assert not output.exists()
    output.write_text(body, encoding="utf-8", newline="\n")
    namespace = {"__name__": "y94_terminal_generation", "__file__": __file__}
    exec(compile(body, str(output), "exec"), namespace)
    assert namespace["CID"] == "AdaRES-Y94"
    assert namespace["ADMISSION"] == ADMISSION and namespace["FREEZE"] == FREEZE
    assert "parent_candidate" in namespace["EXTRA_TESTS"]
    assert "AdaRES-Y92" in namespace["EXTRA_TESTS"]
    assert "AdaRES-Y90" not in namespace["EXTRA_TESTS"]
    namespace["create"](token)
    assert ref(source) == original
    print(json.dumps(dict(wrapper=ref(output), immutable_generator=original,
                          lifecycle_guards_prepared=30, checkpoint_loaded=False, archived=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
