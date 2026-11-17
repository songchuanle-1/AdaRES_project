"""Mechanical AST identity adaptation of established native/resource protocols."""

import ast
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/"scripts"
replacements = {
    "configs_song/model_paper5/adares_y28_mask_readback_res.yaml":
        "configs_song/model_paper5/adares_y30_query_trajectory_res.yaml",
    ".mask_readback.": ".trajectory_readout.",
    "AdaRES-Y28": "AdaRES-Y30",
}
old_attr = ast.parse("model.dec.indi_embedding.mask_readback", mode="eval").body
new_attr = ast.parse("model.dec.trajectory_readout", mode="eval").body


class Adapter(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "readback_input":
            return ast.parse("""
def readback_input(module, args):
    queries, masks, padding, mass = args
    if mode == "rolled":
        return ([v.roll(1, 1) for v in queries[:-1]]+queries[-1:],
                [v.roll(1, 1) for v in masks[:-1]]+masks[-1:], padding, mass)
    if mode == "reverse":
        return (list(reversed(queries[:-1]))+queries[-1:],
                list(reversed(masks[:-1]))+masks[-1:], padding, mass)
    if mode == "terminal":
        return [queries[-1]]*len(queries), [masks[-1]]*len(masks), padding, mass
""").body[0]
        return self.generic_visit(node)

    def visit_Tuple(self, node):
        if (len(node.elts) == 3 and all(isinstance(n, ast.Constant) for n in node.elts)
                and [n.value for n in node.elts] == ["enabled", "rolled", "uniform"]):
            return ast.copy_location(ast.Tuple(elts=[ast.Constant(value=s) for s in
                                                     ("enabled", "rolled", "reverse", "terminal")],
                                              ctx=ast.Load()), node)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = replacements.get(node.value, node.value)
            value = value.replace("Y28", "Y30").replace("readback", "trajectory readout")
            # Parameter filters must remain literal qualified attribute names.
            if node.value == ".mask_readback.":
                value = ".trajectory_readout."
            return ast.copy_location(ast.Constant(value=value), node)
        return node

    def visit_Attribute(self, node):
        if ast.dump(node) == ast.dump(old_attr):
            return ast.copy_location(ast.parse("model.dec.trajectory_readout", mode="eval").body, node)
        return self.generic_visit(node)

    def visit_Compare(self, node):
        self.generic_visit(node)
        left = node.left
        if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                and left.func.id == "len" and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == 4):
            arg = left.args[0]
            if ((isinstance(arg, ast.Name) and arg.id in ("norms", "affine_changes"))
                    or (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                        and arg.func.id == "list")):
                node.comparators[0] = ast.Constant(value=8)
        return node


parser = argparse.ArgumentParser()
parser.add_argument("--stem", choices=("initial_loss_investigation",))
args = parser.parse_args()
for stem in (args.stem,) if args.stem else ("physical_native", "resources"):
    source = SCRIPTS/("probe_y28_"+stem+".py")
    destination = SCRIPTS/("probe_y30_"+stem+".py")
    assert not destination.exists()
    tree = Adapter().visit(ast.parse(source.read_text(encoding="utf-8")))
    ast.fix_missing_locations(tree)
    if stem == "initial_loss_investigation":
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("Wrong controls here"):
                n.value = "Wrong controls are earlier-query history roll/reverse/terminal repetition with the current final state fixed."
    text = ast.unparse(tree)+"\n"
    assert ".mask_readback." not in text and "adares_y28_" not in text
    compile(text, str(destination), "exec")
    destination.write_text(text, encoding="utf-8")
    assert ast.dump(ast.parse(text)) == ast.dump(tree)
    print(destination.name, hashlib.sha256(destination.read_bytes()).hexdigest())
