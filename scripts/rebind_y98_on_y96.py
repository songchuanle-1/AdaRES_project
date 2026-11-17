"""Bind identical original Y98 bytes to actual-native Y96 and exact owners."""
import argparse
import ast
import json
import re
from prepare_y60_stage_heads import ROOT, lease, ref
from y96_deployment import Y95, Y96, git

OLD = "03799b0b2cacca31987a604c539a4457498175a1"
OWNERS = {"server-1": "AdaRES-Y96", "server-2": "AdaRES-Y95"}
REPLACEMENTS = {
    "afc93861cf66b0da4835391597ee2d5ce90f6584": OLD,
    "refs/heads/adares-y95-query-set-mean-on-y94": "refs/heads/adares-y98-appearance-mask-skip-on-y96",
    "refs/heads/adares-y95-query-set-mean": "refs/heads/adares-y98-appearance-mask-skip",
    "run_y95_bound_native_y94": "run_y98_bound_native_y96",
    "run_y95_native": "run_y98_native",
    "retain_y95_cpu_native": "retain_y98_cpu_native",
    "test_y95_runtime_plan": "test_y98_runtime_plan",
    "rebind_y95_on_y94": "rebind_y98_on_y96",
    "prepare_y95_query_set_mean": "prepare_y98_appearance_mask_skip",
    "y95_contract": "y98_contract",
    "advance_y94": "advance_y96",
    "y94_deployment": "y96_deployment",
    "y95_y94_additive_binding_": "y98_y96_additive_binding_",
    "original_pre_y94_binding": "original_pre_y96_binding",
    "AdaRES-Y95": "AdaRES-Y98", "AdaRES-Y94": "AdaRES-Y96",
    "AdaRES-Y93": "AdaRES-Y95", "AdaRES-Y92": "AdaRES-Y94",
    "HEADY94": "HEADY96", "Y94/server1": "Y96/server1", "Y93/server2": "Y95/server2",
    "Y95 bytes": "Y98 bytes", "Y95 additions": "Y98 additions",
    "native-tested Y94": "native-tested Y96",
    "Current Y94/server1 and original Y93/server2 occupy both authorized GPU0 resources; identical Y95 bytes now have a tested HEADY94/current-owner execution binding.":
        "Original Y96/server1 is preserved and Y95/server2 completed; identical Y98 bytes have a tested HEADY96/original-owner binding. Actual native requires a fresh exclusive idle/unreserved authorized GPU0.",
}
PATTERN = re.compile("|".join(re.escape(key) for key in sorted(REPLACEMENTS, key=len, reverse=True)))
RUNNER = '''
class Runner(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == "daa93b4375f144f0e79da92f85fe1fe27e877e18":
            node.value = Y96
        elif isinstance(node.value, str):
            node.value = node.value.replace("run_y98_native.py", "run_y98_bound_native_y96.py")
        return node
    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(target, ast.Name) and target.id == "OWNERS" for target in node.targets):
            node.value = ast.parse(repr(OWNERS), mode="eval").body
        return node
    def visit_FunctionDef(self, node):
        if node.name == "validate_binding":
            return ast.parse(BINDING).body[0]
        return self.generic_visit(node)
'''
CURRENT_SOURCE_TEST = '''
def test_actual_parent_native_and_preserved_current_run_sources(self):
    self.assertEqual(parent_evidence(), self.proposal["parent_native"])
    self.assertEqual(list(contract(Y95, Y96, "AdaRES-Y95")), [Y95, Y96])
    queue = monitor.load_queue(str(ROOT))
    task = next(row for row in queue["gpu_queues"]["server-1"] if row["id"] == "AdaRES-Y96")
    self.assertEqual((task["status"], task["pid"], task["code_commit"]), ("running",3893460,Y96))
    self.assertTrue(task["launch_verified"])
    closure.admission(ROOT,task)
    done = monitor.load_experiment(str(ROOT),"AdaRES-Y95")
    self.assertEqual((done["status"],done["server"],str(done["gpu"])),("done_complete","server-2","0"))
    proof = tick.load(closure.artifact(ROOT,done["terminal_evidence"]))
    runner_module(ROOT/RUNNER).validate_terminal(proof,"server-2")
    self.assertEqual(ref("pipeline/admissions/AdaRES-Y95.json")["sha256"],"de4e2620341bcd651a6d2ad3d8bfa98dc7e2e8523a7da9f1ffaea8c11d7597c5")
'''

class Specialize(ast.NodeTransformer):
    def text(self, value):
        return PATTERN.sub(lambda match: REPLACEMENTS[match.group()], value)
    def visit_Name(self, node):
        node.id = {"Y93":"Y95","Y94":"Y96"}.get(node.id,node.id)
        return node
    def visit_ImportFrom(self,node):
        node.module = self.text(node.module)
        for alias in node.names:
            alias.name = {"Y93":"Y95","Y94":"Y96"}.get(alias.name,alias.name)
        return node
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value = self.text(node.value)
        return node
    def visit_Assign(self,node):
        if any(isinstance(target,ast.Name) and target.id=="OWNERS" for target in node.targets):
            node.value=ast.parse(repr(OWNERS),mode="eval").body
            return node
        return self.generic_visit(node)
    def visit_ClassDef(self,node):
        if node.name=="Runner":
            return ast.parse(RUNNER).body[0]
        return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        if node.name=="test_actual_parent_native_and_preserved_current_run_sources":
            return ast.parse(CURRENT_SOURCE_TEST).body[0]
        return self.generic_visit(node)

def rebind(token):
    lease(token,300)
    assert git("rev-parse","HEAD").decode().strip()==Y96
    source=ROOT/"scripts/rebind_y95_on_y94.py"
    original=ref(source)
    tree=Specialize().visit(ast.parse(source.read_text(encoding="utf-8")))
    body=ast.unparse(ast.fix_missing_locations(tree))+"\n"
    ast.parse(body,feature_version=(3,8))
    output=ROOT/("pipeline/checks/y98_y96_binding_wrapper_"+token+".py")
    assert not output.exists()
    output.write_text(body,encoding="utf-8",newline="\n")
    namespace={"__name__":"y98_y96_binding_generation","__file__":__file__}
    exec(compile(body,str(output),"exec"),namespace)
    assert namespace["CID"]=="AdaRES-Y98" and namespace["OLD"]==OLD
    assert namespace["Y95"]==Y95 and namespace["Y96"]==Y96 and namespace["OWNERS"]==OWNERS
    assert namespace["BRANCH"]=="refs/heads/adares-y98-appearance-mask-skip-on-y96"
    assert namespace["RUNNER"]=="scripts/run_y98_bound_native_y96.py"
    namespace["rebind"](token)
    assert ref(source)==original
    print(json.dumps(dict(wrapper=ref(output),scientific_CPU_replayed=False,native_GPU_executed=False,HEAD_preserved=Y96)))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--token",required=True)
    rebind(parser.parse_args().token)
