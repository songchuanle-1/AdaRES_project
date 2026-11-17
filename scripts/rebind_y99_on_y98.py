"""Bind unchanged original Y99 files to actual-native Y98 and live owners."""
import argparse
import ast
import json
import re
from prepare_y60_stage_heads import ROOT,lease,ref
from y98_deployment import Y96,Y98,git

OLD="e5151a722778af3eecb28d0d4f6ae75e18a60e0c"
OWNERS={"server-1":"AdaRES-Y96","server-2":"AdaRES-Y98"}
REPLACEMENTS={
    "afc93861cf66b0da4835391597ee2d5ce90f6584":OLD,
    "refs/heads/adares-y95-query-set-mean-on-y94":"refs/heads/adares-y99-spatial-mask-skip-on-y98",
    "refs/heads/adares-y95-query-set-mean":"refs/heads/adares-y99-spatial-mask-skip",
    "run_y95_bound_native_y94":"run_y99_bound_native_y98",
    "run_y95_native":"run_y99_native", "retain_y95_cpu_native":"retain_y99_cpu_native",
    "test_y95_runtime_plan":"test_y99_runtime_plan", "rebind_y95_on_y94":"rebind_y99_on_y98",
    "prepare_y95_query_set_mean":"prepare_y99_spatial_mask_skip", "y95_contract":"y99_contract",
    "advance_y94":"advance_y98_v2", "y94_deployment":"y98_deployment",
    "y95_y94_additive_binding_":"y99_y98_additive_binding_", "original_pre_y94_binding":"original_pre_y98_binding",
    "AdaRES-Y95":"AdaRES-Y99", "AdaRES-Y94":"AdaRES-Y98", "AdaRES-Y93":"AdaRES-Y96", "AdaRES-Y92":"AdaRES-Y94",
    "HEADY94":"HEADY98", "Y94/server1":"Y96/server1", "Y93/server2":"Y98/server2",
    "Y95 bytes":"Y99 bytes", "Y95 additions":"Y99 additions", "native-tested Y94":"native-tested Y98",
    "Current Y94/server1 and original Y93/server2 occupy both authorized GPU0 resources; identical Y95 bytes now have a tested HEADY94/current-owner execution binding.":
        "Current original Y96/server1 and Y98/server2 occupy both authorized GPU0 resources; identical Y99 bytes have a tested HEADY98/current-owner execution binding.",
}
PATTERN=re.compile("|".join(re.escape(key) for key in sorted(REPLACEMENTS,key=len,reverse=True)))
RUNNER='''
class Runner(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value=="3f1c351f8a04e12745526e9abb3dbac46af806f1":
            node.value=Y98
        elif isinstance(node.value,str):
            node.value=node.value.replace("run_y99_native.py","run_y99_bound_native_y98.py")
        return node
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if any(isinstance(target,ast.Name) and target.id=="OWNERS" for target in node.targets):
            node.value=ast.parse(repr(OWNERS),mode="eval").body
        return node
    def visit_FunctionDef(self,node):
        if node.name=="validate_binding":
            return ast.parse(BINDING).body[0]
        return self.generic_visit(node)
'''
CURRENT_SOURCE_TEST='''
def test_actual_parent_native_and_preserved_current_run_sources(self):
    self.assertEqual(parent_evidence(),self.proposal["parent_native"])
    self.assertEqual(list(contract(Y96,Y98,"AdaRES-Y96")),[Y96,Y98])
    queue=monitor.load_queue(str(ROOT))
    for name,cid,pid,version in (("server-1","AdaRES-Y96",3893460,Y96),("server-2","AdaRES-Y98",25962,Y98)):
        task=next(row for row in queue["gpu_queues"][name] if row["id"]==cid)
        self.assertEqual((task["status"],task["pid"],task["code_commit"]),("running",pid,version))
        self.assertTrue(task["launch_verified"])
        closure.admission(ROOT,task)
'''

class Specialize(ast.NodeTransformer):
    def text(self,value):
        return PATTERN.sub(lambda match:REPLACEMENTS[match.group()],value)
    def visit_Name(self,node):
        node.id={"Y93":"Y96","Y94":"Y98"}.get(node.id,node.id)
        return node
    def visit_ImportFrom(self,node):
        node.module=self.text(node.module)
        for alias in node.names:
            alias.name={"Y93":"Y96","Y94":"Y98"}.get(alias.name,alias.name)
        return node
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=self.text(node.value)
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
    assert git("rev-parse","HEAD").decode().strip()==Y98
    source=ROOT/"scripts/rebind_y95_on_y94.py"
    original=ref(source)
    tree=Specialize().visit(ast.parse(source.read_text(encoding="utf-8")))
    body=ast.unparse(ast.fix_missing_locations(tree))+"\n"
    ast.parse(body,feature_version=(3,8))
    output=ROOT/("pipeline/checks/y99_y98_binding_wrapper_"+token+".py")
    assert not output.exists()
    output.write_text(body,encoding="utf-8",newline="\n")
    namespace={"__name__":"y99_y98_binding_generation","__file__":__file__}
    exec(compile(body,str(output),"exec"),namespace)
    assert namespace["CID"]=="AdaRES-Y99" and namespace["OLD"]==OLD
    assert namespace["Y96"]==Y96 and namespace["Y98"]==Y98 and namespace["OWNERS"]==OWNERS
    assert namespace["BRANCH"]=="refs/heads/adares-y99-spatial-mask-skip-on-y98"
    assert namespace["RUNNER"]=="scripts/run_y99_bound_native_y98.py"
    namespace["rebind"](token)
    assert ref(source)==original
    print(json.dumps(dict(wrapper=ref(output),original_rebinding_helper=original,
        scientific_CPU_replayed=False,native_GPU_executed=False,production_HEAD_preserved=Y98)))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--token",required=True)
    rebind(parser.parse_args().token)
