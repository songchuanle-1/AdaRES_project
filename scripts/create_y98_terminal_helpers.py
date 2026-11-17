"""Prepare exact Y98 lifecycle tools from the preserved server2 generator."""
import argparse
import ast
import json
from prepare_y60_stage_heads import ROOT, lease, ref, tick
from y98_deployment import Y98, git
from advance_y98_v2 import prepared
from y98_contract import validate_result
import closure

ADMISSION="dc1a8a6dd3afde694d255cceca30b1b16009e8249390aa706632e2a567136a62"
FREEZE="bcd19eedca805c8491568b3f81cf9cefa8dc4b0def8f6a09dd87c58f5e29d51c"
LOG="/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y98_appearance_mask_skip_res_20261110_210741.log"

class CurrentRun(ast.NodeTransformer):
    @staticmethod
    def text(value):
        for old,new in (
            ("adares_y91_hidden_confidence_gate","adares_y98_appearance_mask_skip"),
            ("20261107_232228","20261110_210741"),
            ("4d59825df7e4d0cfe428fc47c3ff6102749f2b1e5f12d84b71aaf5238aa96eaf",ADMISSION),
            ("e5e6eef716ac892e827fe9131907ec44add8fdad1f619da8a568dacfc6935ca1",FREEZE),
            ("83827","25962"),("Y91","Y98"),("y91","y98"),
        ):
            value=value.replace(old,new)
        return value
    def visit_Name(self,node):
        node.id=self.text(node.id)
        return node
    def visit_ImportFrom(self,node):
        node.module=self.text(node.module)
        for alias in node.names:
            alias.name=self.text(alias.name)
        return node
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=self.text(node.value)
        elif type(node.value) is int and node.value==83827:
            node.value=25962
        return node

def create(token):
    lease(token,300)
    assert git("rev-parse","HEAD").decode().strip()==Y98
    assert ref("pipeline/admissions/AdaRES-Y98.json")["sha256"]==ADMISSION
    work=tick.load(ROOT/"pipeline/research_work.json")
    candidate,_,_,frozen=prepared(work)
    assert candidate["stage"]=="training" and not candidate.get("terminal_handoff_plan")
    assert candidate["post_training_freeze"]["sha256"]==FREEZE
    validate_result(tick.load(closure.artifact(ROOT,candidate["native_smoke_result"])),candidate,frozen)
    source=ROOT/"scripts/create_y91_terminal_helpers.py"
    original=ref(source)
    tree=CurrentRun().visit(ast.parse(source.read_text(encoding="utf-8")))
    body=ast.unparse(ast.fix_missing_locations(tree))+"\n"
    ast.parse(body,feature_version=(3,8))
    output=ROOT/("pipeline/checks/y98_terminal_wrapper_"+token+".py")
    assert not output.exists()
    output.write_text(body,encoding="utf-8",newline="\n")
    namespace={"__name__":"y98_terminal_generation","__file__":__file__}
    exec(compile(body,str(output),"exec"),namespace)
    spec=namespace["SPEC"]
    assert namespace["CID"]=="AdaRES-Y98" and spec["code_commit"]==Y98
    assert spec["pid"]==25962 and spec["log"]==LOG
    assert spec["admission_sha256"]==ADMISSION and spec["freeze_sha256"]==FREEZE
    namespace["create"](token)
    assert ref(source)==original
    print(json.dumps(dict(wrapper=ref(output),immutable_generator=original,checkpoint_loaded=False,archived=False)))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--token",required=True)
    create(parser.parse_args().token)
