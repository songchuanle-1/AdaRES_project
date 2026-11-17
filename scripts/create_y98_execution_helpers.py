"""Specialize established execution tools against actual Y98 server2 evidence."""
import argparse
import ast
import json
import re
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y96_deployment import Y96, git
from check_ef_execution import validate_frozen
from y98_contract import validate_result
import closure

Y98 = "15c3481ff32456160b5a94588565f68f0d2d2c5b"
PAUSE = "pipeline/authorizations/server2_gpu1_pause_20261104.json"
PAUSE_SHA = "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45"

class Specialize(ast.NodeTransformer):
    def __init__(self, pair=False, advance=False):
        self.pair, self.advance = pair, advance
        self.names = {"Y93":"Y95", "Y94":"Y96", "Y95":"Y98"} if pair else {
            "Y94":"Y95", "Y95":"Y96", "Y96":"Y98"}
        self.replacements = {
            "advance_y96":"advance_y98", "advance_y95":"advance_y98" if pair else "advance_y95",
            "y96_deployment":"y98_deployment", "y95_deployment":"y98_deployment" if pair else "y96_deployment",
            "run_y96_bound_native_y95":"run_y98_bound_native_y96",
            "prepare_y96_seed_query_skip":"prepare_y98_appearance_mask_skip",
            "y96_contract":"y98_contract", "test_y96_admission":"test_y98_admission",
            "prepare_y94_y95_terminal_binding":"prepare_y96_y98_terminal_binding",
            "y94_y95_terminal_contract":"y96_y98_terminal_contract",
            "test_y94_y95_terminal_binding":"test_y96_y98_terminal_binding",
            "finalize_y94_fixed30_y95":"finalize_y96_fixed30_y98",
            "test_y94_terminal_handoff_y95":"test_y96_terminal_handoff_y98",
            "finalize_y94_fixed30":"finalize_y96_fixed30",
            "test_y94_terminal_handoff":"test_y96_terminal_handoff",
            "prepare_y94_terminal_handoff":"prepare_y96_terminal_handoff",
            "y94_terminal_contract":"y96_terminal_contract",
            "y94_server1_fixed30_y95_":"y96_server1_fixed30_y98_",
            "y94_server1_fixed30_":"y96_server1_fixed30_",
            "y94_y95_terminal_binding_":"y96_y98_terminal_binding_",
            "y94_deployment":"y98_deployment" if pair else "y94_deployment",
            "acb8c1769b7f3d66a92d287da8859a06a138b8b0":Y98,
            "pipeline/checks/adares-y96_":"pipeline/checks/adares-y98_",
            "server-1 GPU0/350W":"server-2 GPU0/280W",
            "--server server-1 --token ":"--server server-2 --token ",
            "server1 protocol":"server2 protocol", "server1 evidence":"server2 evidence",
            "sampled-seed bypass of the original five-layer query generator":"full-existing-appearance bypass of the original static mask MLP",
        }
        if pair:
            self.replacements.update({"AdaRES-Y94":"AdaRES-Y96", "AdaRES-Y95":"AdaRES-Y98",
                "AdaRES-Y93":"AdaRES-Y95", "Y94":"Y96", "Y95":"Y98",
                "20261109_111838":"20261110_165702",
                "b9deb51efa3c274283f8a23ae678d42a4f0c11a29e758d19ae696e80df13cb84":"5dcd15c8da9efca80c530a5c61767fc68f9976a2e0c70aedcef3f5807ea42b1c",
                "7e40371552d170a1d764f9dab160a8dcb12b0b6fe512e5c978313cb7b642241b":"50213e2f1fda98e5e9477eaa5c030952d1d913d3d1aeff86d1fc6aa984dfc154"})
        else:
            self.replacements.update({"AdaRES-Y94":"AdaRES-Y95", "AdaRES-Y95":"AdaRES-Y96",
                                      "AdaRES-Y96":"AdaRES-Y98"})
        self.pattern = re.compile("|".join(re.escape(key) for key in sorted(self.replacements,key=len,reverse=True)))

    def text(self,value):
        return self.pattern.sub(lambda match:self.replacements[match.group()],value)

    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=self.text(node.value)
            if self.advance and node.value=="native_smoke_plan_server1":
                node.value="native_smoke_plan_server2_gpu0"
        elif self.pair and type(node.value) is int and node.value==218513:
            node.value=3893460
        return node

    def visit_Name(self,node):
        node.id=self.names.get(node.id,node.id)
        return node

    def visit_Attribute(self,node):
        node=self.generic_visit(node)
        node.attr=self.names.get(node.attr,node.attr)
        return node

    def visit_alias(self,node):
        node.name=self.names.get(node.name,self.text(node.name))
        return node

    def visit_ImportFrom(self,node):
        node.module=self.text(node.module)
        for alias in node.names:
            alias.name=self.names.get(alias.name,self.text(alias.name))
        return node

    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if self.advance:
            targets={value.id for target in node.targets for value in ast.walk(target) if isinstance(value,ast.Name)}
            if targets=={"CID","NAME","GPU","POWER"}:
                node.value=ast.parse(repr(("AdaRES-Y98","server-2",0,280.0)),mode="eval").body
            for key,value in {"GRANT":PAUSE,"GRANT_SHA":PAUSE_SHA,
                "PARENT_FREEZE_SHA":"2248202035aaa859a4cab6d5b1e4f2ffd947f4775725390562f36a944ceed6c9"}.items():
                if targets=={key}:
                    node.value=ast.Constant(value=value)
        targets={value.id for target in node.targets for value in ast.walk(target) if isinstance(value,ast.Name)}
        if targets=={"name","commit","power"} and isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.Dict):
            node.value=ast.parse('{"AdaRES-Y96":("server-1",Y96,350.0),"AdaRES-Y98":("server-2",Y98,280.0)}[cid]',mode="eval").body
        return node

    def visit_FunctionDef(self,node):
        node=self.generic_visit(node)
        if node.name=="require_running_parent_handoff":
            for value in ast.walk(node):
                if isinstance(value,ast.Constant) and value.value==29:
                    value.value=30
        return node

def create(token):
    lease(token,300)
    writer=monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse","HEAD").decode().strip()==Y96
        before={path:ref(path) for path in ("pipeline/queue.json","pipeline/running.json","pipeline/goal.json","SERVERS.yaml")}
        index=git("ls-files","--stage","-z")
        work=tick.load(ROOT/"pipeline/research_work.json")
        candidate=next(row for row in work["candidates"] if row["id"]=="AdaRES-Y98")
        assert candidate["stage"]=="validation" and candidate["code_commit"]==Y98
        assert candidate["required_server"]=="server-2" and not candidate.get("admission_receipt")
        plan=tick.load(closure.artifact(ROOT,candidate["native_smoke_plan"]))
        program=closure.artifact(ROOT,plan["program"])
        frozen=ast.literal_eval(ast.parse(program.read_text(encoding="utf-8")).body[0].value)
        assert validate_frozen(candidate,plan,frozen)==7
        validate_result(tick.load(closure.artifact(ROOT,candidate["native_smoke_result"])),candidate,frozen)
        outputs=[]
        for source_name,target_name,pair,advance in (
            ("advance_y96.py","advance_y98.py",False,True),
            ("y96_deployment.py","y98_deployment.py",False,False),
            ("y94_y95_terminal_contract.py","y96_y98_terminal_contract.py",True,False),
            ("prepare_y94_y95_terminal_binding.py","prepare_y96_y98_terminal_binding.py",True,False),
            ("test_y94_y95_terminal_binding.py","test_y96_y98_terminal_binding.py",True,False),
        ):
            source=ROOT/"scripts"/source_name
            target=ROOT/"scripts"/target_name
            assert not target.exists()
            original=ref(source)
            tree=Specialize(pair,advance).visit(ast.parse(source.read_text(encoding="utf-8")))
            text=ast.unparse(ast.fix_missing_locations(tree))+"\n"
            ast.parse(text,feature_version=(3,8))
            target.write_text(text,encoding="utf-8",newline="\n")
            assert ref(source)==original
            outputs.append(ref(target))
        output=ROOT/("pipeline/checks/y98_execution_helpers_"+token+".json")
        assert not output.exists()
        tick.save(output,dict(complete=True,token=token,id="AdaRES-Y98",actual_native=candidate["native_smoke_result"],
            helpers=outputs,helper_tests_executed=False,source_selected=False,admitted=False,launched=False,
            evidence=[ref(__file__)]+outputs))
        candidate["evidence"].append(ref(output))
        tick.save(ROOT/"pipeline/research_work.json",work)
        assert all(ref(path)==entry for path,entry in before.items())
        assert git("ls-files","--stage","-z")==index
        print(json.dumps(dict(helpers=ref(output),actual_native_required=True,old_helpers_preserved=True)))
    finally:
        writer.close()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--token",required=True)
    create(parser.parse_args().token)
