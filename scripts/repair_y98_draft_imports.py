"""Preserve untested drafts and add correctly specialized Y98 contract imports."""
import argparse
import ast
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CHANGES = {
    "y96_y98_terminal_contract":"y96_y98_terminal_contract_v2",
    "test_y96_y98_terminal_binding":"test_y96_y98_terminal_binding_v2",
    "y95_contract":"y98_contract",
    "scripts/y95_server2_pre_admission.py":"scripts/advance_y98.py",
    "scripts/run_y95_bound_native_y94.py":"scripts/run_y98_bound_native_y96.py",
}

class Imports(ast.NodeTransformer):
    def text(self,value):
        for old,new in CHANGES.items():
            value=value.replace(old,new)
        return value
    def visit_ImportFrom(self,node):
        node.module=self.text(node.module)
        return node
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=self.text(node.value)
        return node

def repair(token):
    lease(token,180)
    writer=monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        entries=[]
        for name in ("y96_y98_terminal_contract", "test_y96_y98_terminal_binding", "prepare_y96_y98_terminal_binding"):
            source=ROOT/"scripts"/(name+".py")
            target=ROOT/"scripts"/(name+"_v2.py")
            original=ref(source)
            assert not target.exists()
            tree=Imports().visit(ast.parse(source.read_text(encoding="utf-8")))
            text=ast.unparse(ast.fix_missing_locations(tree))+"\n"
            ast.parse(text,feature_version=(3,8))
            target.write_text(text,encoding="utf-8",newline="\n")
            assert ref(source)==original
            entries.append(dict(original=original,corrected=ref(target)))
        output=ROOT/("pipeline/checks/y98_draft_import_correction_"+token+".json")
        assert not output.exists()
        tick.save(output,dict(complete=True,token=token,entries=entries,
            reason="Pre-test review found inherited y95_contract import instead of y98_contract. Original untested drafts remain unchanged; v2 specialization uses the actual Y98 contract and runner.",
            scientific_files_changed=False,native_replayed=False,tests_executed=False,evidence=[ref(__file__)]))
        work=tick.load(ROOT/"pipeline/research_work.json")
        next(row for row in work["candidates"] if row["id"]=="AdaRES-Y98")["evidence"].append(ref(output))
        tick.save(ROOT/"pipeline/research_work.json",work)
        print(json.dumps(dict(correction=ref(output),original_drafts_preserved=True)))
    finally:
        writer.close()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--token",required=True)
    repair(parser.parse_args().token)
