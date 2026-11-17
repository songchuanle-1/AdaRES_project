"""Resume exact binding after a test alias-import failure; preserve all drafts."""
import argparse
import ast
import copy
from datetime import datetime
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y98_deployment import Y96, Y98, git
import y96_y98_terminal_contract_v2 as contract
import closure

class TestAlias(ast.NodeTransformer):
    def visit_alias(self,node):
        if node.name=="y96_y98_terminal_contract":
            node.name="y96_y98_terminal_contract_v2"
        return node

class ResourceText(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=node.value.replace("independently admitted Y96 owns the card", "independently admitted Y98 owns the card")
            node.value=node.value.replace("Fresh exclusive authorized idle server-1 GPU0 without compute or reservation.",
                                          "Fresh exclusive authorized idle server-2 GPU0 without compute or reservation.")
            node.value=node.value.replace("original Y95 and both pauses", "original Y96 and both pauses")
        return node

def repair(token):
    lease(token,240)
    writer=monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse","HEAD").decode().strip()==Y96
        before={path:ref(path) for path in ("pipeline/queue.json","pipeline/running.json","pipeline/goal.json","SERVERS.yaml")}
        index=git("ls-files","--stage","-z")
        work=tick.load(ROOT/"pipeline/research_work.json")
        original=next(row for row in work["candidates"] if row["id"]=="AdaRES-Y96")
        additive=next(row for row in work["candidates"] if row["id"]=="AdaRES-Y98")
        assert original["stage"]=="training" and additive["stage"]=="validation"
        assert not original.get("pre_y98_terminal_handoff_plan")
        actual=contract.native_evidence(work)
        oldref=original["terminal_handoff_plan"]
        old=tick.load(closure.artifact(ROOT,oldref))
        base=ROOT/("pipeline/checks/y96_y98_terminal_binding_"+token)
        lifecycle=base.with_name(base.name+"_lifecycle.log")
        failed=base.with_name(base.name+"_additive.log")
        assert "Ran 30 tests" in lifecycle.read_text(encoding="utf-8") and lifecycle.read_text(encoding="utf-8").rstrip().endswith("OK")
        assert "FAILED (failures=2)" in failed.read_text(encoding="utf-8")
        originals=[]
        for source_name,target_name,transform in (
            ("test_y96_y98_terminal_binding_v2.py","test_y96_y98_terminal_binding_v3.py",TestAlias()),
            ("advance_y98.py","advance_y98_v2.py",ResourceText()),
        ):
            source=ROOT/"scripts"/source_name
            target=ROOT/"scripts"/target_name
            entry=ref(source)
            assert not target.exists()
            tree=transform.visit(ast.parse(source.read_text(encoding="utf-8")))
            text=ast.unparse(ast.fix_missing_locations(tree))+"\n"
            ast.parse(text,feature_version=(3,8))
            target.write_text(text,encoding="utf-8",newline="\n")
            assert ref(source)==entry
            originals.append(entry)
        log=base.with_name(base.name+"_additive_v3.log")
        assert not log.exists()
        result=subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","scripts",
            "-p","test_y96_y98_terminal_binding_v3.py","-v"],cwd=ROOT,capture_output=True,timeout=90)
        log.write_bytes(result.stdout+result.stderr)
        assert result.returncode==0 and b"Ran 8 tests" in log.read_bytes() and b"skipped" not in log.read_bytes(),str(log)
        import finalize_y96_fixed30_y98 as archive
        server,_,task,_,_=archive.current()
        payload=copy.deepcopy(old["payload"])
        payload.update(archive.build_contract(task,server))
        assert set(payload)==set(old["payload"])
        assert all(payload[key]==old["payload"][key] for key in set(payload)-{"allowed_markers","deployments"})
        assert payload["allowed_markers"]==[Y96,Y98]
        dependencies=old["dependencies"]+[ref("scripts/"+name) for name in (
            "finalize_y96_fixed30_y98.py","test_y96_terminal_handoff_y98.py",
            "y96_y98_terminal_contract_v2.py","test_y96_y98_terminal_binding_v3.py",
            "advance_y98_v2.py","y98_deployment.py","y98_contract.py","run_y98_bound_native_y96.py")]
        dependencies += [actual,additive["additive_version_binding"],additive["native_smoke_plan"],ref(__file__),ref(lifecycle),ref(log),ref(failed)]
        for entry in dependencies:
            closure.artifact(ROOT,entry)
        output=base.with_name(base.name+"_v3.json")
        assert not output.exists()
        plan=copy.deepcopy(old)
        plan.update(token=token,observed_at=datetime.now().astimezone().isoformat(),payload=payload,
            dependencies=dependencies,supersedes=oldref,CPU_guard_tests_passed=30,additive_guard_tests_passed=8,
            tested_additive_native_evidence=actual,next_command="python -B scripts/finalize_y96_fixed30_y98.py --token <fresh-owned-token>",
            original_helpers_unchanged=True,original_run_and_protocol_unchanged=True,
            full_terminal_verification_executed=False,checkpoint_loaded=False,archived=False,
            training_stopped=False,queue_changed=False,remote_mutations=False,
            operational_failure_scope="test alias import; actual native/data/scientific controls unchanged",
            preserved_failed_test=ref(failed),evidence=old["evidence"]+dependencies+originals)
        tick.save(output,plan)
        original["pre_y98_terminal_handoff_plan"]=oldref
        original["terminal_handoff_plan"]=ref(output)
        original["evidence"].extend([ref(output)]+dependencies)
        additive["evidence"].extend([ref(output),ref(failed),ref(log),ref(__file__)]+originals)
        tick.save(ROOT/"pipeline/research_work.json",work)
        assert all(ref(path)==entry for path,entry in before.items())
        assert git("ls-files","--stage","-z")==index
        print(json.dumps(dict(binding=ref(output),lifecycle_guards=30,additive_guards=8,
            failed_alias_tests_preserved=ref(failed),scientific_or_native_replayed=False)))
    finally:
        writer.close()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--token",required=True)
    repair(parser.parse_args().token)
