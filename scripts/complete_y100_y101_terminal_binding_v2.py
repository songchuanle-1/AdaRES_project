"""Repair only stale qualified commit attributes in the operational test draft."""
import argparse,ast,copy,importlib,json,subprocess,sys
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_deployment import Y100,Y101,git
from y100_y101_terminal_contract import native_evidence
import closure
class Attributes(ast.NodeTransformer):
    def visit_Attribute(self,n):
        n=self.generic_visit(n)
        if isinstance(n.value,ast.Name) and n.value.id=='deploy':n.attr={'Y98':'Y99','Y99':'Y100','Y100':'Y101'}.get(n.attr,n.attr)
        return n
def finish(token):
    lease(token,300);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        assert git('rev-parse','HEAD').decode().strip()==Y100
        before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml','pipeline/admissions/AdaRES-Y100.json')};index=git('ls-files','--stage','-z')
        work=tick.load(ROOT/'pipeline/research_work.json');prior=copy.deepcopy(work)
        original=next(r for r in work['candidates'] if r['id']=='AdaRES-Y100');additive=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101')
        assert original['stage']=='training' and additive['stage']=='validation' and not additive.get('admission_receipt')
        oldref=original['terminal_handoff_plan'];old=tick.load(closure.artifact(ROOT,oldref));actual=native_evidence(work)
        base='pipeline/checks/y100_y101_terminal_binding_'+token
        lifecycle=ROOT/(base+'_lifecycle.log');failure=ROOT/(base+'_additive.log')
        assert b'Ran 29 tests' in lifecycle.read_bytes() and b'\nOK' in lifecycle.read_bytes()
        assert b'FAILED (failures=4, errors=1)' in failure.read_bytes()
        source=ROOT/'scripts/test_y100_y101_terminal_binding.py';target=ROOT/'scripts/test_y100_y101_terminal_binding_v2.py';assert not target.exists()
        source_before=ref(source);tree=Attributes().visit(ast.parse(source.read_text(encoding='utf-8')));body=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(body,feature_version=(3,8));target.write_text(body,encoding='utf-8',newline='\n');assert ref(source)==source_before
        log=ROOT/(base+'_additive_v2.log');assert not log.exists()
        result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','scripts','-p',target.name,'-v'],cwd=ROOT,capture_output=True,timeout=90);log.write_bytes(result.stdout+result.stderr)
        assert result.returncode==0 and b'Ran 8 tests' in log.read_bytes() and b'skipped' not in log.read_bytes(),str(log)
        archive=importlib.import_module('finalize_y100_fixed30_y101');server,_,task,_,_=archive.current()
        payload=copy.deepcopy(old['payload']);payload.update(archive.build_contract(task,server));assert payload['allowed_markers']==[Y100,Y101]
        assert all(payload[k]==old['payload'][k] for k in set(old['payload'])-{'allowed_markers','deployments'}) and set(payload)==set(old['payload'])
        entries=[ref('scripts/'+p) for p in ['finalize_y100_fixed30_y101.py','test_y100_terminal_handoff_y101.py','create_y101_launch_helpers.py','prepare_y100_y101_terminal_binding.py','test_y100_y101_terminal_binding.py','y100_y101_terminal_contract.py','test_y100_y101_terminal_binding_v2.py','y101_deployment.py','advance_y101.py','y101_contract.py','run_y101_native.py']]+[ref(__file__),ref(lifecycle),ref(failure),ref(log),actual,additive['additive_version_binding'],additive['native_smoke_plan']]
        dependencies=old['dependencies']+entries
        for entry in dependencies:closure.artifact(ROOT,entry)
        out=ROOT/(base+'_v2.json');assert not out.exists()
        plan=copy.deepcopy(old);plan.update(token=token,observed_at=datetime.now().astimezone().isoformat(),payload=payload,dependencies=dependencies,supersedes=oldref,CPU_guard_tests_passed=29,additive_guard_tests_passed=8,tested_additive_native_evidence=actual,next_command='python -B scripts/finalize_y100_fixed30_y101.py --token <fresh-owned-token>',original_helpers_unchanged=True,original_run_and_protocol_unchanged=True,full_terminal_verification_executed=False,checkpoint_loaded=False,archived=False,training_stopped=False,queue_changed=False,remote_mutations=False,operational_test_repair='Qualified deploy.Y99/Y100 attributes in the draft test still named the previous commits; corrected to exact original Y100/actual-native Y101. All scientific files and native evidence unchanged; original failed draft/log retained; no native replay.',evidence=old['evidence']+entries)
        tick.save(out,plan);assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
        assert not original.get('pre_y101_terminal_handoff_plan');original['pre_y101_terminal_handoff_plan']=oldref;original['terminal_handoff_plan']=ref(out);original['evidence'].extend([ref(out)]+entries)
        assert [r for r in work['candidates'] if r['id']!=original['id']]==[r for r in prior['candidates'] if r['id']!=original['id']]
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(binding=ref(out),lifecycle_guards=29,additive_guards=8,scientific_replays=0,negative_operational_evidence_preserved=ref(failure))))
    finally:lock.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);finish(p.parse_args().token)
