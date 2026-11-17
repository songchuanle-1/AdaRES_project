"""Bind original Y100 lifecycle to the unchanged actual-native Y101 additions."""
import argparse, ast, copy, importlib, json, subprocess, sys
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_deployment import Y100,Y101,git
from y100_y101_terminal_contract import native_evidence
import closure

class Specialize(ast.NodeTransformer):
    def __init__(self):self.updates=0
    def visit_ImportFrom(self,n):
        if n.module=='y100_terminal_contract' and not any(a.name=='validate_delta' for a in n.names):n.module='y100_y101_terminal_contract'
        return self.generic_visit(n)
    def visit_alias(self,n):
        if n.name=='finalize_y100_fixed30':n.name='finalize_y100_fixed30_y101'
        return n
    def visit_Constant(self,n):
        if isinstance(n.value,str):
            for a,b in [('test_y100_terminal_handoff.py','test_y100_terminal_handoff_y101.py'),('scripts/finalize_y100_fixed30.py','scripts/finalize_y100_fixed30_y101.py'),('scripts/y100_terminal_contract.py','scripts/y100_y101_terminal_contract.py'),('pipeline/checks/y100_server2_fixed30_','pipeline/checks/y100_server2_fixed30_y101_')]:n.value=n.value.replace(a,b)
        return n
    def visit_Call(self,n):
        n=self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and n.func.attr=='assertEqual' and len(n.args)==2 and isinstance(n.args[1],ast.List) and len(n.args[1].elts)==1 and isinstance(n.args[1].elts[0],ast.Name) and n.args[1].elts[0].id=='Y100':
            assert isinstance(n.args[0],ast.Subscript) and n.args[0].slice.value=='allowed_markers'
            n.args[1].elts.append(ast.Constant(Y101));self.updates+=1
        return n

def prepare(token):
    lease(token,300);writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert git('rev-parse','HEAD').decode().strip()==Y100
        before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml','pipeline/admissions/AdaRES-Y100.json')};index=git('ls-files','--stage','-z')
        work=tick.load(ROOT/'pipeline/research_work.json');prior=copy.deepcopy(work)
        original=next(r for r in work['candidates'] if r['id']=='AdaRES-Y100')
        additive=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101')
        assert original['stage']=='training' and not original.get('terminal_evidence')
        assert additive['stage']=='validation' and not additive.get('admission_receipt')
        actual=native_evidence(work);oldref=original['terminal_handoff_plan'];old=tick.load(closure.artifact(ROOT,oldref));generated=[]
        originals=[ref('scripts/'+p) for p in ('finalize_y100_fixed30.py','test_y100_terminal_handoff.py','y100_terminal_contract.py')]
        for source,target,count in [('finalize_y100_fixed30.py','finalize_y100_fixed30_y101.py',0),('test_y100_terminal_handoff.py','test_y100_terminal_handoff_y101.py',1)]:
            out=ROOT/'scripts'/target;assert not out.exists();transform=Specialize();tree=transform.visit(ast.parse((ROOT/'scripts'/source).read_text(encoding='utf-8')));assert transform.updates==count
            text=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(text,feature_version=(3,8));out.write_text(text,encoding='utf-8',newline='\n');generated.append(ref(out))
        archive=importlib.import_module('finalize_y100_fixed30_y101');server,_,task,_,_=archive.current()
        payload=copy.deepcopy(old['payload']);payload.update(archive.build_contract(task,server))
        assert all(payload[k]==old['payload'][k] for k in set(old['payload'])-{'allowed_markers','deployments'}) and set(payload)==set(old['payload'])
        assert payload['allowed_markers']==[Y100,Y101]
        base=ROOT/('pipeline/checks/y100_y101_terminal_binding_'+token);output=base.with_suffix('.json');assert not output.exists();logs=[]
        for pattern,count,label in [('test_y100_terminal_handoff_y101.py',29,'lifecycle'),('test_y100_y101_terminal_binding.py',8,'additive')]:
            log=base.with_name(base.name+'_'+label+'.log');assert not log.exists()
            r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','scripts','-p',pattern,'-v'],cwd=ROOT,capture_output=True,timeout=90);log.write_bytes(r.stdout+r.stderr)
            assert r.returncode==0 and ('Ran '+str(count)+' tests').encode() in log.read_bytes() and b'skipped' not in log.read_bytes(),str(log)
            logs.append(ref(log))
        dependencies=old['dependencies']+generated+[ref(__file__),ref('scripts/y100_y101_terminal_contract.py'),ref('scripts/test_y100_y101_terminal_binding.py'),ref('scripts/y101_deployment.py'),ref('scripts/advance_y101.py'),ref('scripts/y101_contract.py'),ref('scripts/run_y101_native.py'),actual,additive['additive_version_binding'],additive['native_smoke_plan']]
        for r in dependencies:closure.artifact(ROOT,r)
        plan=copy.deepcopy(old);plan.update(token=token,observed_at=datetime.now().astimezone().isoformat(),payload=payload,dependencies=dependencies,supersedes=oldref,CPU_guard_tests_passed=29,additive_guard_tests_passed=8,tested_additive_native_evidence=actual,next_command='python -B scripts/finalize_y100_fixed30_y101.py --token <fresh-owned-token>',original_helpers_unchanged=True,original_run_and_protocol_unchanged=True,full_terminal_verification_executed=False,checkpoint_loaded=False,archived=False,training_stopped=False,queue_changed=False,remote_mutations=False,evidence=old['evidence']+dependencies+logs)
        tick.save(output,plan)
        assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index and [ref(r['path']) for r in originals]==originals
        assert not original.get('pre_y101_terminal_handoff_plan');original['pre_y101_terminal_handoff_plan']=oldref;original['terminal_handoff_plan']=ref(output);original['evidence'].extend([ref(output)]+generated+logs+[ref(__file__)])
        assert [r for r in work['candidates'] if r['id']!=original['id']]==[r for r in prior['candidates'] if r['id']!=original['id']]
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(binding=ref(output),lifecycle_guards=29,additive_guards=8,checkpoint_loaded=False)))
    finally:writer.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);prepare(p.parse_args().token)
