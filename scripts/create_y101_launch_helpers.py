"""Prepare scoped actual-native Y101 launch tools without changing scientific files."""
import argparse, ast, json, re
from prepare_y60_stage_heads import ROOT, lease, ref, tick
from create_y99_launch_helpers import transformed
from y100_deployment import Y99, Y100, git
Y101='6fb26984ae652efbba177ab20c39575c44a69517'

def create(token):
    lease(token,600)
    assert git('rev-parse','HEAD').decode().strip()==Y100
    assert git('rev-parse',Y101+'^').decode().strip()==Y100
    before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/research_work.json','pipeline/goal.json','SERVERS.yaml')}
    index=git('ls-files','--stage','-z')
    work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101')
    assert c['stage']=='validation' and c.get('native_smoke_result') and not c.get('admission_receipt')
    changes={'run_y99_bound_native_y98':'run_y101_native','advance_y99_v2':'advance_y101',
        'y99_deployment_v2':'y101_deployment','y99_contract':'y101_contract',
        'test_y99_admission_v2.py':'test_y101_admission.py',
        '50213e2f1fda98e5e9477eaa5c030952d1d913d3d1aeff86d1fc6aa984dfc154':'fd07c0536891d905a7cd89e2b4211b32ed31a532b7b70ac9d71a8d7a66e97dbc',
        'Y96':'Y99','y96':'y99','Y98':'Y100','y98':'y100','Y99':'Y101','y99':'y101'}
    advance=transformed('scripts/advance_y99_v2.py',changes)
    advance=advance.replace('full32 sparse-input bypass beside the complete original static-mask MLP',
        'shared full256 current-query bypass after each complete original query FFN')
    outputs={'scripts/advance_y101.py':advance}
    header='''"""Select actual-native Y101 additions; preserve original Y100 training."""
import argparse, ast, hashlib, inspect, json, subprocess
from datetime import datetime
from pathlib import Path
import yaml
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from observe_y69_server1_phase import parse_phase,phase_key,progressed,REMOTE
import closure
import y100_deployment as previous
Y99,Y100=previous.Y99,previous.Y100
Y101="6fb26984ae652efbba177ab20c39575c44a69517"
VERSIONS=previous.VERSIONS+(Y101,)
ADDITIONS=dict(previous.ADDITIONS,**{Y101:"AdaRES-Y101"})
git,adapt=previous.git,previous.adapt
def paths(cid):
    if cid!="AdaRES-Y101":return previous.paths(cid)
    from prepare_y101_query_ffn_skip import files
    return [p.relative_to(ROOT).as_posix() for p in files()]
def require_running_parent_handoff(work,candidate):
    original=next(r for r in work["candidates"] if r["id"]=="AdaRES-Y100")
    assert original["stage"]=="training"
    plan=tick.load(closure.artifact(ROOT,original["terminal_handoff_plan"]))
    assert plan["complete"] is True and plan["id"]==original["id"]
    assert plan["tested_additive_native_evidence"]==candidate["native_smoke_result"]
    assert plan["payload"]["allowed_markers"]==[Y100,Y101]
    assert plan["CPU_guard_tests_passed"]==29 and plan["additive_guard_tests_passed"]==8
    for entry in plan["dependencies"]:closure.artifact(ROOT,entry)
'''
    deploy_changes={'advance_y100':'advance_y101','Y99':'Y100','Y100':'Y101','y99':'y100','y100':'y101'}
    body=header+transformed('scripts/y100_deployment.py',deploy_changes,{'validate_delta','select'})
    body+=transformed('scripts/y100_deployment.py',{'UNUSED':'UNUSED'},{'contract'})
    observe=transformed('scripts/y100_deployment.py',{'Y99':'Y100','Y100':'Y101','y99':'y100','y100':'y101'},{'observe'})
    tree=ast.parse(observe)
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) and [getattr(x,'id',None) for x in t.elts]==['name','commit','power'] for t in n.targets)]
    assert len(assignments)==1
    assignments[0].value=ast.parse("{'AdaRES-Y100':('server-2',Y100,280.0),'AdaRES-Y101':('server-1',Y101,350.0)}[cid]",mode='eval').body
    body+=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    body+='''
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("operation",choices=("select","observe"));p.add_argument("--id",choices=("AdaRES-Y100","AdaRES-Y101"),default="AdaRES-Y101");p.add_argument("--token",required=True);p.add_argument("--label",default="progress");a=p.parse_args()
    lease(a.token,240);writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:select(a.token) if a.operation=="select" else observe(a.token,a.id,a.label)
    finally:writer.close()
'''
    outputs['scripts/y101_deployment.py']=body
    pair={'y99_terminal_contract':'y100_terminal_contract','y99_y100_terminal_contract':'y100_y101_terminal_contract',
        'y100_deployment':'y101_deployment','advance_y100':'advance_y101','y100_admission_contract':'y101_contract',
        'Y99':'Y100','Y100':'Y101','y99':'y100','y100':'y101'}
    binding=transformed('scripts/y99_y100_terminal_contract.py',pair)
    binding=binding.replace('from y100_terminal_contract import SPEC, CID, CHECKPOINT_DIRECTORY, CONFIG, IDENTITY, validate_delta, paths, validate_server',
        'from y100_terminal_contract import SPEC, CID, CHECKPOINT_DIRECTORY, CONFIG, IDENTITY')
    binding+='''
def validate_server(server):
    closure.validate_adares_server(ROOT,server)
    for k,v in dict(host='192.168.1.163',expected_hostname='dl01-MS-7C60',python='/home/songcl/.conda/envs/adares_project/bin/python',remote_code_root='/bak_home/songcl/Project/AdaRES_project/AdaRES',work_dir='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints',allowed_gpus=[0],max_tasks_per_gpu=1).items():assert server[k]==v
    assert not server.get('runtime_path_mappings') and not server.get('runtime_library_path')
'''
    outputs['scripts/y100_y101_terminal_contract.py']=binding
    tests=transformed('scripts/test_y99_y100_terminal_binding.py',pair)
    tests=tests.replace('2980836','19305').replace('20261111_180638','20261111_085241').replace('server-1','server-2')
    tests=tests.replace('4d87ad8b3a6efe78be8dc9e8214bba9226160a22b649f64f8e1dd26d73988ede','7091f206c21429c02349c73e68d5e2b06ca3adc9f585465450870ac2a0e019f2').replace('fd07c0536891d905a7cd89e2b4211b32ed31a532b7b70ac9d71a8d7a66e97dbc','cb9b9a80fb6f9f9b116f9e9a35a515b3c4fb542eb6f4128cbee0ed5d389eff01')
    tests=tests.replace("('server', 'server-2')","('server', 'server-1')").replace("('runtime_path_mappings', [])","('runtime_path_mappings', [{'source':'wrong','target':'wrong'}])")
    outputs['scripts/test_y100_y101_terminal_binding.py']=tests
    originals=[ref(p) for p in ('scripts/advance_y99_v2.py','scripts/y100_deployment.py','scripts/y99_y100_terminal_contract.py','scripts/test_y99_y100_terminal_binding.py')]
    for path,text in outputs.items():
        assert not (ROOT/path).exists(),path
        ast.parse(text,feature_version=(3,8))
    for path,text in outputs.items():(ROOT/path).write_text(text,encoding='utf-8',newline='\n')
    assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
    out=ROOT/('pipeline/checks/y101_launch_helpers_'+token+'.json');assert not out.exists()
    tick.save(out,dict(complete=True,token=token,actual_native=c['native_smoke_result'],generated=[ref(p) for p in outputs],original_inputs=originals,main_index_and_scientific_files_unchanged=True,guard_tests_pending=True))
    print(json.dumps(dict(manifest=ref(out),generated=len(outputs),guards_pending=True)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);create(p.parse_args().token)
