"""Use the original LF-bound reference manifest to distinguish formatting from drift."""
import argparse,ast,json,yaml
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y102_deployment import Y101,Y102,git,adapt
from reconcile_y62_y63_server1 import remote
import closure

REMOTE='''import datetime,hashlib,json,pathlib,subprocess,sys,yaml
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
root=pathlib.Path(P['root']);proc=pathlib.Path('/proc/2263581');assert proc.is_dir() and (proc/'cwd').resolve()==root
assert P['active_config'].encode() in (proc/'cmdline').read_bytes().split(b'\\0') and 'torch' not in sys.modules
sources=[]
for path,digest in P['expected_lf'].items():
    f=root/path;exists=f.is_file();raw=f.read_bytes() if exists else b''
    actual=hashlib.sha256(raw).hexdigest() if exists else None
    lf=hashlib.sha256(raw.replace(b'\\r\\n',b'\\n')).hexdigest() if exists else None
    sources.append(dict(path=path,exists=exists,actual_sha256=actual,actual_lf_sha256=lf,expected_frozen_lf_sha256=digest,matches_frozen_lf=exists and lf==digest,format_only_difference=exists and actual!=digest and lf==digest))
def differences(expected,actual,prefix=''):
    result=[]
    if isinstance(expected,dict) and isinstance(actual,dict):
        for key in sorted(set(expected)|set(actual)):result.extend(differences(expected.get(key),actual.get(key),prefix+'.'+str(key)))
    elif expected!=actual:result.append(dict(field=prefix,expected=expected,actual=actual))
    return result
configs=[]
for path,value in P['expected_configs'].items():
    f=root/path;exists=f.is_file();actual=yaml.safe_load(f.read_text()) if exists else None
    configs.append(dict(path=path,exists=exists,differences=differences(value,actual)))
print(json.dumps(dict(complete=True,observed_at=datetime.datetime.now().astimezone().isoformat(),sources=sources,all_reference_sources_match_original_frozen_lf=all(r['matches_frozen_lf'] for r in sources),configs=configs,CUDA_used=False,checkpoint_loaded=False,remote_mutations=False,torch_imported='torch' in sys.modules)))
'''
def run(token):
    lease(token,300);assert git('rev-parse','HEAD').decode().strip()==Y102
    before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml','pipeline/admissions/AdaRES-Y101.json')}
    work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};candidate=c['AdaRES-Y101'];inventoryref=candidate['frozen_review_asset_inventory'];inventory=tick.load(closure.artifact(ROOT,inventoryref))
    binding=ROOT/'pipeline/analyses/terminal_bindings_6debf88fed634a13b772f27bdc244bbd_program.py'
    tree=ast.parse(binding.read_text(encoding='utf-8'));original=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BINDINGS' for t in n.targets)))
    assert original['commit']=='8943c8c460501b200a8dea958964579ffea65f76'
    historical={r['path']:r['lf_sha256'] for r in original['sources'] if r['path'].startswith(('ipdn/model_paper5/adares_y36_local_adjacency/','ipdn/model_paper5/adares_w2_pointmass_competition/','ipdn/model_paper5/adares_b1_standard/'))}
    assert len(historical)==18
    server=monitor.load_servers(str(ROOT))['server-1'];task=next(r for r in monitor.load_queue(str(ROOT))['gpu_queues']['server-1'] if r['id']==candidate['id']);assert task['pid']==2263581 and task['status']=='running'
    names=[r['config'] for r in inventory['references']]+[task['config']]
    configs={p:adapt(yaml.safe_load(git('show',Y101+':AdaRES/'+p)),server) for p in names}
    payload=dict(root=server['remote_code_root'],active_config=task['config'],expected_lf=historical,expected_configs=configs)
    rawpath=ROOT/('pipeline/checks/y101_review_asset_clarification_'+token+'_remote.json');assert not rawpath.exists();raw=remote(server,'P='+repr(payload)+'\n'+REMOTE,rawpath);tick.save(rawpath,raw)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert all(ref(p)==v for p,v in before.items())
        out=ROOT/('pipeline/checks/y101_review_asset_clarification_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,id=candidate['id'],observed_at=datetime.now().astimezone().isoformat(),original_frozen_reference_binding=ref(binding),inventory=inventoryref,source_lf_checks=raw['sources'],all_reference_sources_match_original_frozen_lf=raw['all_reference_sources_match_original_frozen_lf'],native_YAML_differences=raw['configs'],missing_reference_weights=inventory['missing_reference_weights'],original_failed_byte_guard=ref('pipeline/checks/y101_review_assets_'+token+'_remote.log'),initial_byte_guard_not_passed=True,original_LF_manifest_not_changed=True,native_or_scientific_tests_replayed=False,frozen_evaluation_executed=False,checkpoint_loaded=False,CUDA_used=False,remote_mutations=False,evidence=[ref(__file__),ref(rawpath),ref(rawpath.with_suffix('.log')),ref(binding),inventoryref,candidate['post_training_freeze']]))
        work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');candidate['frozen_review_asset_clarification']=ref(out);candidate['evidence'].extend([ref(out),ref(__file__)])
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(clarification=ref(out),all18_sources_match_original_frozen_LF=raw['all_reference_sources_match_original_frozen_lf'],format_only_paths=[r['path'] for r in raw['sources'] if r['format_only_difference']],YAML_differences=raw['configs'],missing_weights=inventory['missing_reference_weights'],evaluation_executed=False)))
    finally:writer.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);run(p.parse_args().token)
