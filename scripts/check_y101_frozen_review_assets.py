"""Inventory original server1 frozen-review assets without weights, models or GPU use."""
import argparse,json
from datetime import datetime
from pathlib import PurePosixPath
import yaml
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y102_deployment import Y101,Y102,git,adapt
from reconcile_y62_y63_server1 import remote
import closure

REMOTE='''import datetime,hashlib,json,pathlib,shutil,subprocess,sys,yaml
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
root=pathlib.Path(P['root']);proc=pathlib.Path('/proc/2263581')
assert proc.is_dir() and (proc/'cwd').resolve()==root
assert P['active_config'].encode() in (proc/'cmdline').read_bytes().split(b'\\0')
assert (root/'.codex_source_commit').read_text().strip()=='6fb26984ae652efbba177ab20c39575c44a69517'
assert 'torch' not in sys.modules
sources=[]
for relative,digest in P['sources'].items():
    path=root/relative;assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    sources.append(relative)
configs=[]
for relative,value in P['configs'].items():
    path=root/relative;assert path.is_file() and yaml.safe_load(path.read_text())==value
    configs.append(relative)
weights=[]
for item in P['references']:
    path=pathlib.Path(item['mapped_checkpoint']);exists=path.is_file()
    weights.append(dict(item,exists=exists,bytes=path.stat().st_size if exists else None,checkpoint_loaded=False,weights_sha256_verified=False))
evaluator=root/'tools/evaluate_adares_frozen_full.py';present=evaluator.is_file()
digest=hashlib.sha256(evaluator.read_bytes()).hexdigest() if present else None
if present:assert digest==P['evaluator_sha256']
assert pathlib.Path(P['data_root']).is_dir()
assert 'torch' not in sys.modules
print(json.dumps(dict(complete=True,observed_at=datetime.datetime.now().astimezone().isoformat(),hostname='dl02-MS-7B73',references=weights,configs_verified=configs,source_files_verified=sources,evaluator_present=present,evaluator_sha256=digest,original_data_root=P['data_root'],free_bytes=shutil.disk_usage(P['work']).free,torch_imported=False,checkpoint_loaded=False,CUDA_used=False,remote_mutations=False,active_training_preserved=True)))
'''

def run(token):
    lease(token,300);assert git('rev-parse','HEAD').decode().strip()==Y102
    before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml','pipeline/admissions/AdaRES-Y101.json')};index=git('ls-files','--stage','-z')
    work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};candidate=c['AdaRES-Y101']
    freeze=tick.load(closure.artifact(ROOT,candidate['post_training_freeze']))
    assert candidate['stage']=='training' and freeze['complete'] and freeze['id']==candidate['id']
    assert freeze['resource_condition']['server']=='server-1' and freeze['resource_condition']['gpu']==0 and freeze['resource_condition']['power_limit_watts']==350.
    assert freeze['evaluation']['methods']==['AdaRES-Y101','AdaRES-Y36','AdaRES-W2','AdaRES-B1'] and freeze['evaluation']['repeats']==2 and freeze['evaluation']['primary']=='FIRST'
    history_ref=c['AdaRES-Y54']['post_training_freeze'];history=tick.load(closure.artifact(ROOT,history_ref))
    assert history['comparators']['Y36']['sha256']=='f504f6ed548affa54eb7292cf5a7d930158be5dfaac291417f4852ad0f0698d1'
    assert history['comparators']['W2']['sha256']=='e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
    assert history['comparators']['B1']['sha256']=='cfc89ec0dc64f240913968f307d19c623f5ff0dc51c8e61de64e35abea05dd8b'
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server)
    task=next(r for r in monitor.load_queue(str(ROOT))['gpu_queues']['server-1'] if r['id']==candidate['id'])
    assert task['status']=='running' and task['launch_verified'] and task['pid']==2263581 and task['code_commit']==Y101
    configs={};sources={};references=[]
    packages={'Y36':'adares_y36_local_adjacency','W2':'adares_w2_pointmass_competition','B1':'adares_b1_standard'}
    import hashlib
    for name,package in packages.items():
        row=history['comparators'][name];mapped=adapt(row['path'],server)
        assert mapped.startswith(server['work_dir'].rstrip('/')+'/') and str(PurePosixPath(mapped)).endswith('/best.pth')
        references.append(dict(id={'Y36':'AdaRES-Y36','W2':'AdaRES-W2','B1':'AdaRES-B1'}[name],original_checkpoint=row['path'],mapped_checkpoint=mapped,expected_sha256=row['sha256'],config=row['config']))
        configs[row['config']]=adapt(yaml.safe_load(git('show',Y101+':AdaRES/'+row['config'])),server)
        paths=git('ls-tree','-r','--name-only',Y101,'--','AdaRES/ipdn/model_paper5/'+package).decode().splitlines();paths=[p for p in paths if p.endswith('.py')];assert paths
        for path in paths:sources[path[len('AdaRES/'):]]=hashlib.sha256(git('show',Y101+':'+path)).hexdigest()
    configs[task['config']]=adapt(yaml.safe_load(git('show',Y101+':AdaRES/'+task['config'])),server)
    assert history['evaluator']['sha256']==ref('AdaRES/tools/evaluate_adares_frozen_full.py')['sha256']=='7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13'
    payload=dict(root=server['remote_code_root'],work=server['work_dir'],data_root=server['data_root'],active_config=task['config'],configs=configs,sources=sources,references=references,evaluator_sha256=history['evaluator']['sha256'])
    rawpath=ROOT/('pipeline/checks/y101_review_assets_'+token+'_remote.json');assert not rawpath.exists()
    raw=remote(server,'P='+repr(payload)+'\n'+REMOTE,rawpath);tick.save(rawpath,raw)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
        out=ROOT/('pipeline/checks/y101_review_assets_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,id=candidate['id'],observed_at=datetime.now().astimezone().isoformat(),freeze=candidate['post_training_freeze'],original_comparator_manifest=history_ref,original_resource=dict(server='server-1',gpu=0,power_limit_watts=350.),references=raw['references'],missing_reference_weights=[r['id'] for r in raw['references'] if not r['exists']],baseline_source_and_native_YAML_verified=True,evaluator_present=raw['evaluator_present'],old_evaluator_requires_resource_adapter=True,legacy_lock_path='/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock',future_resource_adapter_must_preserve_evaluator_science=True,readiness='inventory_only_not_an_executable_evaluation_plan',candidate_checkpoint_verified=False,wait_for_original_Y101_full_finite_terminal_archive=True,formal_FIRST_or_cost_executed=False,CUDA_used=False,checkpoint_loaded=False,remote_mutations=False,evidence=[ref(__file__),ref(rawpath),ref(rawpath.with_suffix('.log')),candidate['post_training_freeze'],history_ref,history['evaluator'],ref('scripts/prepare_terminal_eval_6deb.py')]))
        work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');candidate['frozen_review_asset_inventory']=ref(out);candidate['evidence'].extend([ref(out),ref(rawpath),ref(__file__)])
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(inventory=ref(out),missing_reference_weights=[r['id'] for r in raw['references'] if not r['exists']],evaluator_present=raw['evaluator_present'],baseline_source_files=len(sources),checkpoint_loaded=False,GPU_used=False,formal_evaluation_executed=False)))
    finally:writer.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);run(p.parse_args().token)
