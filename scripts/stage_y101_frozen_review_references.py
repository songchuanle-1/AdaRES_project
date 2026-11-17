"""Stage exact frozen reference assets outside active source, without GPU/model use."""
import argparse,hashlib,json,subprocess,sys,time
from datetime import datetime
from pathlib import PurePosixPath
import yaml
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y102_deployment import Y101,Y102,git,adapt
from reconcile_y62_y63_server1 import remote
import closure

SOURCE='''import hashlib,json,os,pathlib,subprocess,sys
os.nice(10)
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path(P['work']).resolve();rows=[]
for row in P['references']:
    path=pathlib.Path(row['original_checkpoint']);assert os.path.commonpath([str(root),str(path.resolve())])==str(root)
    assert path.is_file() and path.name=='best.pth'
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert h.hexdigest()==row['expected_sha256']
    rows.append(dict(id=row['id'],path=str(path),bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=h.hexdigest()))
assert 'torch' not in sys.modules
print(json.dumps(dict(complete=True,references=rows,checkpoint_loaded=False,CUDA_used=False,remote_mutations=False)))
'''

PREPARE='''import hashlib,json,os,pathlib,shutil,subprocess,sys
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
root=pathlib.Path(P['work']).resolve();proc=pathlib.Path('/proc/2263581')
assert proc.is_dir() and (proc/'cwd').resolve()==pathlib.Path(P['code'])
assert P['active_config'].encode() in (proc/'cmdline').read_bytes().split(b'\\0')
assert shutil.disk_usage(root).free>P['total_bytes']+4*1024**3
rows=[]
for row in P['references']:
    dest=pathlib.Path(row['mapped_checkpoint']);part=pathlib.Path(str(dest)+'.'+P['token']+'.part')
    assert os.path.commonpath([str(root),str(dest.resolve())])==str(root)
    assert dest.name=='best.pth' and not dest.exists() and not part.exists()
    dest.parent.mkdir(parents=True,exist_ok=True)
    assert os.path.commonpath([str(root),str(dest.parent.resolve())])==str(root)
    rows.append(dict(id=row['id'],destination=str(dest),temporary=str(part),expected_sha256=row['expected_sha256']))
stage=pathlib.Path(P['stage']);project=pathlib.Path(P['project']).resolve()
assert os.path.commonpath([str(project),str(stage.resolve())])==str(project) and P['token'] in str(stage)
assert not stage.exists();stage.mkdir(parents=True)
for row in P['configs']:
    path=stage/row['name'];assert not path.exists();raw=row['text'].encode();assert hashlib.sha256(raw).hexdigest()==row['sha256'];path.write_bytes(raw)
assert 'torch' not in sys.modules
print(json.dumps(dict(complete=True,destinations=rows,staged_config_root=str(stage),config_files=[dict(name=r['name'],sha256=r['sha256']) for r in P['configs']],active_training_unchanged=True,active_source_or_native_configs_modified=False,checkpoint_loaded=False,CUDA_used=False)))
'''

PUBLISH='''import hashlib,json,os,pathlib,subprocess,sys
os.nice(10)
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
root=pathlib.Path(P['work']).resolve();row=P['reference'];dest=pathlib.Path(row['mapped_checkpoint']);part=pathlib.Path(str(dest)+'.'+P['token']+'.part')
assert os.path.commonpath([str(root),str(dest.resolve())])==str(root) and os.path.commonpath([str(root),str(part.resolve())])==str(root)
assert part.name.endswith('.'+P['token']+'.part') and part.is_file() and not dest.exists()
before=part.stat();h=hashlib.sha256()
with part.open('rb') as f:
    for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
after=part.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
assert before.st_size==P['source_bytes'] and h.hexdigest()==row['expected_sha256']
os.link(str(part),str(dest))
assert dest.is_file() and dest.stat().st_ino==part.stat().st_ino
part.unlink()
assert not part.exists() and 'torch' not in sys.modules
print(json.dumps(dict(complete=True,id=row['id'],destination=str(dest),bytes=before.st_size,sha256=h.hexdigest(),published_without_overwrite=True,temporary_removed=True,checkpoint_loaded=False,CUDA_used=False,active_training_unchanged=True)))
'''

def run(token):
    lease(token,1200);assert git('rev-parse','HEAD').decode().strip()==Y102
    immutable={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y102.json')};index=git('ls-files','--stage','-z')
    work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert candidate['stage']=='training'
    clarification=tick.load(closure.artifact(ROOT,candidate['frozen_review_asset_clarification']));assert clarification['all_reference_sources_match_original_frozen_lf'] is True
    inventory=tick.load(closure.artifact(ROOT,candidate['frozen_review_asset_inventory']));assert inventory['missing_reference_weights']==['AdaRES-Y36','AdaRES-W2','AdaRES-B1']
    freeze=tick.load(closure.artifact(ROOT,candidate['post_training_freeze']));assert freeze['evaluation']['methods']==['AdaRES-Y101','AdaRES-Y36','AdaRES-W2','AdaRES-B1'] and freeze['resource_condition']['server']=='server-1' and freeze['resource_condition']['gpu']==0 and freeze['resource_condition']['power_limit_watts']==350.
    servers=monitor.load_servers(str(ROOT));s1,s2=servers['server-1'],servers['server-2'];closure.validate_adares_server(ROOT,s1);closure.validate_adares_server(ROOT,s2)
    assert s1['allowed_gpus']==s2['allowed_gpus']==[0]
    references=inventory['references'];assert len(references)==3
    for row in references:assert row['mapped_checkpoint']==adapt(row['original_checkpoint'],s1) and row['original_checkpoint'].startswith(s2['work_dir'].rstrip('/')+'/')
    base='pipeline/checks/y101_reference_stage_'+token
    origin=ROOT/(base+'_source.json');assert not origin.exists();source=remote(s2,'P='+repr(dict(work=s2['work_dir'],references=references))+'\n'+SOURCE,origin);tick.save(origin,source)
    assert source['complete'] and len(source['references'])==3
    byid={r['id']:r for r in source['references']}
    configs=[];configrefs=[]
    folder=ROOT/(base+'_configs');assert not folder.exists();folder.mkdir()
    for row in references:
        raw=git('show',Y101+':AdaRES/'+row['config']);cfg=yaml.safe_load(raw);mapped=adapt(cfg,s1)
        def diff(a,b,p=''):
            if isinstance(a,dict) and isinstance(b,dict):return [v for k in sorted(set(a)|set(b)) for v in diff(a.get(k),b.get(k),p+'.'+str(k))]
            return [] if a==b else [p]
        assert set(diff(cfg,mapped))=={'.data.train.data_root','.data.val.data_root','.train.pretrain','.work_dir'}
        text=yaml.safe_dump(mapped,sort_keys=False,allow_unicode=True);name=PurePosixPath(row['config']).name;path=folder/name;path.write_text(text,encoding='utf-8',newline='\n');configrefs.append(ref(path))
        configs.append(dict(name=name,text=text,sha256=ref(path)['sha256'],original_config=row['config'],only_original_path_mappings_changed=True))
    project=s1['remote_code_root'].rsplit('/',1)[0];stage=project+'/pipeline/review_assets/AdaRES-Y101/'+token+'/configs'
    prepare_payload=dict(work=s1['work_dir'],code=s1['remote_code_root'],project=project,stage=stage,token=token,references=references,configs=configs,total_bytes=sum(r['bytes'] for r in source['references']),active_config=candidate['config'])
    preparation=ROOT/(base+'_destination.json');assert not preparation.exists();dest=remote(s1,'P='+repr(prepare_payload)+'\n'+PREPARE,preparation);tick.save(preparation,dest)
    published=[];transfers=[]
    for row in references:
        lease(token,400)
        part=row['mapped_checkpoint']+'.'+token+'.part';log=ROOT/(base+'_'+row['id'].lower()+'_transfer.log');assert not log.exists()
        argv=['scp','-3','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',s2['user']+'@'+s2['host']+':'+row['original_checkpoint'],s1['user']+'@'+s1['host']+':'+part]
        start=time.monotonic()
        with log.open('xb') as stream:response=subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        assert response.returncode==0,'Preserve partial and log; do not overwrite or blindly retry'
        proof=ROOT/(base+'_'+row['id'].lower()+'_published.json');assert not proof.exists()
        value=remote(s1,'P='+repr(dict(work=s1['work_dir'],reference=row,source_bytes=byid[row['id']]['bytes'],token=token))+'\n'+PUBLISH,proof);tick.save(proof,value)
        assert value['sha256']==row['expected_sha256'] and value['published_without_overwrite'] and value['bytes']==byid[row['id']]['bytes'];published.append(ref(proof));transfers.append(dict(id=row['id'],command_argv=argv,seconds=time.monotonic()-start,log=ref(log)))
        print(json.dumps(dict(published_reference=row['id'],bytes=value['bytes'],frozen_SHA_verified=True)),flush=True)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert all(ref(p)==v for p,v in immutable.items()) and git('ls-files','--stage','-z')==index
        out=ROOT/(base+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,id=candidate['id'],observed_at=datetime.now().astimezone().isoformat(),purpose='Only the original AdaRES-Y101 full frozen server1GPU0/350W comparison with Y36/W2/B1; no old GPU1 review migration.',freeze=candidate['post_training_freeze'],references=references,source_hashes=ref(origin),destinations=ref(preparation),published=published,transfers=transfers,adapted_native_YAML=configrefs,remote_adapted_config_root=stage,only_two_original_SERVERS_mappings_used=True,all_three_reference_bytes_match_frozen_SHA=True,checkpoint_loaded=False,CUDA_used=False,active_source_or_native_configs_modified=False,queue_or_running_modified=False,formal_FIRST_or_cost_executed=False,evaluation_ready=False,remaining='Require natural original Y101 full finite exact terminal proof/canonical archive, exclusive server1GPU0/350W, tested resource-only evaluator lock adapter and complete prediction/paired bootstrap/subgroup/cost execution. No target achievement.',evidence=[ref(__file__),candidate['frozen_review_asset_inventory'],candidate['frozen_review_asset_clarification'],candidate['post_training_freeze'],ref(origin),ref(preparation)]+published+configrefs+[r['log'] for r in transfers]))
        work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');candidate['frozen_review_reference_stage']=ref(out);candidate['evidence'].extend([ref(out),ref(__file__)])
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(stage=ref(out),reference_files=3,total_bytes=sum(r['bytes'] for r in source['references']),all_reference_SHA_verified=True,adapted_configs=3,active_source_unchanged=True,GPU_used=False,evaluation_executed=False)))
    finally:writer.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);run(p.parse_args().token)
