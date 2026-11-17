"""Exact registered Y79/Y80/Y81 deployments; historical helpers stay immutable."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from observe_y69_server1_phase import parse_phase, phase_key, progressed, REMOTE as BASE_REMOTE
import closure

BASE='87019e7c2f640f99cb82a3e5b396321fc5b54d25'
Y72='c60c903941d6cd7ffcc343d7c13610f2ba416caa'
Y73='7da625ab750447eaad76e82309c89d887df93994'
Y74='3c91105459f773326910b1eeaf16450e2d6b90ac'
Y75='9bf15a255ed2c960429ecf57ab47de05c484b631'
Y76='c3849ead0b8e4f9320d3e353d561016b3b495ce1'
Y77='498e721ea05e96aeb6f948d9cf8cd7a91b65ae22'
Y78='5eee1e50c9e1d65bf113a4b8757bfc27d028a821'
Y79='e0c7dba199fcf1a009941450d04aeb799418a7e6'
Y80='4e68cd84d56d890222adc2bec8ab59cb0c8715b6'
Y81='4157937d0c3cf344a44394ce15c95349aa94d50f'
VERSIONS=(BASE,Y72,Y73,Y74,Y75,Y76,Y77,Y78,Y79,Y80,Y81)
PACKAGES={'AdaRES-Y71':'adares_y71_visual_product_fusion',
          'AdaRES-Y72':'adares_y72_visual_mask_refinement','AdaRES-Y73':'adares_y73_initial_query_anchor',
          'AdaRES-Y74':'adares_y74_language_update_gate','AdaRES-Y75':'adares_y75_language_query_init','AdaRES-Y76':'adares_y76_stage_confidence_conditioning','AdaRES-Y77':'adares_y77_geometry_query_init','AdaRES-Y78':'adares_y78_mask_logit_residual','AdaRES-Y79':'adares_y79_query_language_feedback','AdaRES-Y80':'adares_y80_query_visual_feedback','AdaRES-Y81':'adares_y81_query_mask_refinement'}
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,timeout=30)

def paths(cid):
    package=PACKAGES[cid]
    return ['AdaRES/ipdn/model_paper5/'+package+'/'+n for n in NAMES]+['AdaRES/configs_song/model_paper5/'+package+'_res.yaml']

def validate_delta(rows,cid):
    assert cid in ('AdaRES-Y72','AdaRES-Y73','AdaRES-Y74','AdaRES-Y75','AdaRES-Y76','AdaRES-Y77','AdaRES-Y78','AdaRES-Y79','AdaRES-Y80','AdaRES-Y81')
    assert len(rows)==7 and all(len(r)==2 and r[0]=='A' for r in rows)
    assert {r[1] for r in rows}==set(paths(cid))

def select(token,cid):
    assert cid=='AdaRES-Y81'
    target,parent=Y81,Y80
    assert contract(Y79,parent,'AdaRES-Y79') and contract(Y80,parent,'AdaRES-Y80')
    from advance_y81 import prepared,validate_result
    work=tick.load(ROOT/'pipeline/research_work.json');c,plan,program,frozen=prepared(cid,work)
    assert c['stage']=='validation' and not c.get('admission_receipt')
    validate_result(tick.load(closure.artifact(ROOT,c['native_smoke_result'])),c,frozen)
    assert git('rev-parse','HEAD').decode().strip()==parent
    assert git('rev-parse',target+'^').decode().strip()==parent
    rows=[r.split('\t') for r in git('diff','--no-renames','--name-status',parent,target).decode().splitlines()]
    validate_delta(rows,cid)
    index_before=git('ls-files','--stage','-z')
    def entries(raw):return {r.split(b'\t',1)[1]:r for r in raw.split(b'\0') if r}
    before=entries(index_before)
    for _,path in rows:
        assert path.encode() not in before, 'Preserve existing index entries'
        assert (ROOT/path).read_bytes()==git('show',target+':'+path)
    branch=git('symbolic-ref','HEAD').decode().strip();assert branch=='refs/heads/master'
    out=ROOT/('pipeline/checks/'+cid.lower()+'_source_selection_'+token+'.json');assert not out.exists()
    snapshot=out.with_suffix('.index');assert not snapshot.exists();snapshot.write_bytes(index_before)
    index_path=Path(git('rev-parse','--git-path','index').decode().strip())
    if not index_path.is_absolute():index_path=ROOT/index_path
    binary=out.with_suffix('.index.bin');assert not binary.exists();binary.write_bytes(index_path.read_bytes())
    # Select only the seven previously retained and actually tested files. All
    # other index entries and every working file remain byte-identical.
    for _,path in rows:
        blob=git('rev-parse',target+':'+path).decode().strip()
        subprocess.run(['git','update-index','--add','--cacheinfo','100644,'+blob+','+path],cwd=ROOT,check=True)
    subprocess.run(['git','update-ref','-m','AdaRES tested additive '+cid,branch,target,parent],cwd=ROOT,check=True)
    after=entries(git('ls-files','--stage','-z'));added={p.encode() for _,p in rows}
    assert {p:r for p,r in after.items() if p not in added}==before
    assert all((ROOT/p).read_bytes()==git('show',target+':'+p) for _,p in rows)
    tick.save(out,dict(complete=True,token=token,id=cid,parent=parent,target=target,only_added_files=rows,
        previous_index=ref(snapshot),previous_binary_index=ref(binary),other_index_entries_preserved=True,working_files_modified=False,
        helper=ref(__file__)))
    print(json.dumps(dict(selection=ref(out),HEAD=target)))

def contract(commit,head,cid):
    assert commit in VERSIONS and head in VERSIONS
    assert VERSIONS.index(commit)<=VERSIONS.index(head)
    deployments={}
    for version in VERSIONS[VERSIONS.index(commit):VERSIONS.index(head)+1]:
        rows=[r.split('\t') for r in git('diff','--no-renames','--name-status',commit,version).decode().splitlines()]
        allowed=set()
        if VERSIONS.index(commit)<1<=VERSIONS.index(version):allowed.update(paths('AdaRES-Y72'))
        if VERSIONS.index(commit)<2<=VERSIONS.index(version):allowed.update(paths('AdaRES-Y73'))
        if VERSIONS.index(commit)<3<=VERSIONS.index(version):allowed.update(paths('AdaRES-Y74'))
        if VERSIONS.index(commit)<4<=VERSIONS.index(version):allowed.update(paths('AdaRES-Y75'))
        if VERSIONS.index(commit)<5<=VERSIONS.index(version):allowed.update(paths('AdaRES-Y76'))
        if VERSIONS.index(commit)<6<=VERSIONS.index(version):allowed.update(paths('AdaRES-Y77'))
        if VERSIONS.index(commit)<7<=VERSIONS.index(version):allowed.update(paths('AdaRES-Y78'))
        if VERSIONS.index(commit)<8<=VERSIONS.index(version):allowed.update(paths('AdaRES-Y79'))
        if VERSIONS.index(commit)<9<=VERSIONS.index(version):allowed.update(paths('AdaRES-Y80'))
        if VERSIONS.index(commit)<10<=VERSIONS.index(version):allowed.update(paths('AdaRES-Y81'))
        assert len(rows)==len(allowed) and all(s=='A' for s,p in rows) and {p for s,p in rows}==allowed
        d=dict(source_files={},configs={})
        for _,path in rows:
            raw=git('show',version+':'+path);assert (ROOT/path).read_bytes()==raw
            relative=path[len('AdaRES/'):]
            if path.endswith('.yaml'):d['configs'][relative]=yaml.safe_load(raw)
            else:d['source_files'][relative]=hashlib.sha256(raw).hexdigest()
        deployments[version]=d
    for path in paths(cid):
        assert (ROOT/path).read_bytes()==git('show',commit+':'+path)==git('show',head+':'+path)
    return deployments

def observe(token,cid,label):
    assert cid in ('AdaRES-Y72','AdaRES-Y73','AdaRES-Y74','AdaRES-Y75','AdaRES-Y76','AdaRES-Y77','AdaRES-Y78','AdaRES-Y79','AdaRES-Y80','AdaRES-Y81') and label.replace('_','').isalnum()
    queue=monitor.load_queue(str(ROOT));task=next(t for ts in queue['gpu_queues'].values() for t in ts if t['id']==cid)
    assert task['status']=='running' and task['launch_verified'] is True
    expected={'AdaRES-Y71':('server-2','0',BASE,280.),'AdaRES-Y72':('server-1','0',Y72,350.),'AdaRES-Y73':('server-2','1',Y73,390.),'AdaRES-Y74':('server-2','0',Y74,280.),'AdaRES-Y75':('server-2','1',Y75,390.),'AdaRES-Y76':('server-1','0',Y76,350.),'AdaRES-Y77':('server-2','0',Y77,280.),'AdaRES-Y78':('server-2','1',Y78,390.),'AdaRES-Y79':('server-1','0',Y79,350.),'AdaRES-Y80':('server-2','0',Y80,280.),'AdaRES-Y81':('server-2','1',Y81,390.)}[cid]
    name,gpu,commit,power=expected
    assert (task['server'],str(task['gpu']),task['code_commit'])==(name,gpu,commit)
    if cid=='AdaRES-Y72':assert task['pid']==311445
    if cid=='AdaRES-Y73':assert task['pid']==62829
    if cid=='AdaRES-Y74':assert task['pid']==49891
    server=monitor.load_servers(str(ROOT))[name];closure.validate_adares_server(ROOT,server);closure.admission(ROOT,task)
    assert monitor.load_running(str(ROOT))[name][gpu]==cid
    head=git('rev-parse','HEAD').decode().strip();deployments=contract(commit,head,cid)
    def adapt(v):
        if isinstance(v,dict):return {k:adapt(x) for k,x in v.items()}
        if isinstance(v,list):return [adapt(x) for x in v]
        if isinstance(v,str):
            for m in server.get('runtime_path_mappings',[]):
                old,new=m['source'].rstrip('/'),m['target'].rstrip('/')
                if v==old or v.startswith(old+'/'):v=new+v[len(old):]
        return v
    for d in deployments.values():d['configs']={p:adapt(v) for p,v in d['configs'].items()}
    payload=dict(root=server['remote_code_root'],task=task,source_files={p[len('AdaRES/'):]:ref(p)['sha256'] for p in paths(cid)[:-1]},
        expected_config=adapt(yaml.safe_load((ROOT/'AdaRES'/task['config']).read_bytes())),deployments=deployments)
    body=BASE_REMOTE.replace('dl02-MS-7B73',server['expected_hostname']).replace("'nvidia-smi','-i','0'","'nvidia-smi','-i',"+repr(gpu))
    body=body.replace("assert (root/'.codex_source_commit').read_text().strip()==task['code_commit']", """marker=(root/'.codex_source_commit').read_text().strip()
assert marker in P['deployments']
for relative,digest in P['deployments'][marker]['source_files'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
for relative,value in P['deployments'][marker]['configs'].items():
    assert yaml.safe_load((root/relative).read_text())==value
""")
    body=body.replace("print(json.dumps(dict(complete=True", "assert float(subprocess.check_output(['nvidia-smi','-i',"+repr(gpu)+",'--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())=="+repr(power)+"\nprint(json.dumps(dict(complete=True")
    body=body.replace("source_marker=task['code_commit']","source_marker=marker")
    program='P='+repr(payload)+'\n'+'\n\n'.join(inspect.getsource(f) for f in (parse_phase,phase_key,progressed))+'\n'+body
    ast.parse(program,feature_version=(3,8))
    out=ROOT/('pipeline/checks/production_'+cid.lower()+'_'+token+'_'+label+'.json');assert not out.exists() and not out.with_suffix('.log').exists()
    before_q,before_r=ref('pipeline/queue.json'),ref('pipeline/running.json')
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',server['user']+'@'+server['host'],
        'PYTHONDONTWRITEBYTECODE=1 '+server['python']+' -B -'],input=program.encode(),capture_output=True,timeout=45)
    out.with_suffix('.log').write_bytes(result.stdout+result.stderr);assert result.returncode==0
    v=json.loads(result.stdout);assert v['native_progress_confirmed'] and progressed(v['before'],v['after'])
    v.update(id=cid,token=token,observed_local=datetime.now().astimezone().isoformat(),evidence=[ref(__file__),ref(out.with_suffix('.log'))]);tick.save(out,v)
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(c for c in work['candidates'] if c['id']==cid)
    c['evidence'].append(ref(out));c['latest_training_observation']=ref(out);tick.save(ROOT/'pipeline/research_work.json',work)
    assert ref('pipeline/queue.json')==before_q and ref('pipeline/running.json')==before_r
    print(json.dumps(dict(evidence=ref(out),phase=v['after']['latest'],pid=task['pid'],source_marker=v['source_marker'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('select','observe'));p.add_argument('--id',required=True,choices=tuple(PACKAGES));p.add_argument('--token',required=True);p.add_argument('--label',default='progress');a=p.parse_args()
    lease(a.token,240);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        if a.operation=='select':select(a.token,a.id)
        else:observe(a.token,a.id,a.label)
    finally:lock.close()
