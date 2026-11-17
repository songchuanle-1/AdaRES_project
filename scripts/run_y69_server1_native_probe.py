"""Run the frozen Y69 native check under the verified server-1 GPU0 contract."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import subprocess
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from prepare_refinement_server1_alternative import assert_slot_unreserved
import closure


def main():
    p=argparse.ArgumentParser()
    p.add_argument('id',choices=('AdaRES-Y69',))
    p.add_argument('--token',required=True)
    p.add_argument('--attempt',type=int,default=1,choices=(1,2,3))
    a=p.parse_args()
    lease(a.token,800)
    server=monitor.load_servers(str(ROOT))['server-1']
    closure.validate_adares_server(ROOT,server)
    work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(x for x in work['candidates'] if x['id']==a.id)
    assert c['code_commit']=='9acd73cdc1b96be003c8450a9e22057248a0f531'
    assert c['required_gpu']==0
    assert c['stage']=='validation' and c['required_server']=='server-1'
    assert c['native_smoke_plan'] == c['native_smoke_plan_server1']
    plan=tick.load(closure.artifact(ROOT,c['native_smoke_plan']))
    program_path=closure.artifact(ROOT,plan['program'])
    program=program_path.read_text(encoding='utf-8')
    frozen=ast.literal_eval(ast.parse(program).body[0].value)
    assert frozen['runtime_server']=='server-1' and frozen['runtime_power_limit_watts']==350.0
    assert frozen['runtime_path_mappings']==server['runtime_path_mappings']
    closure.artifact(ROOT,frozen['resource_authorization'])
    assert validate_frozen(c,plan,frozen)==7
    basename=a.id.lower().replace('-','_')+'_server1_smoke_'+a.token
    if a.attempt>1:basename+='_attempt'+str(a.attempt)
    out=ROOT/('pipeline/analyses/'+basename+'.json')
    log=out.with_suffix('.log')
    assert not out.exists() and not log.exists()
    remote='/data1/songcl/Project/AdaRES_project/pipeline/probes/'+a.token+'/'+basename+'.json'
    payload=dict(output=remote,token=a.token,root=server['remote_code_root'],physical_gpu=0,
        source_sha256=ref(program_path)['sha256'],path_mappings=server['runtime_path_mappings'])
    prefix="""import os,sys,pathlib,json,subprocess,fcntl
P=PAYLOAD
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
gpu_lease=open('/data1/songcl/Project/AdaRES_project/.foreground_gpu0_diagnostic.lock','a')
fcntl.flock(gpu_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
used,util=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')]
assert used<1024 and util<10,'GPU0 occupied; preserve foreign jobs'
assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
assert not pathlib.Path(P['output']).exists()
pathlib.Path(P['output']).parent.mkdir(parents=True,exist_ok=True)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.chdir(P['root'])
sys.path[:0]=PYTHON_PATHS
""".replace('PAYLOAD',repr(payload)).replace('PYTHON_PATHS',repr([server['remote_code_root'],server['remote_code_root']+'/tools',server['remote_code_root']+'/ipdn/lib',server['site_packages'],server['pointnet_egg']]))
    compile(prefix+program,str(program_path),'exec')
    lock=monitor.acquire_lock(str(ROOT))
    assert lock is not None
    started=datetime.now().astimezone().isoformat()
    try:
        assert_slot_unreserved(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)))
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==c['code_commit']
        with log.open('xb') as stream:
            r=subprocess.run(['ssh','-o','BatchMode=yes','server-1',
                'LD_LIBRARY_PATH='+server.get('runtime_library_path','')+' timeout --signal=TERM --kill-after=10s 600 '+server['python']+' -B -S -'],
                input=(prefix+program).encode(),stdout=stream,stderr=subprocess.STDOUT,timeout=625)
        lease(a.token,60)
        if r.returncode:raise RuntimeError('Native check failed; preserve '+str(log))
        subprocess.run(['scp','-q','server-1:'+remote,str(out)],check=True,timeout=30)
        raw=tick.load(out)
        assert raw['complete'] and raw['id']==a.id
        work=tick.load(ROOT/'pipeline/research_work.json')
        current=next(x for x in work['candidates'] if x['id']==a.id)
        current['native_smoke_result']=ref(out)
        current['evidence'].extend([ref(out),ref(log)])
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(evidence=ref(out),log=ref(log),started_at=started,
            observed_at=datetime.now().astimezone().isoformat(),result=raw)))
    finally:lock.close()

if __name__=='__main__':main()
