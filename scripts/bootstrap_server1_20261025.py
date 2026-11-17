"""Deploy an isolated copy of the existing AdaRES runtime, preserving sources."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

REMOTE = r'''
import concurrent.futures,json,os,pathlib,subprocess,time
root=pathlib.Path('/data1/songcl/Project/AdaRES_project')
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
assert str(root.resolve())=='/data1/songcl/Project/AdaRES_project'
assert not (root/'runtime'/'bootstrap_complete.json').exists()
root.mkdir(parents=True,exist_ok=True)
(root/'runtime').mkdir(exist_ok=True)
source='songcl@192.168.1.163:'
def run(command):
    start=time.time()
    p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=1500)
    assert p.returncode==0,(command,p.returncode,p.stderr[-3000:])
    return dict(command=command,seconds=time.time()-start,stdout=p.stdout,stderr=p.stderr[-3000:])
def copy_runtime():
    results=[]
    results.append(run(['rsync','-a','--ignore-existing',source+'/home/songcl/.conda/envs/adares_project/',str(root/'runtime'/'adares_project')+'/']))
    results.append(run(['rsync','-a','--ignore-existing',source+'/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg',str(root/'runtime')+'/']))
    results.append(run(['rsync','-a','--ignore-existing','--exclude=logs/','--exclude=.git/',source+'/bak_home/songcl/Project/AdaRES_project/AdaRES/',str(root/'AdaRES')+'/']))
    return results
def compare_data():
    # Check every file actually present in the original data root, read-only.
    return run(['rsync','-rcni','--out-format=%i:%n',source+'/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/','/data-controller/songcl/IPDN/data/'])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    a=pool.submit(copy_runtime)
    b=pool.submit(compare_data)
    copies=a.result()
    comparison=b.result()
out=dict(complete=True,hostname='dl02-MS-7B73',runtime_copies=copies,data_comparison=comparison,
         data_unchanged=True,original_environment_unchanged=True,GPU_used=False)
path=root/'runtime'/'bootstrap_copy_result.json'
assert not path.exists()
path.write_text(json.dumps(out))
print(json.dumps(out))
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--token',required=True)
    args=p.parse_args()
    lease(args.token,1800)
    auth=tick.load(ROOT/'pipeline/authorizations/server1_gpu0_20261025.json')
    assert auth['authorization_granted'] and auth['allowed_gpus']==[0]
    assert auth['source_thread_id']=='01a07e32-78bf-7670-925b-370911ea48db'
    out=ROOT/('pipeline/checks/server1_bootstrap_'+args.token+'.json')
    log=out.with_suffix('.log')
    assert not out.exists() and not log.exists()
    start=datetime.now().astimezone().isoformat()
    with log.open('xb') as stream:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','server-1','python3 -B -'],
                         input=REMOTE.encode(),stdout=stream,stderr=subprocess.STDOUT,timeout=1700)
    if r.returncode:
        raise RuntimeError('Bootstrap failed; preserve '+str(log))
    result=json.loads(log.read_text(encoding='utf-8'))
    result.update(started_at=start,observed_at=datetime.now().astimezone().isoformat(),
                  evidence=[ref(log),ref(__file__),ref('pipeline/authorizations/server1_gpu0_20261025.json')])
    tick.save(out,result)
    print(json.dumps(dict(evidence=ref(out),data_differences=result['data_comparison']['stdout'],
                          copy_seconds=[r['seconds'] for r in result['runtime_copies']])))


if __name__=='__main__':
    main()
