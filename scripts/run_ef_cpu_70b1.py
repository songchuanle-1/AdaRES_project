"""Run one CPU-only implementation check in project-scoped remote storage."""

import json
from pathlib import Path
import subprocess
from experiment_first_70b1 import ROOT,TOKEN,lease,ref,monitor

lease(400)
server=monitor.load_servers(str(ROOT))['server-2']
assert server['host']=='192.168.1.163'
source=ROOT/('pipeline/analyses/ef_cpu_'+TOKEN+'_program.py')
stage='/bak_home/songcl/Project/AdaRES_project/pipeline/probes/'+TOKEN
remote_output=stage+'/cpu_implementation.json'
output=ROOT/('pipeline/analyses/ef_cpu_'+TOKEN+'.json')
assert not output.exists() and not output.with_suffix('.log').exists()
payload=dict(root=server['remote_code_root'],stage=stage,output=remote_output,physical_gpu=None)
prefix=('import os,sys,pathlib,json,subprocess\nP='+repr(payload)+'\n'
    "assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'\n"
    "assert not pathlib.Path(P['stage']).exists()\npathlib.Path(P['stage']).mkdir(parents=True)\n"
    "os.environ['PYTHONDONTWRITEBYTECODE']='1'\nos.environ['CUDA_VISIBLE_DEVICES']=''\n"
    "os.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\n"
    "os.chdir(P['root'])\nsys.path[:0]="+repr([server['remote_code_root'],server['remote_code_root']+'/tools',
        server['remote_code_root']+'/ipdn/lib',server['site_packages'],server['pointnet_egg']])+"\n")
program=prefix+source.read_text(encoding='utf-8')
compile(program,str(source),'exec')
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    with output.with_suffix('.log').open('xb') as log:
        response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
            'songcl@192.168.1.163','timeout --signal=TERM --kill-after=10s 300 '+server['python']+' -S -B -'],
            input=program.encode('utf-8'),stdout=log,stderr=subprocess.STDOUT,timeout=325)
    lease()
    assert response.returncode==0,'CPU implementation check failed; original raw log retained.'
    subprocess.run(['scp','-q','songcl@192.168.1.163:'+remote_output,str(output)],check=True,timeout=30)
    result=json.loads(output.read_text(encoding='utf-8'))
    assert result['complete']
    print(json.dumps(dict(evidence=ref(output),log=ref(output.with_suffix('.log')),
        summary=result),indent=2))
finally:
    lock.close()
