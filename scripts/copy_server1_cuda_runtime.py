"""Copy the existing CUDA 11.3 shared libraries into the private runtime."""
import argparse,json,subprocess
from prepare_y60_stage_heads import ROOT,lease,ref,tick

PROGRAM=r'''
import pathlib,subprocess,json,time
root=pathlib.Path('/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64')
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
root.mkdir(exist_ok=True)
cmd=['rsync','-a','--ignore-existing','--include=*/','--include=*.so*','--exclude=*',
 'songcl@192.168.1.163:/usr/local/cuda-11.3/lib64/',str(root)+'/']
start=time.time()
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=400)
assert r.returncode==0,r.stderr
assert (root/'libcusparse.so.11').is_file()
print(json.dumps(dict(complete=True,seconds=time.time()-start,command=cmd,libraries=[p.name for p in root.glob('*.so*')],original_libraries_unchanged=True)))
'''
p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,600)
out=ROOT/('pipeline/checks/server1_cuda_runtime_'+a.token+'.json');assert not out.exists()
r=subprocess.run(['ssh','-o','BatchMode=yes','server-1','python3 -B -'],input=PROGRAM.encode(),capture_output=True,timeout=430)
out.with_suffix('.log').write_bytes(r.stdout+r.stderr)
assert r.returncode==0,'Preserve CUDA copy log'
d=json.loads(r.stdout);d['evidence']=[ref(__file__),ref(out.with_suffix('.log'))];tick.save(out,d)
print(json.dumps(dict(evidence=ref(out),seconds=d['seconds'])))
