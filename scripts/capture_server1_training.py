"""Observe the exact new server-1 process, source/YAML, and increasing steps."""
import argparse
from datetime import datetime
import json
import subprocess
import yaml
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

BODY=r'''
import datetime,hashlib,json,pathlib,re,subprocess,time,yaml
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
root=pathlib.Path(P['root'])
task=P['task']
def sample():
    proc=pathlib.Path('/proc')/str(task['pid'])
    assert proc.is_dir() and (proc/'cwd').resolve()==root
    args=(proc/'cmdline').read_bytes().split(b'\0')
    assert task['config'].encode() in args
    log=pathlib.Path(task['log'])
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-24000));text=stream.read().decode(errors='replace')
    assert not any(v in text for v in ['Traceback','CUDA out of memory','RuntimeError'])
    steps=[tuple(map(int,x)) for x in re.findall(r'Train\[(\d+)/(\d+)\].*?(\d+)/(\d+)\s*\[',text)]
    assert steps
    return dict(observed_at=datetime.datetime.now().astimezone().isoformat(),latest=steps[-1],log_bytes=log.stat().st_size)
before=sample();time.sleep(5);after=sample()
assert (after['latest'][0],after['latest'][2])>(before['latest'][0],before['latest'][2])
for relative,digest in P['source_files'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
cfg=yaml.safe_load((root/task['config']).read_text())
assert cfg==P['expected_config']
assert cfg['train']['seed']==1999 and cfg['train']['epochs']==30
print(json.dumps(dict(complete=True,before=before,after=after,pid=task['pid'],log=task['log'],
 source_files_verified=len(P['source_files']),native_YAML_semantics_verified=True,
 source_marker=(root/'.codex_source_commit').read_text().strip(),
 gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.limit','--format=csv,noheader,nounits']).decode().strip(),
 no_bounded_tail_error=True,training_efficacy='Unknown; no validation result yet.')))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,120)
    server=monitor.load_servers(str(ROOT))['server-1'];queue=monitor.load_queue(str(ROOT))
    task=next(t for t in queue['gpu_queues']['server-1'] if t['id']=='AdaRES-Y62')
    assert task['status']=='running' and task['launch_verified']
    config=yaml.safe_load((ROOT/'AdaRES'/task['config']).read_text(encoding='utf-8'))
    def adapt(v):
        if isinstance(v,dict):return {k:adapt(x) for k,x in v.items()}
        if isinstance(v,list):return [adapt(x) for x in v]
        if isinstance(v,str):
            for m in server['runtime_path_mappings']:
                old,new=m['source'].rstrip('/'),m['target'].rstrip('/')
                if v==old or v.startswith(old+'/'):v=new+v[len(old):]
        return v
    folder=ROOT/'AdaRES/ipdn/model_paper5/adares_y62_overlap_density'
    files={p.relative_to(ROOT/'AdaRES').as_posix():ref(p)['sha256'] for p in folder.glob('*.py')}
    payload=dict(root=server['remote_code_root'],task=task,source_files=files,expected_config=adapt(config))
    out=ROOT/('pipeline/checks/server1_training_progress_'+a.token+'.json');assert not out.exists()
    r=subprocess.run(['ssh','-o','BatchMode=yes','server-1',server['python']+' -B -'],
        input=('P='+repr(payload)+'\n'+BODY).encode(),capture_output=True,timeout=45)
    out.with_suffix('.log').write_bytes(r.stdout+r.stderr)
    assert r.returncode==0,'Preserve progress observer failure; do not relaunch'
    result=json.loads(r.stdout);result['evidence']=[ref(__file__),ref(out.with_suffix('.log'))]
    tick.save(out,result)
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(c for c in work['candidates'] if c['id']==task['id'])
    c['evidence'].append(ref(out));c['latest_training_observation']=ref(out)
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(evidence=ref(out),result=result)))

if __name__=='__main__':main()
