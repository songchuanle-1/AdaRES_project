"""Lease-bound helpers for prospective experiment-first preparation."""

import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
TOKEN='70b1ba64384f498697815ca88c2c232a'
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
import experiment_first
from dispatch_prepared_candidates import snapshot

PACKAGES={58:'adares_y58_fixed_backbone_statistics',59:'adares_y59_frozen_text_mode'}
FILES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')

def lease(reserve=90):
    base=ROOT/'pipeline/headless'
    owner=tick.load(base/'.foreground.lockdir/owner.json')
    assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
    assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds()>reserve
    for name,key in (('.tick_context.json','context_sha256'),('.tick_context_state.next.json','pending_sha256')):
        assert tick.digest(base/name)==owner[key]
    assert experiment_first.effective(ROOT)['enabled']
    return owner

def ref(path):
    path=Path(path)
    path=path if path.is_absolute() else ROOT/path
    assert path.resolve().is_relative_to(ROOT.resolve()) and path.is_file()
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def save(path,value):
    assert not (ROOT/path).exists()
    tick.save(ROOT/path,value)
    return ref(path)

def bind(candidate,*rows):
    for row in rows:
        if row not in candidate['evidence']:candidate['evidence'].append(row)

def copy_sources():
    lease()
    parent=ROOT/'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
    parent_config=ROOT/'AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
    for name in PACKAGES.values():
        target=parent.parent/name
        config=parent_config.parent/(name+'_res.yaml')
        assert not target.exists() and not config.exists()
    for name in PACKAGES.values():
        target=parent.parent/name
        target.mkdir()
        for filename in FILES:shutil.copyfile(parent/filename,target/filename)
        shutil.copyfile(parent_config,parent_config.parent/(name+'_res.yaml'))
    print('Mechanically copied two independent six-file packages and native YAML files; no manual changes yet.')

def observe():
    lease()
    server=monitor.load_servers(str(ROOT))['server-2']
    assert server['host']=='192.168.1.163' and server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
    queue=monitor.load_queue(str(ROOT))
    tasks=[t for t in queue['gpu_queues']['server-2'] if t['id'] in ('AdaRES-Y56','AdaRES-Y57')]
    assert len(tasks)==2
    live=snapshot(server,tasks,[])
    assert live['gpu_query_rc']==live['compute_query_rc']==0
    import closure
    summary=[]
    for task in tasks:
        row=live['tasks'][task['id']]
        assert row['process_live'] and row['cwd']==server['remote_code_root']
        assert task['config'] in row['cmdline'] and 'tools/train.py' in row['cmdline']
        steps=closure.steps(row['tail'])
        assert len(set(steps))>=2
        metric=monitor.parse_running_miou([task['log']],task['pid'],Path(task['config']).stem,
            server['work_dir'],server['host'],server['user'],registered_native=True)
        summary.append(dict(id=task['id'],gpu=task['gpu'],pid=task['pid'],
            first_step=steps[0],last_step=steps[-1],best_mIoU=metric[1],validation_values=metric[0]))
    now=datetime.now().astimezone()
    result=save('pipeline/analyses/ef_occupation_'+TOKEN+'_'+now.strftime('%H%M%S')+'.json',
        dict(complete=True,observed_at=now.isoformat(),read_only_remote=True,live=live,summary=summary))
    print(json.dumps(dict(evidence=result,observed_at=now.isoformat(),tasks=summary,
        gpu=live['authorized_device_inventory'],compute=live['compute']),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('copy-sources','observe'))
    args=parser.parse_args()
    (copy_sources if args.operation=='copy-sources' else observe)()
