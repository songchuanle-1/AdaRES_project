"""Validate prepared smoke identities and observe current authorized resources."""

import argparse
import ast
import base64
import copy
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import closure
import experiment_first
import foreground_tick as tick
import pipeline_monitor as monitor
from dispatch_prepared_candidates import snapshot

FILES={'__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py'}

def ref(path):
    path=Path(path)
    path=path if path.is_absolute() else ROOT/path
    assert path.resolve().is_relative_to(ROOT.resolve())
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def validate_frozen(candidate,plan,frozen):
    assert frozen['goal_sha256']==ref('pipeline/goal.json')['sha256'],'Current goal differs from frozen smoke'
    assert frozen['code_commit']==candidate['code_commit']==plan['code_commit']
    row=frozen['variant']
    assert row['id']==candidate['id'] and row['config']==candidate['config']==plan['config']
    assert row['package']==candidate['implementation_package']
    assert set(row['python_files'])==FILES
    files={f'AdaRES/ipdn/model_paper5/{row["package"]}/{name}':value
           for name,value in row['python_files'].items()}
    files['AdaRES/'+row['config']]=row['config_data']
    for path,value in files.items():
        expected=value['sha256']
        payload=base64.b64decode(value['base64'])
        assert hashlib.sha256(payload).hexdigest()==expected,'Corrupt frozen payload'
        assert ref(path)['sha256']==expected,'Working bytes changed since test'
        committed=subprocess.check_output(['git','show',candidate['code_commit']+':'+path],cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest()==expected,'Commit differs from tested bytes'
    train=(ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(train).hexdigest()==frozen['train_source_lf_sha256']
    return len(files)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--token',required=True)
    args=parser.parse_args()
    base=ROOT/'pipeline/headless'
    owner=tick.load(base/'.foreground.lockdir/owner.json')
    assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
    assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds()>120
    for name,key in (('.tick_context.json','context_sha256'),('.tick_context_state.next.json','pending_sha256')):
        assert tick.digest(base/name)==owner[key]
    assert experiment_first.effective(ROOT)['enabled']
    lock=monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json')
        candidates=[c for c in work['candidates'] if c['id'] in ('AdaRES-Y58','AdaRES-Y59')]
        assert len(candidates)==2 and all(c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' for c in candidates)
        checks=[]
        for c in candidates:
            plan=tick.load(closure.artifact(ROOT,c['native_smoke_plan']))
            program=closure.artifact(ROOT,plan['program'])
            frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
            count=validate_frozen(c,plan,frozen)
            wrong_goal=copy.deepcopy(frozen)
            wrong_goal['goal_sha256']='0'*64
            refused_goal=False
            try:validate_frozen(c,plan,wrong_goal)
            except AssertionError:refused_goal=True
            assert refused_goal
            wrong_source=copy.deepcopy(frozen)
            wrong_source['variant']['python_files']['ipdn.py']['base64']=base64.b64encode(b'changed').decode()
            refused_source=False
            try:validate_frozen(c,plan,wrong_source)
            except AssertionError:refused_source=True
            assert refused_source
            checks.append(dict(id=c['id'],files=count,program=ref(program),
                commit=c['code_commit'],current_goal_and_worktree_match=True,
                stale_goal_control_refused=True,tampered_source_control_refused=True))
        server=monitor.load_servers(str(ROOT))['server-2']
        assert server['host']=='192.168.1.163' and server['allowed_gpus']==[0,1] and server['max_tasks_per_gpu']==1
        queue=monitor.load_queue(str(ROOT))
        tasks=[t for t in queue['gpu_queues']['server-2'] if t['id'] in ('AdaRES-Y56','AdaRES-Y57')]
        assert len(tasks)==2
        live=snapshot(server,tasks,[])
        assert live['gpu_query_rc']==live['compute_query_rc']==0
        inventory={int(row[0]):row for row in csv.reader(io.StringIO(live['authorized_device_inventory']))}
        assert set(inventory)=={0,1}
        compute=list(csv.reader(io.StringIO(live['compute'])))
        busy_uuids={row[0].strip() for row in compute}
        idle=[gpu for gpu,row in inventory.items() if float(row[3])<10 and float(row[4])<1024
              and row[1].strip() not in busy_uuids]
        progress=[]
        for task in tasks:
            row=live['tasks'][task['id']]
            assert row['process_live'] and row['cwd']==server['remote_code_root']
            assert task['config'] in row['cmdline'] and 'tools/train.py' in row['cmdline']
            steps=closure.steps(row['tail'])
            assert len(set(steps))>=2
            assert not any(v in row['tail'] for v in ('Traceback','CUDA out of memory','RuntimeError'))
            metric=monitor.parse_running_miou([task['log']],task['pid'],Path(task['config']).stem,
                server['work_dir'],server['host'],server['user'],registered_native=True)
            progress.append(dict(id=task['id'],gpu=task['gpu'],pid=task['pid'],
                first_step=steps[0],last_step=steps[-1],best_mIoU=metric[1],validation_values=metric[0]))
        now=datetime.now().astimezone()
        path=ROOT/('pipeline/checks/ef_execution_'+args.token+'.json')
        assert not path.exists()
        path.parent.mkdir(parents=True,exist_ok=True)
        tick.save(path,dict(complete=True,observed_at=now.isoformat(),
            command='python -B scripts/check_ef_execution.py --token '+args.token,
            checks=checks,live=live,training=progress,idle_authorized_gpus=idle,
            native_smoke_executed=False,queue_changed=False,
            result='passed_ready_for_native' if idle else 'blocked_no_exclusive_GPU',
            limitation='Identity/negative-control and resource checks only; not actual native task backward.'))
        evidence=ref(path)
        for c in candidates:
            for row in (evidence,ref(__file__)):
                if row not in c['evidence']:c['evidence'].append(row)
            if idle:
                c['stage']='validation'
                for key in ('blocker','blocker_kind','retry_after','unblock_condition'):
                    c.pop(key,None)
            else:
                c.update(retry_after=(now+timedelta(hours=1)).isoformat(),occupation_evidence=evidence,
                    blocker='Both authorized GPUs are occupied by the verified original Y56/Y57 training. '
                            'All fourteen prepared YAML/source bytes match tested commits and current goal; '
                            'no free whole-GPU slot for the required two-step native smoke.')
        for c in work['candidates']:
            if c['id'] in ('AdaRES-Y56','AdaRES-Y57') and evidence not in c['evidence']:
                c['evidence'].append(evidence)
        work['tick_attempt']=dict(tick_started_at=tick.load(base/'.tick_model_view.json')['generated_at'],
            candidate_id='AdaRES-Y59',kind='resource_check',result='passed' if idle else 'blocked',
            command='python -B scripts/check_ef_execution.py --token '+args.token,
            observation='Actual committed/current YAML and source checks plus stale-goal/tampered-source '
                        'negative controls passed; '+('authorized idle GPU found.' if idle else
                        'both authorized whole GPUs occupied by original progressing training; no native smoke executed.'),
            evidence=[evidence])
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(evidence=evidence,checks=checks,training=progress,
            idle_authorized_gpus=idle,native_smoke_executed=False),indent=2))
    finally:
        lock.close()

if __name__=='__main__':main()
