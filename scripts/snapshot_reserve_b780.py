"""Read live training/GPU evidence without dispatching or changing the queue."""

from datetime import datetime
import json
from pathlib import Path
from research_b780 import ROOT,TOKEN,lease,save,monitor
from dispatch_prepared_candidates import snapshot
import closure

lease()
queue=monitor.load_queue(str(ROOT))
server=monitor.load_servers(str(ROOT))['server-2']
assert server['host']=='192.168.1.163' and server['allowed_gpus']==[0,1]
assert server['max_tasks_per_gpu']==1
tasks=[t for t in queue['gpu_queues']['server-2'] if t['id'] in ('AdaRES-Y56','AdaRES-Y57')]
assert len(tasks)==2 and all(t['status']=='running' for t in tasks)
started=datetime.now().astimezone().isoformat()
live=snapshot(server,tasks,[])
assert live['gpu_query_rc']==live['compute_query_rc']==0
summary=[]
for task in tasks:
    row=live['tasks'][task['id']]
    assert row['process_live'] and row['pid']==task['pid']
    assert row['cwd']==server['remote_code_root']
    assert task['config'] in row['cmdline'] and 'tools/train.py' in row['cmdline']
    assert not any(marker in row['tail'] for marker in ('Traceback','CUDA out of memory','RuntimeError'))
    steps=closure.steps(row['tail'])
    assert len(set(steps))>=2
    metric=monitor.parse_running_miou([task['log']],task['pid'],Path(task['config']).stem,
        '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints',
        server['host'],server['user'],registered_native=True)
    summary.append(dict(id=task['id'],gpu=task['gpu'],pid=task['pid'],
                        first_observed_step=steps[0],last_observed_step=steps[-1],
                        distinct_native_steps=len(set(steps)),validation_values=metric[0],
                        best_mIoU=metric[1],no_improve=metric[2],
                        metric_scope='Exact registered native validation lineage if available; otherwise unknown.'))
now=datetime.now().astimezone()
result=save('pipeline/analyses/reserve_occupation_'+TOKEN+'_'+now.strftime('%H%M%S')+'.json',
            dict(complete=True,observed_from=started,observed_until=now.isoformat(),
                 read_only_remote=True,queue_unchanged=True,live=live,summary=summary))
print(json.dumps(dict(evidence=result,observed_until=now.isoformat(),tasks=summary,
                     gpu=live['authorized_device_inventory'],compute=live['compute']),indent=2))
