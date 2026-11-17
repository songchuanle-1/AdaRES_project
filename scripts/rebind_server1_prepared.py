"""Rebind unchanged Y62/Y63 to the explicit resource-only goal update."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--token',required=True)
    a=p.parse_args()
    lease(a.token,300)
    lock=monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json')
        goal=ref('pipeline/goal.json')['sha256']
        records=[]
        for cid in ('AdaRES-Y62','AdaRES-Y63'):
            c=next(x for x in work['candidates'] if x['id']==cid)
            assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot'
            assert not c.get('admission_receipt')
            previous_plan=c['native_smoke_plan']
            plan=tick.load(closure.artifact(ROOT,previous_plan))
            old=closure.artifact(ROOT,plan['program'])
            source=old.read_text(encoding='utf-8')
            frozen=ast.literal_eval(ast.parse(source).body[0].value)
            assert c['code_commit']==frozen['code_commit']==closure.git(ROOT,'rev-parse','HEAD')
            previous_goal=frozen['goal_sha256']
            frozen['goal_sha256']=goal
            frozen['runtime_server']='server-1'
            frozen['resource_authorization']=ref('pipeline/authorizations/server1_gpu0_20261025.json')
            # Only deterministic deployment path adaptation is added to the old native check.
            body='\n'.join(source.splitlines()[1:])+'\n'
            anchor='cfg = gorilla.Config.fromfile(str(config_path))\n'
            assert body.count(anchor)==1
            adapt="""def adapt_path(value):
    for mapping in P['path_mappings']:
        old,new=mapping['source'].rstrip('/'),mapping['target'].rstrip('/')
        if value==old or value.startswith(old+'/'):value=new+value[len(old):]
    return value
cfg.train.pretrain=adapt_path(cfg.train.pretrain)
cfg.work_dir=adapt_path(cfg.work_dir)
for split in ('train','val'):
    cfg.data[split].data_root=adapt_path(cfg.data[split].data_root)
"""
            body=body.replace(anchor,anchor+adapt)
            prefix='pipeline/checks/'+cid.lower().replace('-','_')+'_server1_'+a.token
            program=ROOT/(prefix+'_program.py')
            plan_path=ROOT/(prefix+'_plan.json')
            assert not program.exists() and not plan_path.exists()
            program.write_text('FROZEN='+repr(frozen)+'\n'+body,encoding='utf-8',newline='\n')
            ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
            new_plan=copy.deepcopy(plan)
            new_plan.update(program=ref(program),goal_sha256=goal,server='server-1',gpu=0,
                previous_plan=previous_plan,resource_authorization=frozen['resource_authorization'])
            tick.save(plan_path,new_plan)
            assert validate_frozen(c,new_plan,frozen)==7
            c.update(stage='validation',required_server='server-1',native_smoke_plan=ref(plan_path),
                next_action='Run the current server-1 native YAML/TRAIN two-step check; only actual PASS permits admission and gated dispatch. Preserve Y57 pause and foreign compute.')
            for key in ('blocker','blocker_kind','blocked_from_stage','retry_after','unblock_condition'):
                c.pop(key,None)
            additions=[ref(program),ref(plan_path),frozen['resource_authorization'],ref(__file__)]
            c['evidence'].extend(x for x in additions if x not in c['evidence'])
            records.append(dict(id=cid,previous_plan=previous_plan,current_plan=ref(plan_path),
                previous_goal_sha256=previous_goal,current_goal_sha256=goal,unchanged_tested_files=7,
                code_commit=c['code_commit'],scientific_protocol_changed=False,
                changes='Resource grant/goal binding and deterministic deployment paths only; old artifacts retained.'))
        out=ROOT/('pipeline/checks/server1_prepared_binding_'+a.token+'.json')
        assert not out.exists()
        tick.save(out,dict(complete=True,observed_at=datetime.now().astimezone().isoformat(),records=records,
            goal=ref('pipeline/goal.json'),authorization=ref('pipeline/authorizations/server1_gpu0_20261025.json')))
        for c in work['candidates']:
            if c['id'] in ('AdaRES-Y62','AdaRES-Y63'):c['evidence'].append(ref(out))
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(evidence=ref(out),records=records)))
    finally:lock.close()

if __name__=='__main__':main()
