"""Bind the Y63 diagnostic to server-1's observed, unchanged 350W limit."""
import argparse,ast,copy,json,shlex
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from check_ef_execution import validate_frozen
import closure

p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
lock=monitor.acquire_lock(str(ROOT));assert lock is not None
try:
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server)
    probe="import subprocess; assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'; print(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())"
    power=float(monitor.ssh(server['host'],server['user'],'python3 -B -c '+shlex.quote(probe)))
    assert power==350
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y63')
    old_plan_ref=c['native_smoke_plan'];old_plan=tick.load(closure.artifact(ROOT,old_plan_ref))
    old_program=closure.artifact(ROOT,old_plan['program']);source=old_program.read_text(encoding='utf-8')
    frozen=ast.literal_eval(ast.parse(source).body[0].value)
    assert validate_frozen(c,old_plan,frozen)==7
    body='\n'.join(source.splitlines()[1:])+'\n'
    old="'--format=csv,noheader,nounits']).decode().strip()) == 280"
    assert body.count(old)==1
    body=body.replace(old,"'--format=csv,noheader,nounits']).decode().strip()) == FROZEN['runtime_power_limit_watts']")
    frozen['runtime_power_limit_watts']=power
    prefix='pipeline/checks/adares_y63_server1_power350_'+a.token
    program=ROOT/(prefix+'_program.py');plan_path=ROOT/(prefix+'_plan.json');out=ROOT/(prefix+'_binding.json')
    assert not any(p.exists() for p in (program,plan_path,out))
    program.write_text('FROZEN='+repr(frozen)+'\n'+body,encoding='utf-8',newline='\n')
    plan=copy.deepcopy(old_plan);plan.update(program=ref(program),previous_plan=old_plan_ref,runtime_power_limit_watts=power)
    tick.save(plan_path,plan);assert validate_frozen(c,plan,frozen)==7
    tick.save(out,dict(complete=True,observed_at=datetime.now().astimezone().isoformat(),previous_plan=old_plan_ref,
        current_plan=ref(plan_path),failure_log=ref('pipeline/analyses/adares_y63_server1_smoke_'+a.token+'.log'),
        failure_scope='Runtime resource guard inherited server-2 280W; before model/data/backward execution.',
        observed_server1_power=power,hardware_changed=False,model_and_YAML_changed=False,
        limitation='This changes only the frozen host-specific guard to the observed authorized server-1 default, not a scientific or numeric model gate.'))
    c['native_smoke_plan']=ref(plan_path);c['evidence'].extend([ref(program),ref(plan_path),ref(out),ref(__file__)])
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(evidence=ref(out),plan=ref(plan_path))))
finally:lock.close()
