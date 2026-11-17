"""Preserve the observed metadata failure and add a scoped closeout correction."""
import argparse,ast,json
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y102_deployment import git,Y102

def prepare(token):
    lease(token,300);writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert git('rev-parse','HEAD').decode().strip()==Y102
        paths=('scripts/y102_deployment.py','scripts/finish_y102_launch_tick.py','pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml','pipeline/admissions/AdaRES-Y102.json')
        before={p:ref(p) for p in paths};index=git('ls-files','--stage','-z')
        out=ROOT/('pipeline/checks/y102_observer_mapping_failure_'+token+'.json');assert not out.exists()
        from y102_observer_v2 import OWNERS,corrected_observer
        assert OWNERS['AdaRES-Y101']==('server-1','6fb26984ae652efbba177ab20c39575c44a69517',350.) and OWNERS['AdaRES-Y102']==('server-2',Y102,280.)
        corrected_observer()
        tick.save(out,dict(complete=True,token=token,recorded_at=datetime.now().astimezone().isoformat(),failure_scope='operational_observer_mapping',actual_command='python -B scripts/finish_y102_launch_tick.py --token '+token,failed_child_command='python -B scripts/y102_deployment.py observe --id AdaRES-Y101 --token '+token+' --label release',actual_error="AssertionError at y102_deployment.py line80: assert (task['server'], str(task['gpu']), task['code_commit']) == (name, '0', commit)",reason='Template observer retained server-2 for original Y101 and server-1 for new Y102. Actual queue/receipts remain Y101/server-1 and Y102/server-2.',failure_before_remote_activation=True,original_helpers=[before[p] for p in paths[:2]],corrected_observer=ref('scripts/y102_observer_v2.py'),no_training_restart=True,no_native_or_scientific_replay=True,old_receipts_and_terminal_proof_preserved=True))
        source=(ROOT/'scripts/finish_y102_launch_tick.py').read_text(encoding='utf-8')
        source=source.replace('from y102_deployment import Y100,Y101,Y102,git,require_running_parent_handoff','from y102_observer_v2 import Y100,Y101,Y102,git,require_running_parent_handoff')
        source=source.replace("['scripts/y102_deployment.py','observe'","['scripts/y102_observer_v2.py','observe'")
        source=source.replace('Current observer=y102_deployment.py','Current observer=y102_observer_v2.py')
        source=source.replace('preserved_observer_failure=failedobserver,','preserved_observer_failure=failedobserver,preserved_mapping_failure=ref(\'pipeline/checks/y102_observer_mapping_failure_\'+token+\'.json\'),')
        source=source.replace('guards,failedobserver]+','guards,failedobserver,ref(\'pipeline/checks/y102_observer_mapping_failure_\'+token+\'.json\')]+')
        source=source.replace("'- Next tick first observe Y101/Y102", "'- Original closeout observer had reversed server mapping, failed before SSH/source writes; original helpers/failure retained '+evidence(ref('pipeline/checks/y102_observer_mapping_failure_'+token+'.json'))+'; y102_observer_v2.py corrects only physical mapping and verifies actual current source/runtime/power/increasing counters. No native/archive/admission/scientific replay.',\n            '- Next tick first observe Y101/Y102")
        source=source.replace("'Y102首次真实TRAIN两步有限loss/backward及新旧路径检查通过；", "'收尾观察模板的服务器映射反转导致的失败及原工具已保留，v2仅纠正观察映射并以两条实际训练验证；不重跑科学检查/准入/归档。Y102首次真实TRAIN两步有限loss/backward及新旧路径检查通过；")
        target=ROOT/'scripts/finish_y102_launch_tick_v2.py';assert not target.exists();ast.parse(source,feature_version=(3,8));target.write_text(source,encoding='utf-8',newline='\n')
        work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y102');candidate['evidence'].extend([ref(out),ref('scripts/y102_observer_v2.py'),ref(__file__)])
        tick.save(ROOT/'pipeline/research_work.json',work)
        assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
        print(json.dumps(dict(preserved_failure=ref(out),corrected_closeout=ref(target),original_helpers_preserved=True,actual_observations_pending=True)))
    finally:writer.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);prepare(p.parse_args().token)
