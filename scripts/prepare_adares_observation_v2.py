"""Extend observation closeout to unchanged, exactly identified parked due holds."""
import argparse,ast,json
from prepare_y60_stage_heads import ROOT,lease,ref,tick
from y102_deployment import Y102,git

DUE='''
def validate_due_holds(view):
    rows=view['research']['shown']
    assert view['research']['omitted_due_count']==0 and len(rows)==view['research']['due_count']
    assert {r['id'] for r in rows}<={'AdaRES-Y57','AdaRES-Y97'}
    work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};q=monitor.load_queue(str(ROOT));result=[]
    for row in rows:
        candidate=c[row['id']];assert row['stage']==candidate['stage']=='blocked'
        if row['id']=='AdaRES-Y57':
            assert candidate['blocker_kind']=='human_authorization' and candidate['blocked_from_stage']=='training'
            task=next(r for r in q['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57')
            assert task['status']=='paused' and task['dispatch_hold'] and task['resume_requires_user_authorization'] is True and str(task['gpu'])=='1'
            result.append(dict(id=row['id'],stage='blocked',reason='unchanged_separate_user_pause',resume_authorized=False,GPU1_used=False))
        else:
            assert candidate['blocker_kind']=='numerical_control_failure' and candidate['failure_scope']=='implementation'
            assert candidate['CPU_failure']['sha256']=='d9875d4734907fca95bd7ae43f731d5af5feed79af3ffe35bf4e9bea2441ca9a'
            closure.artifact(ROOT,candidate['CPU_failure'])
            last=candidate['revision_history'][-1];assert last['revision']==1 and last['result']=='failed'
            for entry in last['evidence']:closure.artifact(ROOT,entry)
            result.append(dict(id=row['id'],stage='blocked',reason='unchanged_original_and_revision1_numerical_failures',original_failure=candidate['CPU_failure'],revision1_evidence=last['evidence'],scientific_tests_replayed=False,reopened=False))
    return result

'''

def prepare(token):
    lease(token,300);assert git('rev-parse','HEAD').decode().strip()==Y102
    source=ROOT/'scripts/finish_adares_zero_budget_observation.py';before=ref(source);body=source.read_text(encoding='utf-8')
    guard=" and view['research']['due_count']==0";assert body.count(guard)==1;body=body.replace(guard,'')
    marker="    assert git('rev-parse','HEAD').decode().strip()==Y102";assert body.count(marker)==1
    body=body.replace(marker,"    due_holds=validate_due_holds(view)\n"+marker)
    marker2="or line.startswith('- Y102 current progress '):";assert body.count(marker2)==1
    body=body.replace(marker2,"or line.startswith('- Y102 current progress ') or line.startswith('- Y102 latest progress '):")
    body=body.replace("snapshot_goal_rc=4,admission_budget=0,","snapshot_goal_rc=4,due_hold_verification=due_holds,admission_budget=0,")
    body=body.replace("': event research_work_changed only, due_count0, phase/admission/creation/reserve budgets0.","': event; due_count'+str(view['research']['due_count'])+' exactly verified parked holds; phase/admission/creation/reserve budgets0.")
    body=body.replace("'本轮event仅research_work_changed，due_count0、无可派发待跑任务", "'本轮event，due_count'+str(view['research']['due_count'])+'项为已核验挂起候选、无可派发待跑任务")
    body=body.replace('def finish(token):',DUE+'def finish(token):')
    ast.parse(body,feature_version=(3,8))
    out=ROOT/'scripts/finish_adares_zero_budget_observation_v2.py';assert not out.exists();out.write_text(body,encoding='utf-8',newline='\n')
    assert ref(source)==before
    manifest=ROOT/('pipeline/checks/adares_observation_v2_'+token+'.json');assert not manifest.exists()
    tick.save(manifest,dict(complete=True,token=token,original_helper=before,successor=ref(out),only_scope='Handle repeated memory prefix and exact unchanged Y57/Y97 due holds; retain zero-budget, original-source and resource checks.',original_helper_preserved=True,scientific_or_numeric_guards_unchanged=True,tests_or_permission_requests_replayed=False,evidence=[ref(__file__),before,ref(out)]))
    print(json.dumps(dict(preparation=ref(manifest),successor=ref(out),original_preserved=True)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);prepare(p.parse_args().token)
