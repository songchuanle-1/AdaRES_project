"""Add immutable GPU1 review holds and original Y86 failure to bounded observation."""
import argparse,ast,json
from prepare_y60_stage_heads import ROOT,lease,ref,tick
from y102_deployment import Y102,git

DUE='''
def validate_due_holds(view):
    frozen=tick.load(ROOT/'pipeline/headless/.tick_context.json')
    ids=frozen['research_work']['due_ids'];shown=[r['id'] for r in view['research']['shown']]
    assert len(ids)==view['research']['due_count'] and ids[:len(shown)]==shown and len(ids)-len(shown)==view['research']['omitted_due_count']
    gpu1={'AdaRES-Y67','AdaRES-Y70','AdaRES-Y73','AdaRES-Y75','AdaRES-Y78','AdaRES-Y81'}
    assert set(ids)<=gpu1|{'AdaRES-Y57','AdaRES-Y97','AdaRES-Y86'}
    work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};q=monitor.load_queue(str(ROOT));result=[]
    assert monitor.load_servers(str(ROOT))['server-2']['allowed_gpus']==[0]
    assert ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
    for cid in ids:
        candidate=c[cid];assert candidate['stage']=='blocked'
        if cid=='AdaRES-Y57':
            assert candidate['blocker_kind']=='human_authorization' and candidate['blocked_from_stage']=='training'
            task=next(r for r in q['gpu_queues']['server-2'] if r['id']==cid)
            assert task['status']=='paused' and task['dispatch_hold'] and task['resume_requires_user_authorization'] is True and str(task['gpu'])=='1'
            result.append(dict(id=cid,stage='blocked',reason='unchanged_separate_user_pause',resume_authorized=False,GPU1_used=False))
        elif cid=='AdaRES-Y97':
            assert candidate['blocker_kind']=='numerical_control_failure' and candidate['failure_scope']=='implementation'
            assert candidate['CPU_failure']['sha256']=='d9875d4734907fca95bd7ae43f731d5af5feed79af3ffe35bf4e9bea2441ca9a'
            closure.artifact(ROOT,candidate['CPU_failure'])
            last=candidate['revision_history'][-1];assert last['revision']==1 and last['result']=='failed'
            for entry in last['evidence']:closure.artifact(ROOT,entry)
            result.append(dict(id=cid,stage='blocked',reason='unchanged_original_and_revision1_numerical_failures',original_failure=candidate['CPU_failure'],revision1_evidence=last['evidence'],scientific_tests_replayed=False,reopened=False))
        elif cid=='AdaRES-Y86':
            assert candidate['blocker_kind']=='numerical_control_failure' and candidate['failure_scope']=='implementation' and candidate['blocked_from_stage']=='validation'
            assert candidate['native_smoke_failure']['sha256']=='8382c924aa1297e5836d53af7c46517917336e269d8ff3656fbaeab547a84b6c'
            closure.artifact(ROOT,candidate['native_smoke_failure']);assert not candidate.get('native_smoke_result')
            result.append(dict(id=cid,stage='blocked',reason='unchanged_original_native_numerical_failure',failure=candidate['native_smoke_failure'],native_replayed=False,reopened=False))
        else:
            assert cid in gpu1 and candidate['blocker_kind']=='human_authorization' and candidate['blocked_from_stage']=='validation'
            assert candidate['required_server']=='server-2' and candidate['required_gpu']==1 and candidate['resource_work_kind']=='frozen_evaluation'
            assert candidate['scientific_status']=='fixed30_complete_frozen_review_pending'
            plan=tick.load(closure.artifact(ROOT,candidate['post_training_freeze']))
            assert plan['complete'] is True and plan['id']==cid
            condition=plan['resource_condition'];assert condition['server']=='server-2' and condition['gpu']==1 and condition['power_limit_watts']==390.
            terminal=tick.load(closure.artifact(ROOT,candidate['terminal_evidence']));assert terminal['complete'] is True and terminal['id']==cid
            receiptref=ref('pipeline/admissions/'+cid+'.json');receipt=tick.load(closure.artifact(ROOT,receiptref))
            assert receipt['id']==cid and receipt['post_training_plan']==candidate['post_training_freeze']
            result.append(dict(id=cid,stage='blocked',reason='unchanged_explicit_GPU1_all_use_pause',original_resource=dict(server='server-2',gpu=1,power_limit_watts=390.),freeze=candidate['post_training_freeze'],original_terminal=candidate['terminal_evidence'],original_receipt=receiptref,artifacts_hash_verified=True,remote_checkpoints_reloaded=False,archive_replayed=False,evaluation_or_cost_executed=False,migrated=False,reauthorized=False))
    return result
'''

def prepare(token):
    lease(token,300);assert git('rev-parse','HEAD').decode().strip()==Y102
    source=ROOT/'scripts/finish_adares_zero_budget_observation_v2.py';original=ref(source);tree=ast.parse(source.read_text(encoding='utf-8'))
    replaced=0
    for i,node in enumerate(tree.body):
        if isinstance(node,ast.FunctionDef) and node.name=='validate_due_holds':tree.body[i]=ast.parse(DUE).body[0];replaced+=1
    assert replaced==1
    body=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(body,feature_version=(3,8))
    out=ROOT/'scripts/finish_adares_zero_budget_observation_v3.py';assert not out.exists();out.write_text(body,encoding='utf-8',newline='\n');assert ref(source)==original
    manifest=ROOT/('pipeline/checks/adares_observation_v3_'+token+'.json');assert not manifest.exists()
    tick.save(manifest,dict(complete=True,token=token,original_helper=original,successor=ref(out),scope='Read exact frozen due IDs, preserve separate Y57 pause, Y97 failures, Y86 native failure, and original six GPU1/390W frozen reviews. No original checkpoint loading, replay or resource migration.',original_helpers_preserved=True,scientific_or_numeric_guards_unchanged=True,evidence=[ref(__file__),original,ref(out)]))
    print(json.dumps(dict(preparation=ref(manifest),successor=ref(out),original_preserved=True)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);prepare(p.parse_args().token)
