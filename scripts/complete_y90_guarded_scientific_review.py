"""Accept only the resource-guarded frozen cost, then apply Y90's unchanged prespecified science."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,tick
import closure

VERIFY=r'''
    assert job['attempt']==3 and c['cost_resource_recovery_status']=='running_awaiting_guarded_cost_postaudit'
    recovery=tick.load(closure.artifact(ROOT,c['frozen_trained_cost_plan']));assert recovery['original_scientific_core_byte_identical'] and recovery['worker_body_byte_identical']
    prior=closure.artifact(ROOT,recovery['original_guarded_program']).read_text(encoding='utf-8');current=program.read_text(encoding='utf-8');oldp=ast.literal_eval(ast.parse(prior).body[0].value)
    assert current.split('\n',1)[1]==prior.split('\n',1)[1] and {k for k in oldp if oldp[k]!=p[k]}=={'token','folder','output'}
    assert p['core']==oldp['core'] and p['imports']==oldp['imports'] and p['checkpoint_audit']==oldp['checkpoint_audit'] and p['capture_files']==oldp['capture_files']
    assert len(c['frozen_trained_cost_attempts'])==3 and not c['frozen_trained_cost_attempts'][0]['scientifically_valid_cost'] and c['frozen_trained_cost_attempts'][1]['status']=='guard_refused_before_remote_creation'
    guard_path=root/'resource_guard.jsonl';samples=[json.loads(line) for line in guard_path.read_text(encoding='utf-8').splitlines() if line.strip()];assert len(samples)>=4
    for row in samples:
        assert row['passed'] is True and row['rc']==0 and row['own_pid']==job['pid']
        assert row['compute_pids']==[int(v) for v in row['stdout'].splitlines() if v.strip()]
        assert all(pid==job['pid'] for pid in row['compute_pids'])
    times=[datetime.fromisoformat(row['observed_at']) for row in samples];assert all(b>=a for a,b in zip(times,times[1:]))
    owned=sum(job['pid'] in row['compute_pids'] for row in samples);assert owned>0
    assert audit['resource_guard_samples_complete'] and audit['no_unowned_GPU0_compute_observed'] and audit['resource_guard_samples']==len(samples) and audit['resource_guard_owned_samples']==owned and audit['resource_guard_log_sha256']==ref(guard_path)['sha256']
'''

def complete(token):
    lease(token,600)
    base=(ROOT/'scripts/complete_y101_scientific_review.py').read_text(encoding='utf-8')
    needle="'worker.log','state.json','launch.json')";assert base.count(needle)==1;base=base.replace(needle,"'worker.log','state.json','launch.json','resource_guard.jsonl')")
    needle="    cost=tick.load(root/'trained_cost.json');audit=tick.load(root/'postaudit.json')";assert base.count(needle)==1;base=base.replace(needle,needle+'\n'+VERIFY)
    needle='full_cost_and_postaudit_complete=True,';assert base.count(needle)==1;base=base.replace(needle,'full_cost_and_postaudit_complete=True,resource_exclusivity_verified_from_guard=True,resource_guard_samples=len(samples),resource_guard_owned_samples=owned,resource_guard_log=ref(guard_path),original_contended_cost_still_unaccepted=True,')
    needle="job['launch'],ref(meta)";assert base.count(needle)==1;base=base.replace(needle,"job['launch'],ref(guard_path),recovery['original_prepared_plan'],ref('scripts/complete_y90_scientific_review.py'),ref(meta)")
    needle="        c['frozen_trained_cost_job']['complete']=True;";assert base.count(needle)==1;base=base.replace(needle,"        c['cost_resource_recovery_status']='complete_guarded_cost_and_scientific_review';c['frozen_trained_cost_job']['accepted_scientific_cost']=True;c['frozen_trained_cost_attempts'][2].update(status='complete_guarded_cost_postaudit_verified',scientifically_valid_cost=True,disposition=ref(review),resource_guard_log=ref(guard_path));c['evidence'].append(ref(guard_path))\n"+needle)
    outer=(ROOT/'scripts/complete_y90_scientific_review.py').read_text(encoding='utf-8')
    needle="source=(ROOT/'scripts/complete_y101_scientific_review.py').read_text(encoding='utf-8')";assert outer.count(needle)==1;outer=outer.replace(needle,'source=GUARDED_SOURCE')
    namespace={'__name__':'y90_guarded_completion','__file__':__file__,'GUARDED_SOURCE':base};exec(compile(ast.parse(outer),__file__,'exec'),namespace);namespace['complete'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);complete(parser.parse_args().token)
