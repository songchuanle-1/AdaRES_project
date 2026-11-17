"""Close only after a failed pre-state observation guard has fresh matching immutable resource/identity evidence."""
import argparse,ast
from prepare_y60_stage_heads import ROOT,lease,ref,tick
from finish_y88_prepared_resource_wait import age
import closure

def finish(token):
    lease(token,300);base=ROOT/'scripts/finish_y88_prepared_resource_wait.py';source=base.read_text(encoding='utf-8')
    identity_ref=ref('pipeline/checks/y88_wait_foreign_identity_'+token+'_reconciled.json');identity=tick.load(closure.artifact(ROOT,identity_ref));assert 0<=age(identity)<180 and identity['compute_pids'] and not identity['remote_mutations']
    refreshed={s:ref('pipeline/checks/gpu_occupation_'+s+'_release_'+token+'.json') for s in ('server-1','server-2')}
    assert all(0<=age(tick.load(closure.artifact(ROOT,r)))<180 for r in refreshed.values())
    assert not (ROOT/('pipeline/checks/y88_resource_wait_'+token+'_y102_diagnostic.json')).exists() and not (ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md')).exists()
    failure=ROOT/('pipeline/checks/y88_resource_observation_guard_recovery_'+token+'.json');assert not failure.exists()
    tick.save(failure,dict(complete=True,token=token,classification='pre_state_operational_observation_guard_failure_not_scientific_or_approval_failure',failed_command='python -B scripts/finish_y88_prepared_resource_wait.py --token '+token,observed_error='AssertionError at line38 fresh identity completeness/absence/timestamp/remote-mutation guard before diagnostic or report writes',original_helper=ref(base),original_identity=ref('pipeline/checks/y88_wait_foreign_identity_'+token+'.json'),original_occupation=ref('pipeline/checks/gpu_occupation_server-1_start_'+token+'.json'),verified_resource_change='Original canonical occupation had PID64587; identity query then observed additional PID395614. Both are independently identified other-project compute. Fresh matching canonical and identity snapshots are now available.',reconciled_identity=identity_ref,reconciled_occupations=refreshed,all_original_checks_unchanged=True,resource_freshness_bound_seconds=300,no_deadline_change=True,diagnostic_memory_report_not_written_by_failed_attempt=True,GPU_jobs_started=False,scientific_tests_replayed=False,approvals_rejected=False,evidence=[ref(base),identity_ref]+list(refreshed.values())))
    failure_ref=ref(failure)
    needle="initial={s:ref('pipeline/checks/gpu_occupation_'+s+'_start_'+token+'.json') for s in ('server-1','server-2')};label='start'";assert source.count(needle)==1;source=source.replace(needle,needle.replace("'_start_'","'_release_'").replace("label='start'","label='release'"))
    needle="identity_path=ROOT/('pipeline/checks/y88_wait_foreign_identity_'+token+'.json')";assert source.count(needle)==1;source=source.replace(needle,"identity_path=ROOT/('pipeline/checks/y88_wait_foreign_identity_'+token+'_reconciled.json')")
    needle='material_change=confirmed,Y102_native_progress_confirmed=';assert source.count(needle)==1;source=source.replace(needle,'material_change=confirmed,operational_guard_failure_preserved=failure_ref,Y102_native_progress_confirmed=')
    needle='evidence=[ref(__file__),ref(diagnostic),identity_ref';assert source.count(needle)==1;source=source.replace(needle,'evidence=[ref(__file__),ref(base),failure_ref,ref(diagnostic),identity_ref')
    needle='tick.save(notes,dict(actions=[';assert source.count(needle)==1;source=source.replace(needle,needle+repr('收尾前一次实际身份/新鲜度保护断言失败，发生于诊断/内存/报告写入之前；随后新增外部GPU进程被确认，已保留旧证据并重新采集一致的不可变占用/身份快照。仅替换当前证据绑定，所有原科学、资源、零容差新鲜度及deadline检查保持，未盲重试同证据或覆盖原产物。')+',')
    needle="'- Closeout '+e(ref(out))";assert source.count(needle)==1;source=source.replace(needle,"'- Pre-state operational observation guard failure preserved '+e(failure_ref)+'; new matching actual resource/identity snapshots bound with all checks unchanged. Closeout '+e(ref(out))")
    namespace={'__name__':'reconciled_y88_resource_wait','__file__':__file__,'base':base,'failure_ref':failure_ref};exec(compile(ast.parse(source),__file__,'exec'),namespace);namespace['finish'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
