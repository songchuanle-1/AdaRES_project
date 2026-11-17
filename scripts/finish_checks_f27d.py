"""Bind execution-check results and preserve operational failure history."""

from datetime import datetime,timezone
import os
from check_ef_execution import ROOT,ref,tick,monitor

TOKEN='f27d10d52a2b4fabadfd1c28afd46c65'
base=ROOT/'pipeline/headless'
owner=tick.load(base/'.foreground.lockdir/owner.json')
assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds()>90
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path='pipeline/checks/ef_execution_'+TOKEN+'_024944.json'
    result=tick.load(ROOT/path)
    assert result['complete'] and result['idle_authorized_gpus']==[]
    assert all(r['current_goal_and_worktree_match'] and r['stale_goal_control_refused'] and
               r['tampered_source_control_refused'] for r in result['checks'])
    assert all(result['resource_logic_controls'][key] for key in (
        'simulated_archived_task_free_card_detected',
        'simulated_uncertain_running_or_claiming_reservation_preserved'))
    decision=monitor.log_decision(str(ROOT),'other',
        '两项最近可入队的Y58/Y59已实现，如何确保空卡后不会用错版本或被旧任务前提卡住？',
        '实测14文件与提交/当前goal/冻结程序一致并拒绝过期goal及篡改源；修复项目检查脚本目录及固定双进程前提，测试归档后空卡识别与未确认槽位保留。',
        '真实查询两卡仍有原Y56/Y57递增训练，无整卡空槽；CPU/逻辑控制不冒充原生GPU后向，不增加训练、准入或重复审批。',path)
    additions=[ref(path),ref(decision),ref(__file__),ref('scripts/check_ef_execution.py'),
        ref('pipeline/analyses/ef_execution_f27d_first_failure.md'),
        ref('pipeline/analyses/check_ef_execution_f27d_initial.py'),
        ref('pipeline/analyses/check_ef_execution_f27d_v1.py'),ref('scripts/archive_ef_checker_f27d.py')]
    work=tick.load(ROOT/'pipeline/research_work.json')
    for c in work['candidates']:
        if c['id'] not in ('AdaRES-Y58','AdaRES-Y59'):continue
        for row in additions:
            if row not in c['evidence']:c['evidence'].append(row)
        c['execution_integrity_check']=ref(path)
        c['next_action']=('Recheck the exact prepared YAML/source/commit/current goal and current authorized '
            'resources (scripts/check_ef_execution.py --token CURRENT_TOKEN). Once a real whole GPU is '
            'available and canonical running/claiming reservations are reconciled, run native_smoke_plan '
            'under the current lease. Require two real finite native TRAIN backward/optimizer steps and '
            'the intended mode path, then admit to the existing queue and try gated dispatch this same tick. '
            'Do not require the original training pair to remain live, old efficacy diagnosis or new routine consent.')
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Bound current resource/integrity checks, source-history preservation and actionable native-smoke next steps; queue unchanged.')
finally:
    lock.close()
