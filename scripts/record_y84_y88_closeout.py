"""Record only verified current-tick outcomes before the official foreground finish."""
import argparse
from datetime import datetime, timedelta
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y88_contract import validate_result
from advance_y88 import prepared
from y88_deployment import Y87, Y88
import closure


def record(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate, _, _, frozen = prepared(work)
    assert candidate['stage'] == 'training'
    raw = tick.load(closure.artifact(ROOT, candidate['native_smoke_result']))
    validate_result(raw, candidate, frozen)
    queue = monitor.load_queue(str(ROOT))
    running = monitor.load_running(str(ROOT))
    active = {}
    for name, cid in (('server-1', 'AdaRES-Y88'), ('server-2', 'AdaRES-Y87')):
        task = next(task for task in queue['gpu_queues'][name] if task['id'] == cid)
        assert task['status'] == 'running' and task['launch_verified'] and str(task['gpu']) == '0'
        assert running[name]['0'] == cid
        closure.admission(ROOT, task)
        current = next(item for item in work['candidates'] if item['id'] == cid)
        observation = tick.load(closure.artifact(ROOT, current['latest_training_observation']))
        assert observation['native_progress_confirmed']
        active[cid] = dict(server=name, pid=task['pid'], phase=observation['after']['latest'],
                           evidence=current['latest_training_observation'])
    paused = next(task for task in queue['gpu_queues']['server-2'] if task['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    done = monitor.load_experiment(str(ROOT), 'AdaRES-Y84')
    assert done['status'] == 'done_complete' and str(done['gpu']) == '0'
    terminal = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
    assert terminal['completed_epochs'] == {phase: list(range(1, 31)) for phase in ('Train', 'Val')}
    assert terminal['checkpoints'][0]['epoch'] == terminal['checkpoints'][0]['scheduler_last_epoch'] == 30
    assert all(row['model_finite'] and row['optimizer_finite'] and row['optimizer_state_count'] > 0
               for row in terminal['checkpoints'])
    current = next(item for item in work['candidates'] if item['id'] == 'AdaRES-Y84')
    assert current['required_server'] == 'server-1' and current['required_gpu'] == 0
    current.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        resource_work_kind='frozen_evaluation',
        blocker='Exact original Y84 fixed30 checkpoints were verified and canonically archived; original full frozen server1GPU0/350W review waits for the progressing independent Y88 reservation. No resource substitution.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        unblock_condition='Fresh exclusive idle/unreserved original server1GPU0/350W after exact current owner terminal verification/archive.',
        next_action='Preserve original Y84 last30/best/model/optimizer/scheduler, receipt and complete frozen protocol. Run only its original full paired FIRST/cost review on original server1GPU0/350W when exclusive. No restart or promotion from training best.')
    assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == Y88
    source = ROOT / ('pipeline/checks/adares-y88_source_selection_' + token + '.json')
    selection = tick.load(source)
    assert selection['other_index_entries_preserved'] and selection['parent'] == Y87
    out = ROOT / ('pipeline/checks/y84_archive_y88_launch_' + token + '.json')
    notes = ROOT / ('pipeline/reports/notes_' + token + '.json')
    backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + token + '.md')
    assert not any(path.exists() for path in (out, notes, backup))
    backup.write_bytes((ROOT / 'pipeline/codex_memory.md').read_bytes())
    helper_repair = dict(first_helper=ref('scripts/create_y88_execution_helpers.py'),
        repaired_helper=ref('scripts/create_y88_execution_helpers_v2.py'),
        first_failure='ValueError: template had no def activate boundary; failed before writing generated files.',
        fix='Use the actual __main__ boundary only. Original failed helper retained; no model/YAML/native guard revision.')
    dependencies = [current['terminal_evidence'], current['terminal_archive_binding'],
        candidate['source_manifest'], candidate['native_preparation'], candidate['native_smoke_result'],
        ref('pipeline/admissions/AdaRES-Y88.json'), candidate['post_training_freeze'],
        ref(source), ref('pipeline/checks/adares-y88_admission_guards_' + token + '.log'),
        ref('pipeline/checks/adares-y88_dispatch_' + token + '.json')]
    for dependency in dependencies:
        closure.artifact(ROOT, dependency)
    tick.save(out, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
        original_Y84=done['terminal_evidence'], Y84_training_best_only=terminal['checkpoints'][0]['best_miou'],
        active=active, Y88_native_losses=raw['losses'], Y88_native_peak_reserved_bytes=raw['peak_reserved_bytes'],
        new_candidates=1, admissions=1, launches=1, archives=1, formal_frozen_FIRST_updated=False,
        GPU1_paused=True, Y57_separately_paused=True, original_data_runtime_paths_power_protocol_retained=True,
        failed_helper_and_repair_preserved=helper_repair, previous_memory=ref(backup),
        evidence=dependencies + [ref(__file__)]))
    current['evidence'].append(ref(out))
    candidate['evidence'].extend([ref(out), ref(__file__), helper_repair['first_helper'], helper_repair['repaired_helper']])
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=candidate['id'], kind='native_smoke', result='passed',
        command='python -B scripts/run_y88_server1_native.py --token ' + token,
        observation='Actual native TRAIN two-step PASS, fifteen admission rejection tests, exact tested source selection/full frozen protocol/receipt/admission/canonical dispatch and increasing native counters completed; Y84 exact fixed30 archive verified. No formal FIRST efficacy update.',
        evidence=[candidate['native_smoke_result'], ref(out)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    phase = active['AdaRES-Y88']['phase']
    prior = active['AdaRES-Y87']['phase']
    tick.save(notes, dict(actions=[
        '核验本会话归属及当前资源暂停；原科学goal字节、数据、私有环境、功耗和调度保持。',
        'Y84原始20261105_032303已自然完成：Train/Val1..30、last30/best及有限完整model/optimizer/scheduler稳定核验，通过25项终态守卫后按原server1GPU0归档；训练best=' + str(terminal['checkpoints'][0]['best_miou']) + '，不是冻结FIRST。',
        '新Y88独立mask-only quadratic读出：CPU原生YAML/完整模型/原TRAIN加载、零锚点/输入梯度/标量置换对照、七个原consumer和两步合成CPU更新通过；只保存七个实测文件。',
        'Y88原server1GPU0/350W原生TRAIN两步通过，losses=' + str(raw['losses']) + '，15项准入守卫通过；完整冻结后评协议、准入收据、安全源码选择、原生派发和步数增长已核实。',
        'Y88最新 ' + str(phase) + '；Y87最新 ' + str(prior) + '。GPU1全部使用和Y57单独实验仍暂停。',
        '执行helper首版仅因不存在的模板边界报错，尚未写生成文件；v2只修边界，原失败helper保留。模型/YAML/科学控制没有修订。'],
        decisions=[
        '冻结正式FIRST保持Y36=0.510725438294，目标0.5150；本轮训练best和两批原生检查均不能替代正式指标或宣称有效。',
        '29个fixed30模型等待原完整冻结FIRST/成本；原GPU1路径继续暂停，不迁移、不重训；Y84原server1GPU0后评等待Y88真实占用释放。',
        'Y86仍为原numerical_control_failure实现挂起，保留失败与诊断，不重放原失败检查、不弱化冻结守卫、不宣称机制否决。',
        '后续先核验Y87/Y88真实进展和既有任务，再按快照补有用独立储备；维持server1GPU0/350W、server2GPU0/280W，一卡一独立实验。']))
    print(json.dumps(dict(closeout=ref(out), notes=ref(notes), previous_memory=ref(backup), active=active,
                          Y84_training_best_only=terminal['checkpoints'][0]['best_miou'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 120)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        record(args.token)
    finally:
        lock.close()
