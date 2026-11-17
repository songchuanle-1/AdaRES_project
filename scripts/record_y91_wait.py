"""Record bounded Y91 preparation and fresh resource holds without changing runs."""
import argparse
import ast
from datetime import datetime, timezone
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y91_contract import validate_cpu
from y88_deployment import Y88
import closure
import gpu_occupation


def record(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y91')
    assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
    assert candidate['required_server'] == 'server-2' and candidate['required_gpu'] == 0
    assert not candidate.get('admission_receipt') and not candidate.get('native_smoke_result')
    validate_cpu(tick.load(closure.artifact(ROOT, candidate['CPU_result'])))
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == Y88
    source = tick.load(closure.artifact(ROOT, candidate['source_manifest']))
    assert source['main_index_unchanged'] and source['earlier_Y89_Y90_preserved']
    assert source['no_untested_parent_candidate_included']
    assert len(source['exact_delta']) == 7 and all(row[0] == 'A' for row in source['exact_delta'])
    assert closure.git(ROOT, 'rev-parse', candidate['code_commit'] + '^') == Y88
    earlier = {}
    for cid, commit in (('AdaRES-Y89', '5f1c4bde4b682fabd84fdc84a4341ba2c845fe71'),
                        ('AdaRES-Y90', '9a64d22c41bb8527ca7ca1abba4340546506b1b1')):
        prior = next(item for item in work['candidates'] if item['id'] == cid)
        assert prior['code_commit'] == commit and prior['stage'] == 'blocked'
        assert not prior.get('admission_receipt') and not prior.get('native_smoke_result')
        closure.artifact(ROOT, prior['source_manifest'])
        for key in ('native_smoke_plan_server1', 'native_smoke_plan_server2_gpu0'):
            closure.artifact(ROOT, prior[key])
        earlier[cid] = prior['source_manifest']
    plans = [tick.load(closure.artifact(ROOT, candidate[key]))
             for key in ('native_smoke_plan_server1', 'native_smoke_plan_server2_gpu0')]
    assert {plan['server'] for plan in plans} == {'server-1', 'server-2'}
    assert all(plan['gpu'] == 0 and plan['native_executed'] is False for plan in plans)
    for plan in plans:
        frozen = ast.literal_eval(ast.parse(
            closure.artifact(ROOT, plan['program']).read_text(encoding='utf-8')).body[0].value)
        assert frozen['variant']['id'] == candidate['id'] and frozen['runtime_physical_gpu'] == 0
    tests = ROOT / ('pipeline/checks/y91_final_preparation_guards_' + token + '.log')
    out = ROOT / ('pipeline/checks/y91_preparation_closeout_' + token + '.json')
    notes = ROOT / ('pipeline/reports/notes_' + token + '.json')
    backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + token + '.md')
    assert not any(path.exists() for path in (tests, out, notes, backup))
    response = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y91_*.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    tests.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and b'Ran 16 tests' in tests.read_bytes()
    queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
    qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
    active, occupations = {}, {}
    now = datetime.now(timezone.utc)
    for name, cid, pid in (('server-1', 'AdaRES-Y88', 3527278), ('server-2', 'AdaRES-Y87', 26281)):
        task = next(task for task in queue['gpu_queues'][name] if task['id'] == cid)
        assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid and str(task['gpu']) == '0'
        assert running[name]['0'] == cid
        closure.admission(ROOT, task)
        current = next(current for current in work['candidates'] if current['id'] == cid)
        value = tick.load(closure.artifact(ROOT, current['latest_training_observation']))
        assert value['native_progress_confirmed'] is True
        assert value['token'] == token
        assert 0 <= (now - datetime.fromisoformat(value['observed_local'])).total_seconds() < 300
        active[cid] = dict(server=name, pid=pid, phase=value['after']['latest'],
                           evidence=current['latest_training_observation'])
        path = ROOT / ('pipeline/checks/gpu_occupation_' + name + '_finish_' + token + '.json')
        occupation = tick.load(path)
        assert occupation['token'] == token and occupation['scope']['server'] == name
        assert occupation['complete'] is True and occupation['read_only'] is True
        assert occupation['scope']['allowed_gpus'] == [0]
        assert 0 <= (now - datetime.fromisoformat(occupation['observed_at'])).total_seconds() < 300
        state = gpu_occupation._raw_state(occupation, {0}, queue, name)
        assert state['busy_gpu_ids'] == [0] and not state['idle_gpu_ids']
        assert state['reserved_gpu_ids'] == [0]
        assert pid in state['compute_pids'][state['inventory'][0]['uuid']]
        occupations[name] = ref(path)
    paused = next(task for task in queue['gpu_queues']['server-2'] if task['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
    backup.write_bytes((ROOT / 'pipeline/codex_memory.md').read_bytes())
    tick.save(out, dict(complete=True, id=candidate['id'], token=token, observed_at=now.isoformat(),
        CPU_result=candidate['CPU_result'], CPU_version=candidate['source_manifest'],
        native_preparation=candidate['native_preparation'], guard_tests=ref(tests), guard_tests_passed=16,
        active=active, occupations=occupations, earlier_preparations=earlier, earlier_Y89_Y90_preserved=True,
        new_candidates=1, admissions=0, launches=0, archives=0, native_GPU_executed=False,
        formal_frozen_FIRST_updated=False, model_revision=False, production_HEAD_preserved=Y88,
        GPU1_paused=True, Y57_separately_paused=True, previous_memory=ref(backup), helper=ref(__file__)))
    candidate['evidence'].extend([ref(out), ref(tests), ref(__file__), ref('scripts/run_y91_native.py'),
        ref('scripts/test_y91_runtime_plan.py'), ref('scripts/run_y90_native.py')] + list(occupations.values()))
    work['tick_attempt'] = dict(
        tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=candidate['id'], kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y91_cpu ' + candidate['CPU_program']['path'] + ' --token ' + token + ' --seconds 480',
        observation='Actual CPU original YAML/full model/released backbone/TRAIN loading, independent scalar/bounded/no-RNG/permutation controls, seven-stage W2/input-gradient anchors, confidence-only intervention/bypass, fourteen original hidden/evidence consumers and two synthetic updates passed first execution. Exact independent seven-file version, matching GPU0-only plans and16 final guards passed. Both GPUs busy; no native GPU/admission/efficacy.',
        evidence=[candidate['CPU_result'], ref(out), ref(tests)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    tick.save(notes, dict(actions=[
        '核验实际所属会话、foreground模式、Disabled后台调度和最新GPU1暂停；仅AdaRES，科学goal原字节、运行收据、数据/私有运行环境/功耗保持。',
        '先观察Y87/Y88并保留原运行，收尾前再核验源码/YAML/收据/功耗及原生步数递增。Y87=' + str(active['AdaRES-Y87']['phase']) + '；Y88=' + str(active['AdaRES-Y88']['phase']) + '。begin已尝试既有队列派发，无就绪waiting。',
        '新增独立Y91，仅对原共享置信度MLP内第二隐状态乘第一隐状态条件门控；原五层/共享方式、竞争统计/证据、查询/掩码/分数/目标/损失/selector及固定seed1999/30轮保持。未组合或重开旧路线。',
        'Y91首次实际CPU原生YAML/完整模型/发布backbone/原TRAIN162738点8表达加载，float32/64标量门控/有界/无RNG/置换及输入参数梯度控制通过。',
        '七阶段W2输出严格相同、输入梯度锚点通过；非零干预只改变置信度且旁路精确恢复；十四次原隐状态/门控/最终线性层和原证据消费者、两次有限梯度及参数更新通过。仅CPU合成decoder，不冒充GPU任务。',
        '只将七个实测Y91文件独立保存为Y88的子版本，生产HEAD/index/原运行及Y89/Y90原版本保持；两套匹配授权GPU0原生计划与16项准备/运行守卫通过。',
        'canonical物理证据确认server1GPU0由Y88、server2GPU0由Y87占用，Y89/Y90/Y91继续等待独占空槽，未GPU原生、未准入或启动；GPU1、外部任务、Y57及调度不动。'],
        decisions=[
        '正式冻结FIRST仍Y36=0.510725438294，目标0.5150；无新正式FIRST/Acc/成本或晋升，CPU正响应和有限梯度不代表训练收益。',
        '本轮只准备一个有实际检查结果的独立候选，不凑额度。Y89、Y90先于Y91，须授权匹配GPU0真实空闲、原属运行完整checkpoint核验并归档后执行原生两步检查，通过才准入及派发。',
        '保留29个fixed30模型的全量原冻结后评及同设备成本协议；GPU1后评仍暂停不迁移、Y57单独暂停、Y86数值实现挂起不重放或弱化控制。',
        '若候选执行前生产HEAD已推进，只允许保持七个实测文件逐字节一致的显式加法版本绑定，不reset、不混合机制、不带未通过候选部署。']))
    assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
    print(json.dumps(dict(closeout=ref(out), notes=ref(notes), previous_memory=ref(backup),
                         tests=16, active=active, native_GPU_executed=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        record(args.token)
    finally:
        writer.close()
