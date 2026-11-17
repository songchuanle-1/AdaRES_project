"""Record a fresh, quiet resource check for the already prepared Y79/Y80 backlog."""
import argparse
import ast
from datetime import datetime, timezone
import json
import subprocess
from prepare_y79_query_language_feedback import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def record(token):
    lease(token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
        context = tick.load(ROOT / 'pipeline/headless/.tick_context.json')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        get = lambda cid: next(c for c in work['candidates'] if c['id'] == cid)
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == '5eee1e50c9e1d65bf113a4b8757bfc27d028a821'
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        tasks = {task['id']: task for rows in queue['gpu_queues'].values() for task in rows}
        observations, phases = {}, {}
        for cid, name, gpu, pid in [('AdaRES-Y76', 'server-1', '0', 3489002), ('AdaRES-Y77', 'server-2', '0', 111334), ('AdaRES-Y78', 'server-2', '1', 22192)]:
            task = tasks[cid]
            assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid and running[name][gpu] == cid
            closure.admission(ROOT, task)
            evidence = get(cid)['latest_training_observation']
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token and value['native_progress_confirmed'] and value['pid'] == pid
            observations[cid], phases[cid] = evidence, value['after']['latest']
        assert tasks['AdaRES-Y57']['status'] == 'paused' and tasks['AdaRES-Y57']['dispatch_hold'] and tasks['AdaRES-Y57']['resume_requires_user_authorization']
        assert 'AdaRES-Y75' not in tasks and monitor.load_experiment(str(ROOT), 'AdaRES-Y75')['status'] == 'done_complete'
        occupations = {}
        for name in ('server-2', 'server-1'):
            evidence = ref('pipeline/checks/gpu_occupation_' + name + '_finish_' + token + '.json')
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token and (datetime.now(timezone.utc) - datetime.fromisoformat(value['observed_at'])).total_seconds() < 300
            candidate = next(c for c in work['candidates'] if c.get('stage') == 'blocked' and c.get('blocker_kind') == 'exclusive_gpu_slot' and c.get('required_server') == name)
            server = next(s for s in context['queue']['servers'] if s['name'] == name)
            assert not gpu_occupation.validated_state(ROOT, context, candidate, server, queue)['idle_gpu_ids']
            occupations[name] = evidence
        retained = {}
        for cid in ('AdaRES-Y79', 'AdaRES-Y80'):
            candidate = get(cid)
            assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot' and candidate['implementation_test_passed']
            for field in ('source_manifest', 'CPU_native_observer_verification', 'native_preparation'):
                closure.artifact(ROOT, candidate[field])
            fields = ['native_smoke_plan_server2_gpu0', 'native_smoke_plan_server2_gpu1']
            if cid == 'AdaRES-Y79':
                fields.append('native_smoke_plan_server1')
            retained[cid] = {}
            for field in fields:
                evidence = candidate[field]
                plan = tick.load(closure.artifact(ROOT, evidence))
                frozen = ast.literal_eval(ast.parse(closure.artifact(ROOT, plan['program']).read_text(encoding='utf-8')).body[0].value)
                assert validate_frozen(candidate, plan, frozen) == 7 and not plan['native_executed']
                retained[cid][field] = evidence
        y76, y79 = get('AdaRES-Y76'), get('AdaRES-Y79')
        terminal_plan = tick.load(closure.artifact(ROOT, y76['terminal_handoff_plan']))
        assert terminal_plan['CPU_guard_tests_passed'] == 20 and not terminal_plan['checkpoint_loaded'] and not terminal_plan['archived']
        for evidence in terminal_plan['dependencies'] + [terminal_plan['admission'], terminal_plan['post_training_freeze']]:
            closure.artifact(ROOT, evidence)
        cpu = tick.load(closure.artifact(ROOT, y79['CPU_server1_runtime_preflight']))
        assert cpu['complete'] and cpu['implementation_test_passed'] and not cpu['CUDA_initialized'] and not cpu['native_smoke_passed']
        verification = tick.load(closure.artifact(ROOT, y79['server1_runtime_verification']))
        assert verification['actual_CPU_result'] == y79['CPU_server1_runtime_preflight'] and verification['plan'] == y79['native_smoke_plan_server1']
        assert verification['original_callback_functions_exact'] == 16 and verification['inactive'] is True
        out = ROOT / ('pipeline/checks/prepared_y79_y80_resource_' + token + '.json')
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, snapshot=view['generated_at'], observed_at=datetime.now().astimezone().isoformat(),
            original_live_training=observations, occupations=occupations, retained_native_plans=retained,
            original_Y76_terminal_preparation=y76['terminal_handoff_plan'], original_Y79_server1_CPU=y79['CPU_server1_runtime_preflight'],
            original_Y79_server1_verification=y79['server1_runtime_verification'], unchanged_preparation_reused=True,
            new_candidates=0, new_admissions=0, new_launches=0, new_archives=0, native_GPU_executed=False, formal_metrics_updated=False,
            Y57_pause_preserved=True, helper=ref(__file__), previous_helper=ref('scripts/record_prepared_y79_y80_wait.py'), evidence_binding='Both current canonical resource observations bind to Y79 and its active/inactive authorized plans.', conclusion='All three original native jobs advance. Both retained candidates require exclusive matching resources; all authorized cards remain busy/reserved. No replay of passed preparation and no new scientific conclusion.'))
        for cid in ('AdaRES-Y76', 'AdaRES-Y77', 'AdaRES-Y78', 'AdaRES-Y79', 'AdaRES-Y80'):
            get(cid)['evidence'].append(ref(out))
        # The inactive server1 route belongs to Y79 even while its active
        # required_server remains server2. Bind both actual observations.
        selected = get('AdaRES-Y79')
        for evidence in occupations.values():
            if evidence not in selected['evidence']:
                selected['evidence'].append(evidence)
        work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y79', kind='resource_check', result='blocked',
            command='python -B scripts/record_prepared_y79_y80_wait_v2.py --token ' + token,
            observation='Fresh canonical physical inventory/compute/reservations and exact retained native-plan source/YAML/goal/version hashes verified; no exclusive matching authorized slot. Original jobs and previously passed CPU/terminal preparation preserved; no GPU task, admission or efficacy claim.',
            evidence=[ref(out), occupations['server-2'], occupations['server-1']])
        assert all(evidence in selected['evidence'] for evidence in work['tick_attempt']['evidence'])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        memory = ROOT / 'pipeline/codex_memory.md'
        backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + token + '.md')
        assert not backup.exists()
        backup.write_bytes(memory.read_bytes())
        def e(value):
            return value['path'] + ' SHA' + value['sha256']
        lines = memory.read_text(encoding='utf-8').splitlines()
        lines = lines[:lines.index('## Last Tick')]
        for index, line in enumerate(lines):
            if line.startswith('- 当前token '):
                lines[index] = '- 当前token ' + token + '，snapshot' + view['generated_at'] + '，deadline' + view['deadline_at'] + '不延长；postflight仅认官方同token收据。'
            elif line.startswith('- 实际CODEX_THREAD_ID=owner='):
                lines[index] = '- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled。experiment_first enabled；snapshot admission2/creation2；本轮新增候选0/准入0/启动0/归档0。仅实际进度/资源与原计划完整性复核，无正式FIRST/cost更新。'
            elif line.startswith('- 前轮完整记忆精确备份：'):
                lines[index] = '- 前轮完整记忆精确备份：' + e(ref(backup)) + '；历史机制/禁用/专用恢复及原证据沿备份保留。'
            elif line.startswith('- Y76独立server1完整终态准备本轮完成：'):
                lines[index] = line.replace('本轮完成', 'd488轮完成')
            elif line.startswith('- Y79新增server1GPU0/350W inactive原生方案：') or line.startswith('- Y79实际server1私有runtime'):
                lines[index] = line.replace('本轮', 'd488轮')
            elif line.startswith('- 本轮fresh occupation server2：'):
                lines[index] = '- 本轮fresh occupation server2：' + e(occupations['server-2']) + '；server1：' + e(occupations['server-1']) + '；三授权卡busy+reserved。finish前须<300秒，超时新label；未知/失败/缺设备/idle不能证明hold。'
            elif any(line.startswith('- ' + cid + ' server') for cid in observations):
                cid = line.split()[1]
                phase = phases[cid]
                resource = {'AdaRES-Y76': 'server1GPU0/350W PID3489002', 'AdaRES-Y77': 'server2GPU0/280W PID111334', 'AdaRES-Y78': 'server2GPU1/390W PID22192'}[cid]
                lines[index] = '- ' + cid + ' ' + resource + ' 原source/YAML/receipt/marker/计算归属/功耗与native递增已核：' + phase['phase'] + str(phase['epoch']) + '/30 step' + str(phase['step']) + '/' + str(phase['count']) + '；' + e(observations[cid]) + '。'
        lines += [
            '## Last Tick',
            '- ' + token + ' begin/design先查并尝试派发waiting0，三原运行正常且native计数递增，无新终态/故障。真实fresh canonical三卡busy+reserved；两已准备候选Y79/Y80继续待独占native GPU检查，不重放CPU/model/data/version/plans/callback或为额度凑新变体。',
            '- 本轮新候选0/准入0/启动0/归档0，无正式FIRST/cost更新。原Y76终态准备20守卫/专用源与依赖/receipt/freeze，Y79 server1方案8守卫/实际CPU预检16原回调与原server2双计划均哈希复核有效。新证据：' + e(ref(out)) + '；tick_attempt绑定Y79当前真实资源检查。',
            '- 下轮先y78_deployment核三原运行并派发现有任务。Y76自然fixed30退出前保持原运行；退出后先核其terminal_handoff_plan依赖/原receipt/freeze，再专用finalize_y76_fixed30.py完整log/last30/best/model/optimizer/scheduler并canonical显式server1GPU0归档。fresh idle/unreserved350W才Y79 server1 alternative activate/run；实际native PASS后仍需匹配Y79/e0c source/admission/原完整后评冻结/server1资源契约与安全派发。Y79优先Y80，Y57继续单独暂停。',
            '- 官方report/finish后只读核FINALIZED/PASS/context_committed/released及锁/pending/archive_pending；此记忆不预签结果，释放后不写。',
        ]
        assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        notes = ROOT / ('pipeline/checks/foreground_notes_' + token + '.json')
        assert not notes.exists()
        status = '；'.join(cid + ' ' + value['phase'] + str(value['epoch']) + '/30 step' + str(value['step']) + '/' + str(value['count']) for cid, value in phases.items())
        tick.save(notes, dict(actions=[
            '运行：' + status + '，原PID/source/YAML/receipt/marker/功耗和实际计数递增核验通过。',
            '真实fresh canonical三授权卡均busy+reserved；Y79/Y80原native五份计划源码/YAML/goal/version与原Y76专用终态依赖/receipt/freeze和Y79 server1已通过CPU预检证据仍有效；未重放已通过准备。',
            '完成本轮新资源证据、七节短记忆备份与更新；新候选0/准入0/启动0/归档0，无正式FIRST/cost或新科学结论。'], decisions=[
            '已有两候选Y79/Y80等待独占native检查，优先Y79；保持Y76/Y77/Y78，不挤占GPU，不重复CPU或为额度新增变体。',
            'Y76自然完整fixed30退出后才核原专用plan依赖/receipt/freeze并完整canonical显式GPU0归档；fresh idle/unreserved350W后Y79 server1 guarded activate/run，实际native PASS再匹配准入/原完整后评/source契约/安全派发。',
            '原数据/独立runtime/mapping/科学goal与收据/全部禁用/seed1999固定30/既有350W-280W-390W保持；Y57单独暂停、禁server6/跨项目/调度变更。无新用户操作事项，健康等待保持静默。']))
        assert qref == ref('pipeline/queue.json') and rref == ref('pipeline/running.json')
        print(json.dumps(dict(evidence=ref(out), notes=str(notes.relative_to(ROOT)), memory_lines=len(lines), new_candidates=0, admitted=0, launched=0, archived=0)))
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    record(parser.parse_args().token)
