"""Retain inactive Y79 server1 preflight and exact Y76 terminal preparation."""
import ast
from datetime import datetime, timezone
import json
import subprocess
from prepare_y79_query_language_feedback import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation

TOKEN = 'd488959562e94ac492b5884cf430f679'
HEAD = '5eee1e50c9e1d65bf113a4b8757bfc27d028a821'


def main():
    lease(TOKEN, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
        context = tick.load(ROOT / 'pipeline/headless/.tick_context.json')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        get = lambda cid: next(c for c in work['candidates'] if c['id'] == cid)
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        tasks = {task['id']: task for rows in queue['gpu_queues'].values() for task in rows}
        observations, phases = {}, {}
        for cid, name, gpu, pid in [('AdaRES-Y76', 'server-1', '0', 3489002), ('AdaRES-Y77', 'server-2', '0', 111334), ('AdaRES-Y78', 'server-2', '1', 22192)]:
            task = tasks[cid]
            assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid and running[name][gpu] == cid
            closure.admission(ROOT, task)
            evidence = get(cid)['latest_training_observation']
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == TOKEN and value['pid'] == pid and value['native_progress_confirmed'] and evidence['path'].endswith('_final.json')
            observations[cid], phases[cid] = evidence, value['after']['latest']
        occupations = {}
        for name in ('server-2', 'server-1'):
            evidence = ref('pipeline/checks/gpu_occupation_' + name + '_finish_' + TOKEN + '.json')
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == TOKEN and (datetime.now(timezone.utc) - datetime.fromisoformat(value['observed_at'])).total_seconds() < 300
            candidate = next(c for c in work['candidates'] if c.get('stage') == 'blocked' and c.get('blocker_kind') == 'exclusive_gpu_slot' and c.get('required_server') == name)
            server = next(s for s in context['queue']['servers'] if s['name'] == name)
            assert not gpu_occupation.validated_state(ROOT, context, candidate, server, queue)['idle_gpu_ids']
            occupations[name] = evidence
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == HEAD
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        assert tasks['AdaRES-Y57']['status'] == 'paused' and tasks['AdaRES-Y57']['dispatch_hold'] and tasks['AdaRES-Y57']['resume_requires_user_authorization']
        assert 'AdaRES-Y75' not in tasks and monitor.load_experiment(str(ROOT), 'AdaRES-Y75')['status'] == 'done_complete'
        for cid in ('AdaRES-Y79', 'AdaRES-Y80'):
            candidate = get(cid)
            assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot' and candidate['implementation_test_passed']
            for field in ('source_manifest', 'CPU_native_observer_verification', 'native_preparation'):
                closure.artifact(ROOT, candidate[field])
            for field in ('native_smoke_plan_server2_gpu0', 'native_smoke_plan_server2_gpu1'):
                plan = tick.load(closure.artifact(ROOT, candidate[field]))
                frozen = ast.literal_eval(ast.parse(closure.artifact(ROOT, plan['program']).read_text(encoding='utf-8')).body[0].value)
                assert validate_frozen(candidate, plan, frozen) == 7
        y76, y79 = get('AdaRES-Y76'), get('AdaRES-Y79')
        terminal_plan = tick.load(closure.artifact(ROOT, y76['terminal_handoff_plan']))
        assert terminal_plan['token'] == TOKEN and terminal_plan['CPU_guard_tests_passed'] == 20
        assert not terminal_plan['checkpoint_loaded'] and not terminal_plan['archived'] and not terminal_plan['full_terminal_verification_executed']
        assert terminal_plan['admission'] == ref('pipeline/admissions/AdaRES-Y76.json') and terminal_plan['post_training_freeze'] == y76['post_training_freeze']
        for evidence in terminal_plan['dependencies']:
            closure.artifact(ROOT, evidence)
        closure.artifact(ROOT, y76['post_training_freeze'])
        location = tick.load(closure.artifact(ROOT, terminal_plan['declared_location']))
        assert not location['checkpoint_loaded'] and not location['remote_mutations']
        alternative_ref = y79['native_smoke_plan_server1']
        alternative = tick.load(closure.artifact(ROOT, alternative_ref))
        native = closure.artifact(ROOT, alternative['program'])
        frozen = ast.literal_eval(ast.parse(native.read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(y79, alternative, frozen) == 7 and alternative['activated'] is False and alternative['native_executed'] is False
        assert (alternative['server'], alternative['gpu'], alternative['runtime_power_limit_watts']) == ('server-1', 0, 350.)
        assert y79['required_server'] == 'server-2' and y79['native_smoke_plan'] == y79['native_smoke_plan_server2_gpu0']
        cpu_ref = y79['CPU_server1_runtime_preflight']
        cpu = tick.load(closure.artifact(ROOT, cpu_ref))
        assert cpu['complete'] and cpu['implementation_test_passed'] and cpu['configured_TRAIN_data_collated'] and cpu['CPU_native_callbacks_executed']
        assert not cpu['CUDA_initialized'] and not cpu['native_task_backward'] and not cpu['native_smoke_passed'] and not cpu['admission']
        assert cpu['runtime_data_root'].rstrip('/') == '/data1/songcl/Project/AdaRES_project/data' and cpu['runtime_pretrain'].endswith('/data/sp_unet_backbone.pth')
        assert cpu['train_points'] == 162738 and cpu['train_expressions'] == 8 and cpu['synthetic_CPU_decoder_optimizer_steps'] == 2
        original_verification = tick.load(closure.artifact(ROOT, y79['CPU_native_observer_verification']))
        original_program = closure.artifact(ROOT, original_verification['CPU_program'])
        adapted_program = closure.artifact(ROOT, alternative['CPU_program'])
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        before, after = functions(original_program), functions(adapted_program)
        assert len(before) == 16 and set(after) == set(before) | {'adapt_path'} and all(after[key] == value for key, value in before.items())
        binding_ref = ref('pipeline/checks/y79_server1_' + TOKEN + '_binding.json')
        binding = tick.load(closure.artifact(ROOT, binding_ref))
        assert binding['guard_tests_passed'] == 8 and binding['original_model_and_YAML_payload_identical'] and binding['diagnostic_scientific_body_reversible_exact'] and binding['CPU_observer_core_reversible_exact']
        closure.artifact(ROOT, binding['exact_original_data'])
        verification = ROOT / ('pipeline/checks/y79_server1_verification_' + TOKEN + '.json')
        assert not verification.exists()
        tick.save(verification, dict(complete=True, token=TOKEN, id='AdaRES-Y79', plan=alternative_ref, binding=binding_ref, actual_CPU_result=cpu_ref,
            original_callback_functions_exact=16, actual_private_runtime_original_data_CPU_tested=True,
            native_GPU_executed=False, inactive=True, active_original_server2_plan_preserved=True,
            evidence=[ref(original_program), ref(adapted_program), ref(cpu_ref['path'].replace('.json', '.log')), ref(__file__)]))
        y79['server1_runtime_verification'] = ref(verification)
        y79['evidence'].append(ref(verification))
        y79['next_action'] = 'Completed original server2 CPU/model/TRAIN/version/plans/callback preparation plus actual private server1 mapped-data CPU preflight with sixteen exact callback functions. Do not replay. Preserve original Y76/Y77/Y78 and paused Y57. Server1 alternative remains inactive; only original Y76 complete fixed30 canonical archive and fresh exclusive idle/no compute/no reservation GPU0/350W permits scripts/y79_server1_alternative.py activate then run --token CURRENT. First verify Y76 terminal_handoff_plan dependencies and original receipt/freeze; use its own finalizer. Original server2 GPU0/280W and GPU1/390W plans remain unchanged. Actual native two-step task PASS on selected device is required before original complete evaluation/cost freeze, matching Y79 source/admission/server/runtime contract, safe source selection/sync/dispatch and increasing steps. Y79 precedes Y80; no CPU-based GPU PASS or efficacy.'
        out = ROOT / ('pipeline/checks/resource_handoff_' + TOKEN + '.json')
        assert not out.exists()
        tick.save(out, dict(complete=True, token=TOKEN, snapshot=view['generated_at'], observed_at=datetime.now().astimezone().isoformat(),
            original_live_training=observations, occupations=occupations, Y76_terminal_preparation=y76['terminal_handoff_plan'],
            Y79_inactive_server1_native_plan=alternative_ref, Y79_actual_server1_CPU=cpu_ref, Y79_exact_callback_verification=ref(verification),
            Y79_original_server2_plans=[y79['native_smoke_plan_server2_gpu0'], y79['native_smoke_plan_server2_gpu1']],
            working_HEAD=HEAD, source_markers_preserved=True, new_candidates=0, new_admissions=0, new_dispatches=0, archived=[],
            prepared_native_pending=['AdaRES-Y79', 'AdaRES-Y80'], native_GPU_executed=False, formal_metrics_updated=False, Y57_pause_preserved=True, helper=ref(__file__)))
        for cid in ('AdaRES-Y76', 'AdaRES-Y77', 'AdaRES-Y78', 'AdaRES-Y79', 'AdaRES-Y80'):
            get(cid)['evidence'].append(ref(out))
        work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y79', kind='implementation_test', result='passed',
            command='python -B scripts/y79_server1_alternative.py cpu --token ' + TOKEN,
            observation='Actual verified private server1 runtime with CUDA hidden loaded unchanged native YAML/full model/released backbone and exact mapped original TRAIN; sixteen original observer functions/classes executed in two synthetic CPU decoder optimizer steps. Native GPU/activation/admission/efficacy remain pending.',
            evidence=[cpu_ref, ref(verification), binding_ref])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        memory = ROOT / 'pipeline/codex_memory.md'
        backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + TOKEN + '.md')
        assert not backup.exists()
        backup.write_bytes(memory.read_bytes())
        def e(value):
            return value['path'] + ' SHA' + value['sha256']
        lines = memory.read_text(encoding='utf-8').splitlines()
        lines = lines[:lines.index('## Last Tick')]
        for index, line in enumerate(lines):
            if line.startswith('- 当前token '):
                lines[index] = '- 当前token ' + TOKEN + '，snapshot' + view['generated_at'] + '，deadline' + view['deadline_at'] + '不延长；postflight仅认官方同token收据。'
            elif line.startswith('- 实际CODEX_THREAD_ID=owner='):
                lines[index] = '- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled。experiment_first enabled；snapshot admission2/creation2；本轮新增候选0/准入0/启动0/归档0：Y79新增inactive server1计划与实际CPU预检，Y76专用终态准备通过。Y75为前轮归档，无正式FIRST/cost更新。'
            elif line.startswith('- 前轮完整记忆精确备份：'):
                lines[index] = '- 前轮完整记忆精确备份：' + e(ref(backup)) + '；历史机制/禁用/专用恢复和原证据沿备份保持。'
            elif line.startswith('- Y78本轮真实') or line.startswith('- Y78本轮已实际') or line.startswith('- Y75本轮专用'):
                lines[index] = line.replace('本轮', '在eee3轮', 1)
            elif line.startswith('- 本轮fresh occupation server2：'):
                lines[index] = '- 本轮fresh occupation server2：' + e(occupations['server-2']) + '；server1：' + e(occupations['server-1']) + '；三授权卡busy+reserved。finish前须<300秒，超时新label；未知/失败/缺设备/idle不能证明hold。'
            elif any(line.startswith('- ' + cid + ' server') for cid in observations):
                cid = line.split()[1]
                phase = phases[cid]
                resource = {'AdaRES-Y76': 'server1GPU0/350W PID3489002', 'AdaRES-Y77': 'server2GPU0/280W PID111334', 'AdaRES-Y78': 'server2GPU1/390W PID22192'}[cid]
                lines[index] = '- ' + cid + ' ' + resource + ' 原source/YAML/receipt/marker/计算归属/功耗与native递增已核：' + phase['phase'] + str(phase['epoch']) + '/30 step' + str(phase['step']) + '/' + str(phase['count']) + '；' + e(observations[cid]) + '。'
        index = lines.index('## Active Hypotheses') + 1
        lines[index:index] = [
            '- Y76独立server1完整终态准备本轮完成：' + e(y76['terminal_handoff_plan']) + '；20生命周期守卫与实际只读原六源码/YAML/receipt/freeze/私有runtime路径、marker c384及声明目录' + location['checkpoint_directory'] + '核验通过，未checkpoint加载/未提前终态验证/未归档。独立finalize_y76_fixed30.py携原350W与私有LD_LIBRARY_PATH，依赖均哈希绑定；自然退出后先核plan.dependencies/原receipt/freeze再运行其专用finalizer。',
            '- Y79新增server1GPU0/350W inactive原生方案：' + e(alternative_ref) + '；8资源/映射/host/power/保留原job负守卫通过，native科学body仅host与两授权path mappings可逆精确变化，model/YAML/commit/goal原字节不变，原server2两计划不变。需Y76自然完整canonical归档且fresh idle/unreserved才activate/run；本轮未激活/未GPU任务/未同步/未准入。',
            '- Y79实际server1私有runtime CUDA隐藏CPU预检通过：原native YAML/full model/released backbone、原映射TRAIN162738点8表达、两合成CPU DEC原optimizer步及原16函数/类逐AST相同、路径/梯度/实际更新通过。CPU result：' + e(cpu_ref) + '；verification：' + e(ref(verification)) + '；不是native sparse TRAIN GPU PASS或效益结论，不重放已通过CPU预检。',
        ]
        lines += [
            '## Last Tick',
            '- ' + TOKEN + ' begin/design先查并尝试派发waiting0；snapshot new_terminal Y75是前轮已完整归档，不重放。Y76/Y77/Y78原运行及native增长保持，无新训练完成/故障。',
            '- 本轮新候选0/准入0/启动0/归档0。优先推进既有Y79跨已授权资源的inactive server1方案，实际private runtime/精确原数据CPU预检与16原回调两步通过；Y76专用20生命周期守卫和只读原声明目录检查通过，仍在训练且未加载checkpoint。两候选Y79/Y80仍待真实native GPU检查。',
            '- 原HEAD/server2 marker5eee，server1 markerc384，原receipt/freeze/data/功耗/Y57暂停保持；handoff：' + e(ref(out)) + '；tick_attempt绑定Y79本轮实际server1 CPU预检。fresh canonical三授权卡busy+reserved，无正式FIRST/cost更新。',
            '- 下轮先y78_deployment核三原运行并派发现有任务。Y76自然fixed30退出时先验证其terminal_handoff_plan依赖/原receipt/freeze，再专用finalize_y76_fixed30.py完整last30/best/model/optimizer/scheduler/full-log及canonical显式server1GPU0归档。fresh idle/unreserved350W后Y79 server1 alternative activate→run原native两步；实际PASS后仍需匹配Y79/e0c source/admission/冻结原完整后评/server1资源契约与安全派发，不套用Y78 helper。不要重放本轮CPU或prepare。',
            '- 官方report/finish后只读核FINALIZED/PASS/context_committed/released及锁/pending/archive_pending；此记忆不预签结果，释放后不写。',
        ]
        assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        notes = ROOT / ('pipeline/checks/foreground_notes_' + TOKEN + '.json')
        assert not notes.exists()
        status = '；'.join(cid + ' ' + value['phase'] + str(value['epoch']) + '/30 step' + str(value['step']) + '/' + str(value['count']) for cid, value in phases.items())
        tick.save(notes, dict(actions=[
            '运行：' + status + '，原PID/source/YAML/receipt/marker/功耗和实际计数递增核验通过，未改变现有训练。',
            '新准备：Y79新增inactive server1GPU0/既有350W原生方案；8资源与路径守卫通过，原model/YAML/commit/goal和server2两计划不变。实际私有runtime CUDA隐藏CPU预检已加载原生模型/发布backbone/精确映射TRAIN162738点8表达，16原回调函数/类及两合成CPU解码器优化步通过；不是GPU任务PASS。',
            '新准备：Y76独立server1终态程序、20生命周期守卫、原six source/YAML/receipt/freeze/marker与声明目录只读核验通过；保持训练，未加载checkpoint、未提前终态验证或归档。',
            '完成本轮证据与七节记忆备份/更新；新候选0/准入0/启动0/归档0，无正式FIRST/cost结果。Y75为前轮完成，未重复处理。'], decisions=[
            '三授权GPU仍busy+reserved，Y79优先Y80；已有两候选CPU准备，继续等待原生空槽，不重复CPU或为额度新增候选。',
            'Y76自然fixed30后先核专用terminal plan依赖/原receipt/freeze并完整canonical显式GPU0归档；fresh idle/unreserved350W才Y79 server1 guarded activate/run。实际native PASS后继续匹配Y79 source/完整后评冻结/收据/安全派发，不以CPU预检签GPU准入。',
            '复用既有项目授权，保持原数据和独立runtime/部署mapping、server1GPU0 350W与server2GPU0 280W/GPU1 390W、固定seed1999/30轮/全部禁用项。Y57单独暂停；禁server6/跨项目/调度变更。当前无新增用户处理事项。']))
        assert qref == ref('pipeline/queue.json') and rref == ref('pipeline/running.json')
        print(json.dumps(dict(handoff=ref(out), notes=str(notes.relative_to(ROOT)), memory_lines=len(lines), new_candidates=0, admitted=0, launched=0, archived=0)))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
