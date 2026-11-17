"""Bind the new Y82 terminal/Y84 inactive server-1 readiness to this tick."""
import argparse
import ast
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure


def record(token):
    lease(token, 180)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try:
        work = tick.load(ROOT/'pipeline/research_work.json')
        get = lambda cid: next(c for c in work['candidates'] if c['id'] == cid)
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        assert monitor.load_running(str(ROOT)) == {'server-2': {'0': 'AdaRES-Y83'}, 'server-1': {'0': 'AdaRES-Y82'}}
        view = tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
        terminal = tick.load(closure.artifact(ROOT, get('AdaRES-Y82')['terminal_handoff_plan']))
        assert terminal['token'] == token and terminal['CPU_guard_tests_passed'] == 25
        assert not terminal['checkpoint_loaded'] and not terminal['archived'] and not terminal['training_stopped']
        assert terminal['payload']['pid'] == 1121817 and terminal['payload']['code_commit'] == '5773768ffb87d7fea7432e85fd0da561556aa2fc'
        location = tick.load(closure.artifact(ROOT, terminal['declared_location']))
        assert location['checkpoint_directory'] == '20261104_010148'
        for entry in terminal['dependencies']: closure.artifact(ROOT, entry)
        c = get('AdaRES-Y84')
        assert c['stage'] == 'blocked' and c['required_server'] == 'server-2' and c['required_gpu'] == 0
        binding = tick.load(closure.artifact(ROOT, c['server1_alternative_preparation']))
        assert binding['token'] == token and binding['guard_tests_passed'] == 8
        assert binding['exact_callback_functions'] == 6 and binding['CPU_callbacks_previously_executed']
        assert not binding['activated'] and not binding['native_executed'] and binding['canonical_reservation'] == 'AdaRES-Y82'
        assert binding['original_model_and_YAML_payload_identical'] and binding['diagnostic_scientific_body_reversible_exact']
        plan = tick.load(closure.artifact(ROOT, c['native_smoke_plan_server1']))
        program = closure.artifact(ROOT, plan['program'])
        frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(c, plan, frozen) == 7
        assert plan['server'] == 'server-1' and plan['gpu'] == 0 and plan['runtime_power_limit_watts'] == 350.
        assert plan['activation_required'] and not plan['activated'] and not plan['native_executed']
        wait = ref('pipeline/checks/y82_y83_prepared_wait_' + token + '.json')
        assert tick.load(closure.artifact(ROOT, wait))['new_admissions'] == 0
        out = ROOT/('pipeline/checks/y82_y84_readiness_' + token + '.json'); assert not out.exists()
        tick.save(out, dict(complete=True, token=token, snapshot=view['generated_at'],
            Y82_terminal_preparation=get('AdaRES-Y82')['terminal_handoff_plan'],
            original_checkpoint_directory='20261104_010148', lifecycle_guards=25,
            full_checkpoint_verification_executed=False, original_Y82_training_preserved=True,
            Y84_inactive_server1_binding=c['server1_alternative_preparation'],
            inactive_plan=c['native_smoke_plan_server1'], resource_guards=8, exact_previously_executed_callbacks=6,
            current_resource_wait=wait, native_GPU_executed=False, production_source_changed=False,
            new_candidates=0, new_admissions=0, new_launches=0, new_archives=0,
            GPU1_pause_preserved=True, original_scientific_goal_and_receipts_preserved=True,
            helpers=[ref(p) for p in ('scripts/create_y82_terminal_helpers.py', 'scripts/y82_terminal_contract.py',
                'scripts/finalize_y82_fixed30.py', 'scripts/prepare_y82_terminal_handoff.py',
                'scripts/test_y82_terminal_handoff.py', 'scripts/y84_server1_alternative.py',
                'scripts/test_y84_server1_alternative.py', __file__)]))
        c['evidence'].append(ref(out)); get('AdaRES-Y82')['evidence'].append(ref(out))
        work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y84',
            kind='implementation_test', result='passed',
            command='python -B scripts/y84_server1_alternative.py prepare --token ' + token,
            observation='Eight host/power/path/idle/reservation controls and exact six previously executed callback identities pass for an inactive server-1 alternative. Original Y82 still live/reserved; no GPU task, activation or admission. Separate Y82 terminal handoff passes 25 lifecycle controls and verifies original source/YAML/native save location without loading checkpoints.',
            evidence=[ref(out), c['server1_alternative_preparation'], binding['guard_log']])
        assert all(entry in c['evidence'] for entry in work['tick_attempt']['evidence'])
        tick.save(ROOT/'pipeline/research_work.json', work)
        memory = ROOT/'pipeline/codex_memory.md'; lines = memory.read_text(encoding='utf-8').splitlines()
        at = lines.index('## Closed Or Failed Directions')
        lines[at:at] = [
            '- Y82本轮专用终态准备：' + get('AdaRES-Y82')['terminal_handoff_plan']['path'] + ' SHA' + get('AdaRES-Y82')['terminal_handoff_plan']['sha256'] + '；25生命周期守卫，精确原server1GPU0/350W、PID1121817/source/YAML/receipt/freeze/原native目录20261104_010148和追加Y83链通过。未加载checkpoint/未停止/未归档；仅自然结束后调用scripts/finalize_y82_fixed30.py完整实核。',
            '- Y84本轮新增未激活server1GPU0/350W原生备选计划：' + c['native_smoke_plan_server1']['path'] + ' SHA' + c['native_smoke_plan_server1']['sha256'] + '；8资源/路径守卫、6原CPU已执行callback AST保持，原model/YAML/科学body/code/goal不变。Y82仍运行且reserved；无native GPU/PASS/准入。Y82完整归档后fresh独占空闲才用y84_server1_alternative.py activate→run，不重放prepare；原server2GPU0计划仍主计划。']
        for i, line in enumerate(lines):
            if line.startswith('- 当前Y84优先Y85；'):
                lines[i] = '- 当前Y84优先Y85；原server2GPU0/280W native主计划保持；Y84现有未激活server1GPU0/350W备选，Y82完整归档且fresh独占空槽后才能activate→真实两步run。Y85尚无server1计划。GPU1旧计划仅保留禁执行。'
            elif line.startswith('- 当前安静复核使用scripts/record_y82_y83_wait.py'):
                lines[i] = '- 两原运行未变时可用scripts/record_y82_y83_wait.py安静复核；它不处理新准备，勿重放本轮prepare/终态。最新readiness：' + str(out.relative_to(ROOT)).replace('\\', '/') + '。'
        lines += ['- 本轮额外准备完成：Y82终态25守卫与原目录核验、Y84未激活server1备选8守卫/6原callback一致；无GPU任务/准入/启动/终态。readiness：' + str(out.relative_to(ROOT)).replace('\\', '/') + '。']
        assert len(lines) < 120 and sum(v.startswith('## ') for v in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        notes = ROOT/('pipeline/checks/foreground_notes_' + token + '.json'); value = tick.load(notes)
        value['actions'].append('新准备：Y82专用终态检查25守卫通过，精确核对原source/YAML/receipt/freeze和native保存目录20261104_010148；未加载checkpoint/未停止/未归档。')
        value['actions'].append('新准备：Y84未激活server1GPU0/350W备选计划完成，8资源/映射守卫和6原已执行callback身份通过，model/YAML/科学body/code/goal保持；无native GPU任务或准入。')
        value['decisions'].append('Y82自然fixed30结束并完整归档后，如server1GPU0首先独占空闲，激活Y84备选并执行真实原native两步；原server2GPU0主计划保持。GPU1不使用。')
        tick.save(notes, value)
        assert qref == ref('pipeline/queue.json') and rref == ref('pipeline/running.json')
        print(json.dumps(dict(readiness=ref(out), memory_lines=len(lines), new_admissions=0, GPU_executed=False)))
    finally:
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--token', required=True); record(p.parse_args().token)
