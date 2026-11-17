"""Record fresh resource holds for retained native plans without replaying probes."""
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
        queue = monitor.load_queue(str(ROOT))
        running = monitor.load_running(str(ROOT))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
        assert head == '498e721ea05e96aeb6f948d9cf8cd7a91b65ae22'
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        tasks = {t['id']: t for ts in queue['gpu_queues'].values() for t in ts}
        observations = {}
        phases = {}
        for cid, name, gpu, pid in [('AdaRES-Y75', 'server-2', '1', 65454), ('AdaRES-Y76', 'server-1', '0', 3489002), ('AdaRES-Y77', 'server-2', '0', 111334)]:
            task = tasks[cid]
            assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid and running[name][gpu] == cid
            closure.admission(ROOT, task)
            evidence = get(cid)['latest_training_observation']
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token and value['native_progress_confirmed'] and value['pid'] == pid
            observations[cid] = evidence
            phases[cid] = value['after']['latest']
        assert tasks['AdaRES-Y57']['status'] == 'paused' and tasks['AdaRES-Y57']['dispatch_hold'] and tasks['AdaRES-Y57']['resume_requires_user_authorization']
        occupations = {}
        for name in ('server-2', 'server-1'):
            evidence = ref('pipeline/checks/gpu_occupation_' + name + '_finish_' + token + '.json')
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token and (datetime.now(timezone.utc) - datetime.fromisoformat(value['observed_at'])).total_seconds() < 300
            candidate = next(c for c in work['candidates'] if c.get('stage') == 'blocked' and c.get('blocker_kind') == 'exclusive_gpu_slot' and c.get('required_server') == name)
            server = next(s for s in context['queue']['servers'] if s['name'] == name)
            assert not gpu_occupation.validated_state(ROOT, context, candidate, server, queue)['idle_gpu_ids']
            occupations[name] = evidence
        plans = {}
        for cid in ('AdaRES-Y78', 'AdaRES-Y79'):
            candidate = get(cid)
            assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot' and candidate['implementation_test_passed']
            for field in ('source_manifest', 'CPU_native_observer_verification', 'native_preparation'):
                closure.artifact(ROOT, candidate[field])
            version = tick.load(closure.artifact(ROOT, candidate['source_manifest']))
            assert version['code_commit'] == candidate['code_commit']
            assert subprocess.check_output(['git', 'rev-parse', candidate['retained_branch']], cwd=ROOT).decode().strip() == candidate['code_commit']
            plans[cid] = {}
            for gpu in (0, 1):
                evidence = candidate['native_smoke_plan_server2_gpu' + str(gpu)]
                plan = tick.load(closure.artifact(ROOT, evidence))
                program = closure.artifact(ROOT, plan['program'])
                frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
                assert validate_frozen(candidate, plan, frozen) == 7
                assert plan['gpu'] == gpu and plan['runtime_power_limit_watts'] == (280. if gpu == 0 else 390.)
                assert not plan['native_executed'] and not plan['admitted']
                plans[cid][str(gpu)] = evidence
        terminal_ref = ref('pipeline/checks/y75_terminal_preparation_909e7705ed74484fbc9bb4d16c7dbb59.json')
        assert terminal_ref['sha256'] == '1e2ec654157b569a6fda192170e42737b7f80e6b216e14424b9ee1e82b624b33'
        terminal = tick.load(closure.artifact(ROOT, terminal_ref))
        for dependency in terminal['dependencies']:
            closure.artifact(ROOT, dependency)
        assert terminal['admission'] == ref('pipeline/admissions/AdaRES-Y75.json')
        assert terminal['post_training_freeze'] == get('AdaRES-Y75')['post_training_freeze']
        closure.artifact(ROOT, terminal['post_training_freeze'])
        assert not terminal['checkpoint_loaded'] and not terminal['archived']
        out = ROOT / ('pipeline/checks/pending_native_resource_' + token + '.json')
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, snapshot=view['generated_at'], observed_at=datetime.now().astimezone().isoformat(),
            candidate_ids=['AdaRES-Y78', 'AdaRES-Y79'], retained_native_plans=plans, original_training=observations, occupations=occupations,
            original_Y75_terminal_preparation=terminal_ref, terminal_dependencies_unchanged=True, working_HEAD=head,
            new_candidates=0, new_admissions=0, new_launches=0, new_archives=0, native_GPU_steps_executed=False, formal_metrics_updated=False,
            evidence=[ref(__file__)], conclusion='All three original native jobs advance. Two retained CPU-tested native-plan candidates still require an exclusive matching GPU; all authorized devices are busy/reserved. No repeat of completed controls, no new admission or efficacy claim.'))
        for cid in ('AdaRES-Y75', 'AdaRES-Y76', 'AdaRES-Y77', 'AdaRES-Y78', 'AdaRES-Y79'):
            get(cid)['evidence'].append(ref(out))
        work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y78', kind='resource_check', result='blocked',
            command='python -B scripts/record_pending_native_resource_tick.py --token ' + token,
            observation='Fresh canonical physical GPU/compute/reservation observations and retained native source/YAML/goal/version checks passed; no authorized exclusive slot. Original Y75/Y76/Y77 continue; no native GPU task/admission/efficacy.',
            evidence=[ref(out), occupations['server-2']])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        memory = ROOT / 'pipeline/codex_memory.md'
        backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + token + '.md')
        assert not backup.exists()
        backup.write_bytes(memory.read_bytes())
        def e(value):
            return value['path'] + ' SHA' + value['sha256']
        lines = memory.read_text(encoding='utf-8').splitlines()
        last = lines.index('## Last Tick')
        lines = lines[:last]
        for index, line in enumerate(lines):
            if line.startswith('- 当前token '):
                lines[index] = '- 当前token ' + token + '，snapshot' + view['generated_at'] + '，deadline' + view['deadline_at'] + '不延长；postflight仅认官方同token收据。'
            elif line.startswith('- 实际CODEX_THREAD_ID=owner='):
                lines[index] = '- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled。experiment_first enabled；snapshot admission2/creation2；本轮新增候选0/准入0/启动0/归档0。两已准备候选Y78/Y79等原native空槽；无正式FIRST/cost更新。'
            elif line.startswith('- 前轮完整记忆精确备份：'):
                lines[index] = '- 前轮完整记忆精确备份：' + e(ref(backup)) + '；历史机制/禁用/专用恢复及已执行CPU/native/终态完整证据沿备份保留。'
            elif line.startswith('- fresh canonical server2 occupation：'):
                lines[index] = '- 本轮fresh canonical server2 occupation：' + e(occupations['server-2']) + '；server1：' + e(occupations['server-1']) + '；三卡busy+reserved。finish前须<300秒；超时新label，未知/失败/缺设备/idle不证明hold。'
            else:
                for cid, resource, pid in [('AdaRES-Y75', 'server2GPU1/390W', 65454), ('AdaRES-Y76', 'server1GPU0/350W', 3489002), ('AdaRES-Y77', 'server2GPU0/280W', 111334)]:
                    if line.startswith('- ' + cid + ' server'):
                        phase = phases[cid]
                        lines[index] = '- ' + cid + ' ' + resource + ' PID' + str(pid) + ' 本轮原源码/YAML/receipt/marker/计算归属与功耗核验通过；实际' + phase['phase'] + str(phase['epoch']) + '/30 step' + str(phase['step']) + '/' + str(phase['count']) + '且递增；evidence：' + e(observations[cid]) + '。'
        lines = [line.replace('Y79本轮CPU/model/原生回调检查首次即通过。', 'Y79在08e82轮CPU/model/原生回调检查首次即通过。').replace('；本轮核全部dependencies', '；本轮仅重核全部dependencies') for line in lines]
        lines += [
            '## Last Tick',
            '- ' + token + ' begin/design先查进展并尝试派发现有waiting0；无新终态/故障。三原实验实际native计数递增，fresh canonical三卡busy+reserved；只做新资源证据和原Y78/Y79源码/YAML/goal/branch/双卡计划哈希核验，不重放已完成CPU、回调或版本准备。',
            '- 本轮新增候选0/准入0/启动0/归档0；已有两CPU准备候选Y78/Y79等待native GPU检查，优先Y78，无新可执行GPU步骤或正式FIRST/cost更新。resource_handoff：' + e(ref(out)) + '；tick_attempt绑定Y78当前真实资源检查。',
            '- Y75专用终态准备原全部依赖/receipt/freeze保持。下轮先用y77_deployment核三运行并派发现有任务；Y75自然fixed30退出才核专用plan依赖并执行finalize_y75_fixed30.py完整last30/best/model/optimizer/scheduler/full-log与canonical显式GPU1归档，fresh idle/unreserved390W后Y78精确native两步PASS→原完整freeze/新准入/5eee源契约→安全派发。Y79随后；保留Y57暂停/原数据/功耗/禁用/调度。',
            '- 官方report/finish后仅只读核FINALIZED/PASS/context_committed/released及锁/pending/archive_pending；此记忆不预签结果，释放后不写。',
        ]
        assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        notes = ROOT / ('pipeline/checks/foreground_notes_' + token + '.json')
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            '运行：Y75 Train28/30 step310/4686、Y76 Train17/30 step1702/4686、Y77 Train6/30 step136/4686，原PID/源码/YAML/receipt/功耗与实际计数递增核验通过。',
            '核验Y78/Y79保留版本和四份原native计划源码/YAML/goal/分支一致；未重放CPU/model/data/callback准备。fresh canonical三授权卡均busy+reserved。',
            '完成本轮真实资源检查记录、七节记忆精确备份与更新；Y75原专用终态plan全部依赖/原receipt/freeze未变，仍在训练，未提前加载checkpoint或归档。'], decisions=[
            '两已准备候选Y78/Y79均等待匹配独占GPU，优先Y78；本轮无新候选/准入/启动/归档/正式评测结果，不为额度重复设计或重放已通过检查。',
            'Y75自然fixed30结束后按专用原程序完整canonical显式GPU1归档，fresh idle/unreserved390W后Y78精确native检查实际PASS才准入和安全派发；Y79随后。',
            'Y57实验继续暂停；server2GPU0 280W/GPU1 390W、server1GPU0 350W保持，禁server6/跨项目/改调度/挤占或重启受保护运行。无新用户处理事项，健康推进保持静默。']))
        assert qref == ref('pipeline/queue.json') and rref == ref('pipeline/running.json')
        print(json.dumps(dict(evidence=ref(out), memory_lines=len(lines), notes=str(notes.relative_to(ROOT)), new_candidates=0, admitted=0, launched=0, archived=0)))
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    record(parser.parse_args().token)
