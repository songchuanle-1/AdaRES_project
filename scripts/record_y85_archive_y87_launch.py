"""Close this real terminal/failure/launch cycle without replaying old work."""
import argparse
import ast
import copy
from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from advance_y87 import prepared, validate_result
from y87_deployment import Y84, Y85, Y87
import closure
import gpu_occupation


def record(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        get = lambda cid: next(c for c in work['candidates'] if c['id'] == cid)
        done, owner = get('AdaRES-Y85'), get('AdaRES-Y87')
        assert done['stage'] == 'validation' and done['resource_work_kind'] == 'frozen_evaluation'
        assert (done['required_server'], done['required_gpu']) == ('server-2', 0)
        assert monitor.load_experiment(str(ROOT), done['id'])['status'] == 'done_complete'
        assert monitor.load_running(str(ROOT))['server-2']['0'] == owner['id']
        observed = tick.load(closure.artifact(ROOT, owner['latest_training_observation']))
        assert observed['token'] == token and observed['native_progress_confirmed']
        terminal = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
        archive = tick.load(closure.artifact(ROOT, done['terminal_archive_binding']))
        assert terminal['completed_epochs'] == {phase: list(range(1, 31)) for phase in ('Train', 'Val')}
        assert archive['only_exact_Y85_archived'] and archive['checkpoint_files_retained']
        assert not archive['full_frozen_evaluation_executed']
        freeze = tick.load(closure.artifact(ROOT, done['post_training_freeze']))
        assert (freeze['resource_condition']['server'], freeze['resource_condition']['gpu'],
                freeze['resource_condition']['power_limit_watts']) == ('server-2', 0, 280.)
        hold = ROOT / ('pipeline/checks/y85_frozen_evaluation_hold_' + token + '.json')
        assert not hold.exists()
        tick.save(hold, dict(complete=True, token=token, id=done['id'], original_server='server-2',
            original_gpu=0, power_limit_watts=280., current_reserved_owner=owner['id'],
            native_owner_observation=owner['latest_training_observation'],
            terminal=done['terminal_evidence'], archive=done['terminal_archive_binding'],
            post_training_freeze=done['post_training_freeze'], full_frozen_evaluation_executed=False,
            helper=ref(__file__)))
        done.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
            blocker='Original complete Y85 checkpoints verified and archived; original full frozen FIRST/cost requires server-2 GPU0/280W, now occupied by verified independent Y87. Preserve all products and live training.',
            retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
            unblock_condition='Fresh exclusive original server-2 GPU0/280W without compute or canonical reservation.')
        done['evidence'].append(ref(hold))
        tick.save(ROOT / 'pipeline/research_work.json', work)
    finally:
        writer.close()
    for name in ('server-2', 'server-1'):
        response = subprocess.run([sys.executable, '-B',
            'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/gpu_occupation.py',
            'AdaRES_project', '--token', token, '--server', name, '--label', 'finish'],
            cwd=ROOT, capture_output=True, timeout=45)
        assert response.returncode == 0, response.stderr.decode(errors='replace')
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        get = lambda cid: next(c for c in work['candidates'] if c['id'] == cid)
        view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
        context = tick.load(ROOT / 'pipeline/headless/.tick_context.json')
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        assert running == {'server-1': {'0': 'AdaRES-Y84'}, 'server-2': {'0': 'AdaRES-Y87'}}
        tasks = {task['id']: task for tasks in queue['gpu_queues'].values() for task in tasks}
        assert set(tasks) == {'AdaRES-Y57', 'AdaRES-Y84', 'AdaRES-Y87'}
        assert tasks['AdaRES-Y57']['status'] == 'paused' and tasks['AdaRES-Y57']['dispatch_hold']
        assert tasks['AdaRES-Y57']['resume_requires_user_authorization']
        assert closure.git(ROOT, 'rev-parse', 'HEAD') == Y87
        assert closure.git(ROOT, 'rev-parse', Y87 + '^') == Y85
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        authority = ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')
        assert authority['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
        servers = monitor.load_servers(str(ROOT))
        assert servers['server-1']['allowed_gpus'] == servers['server-2']['allowed_gpus'] == [0]
        assert servers['server-6']['allowed_gpus'] == []
        observations, phases = {}, {}
        for cid, name, commit in [('AdaRES-Y84', 'server-1', Y84), ('AdaRES-Y87', 'server-2', Y87)]:
            task = tasks[cid]
            assert task['status'] == 'running' and task['launch_verified']
            assert (task['server'], str(task['gpu']), task['code_commit']) == (name, '0', commit)
            closure.admission(ROOT, task)
            evidence = get(cid)['latest_training_observation']
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token and value['pid'] == task['pid'] and value['native_progress_confirmed']
            observations[cid], phases[cid] = evidence, value['after']['latest']
        candidate, _, program, frozen = prepared(work)
        raw = tick.load(closure.artifact(ROOT, candidate['native_smoke_result']))
        validate_result(raw, candidate, frozen)
        evaluation = tick.load(closure.artifact(ROOT, candidate['post_training_freeze']))
        parent = tick.load(closure.artifact(ROOT, evaluation['parent_protocol']))
        restored = copy.deepcopy(evaluation['evaluation'])
        restored['methods'][0] = 'AdaRES-Y64'
        assert restored == parent['evaluation'] and evaluation['training'] == parent['training']
        failed = get('AdaRES-Y86')
        assert failed['stage'] == 'blocked' and failed['blocker_kind'] == 'numerical_control_failure'
        assert failed['failure_scope'] == 'implementation' and not failed.get('admission_receipt')
        failure = tick.load(closure.artifact(ROOT, failed['native_smoke_failure']))
        diagnosis = tick.load(closure.artifact(ROOT, failed['native_failure_diagnosis']))
        assert failure['result'] == diagnosis['result'] == 'failed'
        assert failure['original_guard_preserved'] and diagnosis['original_assertion_preserved']
        assert not diagnosis['native_PASS'] and not diagnosis['admitted']
        occupations = {}
        for name in ('server-2', 'server-1'):
            evidence = ref('pipeline/checks/gpu_occupation_' + name + '_finish_' + token + '.json')
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token
            assert 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(value['observed_at'])).total_seconds() < 300
            blocked = next(c for c in work['candidates'] if c.get('stage') == 'blocked'
                and c.get('blocker_kind') == 'exclusive_gpu_slot' and c.get('required_server') == name)
            server_context = next(s for s in context['queue']['servers'] if s['name'] == name)
            state = gpu_occupation.validated_state(ROOT, context, blocked, server_context, queue)
            assert state['busy_gpu_ids'] == [0] and not state['idle_gpu_ids']
            occupations[name] = evidence
        selection = ref('pipeline/checks/adares-y87_source_selection_' + token + '.json')
        dispatch = ref('pipeline/checks/adares-y87_dispatch_' + token + '.json')
        guards = ref('pipeline/checks/adares-y87_admission_guards_' + token + '.log')
        tests = (ROOT / guards['path']).read_text(encoding='utf-8')
        assert 'Ran 15 tests' in tests and 'OK' in tests and 'skipped=' not in tests
        out = ROOT / ('pipeline/checks/y85_archive_y87_launch_' + token + '.json')
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, snapshot=view['generated_at'],
            Y85_terminal=get('AdaRES-Y85')['terminal_evidence'],
            Y85_archive=get('AdaRES-Y85')['terminal_archive_binding'], Y85_original_full_evaluation_hold=ref(hold),
            Y86_failure=failed['native_smoke_failure'], Y86_diagnosis=failed['native_failure_diagnosis'],
            Y87_native=candidate['native_smoke'], Y87_receipt=ref('pipeline/admissions/AdaRES-Y87.json'),
            Y87_full_freeze=candidate['post_training_freeze'], independent_source_binding=candidate['independent_parent_binding'],
            selection=selection, dispatch=dispatch, admission_guards=guards, live_training=observations,
            occupations=occupations, new_candidates=0, new_admissions=1, new_launches=1, new_archives=1,
            failed_native_checks=2, passed_native_checks=1, scientific_revision=False,
            formal_metrics_updated=False, GPU1_pause_preserved=True, Y57_pause_preserved=True,
            helper=ref(__file__)))
        candidate['evidence'].extend([ref(out), selection, dispatch, guards, ref(program)])
        get('AdaRES-Y85')['evidence'].append(ref(out))
        failed['evidence'].append(ref(out))
        work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y87',
            kind='native_smoke', result='passed',
            command='python -B scripts/run_experiment_first_probe.py y87_gpu0_smoke ' + ref(program)['path']
                + ' --token ' + token + ' --seconds 600 --gpu --gpu-id 0',
            observation='Actual unchanged native YAML/TRAIN two finite losses/backward/updates and original self-language/norm/FFN paths passed. Fifteen actual admission guards passed; original full review frozen, exact seven files selected, admitted and dispatched with increasing native counters. Y86 failure preserved; no efficacy claim.',
            evidence=[candidate['native_smoke'], candidate['post_training_freeze'], ref(out)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        memory = ROOT / 'pipeline/codex_memory.md'
        backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + token + '.md')
        assert not backup.exists()
        backup.write_bytes(memory.read_bytes())
        old = memory.read_text(encoding='utf-8').splitlines()

        def section(name):
            start = old.index('## ' + name) + 1
            end = next((i for i in range(start, len(old)) if old[i].startswith('## ')), len(old))
            return old[start:end]

        def e(evidence):
            return evidence['path'] + ' SHA' + evidence['sha256']

        def phase(cid):
            row = phases[cid]
            return cid + ' ' + row['phase'] + str(row['epoch']) + '/30 step' + str(row['step']) + '/' + str(row['count'])

        goal = section('Goal And Phase')
        goal[0] = '- Current token ' + token + '; snapshot ' + view['generated_at'] + '; deadline ' + view['deadline_at'] + '; never extend.'
        goal = [('- Exact previous seven-section memory: ' + e(ref(backup)) + '; all historical bans/negative evidence/special recovery requirements retained.')
                if line.startswith('- Exact previous seven-section memory:') else line for line in goal]
        best = terminal['checkpoints'][0]['best_miou']
        status = '; '.join(phase(cid) for cid in phases)
        lines = ['# AdaRES_project Research Memory', '## Goal And Phase'] + goal + [
            '## Champion And Baseline'] + section('Champion And Baseline') + [
            '## Active Hypotheses',
            '- Y87 self_language_product remains independent W2: six self+language+(self*language)@zeroW256x256 before SAME active norm_qqa and original FFNs; +65536/total164695173/no RNG. Original coarse/fine half-average, data/2D/backbone/SCG/gates/shared heads/targets/loss/selector/seed1999/fixed30 retained.',
            '- Y87 actual server2GPU0 native two-step PASS: losses ' + repr(raw['losses']) + ', peak_reserved_bytes=' + str(raw['peak_reserved_bytes']) + ';12 self/language/original norm/FFN consumers,14 shared heads/query norms, finite original/new/input gradients and two updates. Two batches do not establish efficacy or full-data memory bounds.',
            '- Y87 native: ' + e(candidate['native_smoke']) + '; receipt: ' + e(ref('pipeline/admissions/AdaRES-Y87.json')) + '; original full freeze: ' + e(candidate['post_training_freeze']) + ';15 admission guards all passed, no skip.',
            '- Y87 exact seven tested files re-retained directly on Y85 without Y86 code. Model/YAML/CPU evidence/native body and controls unchanged; metadata-only independent parent, not a scientific revision. Original branch053aea85a03be4244879c90362455c45327a7d64 and evidence retained. Binding: ' + e(candidate['independent_parent_binding']),
            '- Master/server2 marker=' + Y87 + '; server1 marker=' + Y84 + '. Only seven tested Y87 A files selected; unrelated index/worktree and historical signed helpers preserved. Selection: ' + e(selection),
            '## Closed Or Failed Directions'] + section('Closed Or Failed Directions') + [
            '- Y86 original GPU0 native failed second-batch stage4 positive-response guard. Exact diagnostic repeats SAME assertion: coefficient about1e-4, overlap max1.3414648492471315e-5, float32 confidence delta0. Guard not relaxed; implementation/numerical hold, no native PASS/admission/scientific rejection.',
            '- Y86 failure: ' + e(failed['native_smoke_failure']) + '; diagnosis: ' + e(failed['native_failure_diagnosis']) + '. Original model/CPU version2d2a0b98865977a6e2f37d316979344ffc56817c and both failed raw logs/programs retained; no revision used.',
            '## Running And Queue']
        lines += [line.replace('held by Y85', 'held by Y87') for line in section('Running And Queue')
                  if line.startswith(('- Y83 ', '- Y82 ', '- Y81 ', '- Y57 ', '- server1=', '- server2='))]
        lines += [
            '- Y85 naturally fixed30 complete and canonically archived: original20261104_160627, stable finite last30/best21/model/optimizer545/scheduler30-21, native Train/Val1..30/source/YAML/receipt/freeze verified. Training best ' + repr(best) + ' is not FIRST.',
            '- Y85 terminal: ' + e(get('AdaRES-Y85')['terminal_evidence']) + '; archive: ' + e(get('AdaRES-Y85')['terminal_archive_binding']) + '; original server2GPU0/280W full frozen review held by verified Y87, no migration/retrain.',
            '- 28 fixed30 models Y56/Y58..Y83/Y85 await original full frozen FIRST/cost; original devices retained and GPU1 routes user-held. Do not repeat archived checkpoint loading/terminal helpers.']
        for cid in phases:
            lines.append('- ' + phase(cid) + '; ' + tasks[cid]['server'] + ' GPU0 PID' + str(tasks[cid]['pid'])
                + '; exact source/YAML/receipt/marker/power/increasing counters: ' + e(observations[cid]))
        lines += [
            '- Current observer: scripts/y87_deployment.py observe --id AdaRES-Y84|AdaRES-Y87 --token NEW_TOKEN --label progress. Old y85 observer cannot accept HEADY87; signed helpers remain unchanged.',
            '- Fresh occupation: ' + e(occupations['server-2']) + '; ' + e(occupations['server-1']) + '; both permitted GPU0 busy/reserved; finish needs age<300s.',
            '## Open Questions And Risks',
            '- Full original9508 expressions/141 scenes, candidate+Y36/W2/B1 twice, FIRST primary, scene bootstrap10000/seed1999, TRAIN cuts1487/3325, subgroup harm and mIoU/Acc25/50/activequeries/latency/memory/same-device cost unchanged.',
            '- No new formal FIRST/cost or promotion. Y85 training best and Y87 native finite gradients are not trained efficacy. GPU1 all use and Y57 separate experiment stay paused; no migration or substitution.',
            '- Next tick first observe Y84/Y87, dispatch existing qualified work, then bounded independent preparation if backlog short. Y86 is a numerical implementation hold, not usable reserve; do not replay unchanged failed smoke or relax its guard. Y85 already archived, never restart/re-finalize.',
            '## Last Tick',
            '- ' + token + ': Y85 full checkpoint verification/archive; Y86 original native failure and unchanged-guard actual-value diagnosis preserved; Y87 unchanged original native PASS,15 guards, independent seven-file version/source selection/full freeze/admission/dispatch/increasing native steps.',
            '- ' + status + '; new candidates0/admissions1/launches1/archives1; two failed Y86 checks and one successful Y87 native check; no formal FIRST/cost/scientific revision.',
            '- Handoff: ' + e(ref(out)) + '; original resources/data/private runtimes/path mappings/goal bytes/receipts/seed1999-fixed30/bans/power/schedule and both pauses retained.',
            '- After official report/finish read only FINALIZED/PASS/context_committed/released and absent owned lock/pending/archive_pending; no preclaim or post-release writes.']
        assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        notes = ROOT / ('pipeline/checks/foreground_notes_' + token + '.json')
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            '完成：Y85自然固定30轮，原last30/best21/model/optimizer545/scheduler30-21稳定有限及原Train/Val1..30/source/YAML/receipt/freeze全部核验后归档，原产物保留；训练best=' + repr(best) + '不是FIRST。',
            'Y86原生第二批第4阶段非零响应检查失败；追加同一断言实际值诊断确认微小overlap投影在float32舍入为零。两份失败日志和原模型/控制保留，未准入，实施层数值挂起，不声称机制无效。',
            '运行：Y87原生YAML/TRAIN两步loss=' + repr(raw['losses']) + '，实际路径/有限梯度/更新通过，15准入检查通过；模型/YAML不变，仅重整独立父提交，七测试文件留版、原完整后评冻结/准入/安全派发，训练步数递增。',
            '运行：' + status + '；GPU0功耗/source/YAML/receipt/原生进度核实。新候选0/准入1/启动1/归档1，无正式FIRST/cost。'],
            decisions=[
                'Y85原完整server2GPU0/280W后评等独占空槽；Y86挂起不阻断独立Y87，无修改冻结数值检查。',
                'GPU1全部AdaRES使用继续暂停，原GPU1冻结后评/cost不迁移替换；Y57单独暂停。',
                '原goal字节/receipt/精确数据/私有环境/mapping/seed1999-fixed30/禁用/既有功耗与调度保持，无需用户处理。']))
        assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        print(json.dumps(dict(handoff=ref(out), notes=notes.relative_to(ROOT).as_posix(),
            memory_lines=len(lines), admissions=1, launches=1, archives=1)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    record(parser.parse_args().token)
