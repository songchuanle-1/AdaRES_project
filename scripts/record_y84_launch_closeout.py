"""Record exact Y82 archival and Y84 launch; retain frozen reviews and pauses."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from advance_y84 import prepared, validate_result
from y84_deployment import Y84, Y83
import closure
import gpu_occupation


def record(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        get = lambda cid: next(candidate for candidate in work['candidates'] if candidate['id'] == cid)
        done, owner = get('AdaRES-Y82'), get('AdaRES-Y84')
        assert done['stage'] == 'validation' and done['resource_work_kind'] == 'frozen_evaluation'
        assert (done['required_server'], done['required_gpu']) == ('server-1', 0)
        assert monitor.load_experiment(str(ROOT), done['id'])['status'] == 'done_complete'
        assert monitor.load_running(str(ROOT))['server-1']['0'] == owner['id']
        observed = tick.load(closure.artifact(ROOT, owner['latest_training_observation']))
        assert observed['token'] == token and observed['native_progress_confirmed']
        terminal = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
        archive = tick.load(closure.artifact(ROOT, done['terminal_archive_binding']))
        assert terminal['completed_epochs'] == {phase: list(range(1, 31)) for phase in ('Train', 'Val')}
        assert archive['only_exact_Y82_archived'] and archive['checkpoint_files_retained']
        assert not archive['full_frozen_evaluation_executed']
        freeze = tick.load(closure.artifact(ROOT, done['post_training_freeze']))
        assert freeze['resource_condition']['server'] == 'server-1'
        assert freeze['resource_condition']['gpu'] == 0 and freeze['resource_condition']['power_limit_watts'] == 350
        hold_path = ROOT / ('pipeline/checks/y82_frozen_evaluation_hold_' + token + '.json')
        assert not hold_path.exists()
        tick.save(hold_path, dict(complete=True, token=token, id=done['id'],
            original_server='server-1', original_gpu=0, power_limit_watts=350,
            current_reserved_owner=owner['id'], native_owner_observation=owner['latest_training_observation'],
            terminal=done['terminal_evidence'], archive=done['terminal_archive_binding'],
            post_training_freeze=done['post_training_freeze'], full_frozen_evaluation_executed=False,
            helper=ref(__file__)))
        done.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
            blocker='Original complete Y82 checkpoints verified and archived; original full frozen FIRST/cost '
                'requires server-1 GPU0/350W, now occupied by verified Y84. Preserve all products and live training.',
            retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
            unblock_condition='Fresh exclusive original server-1 GPU0/350W, without compute or canonical reservation.')
        done['evidence'].append(ref(hold_path))
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
        get = lambda cid: next(candidate for candidate in work['candidates'] if candidate['id'] == cid)
        view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
        context = tick.load(ROOT / 'pipeline/headless/.tick_context.json')
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        assert running == {'server-2': {'0': 'AdaRES-Y83'}, 'server-1': {'0': 'AdaRES-Y84'}}
        tasks = {task['id']: task for tasks in queue['gpu_queues'].values() for task in tasks}
        assert set(tasks) == {'AdaRES-Y57', 'AdaRES-Y83', 'AdaRES-Y84'}
        assert tasks['AdaRES-Y57']['status'] == 'paused' and tasks['AdaRES-Y57']['dispatch_hold']
        assert tasks['AdaRES-Y57']['resume_requires_user_authorization']
        assert closure.git(ROOT, 'rev-parse', 'HEAD') == Y84
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        authority = ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')
        assert authority['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
        servers = monitor.load_servers(str(ROOT))
        assert servers['server-1']['allowed_gpus'] == servers['server-2']['allowed_gpus'] == [0]
        assert servers['server-6']['allowed_gpus'] == []
        observations, phases = {}, {}
        for cid, name, commit in [('AdaRES-Y83', 'server-2', Y83), ('AdaRES-Y84', 'server-1', Y84)]:
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
        frozen84 = tick.load(closure.artifact(ROOT, candidate['post_training_freeze']))
        parent = tick.load(closure.artifact(ROOT, frozen84['parent_protocol']))
        restored = dict(frozen84['evaluation'], methods=list(frozen84['evaluation']['methods']))
        restored['methods'][0] = 'AdaRES-Y64'
        assert restored == parent['evaluation'] and frozen84['training'] == parent['training']
        retained = get('AdaRES-Y85')
        assert retained['stage'] == 'blocked' and retained['blocker_kind'] == 'exclusive_gpu_slot'
        assert (retained['required_server'], retained['required_gpu']) == ('server-2', 0)
        assert retained['implementation_test_passed'] and not retained.get('admission_receipt')
        plan = tick.load(closure.artifact(ROOT, retained['native_smoke_plan_server2_gpu0']))
        retained_program = closure.artifact(ROOT, plan['program'])
        retained_frozen = ast.literal_eval(ast.parse(
            retained_program.read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(retained, plan, retained_frozen) == 7 and not plan['native_executed']
        occupations = {}
        for name in ('server-2', 'server-1'):
            evidence = ref('pipeline/checks/gpu_occupation_' + name + '_finish_' + token + '.json')
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token
            assert 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(value['observed_at'])).total_seconds() < 300
            blocked = next(candidate for candidate in work['candidates']
                if candidate.get('stage') == 'blocked' and candidate.get('blocker_kind') == 'exclusive_gpu_slot'
                and candidate.get('required_server') == name)
            server_context = next(server for server in context['queue']['servers'] if server['name'] == name)
            state = gpu_occupation.validated_state(ROOT, context, blocked, server_context, queue)
            assert state['busy_gpu_ids'] == [0] and not state['idle_gpu_ids']
            occupations[name] = evidence
        out = ROOT / ('pipeline/checks/y82_archive_y84_launch_' + token + '.json')
        assert not out.exists()
        terminal = tick.load(closure.artifact(ROOT, get('AdaRES-Y82')['terminal_evidence']))
        selection = ref('pipeline/checks/adares-y84_source_selection_' + token + '.json')
        dispatch = ref('pipeline/checks/adares-y84_dispatch_' + token + '.json')
        guard_log = ref('pipeline/checks/adares-y84_admission_guards_' + token + '.log')
        test_output = (ROOT / guard_log['path']).read_text(encoding='utf-8')
        assert 'Ran 19 tests' in test_output and 'OK' in test_output and 'skipped=' not in test_output
        tick.save(out, dict(complete=True, token=token, snapshot=view['generated_at'],
            Y82_terminal=get('AdaRES-Y82')['terminal_evidence'], Y82_archive=get('AdaRES-Y82')['terminal_archive_binding'],
            Y82_original_frozen_evaluation_hold=ref(hold_path), live_training=observations, occupations=occupations,
            Y84_native=candidate['native_smoke'], Y84_receipt=ref('pipeline/admissions/AdaRES-Y84.json'),
            Y84_full_frozen_protocol=candidate['post_training_freeze'], source_selection=selection, dispatch=dispatch,
            native_admission_guards=guard_log, retained_Y85_GPU0_plan=retained['native_smoke_plan_server2_gpu0'],
            new_candidates=0, new_admissions=1, new_launches=1, new_archives=1, formal_metrics_updated=False,
            GPU1_pause_preserved=True, Y57_pause_preserved=True,
            helpers=[ref(path) for path in ('scripts/advance_y84.py', 'scripts/y84_deployment.py',
                                             'scripts/test_y84_admission.py', __file__)]))
        candidate['evidence'].extend([ref(out), selection, dispatch, guard_log, ref(program)])
        get('AdaRES-Y82')['evidence'].append(ref(out))
        work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y84',
            kind='native_smoke', result='passed',
            command='python -B scripts/y84_server1_alternative.py run --token ' + token,
            observation='Actual original native YAML/TRAIN two steps and path/backward/update checks passed; '
                '19 admission/source/resource guards passed, complete protocol frozen, admitted and safely deployed. '
                'Exact Y84 native training counters increase; efficacy remains unknown.',
            evidence=[candidate['native_smoke'], candidate['post_training_freeze'], ref(out)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        memory = ROOT / 'pipeline/codex_memory.md'
        backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + token + '.md')
        assert not backup.exists()
        backup.write_bytes(memory.read_bytes())
        old = memory.read_text(encoding='utf-8').splitlines()

        def section(name):
            start = old.index('## ' + name) + 1
            end = next((index for index in range(start, len(old)) if old[index].startswith('## ')), len(old))
            return old[start:end]

        def evidence_text(evidence):
            return evidence['path'] + ' SHA' + evidence['sha256']

        def phase_text(cid):
            value = phases[cid]
            return cid + ' ' + value['phase'] + str(value['epoch']) + '/30 step' + str(value['step']) + '/' + str(value['count'])

        status = '; '.join(phase_text(cid) for cid in phases)
        best = terminal['checkpoints'][0]['best_miou']
        lines = ['# AdaRES_project Research Memory', '## Goal And Phase',
            '- Current token ' + token + '; snapshot ' + view['generated_at'] + '; deadline ' + view['deadline_at'] + '; never extend.',
            '- Actual CODEX_THREAD_ID=owner01a07e32-78bf-7670-925b-370911ea48db; Pipeline-AdaRES_project Disabled; experiment_first enabled.',
            '- Only server-1 GPU0 existing350W and server-2 GPU0 fixed280W, one independent experiment per card. GPU1 all use paused by user2026-11-04; Y57 separately paused; no server-6/cross-project/schedule/power changes.',
            '- Resource authority: ' + evidence_text(authority) + '. Supersedes old GPU1 grant without changing scientific goal/receipts.',
            '- Scientific goal SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a remains byte-identical.',
            '- PROJECT_ROOT=D:/project; PYTHONDONTWRITEBYTECODE=1; PYTHONUTF8=1; python -B. SSH keys work; no password file read or sudo assumptions.',
            '- Previous exact seven-section memory: ' + evidence_text(ref(backup)) + '; all historical evidence, bans and dedicated recovery requirements retained.',
            '## Champion And Baseline'] + section('Champion And Baseline') + [
            '## Active Hypotheses',
            '- Y84 visual_query_init: independent W2, initial generated query plus point-mass-weighted full valid original visual mean @ zero W256x256, once before initial prediction; +65536, total164695173, no RNG. Original inputs/heads/gates/sampling/seed1999/fixed30 retained. No efficacy claim.',
            '- Y84 original server1 native YAML/TRAIN two-step PASS: losses ' + repr(raw['losses']) + ', peak_reserved_bytes=' + str(raw['peak_reserved_bytes']) + '; actual identities/initial-zero-then-learned response/shared head/query_norm/finite new-original-input gradients/optimizer updates passed. Only two batches, not full-dataset memory guarantee.',
            '- Y84 native: ' + evidence_text(candidate['native_smoke']) + '; raw: ' + evidence_text(candidate['native_smoke_result']) + ';19 guards all passed including actual record (no skip).',
            '- Y84 receipt: ' + evidence_text(ref('pipeline/admissions/AdaRES-Y84.json')) + '; unchanged full evaluation/training freeze: ' + evidence_text(candidate['post_training_freeze']) + '.',
            '- Y84 source selection: ' + evidence_text(selection) + '; only seven actually tested additive files, other index/working files preserved; old signed helpers unchanged. master/server1 marker now7bef2cd313b870583c970c56580be5a78097234d; server2 marker remains2818322e6049b2541f1590a4fa9ad6854b3c598d.',
            '- Y85 mask_feature_norm remains prepared, blocked on original server2GPU0/280W. Shared LayerNorm256 on full x_mask(x) before batching, +512/total164630149/no RNG. Active changes masks/gate values; only bypass restores W2 outputs/gradients. No rerun CPU preparation.',
            '- Y85 source: ' + evidence_text(retained['source_manifest']) + '; original GPU0 plan: ' + evidence_text(retained['native_smoke_plan_server2_gpu0']) + '; retained GPU1 plan forbidden. No native GPU PASS/admission; no server1 plan.',
            '## Closed Or Failed Directions'] + section('Closed Or Failed Directions') + [
            '- Y84 preparation schema tests initially skipped the unavailable real GPU record honestly; after actual native execution all19 tests passed with no skip. No scientific model revision or falsification.',
            '## Running And Queue',
            '- Y82 naturally fixed30 complete, canonically archived on original server1GPU0; stable finite last30/best21, model/optimizer544/scheduler30-21; native Train/Val1..30 and original source/YAML/receipt/freeze verified. Original20261104_010148 retained. Training best ' + repr(best) + ' is not FIRST.',
            '- Y82 terminal: ' + evidence_text(get('AdaRES-Y82')['terminal_evidence']) + '; archive: ' + evidence_text(get('AdaRES-Y82')['terminal_archive_binding']) + '; original full server1GPU0/350W review held by verified Y84, no substitution/retrain.',
            '- 26 completed fixed30 models Y56 and Y58..Y82 await their original full frozen FIRST/cost; GPU1 routes remain human authorization holds. Do not replay original archival/checkpoint loading.',
            '- Y81 already canonically archived in73fa; original best .5078033804893494 is training-only. Do not live-observe/restart it. GPU1 pause remains.',
            '- Y57 separately paused at original epoch27/model/optimizer543/scheduler27; only explicit Y57 continuation permits28..30, never generic GPU1 reauthorization or restart from pretrain.']
        for cid in phases:
            task = tasks[cid]
            lines.append('- ' + phase_text(cid) + '; ' + task['server'] + ' GPU0 PID' + str(task['pid']) + '; exact source/YAML/receipt/marker/compute/power/increasing counters verified: ' + evidence_text(observations[cid]))
        lines += [line for line in section('Running And Queue') if line.startswith('- server1=') or line.startswith('- server2=')]
        lines += [
            '- Current observer: python -B scripts/y84_deployment.py observe --id AdaRES-Y83|AdaRES-Y84 --token NEW_TOKEN --label progress. Old y83 observer cannot accept new local HEAD; never alter old hash-bound helpers.',
            '- Fresh occupations: ' + evidence_text(occupations['server-2']) + '; ' + evidence_text(occupations['server-1']) + '; both authorized GPU0 busy/reserved. Finish evidence age<300s.',
            '## Open Questions And Risks',
            '- Full original9508 expressions/141 scenes, candidate+Y36/W2/B1 twice each, FIRST primary, scene bootstrap10000/seed1999, TRAIN cuts1487/3325 and original subgroup harm/mIoU/Acc25/50/activequeries/latency/memory/same-device cost remain unchanged.',
            '- No new formal FIRST/cost or promotion. Training best and native finite gradients do not establish benefit, full-data memory bounds or hardware repair.',
            '- GPU1 full evaluation/cost remains paused without migration/substitution. Y57 needs its separate explicit continuation. No other routine project operation needs renewed permission.',
            '- Next tick observe Y83/Y84 and dispatch ready work first. Y85 is nearest prepared original native candidate; execute only on fresh authorized idle unreserved server2GPU0 after exact current owner terminal verification/archival. Do not rerun CPU prep, Y82 finalization or old wait/launch closeout scripts.',
            '## Last Tick',
            '- ' + token + ': Y82 complete checkpoint verification/archive; Y84 actual original native two-step PASS,19 guards, exact seven-file source selection, original complete freeze/receipt/admission/safe dispatch and increasing training verified.',
            '- ' + status + '; new candidates0/admissions1/launches1/archives1; no formal FIRST/cost. Y85 prepared GPU0 plan retained, no GPU1 use.',
            '- Handoff: ' + evidence_text(ref(out)) + '; both pauses, exact data/private runtime/path mapping/scientific goal/old receipts and schedule preserved.',
            '- After official report/finish read only FINALIZED/PASS/context_committed/released and absent owned lock/pending/archive_pending; never preclaim or write after release.']
        assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        notes = ROOT / ('pipeline/checks/foreground_notes_' + token + '.json')
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            '完成：Y82自然结束，完整Train/Val1..30及原last30/best21/model/optimizer544/scheduler30-21稳定有限核验后归档；原产物保留。训练best=' + repr(best) + '，不是FIRST。',
            '运行：Y84原native YAML/TRAIN两步loss=' + repr(raw['losses']) + '及实际路径/有限梯度/参数更新通过，19准入与源码资源守卫全通过；只选择七测试A文件，原完整后评冻结、准入并安全派发，训练计数递增。',
            '运行：' + status + '；两授权GPU0占用与功耗/source/YAML/receipt核实；Y83原训练保持。',
            '新准备：补齐Y84准入/增量源码保护与部署观察工具；Y85既有CPU与原GPU0计划保留，未重放。新候选0/准入1/启动1/归档1，无正式FIRST/cost。'],
            decisions=[
                'GPU1全部AdaRES使用继续暂停，原GPU1冻结后评/cost不迁移；Y57单独暂停。',
                'Y82原完整server1GPU0/350W后评等独占空槽；训练best不能替代FIRST。Y85需原GPU0真实native通过才准入。',
                '保持科学goal字节/原receipt/数据/私有环境/path mapping/seed1999-fixed30/禁用项/既有功耗与调度，无需用户处理。']))
        assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        print(json.dumps(dict(handoff=ref(out), notes=notes.relative_to(ROOT).as_posix(),
                              memory_lines=len(lines), admissions=1, launches=1, archives=1)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    record(parser.parse_args().token)
