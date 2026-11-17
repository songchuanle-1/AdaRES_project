"""Preserve Y83's full review, verify Y84/Y85 and bind the new Y86 preparation."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from advance_y85 import prepared, validate_result
from y85_deployment import Y84, Y85
import closure
import gpu_occupation


def record(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        get = lambda cid: next(candidate for candidate in work['candidates'] if candidate['id'] == cid)
        done, owner = get('AdaRES-Y83'), get('AdaRES-Y85')
        assert done['stage'] == 'validation' and done['resource_work_kind'] == 'frozen_evaluation'
        assert (done['required_server'], done['required_gpu']) == ('server-2', 0)
        assert monitor.load_experiment(str(ROOT), done['id'])['status'] == 'done_complete'
        assert monitor.load_running(str(ROOT))['server-2']['0'] == owner['id']
        observed = tick.load(closure.artifact(ROOT, owner['latest_training_observation']))
        assert observed['token'] == token and observed['native_progress_confirmed']
        terminal = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
        archive = tick.load(closure.artifact(ROOT, done['terminal_archive_binding']))
        assert terminal['completed_epochs'] == {phase: list(range(1, 31)) for phase in ('Train', 'Val')}
        assert archive['only_exact_Y83_archived'] and archive['checkpoint_files_retained']
        assert not archive['full_frozen_evaluation_executed']
        freeze = tick.load(closure.artifact(ROOT, done['post_training_freeze']))
        assert (freeze['resource_condition']['server'], freeze['resource_condition']['gpu'],
                freeze['resource_condition']['power_limit_watts']) == ('server-2', 0, 280)
        hold = ROOT / ('pipeline/checks/y83_frozen_evaluation_hold_' + token + '.json')
        assert not hold.exists()
        tick.save(hold, dict(complete=True, token=token, id=done['id'], original_server='server-2',
            original_gpu=0, power_limit_watts=280, current_reserved_owner=owner['id'],
            native_owner_observation=owner['latest_training_observation'],
            terminal=done['terminal_evidence'], archive=done['terminal_archive_binding'],
            post_training_freeze=done['post_training_freeze'], full_frozen_evaluation_executed=False,
            helper=ref(__file__)))
        done.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
            blocker='Original complete Y83 checkpoints verified and archived; original full frozen FIRST/cost '
                'requires server-2 GPU0/280W, now occupied by verified independent Y85. Preserve all products and live training.',
            retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
            unblock_condition='Fresh exclusive original server-2 GPU0/280W, no compute or canonical reservation.')
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
        get = lambda cid: next(candidate for candidate in work['candidates'] if candidate['id'] == cid)
        view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
        context = tick.load(ROOT / 'pipeline/headless/.tick_context.json')
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        assert running == {'server-1': {'0': 'AdaRES-Y84'}, 'server-2': {'0': 'AdaRES-Y85'}}
        tasks = {task['id']: task for tasks in queue['gpu_queues'].values() for task in tasks}
        assert set(tasks) == {'AdaRES-Y57', 'AdaRES-Y84', 'AdaRES-Y85'}
        assert tasks['AdaRES-Y57']['status'] == 'paused' and tasks['AdaRES-Y57']['dispatch_hold']
        assert tasks['AdaRES-Y57']['resume_requires_user_authorization']
        assert closure.git(ROOT, 'rev-parse', 'HEAD') == Y85
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        authority = ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')
        assert authority['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
        servers = monitor.load_servers(str(ROOT))
        assert servers['server-1']['allowed_gpus'] == servers['server-2']['allowed_gpus'] == [0]
        assert servers['server-6']['allowed_gpus'] == []
        observations, phases = {}, {}
        for cid, name, commit, pid in [('AdaRES-Y84', 'server-1', Y84, 243952),
                                       ('AdaRES-Y85', 'server-2', Y85, 22090)]:
            task = tasks[cid]
            assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid
            assert (task['server'], str(task['gpu']), task['code_commit']) == (name, '0', commit)
            closure.admission(ROOT, task)
            evidence = get(cid)['latest_training_observation']
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token and value['pid'] == pid and value['native_progress_confirmed']
            observations[cid], phases[cid] = evidence, value['after']['latest']
        candidate, _, _, frozen = prepared(work)
        raw = tick.load(closure.artifact(ROOT, candidate['native_smoke_result']))
        validate_result(raw, candidate, frozen)
        freeze85 = tick.load(closure.artifact(ROOT, candidate['post_training_freeze']))
        parent = tick.load(closure.artifact(ROOT, freeze85['parent_protocol']))
        restored = dict(freeze85['evaluation'], methods=list(freeze85['evaluation']['methods']))
        restored['methods'][0] = 'AdaRES-Y64'
        assert restored == parent['evaluation'] and freeze85['training'] == parent['training']
        prepared86 = get('AdaRES-Y86')
        assert prepared86['stage'] == 'blocked' and prepared86['blocker_kind'] == 'exclusive_gpu_slot'
        assert (prepared86['required_server'], prepared86['required_gpu']) == ('server-2', 0)
        assert prepared86['implementation_test_passed'] and not prepared86.get('admission_receipt')
        version86 = tick.load(closure.artifact(ROOT, prepared86['source_manifest']))
        cpu86 = tick.load(closure.artifact(ROOT, version86['CPU_result']))
        prep86 = tick.load(closure.artifact(ROOT, prepared86['native_preparation']))
        assert cpu86['complete'] and cpu86['additional_parameters'] == 2 and cpu86['parameters'] == 164629639
        assert not cpu86['CUDA_initialized'] and not cpu86['native_smoke_passed']
        assert prep86['guard_tests_passed'] == 9 and prep86['exact_callback_functions_and_classes'] == 7
        assert prep86['CPU_native_callbacks_actually_executed'] and not prep86['native_executed']
        plan86 = tick.load(closure.artifact(ROOT, prepared86['native_smoke_plan_server2_gpu0']))
        frozen86 = ast.literal_eval(ast.parse(closure.artifact(ROOT, plan86['program']).read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(prepared86, plan86, frozen86) == 7 and not plan86['native_executed']
        assert set(prep86['native_plans']) == {'0'} and not prepared86.get('native_smoke_plan_server2_gpu1')
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
        guard_log = ref('pipeline/checks/adares-y85_admission_guards_' + token + '.log')
        tests = (ROOT / guard_log['path']).read_text(encoding='utf-8')
        assert 'Ran 19 tests' in tests and 'OK' in tests and 'skipped=' not in tests
        selection = ref('pipeline/checks/adares-y85_source_selection_' + token + '.json')
        dispatch = ref('pipeline/checks/adares-y85_dispatch_' + token + '.json')
        out = ROOT / ('pipeline/checks/y85_launch_y86_preparation_' + token + '.json')
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, snapshot=view['generated_at'],
            Y83_terminal=get('AdaRES-Y83')['terminal_evidence'], Y83_archive=get('AdaRES-Y83')['terminal_archive_binding'],
            original_Y83_frozen_evaluation_hold=ref(hold), live_training=observations, occupations=occupations,
            Y85_native=candidate['native_smoke'], Y85_receipt=ref('pipeline/admissions/AdaRES-Y85.json'),
            Y85_full_frozen_protocol=candidate['post_training_freeze'], source_selection=selection, dispatch=dispatch,
            Y85_admission_guards=guard_log, Y86_CPU=version86['CPU_result'], Y86_version=prepared86['source_manifest'],
            Y86_native_preparation=prepared86['native_preparation'], new_candidates=1, new_admissions=1,
            new_launches=1, new_archives=1, formal_metrics_updated=False,
            GPU1_pause_preserved=True, Y57_pause_preserved=True,
            helpers=[ref(path) for path in ('scripts/create_y83_terminal_helpers.py', 'scripts/y83_terminal_contract.py',
                'scripts/finalize_y83_fixed30.py', 'scripts/prepare_y83_terminal_handoff.py', 'scripts/test_y83_terminal_handoff.py',
                'scripts/create_y85_execution_helpers.py', 'scripts/advance_y85.py', 'scripts/y85_deployment.py',
                'scripts/test_y85_admission.py', 'scripts/prepare_y86_overlap_skip.py', 'scripts/retain_y86_cpu_native.py',
                'scripts/y86_observer_core.py', 'scripts/probe_y86_cpu_body.py', 'scripts/probe_y86_cpu_tail.py',
                'scripts/probe_y86_native_tail.py', 'scripts/test_y86_native_plan.py', __file__)]))
        candidate['evidence'].extend([ref(out), selection, dispatch, guard_log])
        get('AdaRES-Y83')['evidence'].append(ref(out))
        prepared86['evidence'].append(ref(out))
        work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y86',
            kind='implementation_test', result='passed',
            command='python -B scripts/run_experiment_first_probe.py y86_cpu pipeline/checks/y86_cpu_' + token + '_program.py --token ' + token + ' --seconds 480',
            observation='Recorded full CPU model/backbone/original TRAIN loading, independent scalar/zero/RNG/permutation controls, seven-stage W2 anchors and original input-gradient controls, fourteen original max-overlap/readout consumers and two synthetic optimizer steps passed. Exact seven-file version and GPU0-only native plan/seven identical callbacks/nine guards retained; no Y86 GPU PASS/admission/efficacy.',
            evidence=[version86['CPU_result'], prepared86['source_manifest'], prepared86['native_preparation'], ref(out)])
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

        def e(evidence):
            return evidence['path'] + ' SHA' + evidence['sha256']

        def phase(cid):
            value = phases[cid]
            return cid + ' ' + value['phase'] + str(value['epoch']) + '/30 step' + str(value['step']) + '/' + str(value['count'])

        terminal = tick.load(closure.artifact(ROOT, get('AdaRES-Y83')['terminal_evidence']))
        best = terminal['checkpoints'][0]['best_miou']
        status = '; '.join(phase(cid) for cid in phases)
        lines = ['# AdaRES_project Research Memory', '## Goal And Phase',
            '- Current token ' + token + '; snapshot ' + view['generated_at'] + '; deadline ' + view['deadline_at'] + '; never extend.',
            '- Actual CODEX_THREAD_ID=owner01a07e32-78bf-7670-925b-370911ea48db; Pipeline-AdaRES_project Disabled; experiment_first enabled.',
            '- Only server1GPU0 existing350W and server2GPU0 fixed280W, one independent experiment per card. GPU1 all use user-paused; Y57 separately paused. No server6/cross-project/scheduler/power changes.',
            '- Resource authority: ' + e(authority) + '; overrides old GPU1 grant without changing scientific goal/receipts.',
            '- Goal SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a stays byte-identical. PROJECT_ROOT=D:/project, PYTHONDONTWRITEBYTECODE=1, PYTHONUTF8=1, python -B; SSH keys, no password file or sudo.',
            '- Exact previous seven-section memory: ' + e(ref(backup)) + '; all historical bans/negative evidence/special recovery requirements retained.',
            '## Champion And Baseline'] + section('Champion And Baseline') + [
            '## Active Hypotheses',
            '- Y85 mask_feature_norm: independent W2, one shared affine LayerNorm256 on full x_mask(x) before batching, +512/total164630149/no RNG. Active transform changes mask/gate values; only bypass matches W2. Original all data/2D/backbone/query heads/norms/coordinates/sampling/SCG/k/gamma/gate algorithm/targets/loss/selector/seed1999/fixed30 retained.',
            '- Y85 actual original GPU0 native TRAIN PASS: losses ' + repr(raw['losses']) + ', peak_reserved_bytes=' + str(raw['peak_reserved_bytes']) + '; exact raw/norm/batching/prediction consumers, all channels/eps/float64 reference, original shared heads/gates, finite new-original-input gradients and two real updates verified. Two batches are not efficacy or a full-data memory bound.',
            '- Y85 native: ' + e(candidate['native_smoke']) + '; receipt: ' + e(ref('pipeline/admissions/AdaRES-Y85.json')) + '; original full freeze: ' + e(candidate['post_training_freeze']) + ';19 guards passed on actual record, no skip.',
            '- Master/server2 marker=' + Y85 + '; server1 marker=' + Y84 + '. Exact seven tested Y85 A files selected, other index/working files and old signed helpers preserved. Selection: ' + e(selection),
            '- Y86 overlap_readout_skip: independent W2, shared confidence logits += the SAME already consumed max-overlap scalar times zero vector2 at all seven stages; +2/total164629639/no RNG. Original graph/statistic/formula/head/masks/targets/loss/selector unchanged; no missing-information/novelty/efficacy claim. Not rejected Y34/Y47 new-graph information, Y62 density, Y63 bilinear or Y76 stage input.',
            '- Y86 full CPU original model/released backbone/TRAIN162738 points8 expressions; float32/64 independent scalar/zero/RNG/permutation/gradient controls, seven-stage exact W2 and input-gradient anchors, confidence-only intervention/restoration, fourteen original readout/overlap consumers and two synthetic optimizer steps passed. No native sparse/GPU PASS.',
            '- Y86 CPU: ' + e(version86['CPU_result']) + '; version: ' + e(prepared86['source_manifest']) + '; commit=' + prepared86['code_commit'] + '; parent=' + Y85 + '; isolated seven-file A delta retained, current HEAD/index/production unchanged.',
            '- Y86 GPU0-only native preparation: ' + e(prepared86['native_preparation']) + '; nine rejection guards, seven literally identical callbacks actually executed on CPU, stale-goal/source-tamper refusals and Python3.8 syntax passed. No GPU1 plan. Wait original server2GPU0/280W idle/unreserved after Y85 exact natural terminal/archive; do not replay CPU/retain/plan.',
            '## Closed Or Failed Directions'] + section('Closed Or Failed Directions') + [
            '- Y83 terminal25 guards and actual stable finite checkpoint verification passed; Y85 native/admission19 guards and Y86 CPU/native-plan9 guards passed first execution. No new scientific rejection or model revision. Preserve all prior failures.',
            '## Running And Queue',
            '- Y83 naturally fixed30 complete; original20261103_175050 retained, stable finite last30/best20/model/optimizer553/scheduler30-20, native Train/Val1..30/source/YAML/receipt/freeze verified; canonical explicit GPU0 archive. Training best ' + repr(best) + ' is not FIRST.',
            '- Y83 terminal: ' + e(get('AdaRES-Y83')['terminal_evidence']) + '; archive: ' + e(get('AdaRES-Y83')['terminal_archive_binding']) + '; original full server2GPU0/280W evaluation held by Y85, no substitution/retrain.',
            '- 27 fixed30 models Y56 and Y58..Y83 await original full frozen FIRST/cost; original resources retained, GPU1 routes human-authorization held. No repeated archived checkpoint loading/terminal helpers.']
        lines += [line for line in section('Running And Queue')
                  if line.startswith('- Y82 ') or line.startswith('- Y81 ') or line.startswith('- Y57 ')]
        for cid in phases:
            lines.append('- ' + phase(cid) + '; ' + tasks[cid]['server'] + ' GPU0 PID' + str(tasks[cid]['pid'])
                + '; source/YAML/receipt/marker/compute/power/native increase verified: ' + e(observations[cid]))
        lines += [line for line in section('Running And Queue') if line.startswith('- server1=') or line.startswith('- server2=')]
        lines += [
            '- Current observer: scripts/y85_deployment.py observe --id AdaRES-Y84|AdaRES-Y85 --token NEW_TOKEN --label progress; old y84 observer cannot accept HEADY85. Never alter signed historical helpers.',
            '- Fresh occupation: ' + e(occupations['server-2']) + '; ' + e(occupations['server-1']) + '; both authorized GPU0 busy/reserved, finish needs age<300s.',
            '## Open Questions And Risks',
            '- Full original9508 expressions/141 scenes, candidate+Y36/W2/B1 twice each, FIRST primary, scene bootstrap10000/seed1999, TRAIN cuts1487/3325 and subgroup harm/mIoU/Acc25/50/activequeries/latency/memory/same-device cost remain unchanged.',
            '- No new formal FIRST/cost or promotion. Training best/native gradients/CPU controls are not trained efficacy, hardware repair or full-data memory bounds.',
            '- GPU1 all use including full frozen evaluation/cost paused; never migrate/substitute. Y57 needs separate explicit continuation; no repeated ordinary-operation permission request.',
            '- Next tick observe Y84/Y85 and dispatch existing work first; Y86 is nearest prepared GPU0 native candidate. Only exact original YAML/source/goal and fresh authorized exclusive idle/unreserved slot allow actual native task, full original freeze/receipt/admission/safe deployment/increasing counters. Do not replay original Y83 terminal, old closeout/wait helpers or Y86 CPU/retain/plan.',
            '## Last Tick',
            '- ' + token + ': Y83 exact checkpoint/source/full-log terminal verification/archive; Y85 original native two-step PASS/19 guards/seven-file source selection/full freeze/receipt/admission/safe dispatch and increasing counters; Y86 new independent CPU/source/GPU0-only native-plan preparation.',
            '- ' + status + '; new candidates1/admissions1/launches1/archives1. Y86 nine guards/seven exact callbacks, no Y86 GPU PASS/admission. No formal FIRST/cost.',
            '- Handoff: ' + e(ref(out)) + '; both user pauses, exact data/private runtime/mappings/scientific goal/old receipts/schedule preserved.',
            '- Official report/finish then read only FINALIZED/PASS/context_committed/released and absent owned lock/pending/archive_pending; never preclaim or write after release.']
        assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        notes = ROOT / ('pipeline/checks/foreground_notes_' + token + '.json')
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            '完成：Y83自然fixed30，完整Train/Val1..30及原last30/best20/model/optimizer553/scheduler30-20稳定有限核验后归档；训练best=' + repr(best) + '，不是冻结FIRST。',
            '运行：Y85原native YAML/TRAIN两步loss=' + repr(raw['losses']) + '及实际路径/有限梯度/更新通过，19准入/源码/资源守卫通过；七测试A文件安全增量部署，原完整后评冻结后准入/派发并核训练递增。Y84保持原运行。',
            '运行：' + status + '；两GPU0原功耗/source/YAML/receipt/占用核实。',
            '新准备：Y86独立既有max-overlap到confidence的两参数直接读出，完整CPU原model/backbone/TRAIN/float32-64标量/零RNG/七阶段W2与输入梯度/原消费者及两合成优化步通过；七文件版本与GPU0-only原生计划、7精确CPU已执行回调/9拒绝守卫保留。尚未GPU检查/准入。'],
            decisions=[
                'GPU1全部AdaRES使用与Y57单独暂停保持；原GPU1冻结后评不迁移/替换。',
                'Y83原server2GPU0/280W完整后评保留，现由Y85占用。训练best不可晋升为FIRST。',
                'Y86仅准备不抢占；下次授权GPU0独占空闲后实际native通过再准入。数据/环境/mapping/科学goal/原receipt/seed1999-fixed30/禁用/功耗/调度不变，无需用户处理。']))
        assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        print(json.dumps(dict(handoff=ref(out), notes=notes.relative_to(ROOT).as_posix(),
                              memory_lines=len(lines), candidates=1, admissions=1, launches=1, archives=1)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    record(parser.parse_args().token)
