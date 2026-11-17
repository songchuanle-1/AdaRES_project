"""Add only the missing retry/unblock metadata after an explicit postflight error."""
import argparse
from datetime import datetime, timedelta
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure


def repair(token):
    lease(token, 60)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y86')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'numerical_control_failure'
        assert candidate['failure_scope'] == 'implementation' and not candidate.get('admission_receipt')
        failure = closure.artifact(ROOT, candidate['native_smoke_failure'])
        diagnosis = closure.artifact(ROOT, candidate['native_failure_diagnosis'])
        before = dict(retry_after=candidate.get('retry_after'), unblock_condition=candidate.get('unblock_condition'))
        assert not before['retry_after'] and not before['unblock_condition']
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        candidate.update(retry_after=(datetime.now().astimezone() + timedelta(hours=4)).isoformat(),
            unblock_condition='New evidence or a pre-registered bounded implementation revision that passes the original unchanged positive-response guard on an authorized exclusive idle GPU0; a timer is not permission to replay the unchanged failed smoke or relax the control.')
        output = ROOT / ('pipeline/checks/y86_hold_metadata_repair_' + token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, id='AdaRES-Y86', before=before,
            retry_after=candidate['retry_after'], unblock_condition=candidate['unblock_condition'],
            reason='Postflight explicitly required retry and unblock condition for the blocked record.',
            native_failure=ref(failure), native_diagnosis=ref(diagnosis), old_evidence_preserved=True,
            model_or_plan_changed=False, guard_changed=False, native_replayed=False, queue_changed=False,
            helper=ref(__file__)))
        candidate['evidence'].append(ref(output))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        original = ROOT / ('pipeline/checks/foreground_notes_' + token + '.json')
        notes = tick.load(original)
        notes['actions'].append('收尾首检因Y86挂起记录缺retry/unblock失败；仅补明确复查时间与新证据/原控制通过的解锁条件，原失败模型/日志/数值守卫及Y87运行不变。')
        destination = original.with_name(original.stem + '_retry_repair.json')
        assert not destination.exists()
        tick.save(destination, notes)
        memory = ROOT / 'pipeline/codex_memory.md'
        text = memory.read_text(encoding='utf-8')
        assert len(text.splitlines()) < 118
        memory.write_text(text + '- Initial postflight required Y86 retry/unblock metadata; exact non-replay hold conditions added without model/guard changes. Repair: ' + str(output.relative_to(ROOT).as_posix()) + '\n',
                          encoding='utf-8', newline='\n')
        assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        print(json.dumps(dict(repair=ref(output), notes=destination.relative_to(ROOT).as_posix())))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    repair(parser.parse_args().token)
