"""Retain the tested Y79 terminal plan alongside current protected-run checks."""
import argparse
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from record_prepared_y82_y83_wait_v2 import record
import closure


def closeout(token):
    record(token, False)
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y79')
        plan_ref = candidate['terminal_handoff_plan']
        plan = tick.load(closure.artifact(ROOT, plan_ref))
        assert plan['token'] == token and plan['id'] == 'AdaRES-Y79'
        assert plan['CPU_guard_tests_passed'] == 25 and candidate['stage'] == 'training'
        for key in ('full_terminal_verification_executed', 'checkpoint_loaded', 'archived',
                    'training_stopped', 'queue_changed', 'remote_mutations'):
            assert plan[key] is False
        for dependency in plan['dependencies']:
            closure.artifact(ROOT, dependency)
        location = tick.load(closure.artifact(ROOT, plan['declared_location']))
        assert location['checkpoint_directory'] == '20261102_230148'
        handoff = ref('pipeline/checks/prepared_y82_y83_resource_' + token + '.json')
        notes_path = ROOT / ('pipeline/checks/foreground_notes_' + token + '.json')
        notes = tick.load(notes_path)
        description = ('Y79原server-1 GPU0/350W自然fixed30终态契约已准备；25项生命周期守卫通过，'
            '实核原source/YAML/receipt/freeze/原目录20261102_230148和Y80/Y81精确追加版本链。'
            '未加载checkpoint、未停止或归档训练；仍须自然退出、完整Train-Val1..30和稳定finite '
            'last30/best/model/optimizer/scheduler实核后才允许显式GPU0归档。')
        notes['actions'].append('新准备：' + description)
        notes['decisions'].append('Y79自然完成后用scripts/finalize_y79_fixed30.py及其绑定计划；'
            '保留其他运行和Y57暂停。Y82/Y83仍无server-1原生计划，不得只改server字段启动。')
        tick.save(notes_path, notes)
        out = ROOT / ('pipeline/checks/y79_terminal_preparation_closeout_' + token + '.json')
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, terminal_plan=plan_ref,
            protected_training_resource_handoff=handoff, notes=ref(notes_path),
            new_candidates=0, admissions=0, launches=0, archives=0,
            helper=ref(__file__)))
        memory = ROOT / 'pipeline/codex_memory.md'
        lines = memory.read_text(encoding='utf-8').splitlines()
        text = '- ' + description + ' plan：' + plan_ref['path'] + ' SHA' + plan_ref['sha256'] + '。'
        lines.insert(lines.index('## Closed Or Failed Directions'), text)
        for i, line in enumerate(lines):
            if line.startswith('- 下轮先y81_deployment'):
                lines[i] = ('- 下轮先核三原运行及现有队列。Y79已有本轮25守卫原终态计划，'
                    '自然完成后核完整Train-Val/稳定finite checkpoint再显式server1GPU0归档；'
                    '用finalize_y79_fixed30.py，不能停止或提前归档。Y82/Y83无server1计划，'
                    '须先精确私有环境/路径/350W计划及检查才能用该卡。')
        lines.append('- 本轮另完成Y79原终态契约准备；未加载checkpoint/未归档。closeout：'
            + out.relative_to(ROOT).as_posix() + ' SHA' + ref(out)['sha256'] + '。')
        assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        print(__import__('json').dumps(dict(closeout=ref(out), terminal_plan=plan_ref,
            notes=notes_path.relative_to(ROOT).as_posix(), memory_lines=len(lines))))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    closeout(parser.parse_args().token)
