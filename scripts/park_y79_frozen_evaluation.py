"""Bind completed Y79's original evaluation hold to the actual new GPU0 owner."""
import argparse
from datetime import datetime, timedelta
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure


def park(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT)); assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y79')
        assert candidate['stage'] == 'validation' and candidate['resource_work_kind'] == 'frozen_evaluation'
        assert candidate['required_server'] == 'server-1' and candidate['required_gpu'] == 0
        assert monitor.load_experiment(str(ROOT), 'AdaRES-Y79')['status'] == 'done_complete'
        assert monitor.load_running(str(ROOT))['server-1']['0'] == 'AdaRES-Y82'
        for field in ('terminal_evidence', 'terminal_archive_binding', 'post_training_freeze'):
            closure.artifact(ROOT, candidate[field])
        attempt = work['tick_attempt']
        assert attempt['candidate_id'] == 'AdaRES-Y84' and attempt['tick_started_at'] == tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at']
        candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
            blocker='Y79 original fixed30 and stable finite checkpoints verified and canonically archived. Original full frozen FIRST/cost requires server-1 GPU0/350W, now owned by verified independent Y82. Preserve original artifacts and all live training.',
            retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
            unblock_condition='Fresh exclusive original server-1 GPU0/350W with no compute process or canonical reservation; execute the original full frozen protocol.')
        tick.save(ROOT / 'pipeline/research_work.json', work)
    finally:
        writer.close()
    subprocess.run([sys.executable, '-B', 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/gpu_occupation.py',
        'AdaRES_project', '--token', token, '--server', 'server-1', '--label', 'postarchive'], cwd=ROOT, check=True, capture_output=True, timeout=45)
    writer = monitor.acquire_lock(str(ROOT)); assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y79')
        evidence = candidate['occupation_evidence']
        value = tick.load(closure.artifact(ROOT, evidence))
        assert value['token'] == token and 'AdaRES-Y79' in value['candidate_ids']
        out = ROOT / ('pipeline/checks/y79_frozen_evaluation_hold_' + token + '.json'); assert not out.exists()
        tick.save(out, dict(complete=True, token=token, id='AdaRES-Y79', original_server='server-1', original_gpu=0,
            power_limit_watts=350, actual_reserved_owner='AdaRES-Y82', occupation=evidence,
            terminal=candidate['terminal_evidence'], full_frozen_evaluation_executed=False, helper=ref(__file__)))
        candidate['evidence'].append(ref(out))
        # Keep the real recorded Y84 implementation attempt; retain this separate
        # fresh resource check in Y79's evidence rather than replacing that result.
        work['tick_attempt'] = attempt
        tick.save(ROOT / 'pipeline/research_work.json', work)
        notes = ROOT / ('pipeline/checks/foreground_notes_' + token + '.json')
        n = tick.load(notes)
        n['actions'].append('Y79原完整冻结评估/同卡cost明确等待server1GPU0/350W；该卡现由已核Y82独占，fresh occupation已绑定Y79，原checkpoint/freeze保持，无新增用户操作。')
        tick.save(notes, n)
        memory = ROOT / 'pipeline/codex_memory.md'
        lines = memory.read_text(encoding='utf-8').splitlines()
        lines.append('- Y79原冻结完整FIRST/cost已按server1GPU0/350W实际Y82占用挂起；最新server1 occupation：' + evidence['path'] + ' SHA' + evidence['sha256'] + '；hold：' + out.relative_to(ROOT).as_posix() + ' SHA' + ref(out)['sha256'] + '。')
        assert len(lines) < 120
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        print(__import__('json').dumps(dict(hold=ref(out), occupation=evidence, original_evaluation_pending=True)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True)
    park(parser.parse_args().token)
