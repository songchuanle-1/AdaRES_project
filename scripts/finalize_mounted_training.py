"""Capture both admitted native runs and reconcile only their real resource recovery."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import authorize_prepared_candidates as scoped
import closure
import dispatch_prepared_candidates as prepared
import foreground_tick as tick
import pipeline_monitor as monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    scoped.lease(args.token)
    writer = monitor.acquire_lock(str(ROOT))
    if writer is None:
        raise ValueError('Project writer busy')
    try:
        queue = monitor.load_queue(str(ROOT))
        tasks = queue['gpu_queues']['server-2']
        assert {t['id'] for t in tasks} == {'AdaRES-Y52', 'AdaRES-Y54'}
        assert all(t['status'] == 'running' and t.get('launch_verified') for t in tasks)
        assert {str(t['gpu']) for t in tasks} == {'0', '1'}
        server = monitor.load_servers(str(ROOT))['server-2']
        work = tick.load(ROOT / 'pipeline/research_work.json')
        files = {}
        for task in tasks:
            closure.admission(ROOT, task)
            candidate = next(c for c in work['candidates'] if c['id'] == task['id'])
            _, rows = scoped.checked_bundle(task['id'], candidate)
            files.update({r['path'].removeprefix('AdaRES/'): r['sha256'] for r in rows})
        started = datetime.now(timezone(timedelta(hours=8))).isoformat()
        before = prepared.snapshot(server, tasks, list(files))
        time.sleep(5)
        after = prepared.snapshot(server, tasks, list(files))
        progress = []
        for task in tasks:
            samples = []
            for capture in (before, after):
                row = capture['tasks'][task['id']]
                assert row['process_live'] and row['cwd'] == server['remote_code_root']
                assert task['config'] in row['cmdline'] and '--num_gpus' in row['cmdline']
                assert not any(s in row['tail'] for s in ('Traceback', 'CUDA out of memory', 'RuntimeError'))
                matches = re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?\|\s*(\d+)/(\d+)', row['tail'])
                assert matches
                samples.append(list(map(int, matches[-1])))
            assert (samples[1][0], samples[1][2]) > (samples[0][0], samples[0][2])
            progress.append(dict(id=task['id'], pid=task['pid'], gpu=task['gpu'], samples=samples))
        assert after['source_hashes'] == files
        now = datetime.now(timezone(timedelta(hours=8))).isoformat()
        path = f'pipeline/analyses/mounted_training_verified_{args.token}.json'
        skill = Path('C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
        fixes = [{'path': str(skill/name), 'sha256': hashlib.sha256((skill/name).read_bytes()).hexdigest()}
                 for name in ('source_integrity.py', 'test_source_integrity.py')]
        tick.save(ROOT/path, dict(complete=True,token=args.token,observed_from=started,observed_until=now,
            progress=progress,before=before,after=after,previous_design_cycle=work.get('design_cycle'),
            guard_files=fixes,tests=dict(source_integrity=16,closure_foreground_context=109,failed_final=0,
                intermediate_failure='16-test iteration had4 generated-code errors; escaping repaired and final16 passed'),
            repairs=['Python3.8 containment API compatibility', 'YAML parsed structure identity',
                     'Python CRLF/LF identity only; binary and other code changes still rejected'],
            y52_attempt='Original PID15934 stopped at epoch1 step535/4686 with no checkpoint; old logs/history preserved. User requested a fresh fixed30 attempt, not a stateful resume.',
            efficacy='Unknown; increasing native steps prove execution only'))
        evidence = scoped.ref(path)
        for candidate in work['candidates']:
            if candidate['id'] not in ('AdaRES-Y52', 'AdaRES-Y54', 'AdaRES-Y56'):
                continue
            candidate['evidence'].append(evidence)
            if candidate['id'] == 'AdaRES-Y56':
                prior = {k: candidate[k] for k in ('blocker_kind','blocker','retry_after','unblock_condition') if k in candidate}
                prior.update(resolved_at=now,resolved_by=evidence)
                candidate.setdefault('blocker_history', []).append(prior)
                for key in ('blocker_kind','blocker','retry_after','unblock_condition','required_server'):
                    candidate.pop(key, None)
                candidate.update(stage='diagnosis', next_action='Continue SAME Y56 at next research work: prepare source-bound actual TRAIN duplicate incidence and trained-parent controls; GPU capture requires a live idle authorized slot. Synthetic CPU signal is not efficacy or admission.')
        work.pop('design_cycle', None)
        closure.validate_work(ROOT,tick.load(ROOT/'pipeline/headless/.tick_context.json'),work['candidates'],queue)
        scoped.lease(args.token)
        tick.save(ROOT/'pipeline/research_work.json',work)
        dw = monitor.load_deathwatch(str(ROOT))
        dw.setdefault('server-2', {}).pop('adares_y52_point_target_res',None)
        monitor.save_deathwatch(str(ROOT),dw)
        monitor.log_decision(str(ROOT),'manual','User confirms mounted storage and requests training.',
            'Both SAME admitted fixed30 candidates verified with increasing native steps: Y54 GPU0, Y52 fresh checkpoint-free attempt GPU1.',
            'Original partial Y52 logs preserved; guard compatibility/format fixes passed125 tests; no model, metric gate, schedule or other-project change.',path)
        print(__import__('json').dumps(dict(evidence=evidence,observed_from=started,observed_until=now,
            progress=progress,tests_passed=125),indent=2))
    finally:
        writer.close()


if __name__ == '__main__':
    main()
