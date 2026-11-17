"""Archive Y59 only after full fixed30 checkpoint and exact-process verification."""

import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from dispatch_prepared_candidates import snapshot
from check_ef_execution import validate_frozen
from finalize_y58_y59_handoff import REMOTE
import closure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 360)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        queue = monitor.load_queue(str(ROOT))
        tasks = queue['gpu_queues']['server-2']
        old = next(t for t in tasks if t['id'] == 'AdaRES-Y59')
        assert old['status'] == 'running' and old['pid'] == 28712 and int(old['gpu']) == 0
        paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
        assert paused['status'] == 'paused' and paused['dispatch_hold']
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == old['id'])
        assert candidate['stage'] == 'training'
        closure.admission(ROOT, old)
        closure.artifact(ROOT, candidate['post_training_freeze'])
        plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
        program = closure.artifact(ROOT, plan['program'])
        frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(candidate, plan, frozen) == 7
        before = snapshot(server, [old], [])
        assert before['gpu_query_rc'] == before['compute_query_rc'] == 0
        assert not before['tasks'][old['id']]['process_live']
        basename = Path(old['config']).stem
        base = ROOT / ('pipeline/checks/y59_fixed30_' + args.token)
        assert not base.with_suffix('.json').exists() and not base.with_suffix('.log').exists()
        payload = dict(id=old['id'], pid=old['pid'], root=server['remote_code_root'],
            work=server['work_dir'], log=old['log'], config=old['config'], basename=basename,
            checkpoint_directory='20261022_104722')
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            server['user'] + '@' + server['host'],
            'PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=' +
            shlex.quote(server['site_packages']) + ' ' + shlex.quote(server['python']) + ' -S -B -'],
            input=('P=' + repr(payload) + '\n' + REMOTE).encode(), capture_output=True, timeout=240)
        base.with_suffix('.log').write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0, 'Full checkpoint verification failed; preserve raw evidence'
        proof = json.loads(response.stdout)
        assert proof['complete'] and not proof['CUDA_initialized'] and proof['id'] == old['id']
        after = snapshot(server, [old], [])
        assert after['gpu_query_rc'] == after['compute_query_rc'] == 0
        assert not after['tasks'][old['id']]['process_live']
        proof.update(observed_at=datetime.now().astimezone().isoformat(), live=after,
            source_checks=dict(exact_files=7, admission_verified=True, freeze_preserved=True),
            readonly_checkpoint_verifier=ref(ROOT / 'scripts/finalize_y58_y59_handoff.py'),
            command='python -B scripts/finalize_y59_fixed30.py --token ' + args.token,
            explanation='Verified last epoch30/model/optimizer/scheduler and complete native terminal '
                        'log plus absent exact process; no fabricated deathwatch counts or scientific promotion.')
        tick.save(base.with_suffix('.json'), proof)
        destination = ROOT / ('Checkpoints/server-2/' + basename + '/' + args.token + '/fixed30.log')
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        subprocess.run(['scp', '-q', server['user'] + '@' + server['host'] + ':' + old['log'],
                        str(destination)], check=True, timeout=60)
        assert ref(destination)['sha256'] == proof['log_sha256']
        lease(args.token, 120)
        original_pull = monitor.pull_log
        def verified_pull(project_dir, host, user, remote_log, cfg_name, state, srv_name=''):
            assert Path(project_dir) == ROOT and host == server['host'] and user == server['user']
            assert remote_log == old['log'] and cfg_name == basename
            assert state == 'done_complete' and srv_name == 'server-2'
            return str(destination)
        monitor.pull_log = verified_pull
        events = []
        try:
            monitor.finalize_task(str(ROOT), queue, 'server-2', server['host'], server['user'], old,
                basename, 'done_complete', proof['checkpoints'][0]['best_miou'],
                '30/30', '0', old['log'], events)
        finally:
            monitor.pull_log = original_pull
        experiment = monitor.load_experiment(str(ROOT), old['id'])
        experiment.update(terminal_evidence=ref(base.with_suffix('.json')),
                          terminal_checkpoint_metadata=proof['checkpoints'])
        monitor.save_experiment(str(ROOT), experiment)
        deathwatch = monitor.load_deathwatch(str(ROOT))
        deathwatch.get('server-2', {}).pop(basename, None)
        monitor.save_deathwatch(str(ROOT), deathwatch)
        additions = [ref(base.with_suffix('.json')), ref(base.with_suffix('.log')),
                     ref(__file__), proof['readonly_checkpoint_verifier']]
        candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
        candidate.update(stage='validation', scientific_status='fixed30_complete_frozen_review_pending',
            resource_work_kind='frozen_evaluation',
            next_action='Preserve completed fixed30 best/last checkpoints and immutable '
                'post_training_freeze. Perform only the registered full frozen evaluation and '
                'trained-cost comparison on exclusive authorized GPU0, without tuning, restarting '
                'or scientific promotion from the training-log best. First advance prepared Y60 smoke.')
        for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after',
                    'unblock_condition', 'occupation_evidence'):
            candidate.pop(key, None)
        work['tick_attempt'] = dict(
            tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id=old['id'], kind='resource_check', result='passed', command=proof['command'],
            observation='Authoritative full fixed30 checkpoint/log and absent exact Y59 process '
                        'verified; canonical archive completed. Frozen efficacy evaluation remains pending.',
            evidence=[ref(base.with_suffix('.json'))])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        remaining = monitor.load_queue(str(ROOT))['gpu_queues']['server-2']
        assert remaining == [paused]
        assert not monitor.load_running(str(ROOT)).get('server-2', {}).get('0')
        print(json.dumps(dict(proof=ref(base.with_suffix('.json')), state=experiment['status'],
            checkpoints=proof['checkpoints'], Y57_preserved=True, events=events), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
