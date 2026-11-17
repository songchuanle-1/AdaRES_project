"""Refresh stale observation evidence mechanically, retaining the original guard."""
import argparse
from datetime import datetime
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, ref, tick
import closure


def main(token):
    lease(token, 300)
    work = tick.load(ROOT / 'pipeline/research_work.json')
    y92 = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y92')
    frozen = {key: y92[key] for key in ('code_commit', 'CPU_result', 'CPU_program', 'native_preparation')}
    before_q, before_r = ref('pipeline/queue.json'), ref('pipeline/running.json')
    base = ROOT / ('pipeline/checks/y92_closeout_observation_refresh_' + token)
    output = base.with_suffix('.json')
    diagnostic = base.with_name(base.name + '_stale_guard.log')
    refreshed = base.with_name(base.name + '_fresh_guard.log')
    assert not any(path.exists() for path in (output, diagnostic, refreshed))
    closeout = ROOT / ('pipeline/checks/y92_preparation_closeout_' + token + '.json')
    assert not closeout.exists(), 'Never replay a completed closeout'
    age_rows = []
    now = datetime.now().astimezone()
    for cid in ('AdaRES-Y89', 'AdaRES-Y90'):
        candidate = next(row for row in work['candidates'] if row['id'] == cid)
        value = tick.load(closure.artifact(ROOT, candidate['latest_training_observation']))
        age_rows.append(dict(id=cid, evidence=candidate['latest_training_observation'],
            observed_local=value['observed_local'],
            age_seconds=(now - datetime.fromisoformat(value['observed_local'])).total_seconds()))
    assert max(row['age_seconds'] for row in age_rows) >= 300
    command = [sys.executable, '-B', 'scripts/record_y92_wait.py', '--token', token]
    response = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
    diagnostic.write_bytes(response.stdout + response.stderr)
    assert response.returncode != 0
    assert b'Refresh actual native observation under a new label' in diagnostic.read_bytes()
    assert not closeout.exists()
    for cid in ('AdaRES-Y89', 'AdaRES-Y90'):
        subprocess.run([sys.executable, '-B', 'scripts/y90_deployment.py', 'observe',
            '--id', cid, '--token', token, '--label', 'closeout_fresh'], cwd=ROOT,
            check=True, timeout=60)
    response = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
    refreshed.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and closeout.exists(), 'Preserve failure; never relax the 300s guard'
    after = tick.load(ROOT / 'pipeline/research_work.json')
    current = next(row for row in after['candidates'] if row['id'] == 'AdaRES-Y92')
    assert all(current[key] == value for key, value in frozen.items())
    assert (before_q, before_r) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
    tick.save(output, dict(complete=True, token=token,
        observed_at=datetime.now().astimezone().isoformat(), diagnostic_sampled_at=now.isoformat(),
        stale_observations=age_rows, unchanged_max_observation_age_seconds=300,
        original_helper=ref('scripts/record_y92_wait.py'),
        actual_unchanged_guard_diagnostic=ref(diagnostic), fresh_guard_result=ref(refreshed),
        closeout=ref(closeout), original_model_CPU_native_preparation=frozen,
        only_progress_observation_evidence_refreshed=True, model_or_scientific_revision=False,
        CPU_or_native_scientific_checks_replayed=False, admission=False,
        evidence=[ref(__file__), ref(diagnostic), ref(refreshed), ref(closeout)]))
    current['evidence'].append(ref(output))
    tick.save(ROOT / 'pipeline/research_work.json', after)
    print(json.dumps(dict(refresh=ref(output), closeout=ref(closeout),
                          scientific_inputs_unchanged=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    main(parser.parse_args().token)
