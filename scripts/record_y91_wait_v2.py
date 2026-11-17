"""Retry closeout with fresh observations and new immutable artifact names."""
import argparse
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


def record(token):
    original = ROOT / 'scripts/record_y91_wait.py'
    assert ref(original)['sha256'] == 'ebbcd63c1264ac27467308805a95dcc0638bf540d858a1dd0b225e5a5edf9ffc'
    old_tests = ROOT / ('pipeline/checks/y91_final_preparation_guards_' + token + '.log')
    assert old_tests.exists()
    proof = ROOT / ('pipeline/checks/y91_closeout_observation_retry_' + token + '.json')
    assert not proof.exists()
    source = original.read_text(encoding='utf-8')
    replacements = [
        ("'pipeline/checks/y91_final_preparation_guards_'", "'pipeline/checks/y91_final_preparation_guards_attempt2_'"),
        ("'pipeline/checks/gpu_occupation_' + name + '_finish_'", "'pipeline/checks/gpu_occupation_' + name + '_finish2_'"),
    ]
    for old, new in replacements:
        assert source.count(old) == 1
        source = source.replace(old, new)
    assert "total_seconds() < 300" in source
    tick.save(proof, dict(complete=True, token=token, original_helper=ref(original),
        original_guard_log=ref(old_tests), retry_helper=ref(__file__),
        original_failure='record_y91_wait.py:68 rejected latest_training_observation older than300s before any closeout/memory/notes write.',
        diagnosis='Measured at2026-11-06T03:29:18.840502+00:00: Y87 observation age491.28088s and Y88 age271.73874s. This is stale closeout evidence, not a training or model failure.',
        correction='Collect fresh native progress and occupation for both servers; use new immutable guard-log and occupation labels. All300s checks, scientific tests, models, YAML and original failed helper/log remain unchanged.',
        scientific_control_changed=False, model_changed=False, original_evidence_preserved=True))
    namespace = dict(__name__='y91_closeout_attempt2', __file__=__file__)
    exec(compile(source, str(original), 'exec'), namespace)
    namespace['record'](token)
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y91')
    candidate['evidence'].extend([ref(proof), ref(original), ref(old_tests)])
    tick.save(ROOT / 'pipeline/research_work.json', work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        record(args.token)
    finally:
        writer.close()
