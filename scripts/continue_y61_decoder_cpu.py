"""Bind the prepared Y61 source to a full-decoder CPU integration check."""

import argparse
import ast
import base64
from datetime import datetime, timedelta
import hashlib
import json
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y61_peer_values import files
import closure


def source_payload(candidate):
    matches = [entry for entry in candidate['evidence'] if
               entry['path'].startswith('pipeline/checks/y61_cpu_') and
               entry['path'].endswith('_program.py')]
    assert len(matches) == 1
    original = closure.artifact(ROOT, matches[0])
    payload = ast.literal_eval(ast.parse(original.read_text(encoding='utf-8')).body[0].value)
    for path in files():
        entry = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert hashlib.sha256(base64.b64decode(entry['base64'])).hexdigest() == entry['sha256']
        assert ref(path)['sha256'] == entry['sha256']
    return payload, ref(original)


def path_for(token):
    return ROOT / ('pipeline/checks/y61_decoder_cpu_' + token + '_program.py')


def prepare(token, candidate):
    payload, original = source_payload(candidate)
    base = ROOT / 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
    config = ROOT / ('AdaRES/' + payload['parent_config'])
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    def encoded(path):
        committed = subprocess.check_output(['git', 'show', head + ':' + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        assert committed.replace(b'\r\n', b'\n') == path.read_bytes().replace(b'\r\n', b'\n')
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])
    parent = dict(package=base.name, config=payload['parent_config'],
        python_files={name: encoded(base / name) for name in payload['variant']['python_files']},
        config_data=encoded(config))
    packet = dict(variant=payload['variant'], parent=parent, base_commit=head,
                  original_payload=original, goal_sha256=ref('pipeline/goal.json')['sha256'])
    program = path_for(token)
    assert not program.exists()
    body = ROOT / 'scripts/probe_y61_decoder_cpu.py'
    text = 'PACKET=' + repr(packet) + '\n' + body.read_text(encoding='utf-8')
    ast.parse(text, feature_version=(3, 8))
    program.write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps(dict(id=candidate['id'], program=ref(program), original_payload=original,
                         source_changed=False, native_smoke_executed=False)))


def record(token, work, candidate):
    source_payload(candidate)
    program = path_for(token)
    packet = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert packet['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    result_path = ROOT / ('pipeline/analyses/y61_decoder_cpu_' + token + '.json')
    result = tick.load(result_path)
    assert result['complete'] and result['id'] == candidate['id']
    assert result['goal_sha256'] == packet['goal_sha256']
    assert result['actual_decoder_class_executed'] and result['seven_confidence_calls'] == 7
    assert result['all_stage_forward_bitwise_equal'] and result['all_stage_backward_finite']
    assert result['query_norm_input_requires_grad'] == [True, False] * 7
    assert result['direct_mask_gradients_preserved'] and result['final_query_gradient_changed']
    assert not result['CUDA_initialized'] and not result['native_smoke_passed']
    resource_path = ROOT / ('pipeline/checks/prepared_resources_' + token + '.json')
    resource = tick.load(resource_path)
    assert resource['complete'] and not resource['idle_authorized_gpus']
    additions = [ref(result_path), ref(result_path.with_suffix('.log')), ref(program),
                 ref('scripts/probe_y61_decoder_cpu.py'), ref(resource_path), ref(__file__)]
    candidate['evidence'].extend(entry for entry in additions if entry not in candidate['evidence'])
    candidate['decoder_integration_check'] = ref(result_path)
    candidate.update(stage='blocked', blocked_from_stage='implementation',
        required_server='server-2', blocker_kind='exclusive_gpu_slot', occupation_evidence=ref(resource_path),
        blocker='Y58 occupies the sole authorized GPU0; Y59 already waits FIFO. Full Y61 CPU decoder '
                'integration passed, but actual native sparse TRAIN two-step smoke remains unrun.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat())
    work['tick_attempt'] = dict(
        tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=candidate['id'], kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y61_decoder_cpu ' +
                program.relative_to(ROOT).as_posix() + ' --token ' + token + ' --seconds 300',
        observation='Actual full seven-stage native decoder graph on synthetic CPU inputs passed '
                    'common-state forward parity, intended detach-branch tracing and finite backward. '
                    'Not native sparse TRAIN task loss, optimizer steps, GPU resource proof or efficacy.',
        evidence=[ref(result_path), ref(result_path.with_suffix('.log')), ref(resource_path)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=candidate['id'], evidence=ref(result_path), new_candidate=False,
                         source_changed=False, admitted=False)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'record'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 120)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y61')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        if args.operation == 'prepare':
            prepare(args.token, candidate)
        else:
            record(args.token, work, candidate)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
