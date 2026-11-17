"""Prepare a bounded peer-value gradient-routing comparison without GPU use."""

import argparse
import ast
import base64
from datetime import datetime, timedelta
import hashlib
import json
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

PACKAGE = 'adares_y61_peer_value_gradient'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / n for n in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(c['id'] == 'AdaRES-Y61' for c in work['candidates'])
    source = files()
    for path in source[:-1]:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
        if path.name not in ('__init__.py', 'dec.py'):
            assert path.read_bytes() == (path.parent.parent / PARENT / path.name).read_bytes()
    assert source[-1].read_bytes() == (source[-1].parent / (PARENT + '_res.yaml')).read_bytes()
    assert len(list((ROOT / 'AdaRES/configs_song').rglob(source[-1].name))) == 1
    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])
    payload = dict(
        variant=dict(id='AdaRES-Y61', package=PACKAGE,
            config=source[-1].relative_to(ROOT / 'AdaRES').as_posix(),
            python_files={p.name: encoded(p) for p in source[:-1]}, config_data=encoded(source[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        train_source_lf_sha256=hashlib.sha256(
            (ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    program = ROOT / ('pipeline/checks/y61_cpu_' + token + '_program.py')
    assert not program.exists()
    body = ROOT / 'scripts/probe_y61_cpu.py'
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'),
                       encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    work['candidates'].append(dict(
        id='AdaRES-Y61', profile='experiment_first', stage='implementation',
        hypothesis='Direct gradients from a focal confidence through peer query values couple rival '
            'representations. Isolating only that value-input path may change confidence learning while '
            'preserving every forward output and mask-overlap gradient. Task benefit is unmeasured.',
        matched_control='Unchanged W2, shared indicator at all seven stages. Recompute the SAME trainable '
            'query LayerNorm on query.detach() for peer values only; retain focal input, mask gradients, '
            'affine/head parameter gradients, native targets, selector, losses, seed1999 and fixed30.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE,
        config=payload['variant']['config'],
        historical_scope='Y20 rejected a stop-gradient on mask-competition inputs. Y61 does not detach '
            'masks, overlaps or competition weights, and is not a reopening of that intervention. '
            'Y60 separates head parameters; Y61 retains the single shared head and all native losses.',
        next_action='Run frozen CPU native YAML/full-model/released-pretrain/real TRAIN collate and '
            'matched float32/64 forward and gradient-path tests. Native two-step sparse training and '
            'GPU resources must still pass on a genuinely idle authorized GPU0 before admission.',
        evidence=[ref(p) for p in source] + [ref(program), ref(body), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id='AdaRES-Y61', program=ref(program), native_smoke_executed=False)))


def record(token):
    result_path = ROOT / ('pipeline/analyses/y61_cpu_' + token + '.json')
    result = tick.load(result_path)
    assert result['complete'] and result['implementation_test_passed']
    assert result['full_model_CPU_constructed'] and result['configured_TRAIN_data_collated']
    assert not result['CUDA_initialized'] and not result['native_smoke_passed']
    program = ROOT / ('pipeline/checks/y61_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    for path in files():
        expected = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == expected['sha256']
    runtime_path = ROOT / ('pipeline/checks/prepared_resources_' + token + '.json')
    resource = tick.load(runtime_path)
    assert not resource['idle_authorized_gpus'] and not resource['native_smoke_executed']
    assert any(t['id'] == 'AdaRES-Y58' and t['native_progress_verified'] for t in resource['training'])
    binding_path = ROOT / ('pipeline/checks/y61_prepared_' + token + '.json')
    assert not binding_path.exists()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    tick.save(binding_path, dict(complete=True, id='AdaRES-Y61', tested_source=[ref(p) for p in files()],
        cpu_test=ref(result_path), resource_check=ref(runtime_path), base_commit=head,
        code_commit=None, source_committed=False, native_smoke_passed=False, admitted=False,
        reason='Immutable tested payload retained; save the tested candidate version and bind a native '
               'smoke plan before GPU execution. Do not synchronize or disturb deployed Y58/Y59.'))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    c = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y61')
    evidence = [ref(result_path), ref(result_path.with_suffix('.log')), ref(binding_path), ref(runtime_path)]
    c['evidence'].extend(evidence)
    c.update(stage='blocked', blocked_from_stage='implementation', blocker_kind='exclusive_gpu_slot',
        required_server='server-2', occupation_evidence=ref(runtime_path),
        blocker='Y58 occupies the sole authorized GPU0 and Y59 is already FIFO waiting. '
                'Y61 CPU tests passed, but sparse native two-step smoke cannot share the running GPU.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Reconciled canonical queue and current live compute/memory checks show '
                          'exclusive idle authorized GPU0; preserve existing FIFO waiting work.',
        next_action='Recheck exact tested payload/source and goal, save only these seven tested files '
            'as a scoped version, then bind/run the actual native YAML/TRAIN two-step smoke with '
            'explicit --gpu-id 0 on an exclusive authorized idle slot. Require real path/finite '
            'loss/backward/resource evidence before admission; preserve Y58/Y59 receipts and FIFO. '
            'No repeated routine approval, GPU1 use or synchronization while preparing.')
    work['tick_attempt'] = dict(
        tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=c['id'], kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y61_cpu ' +
                program.relative_to(ROOT).as_posix() + ' --token ' + token + ' --seconds 300',
        observation='Real TRAIN data/native YAML/full CPU construction and float32/64 peer-value '
                    'gradient routing controls passed. No GPU sparse task smoke or trained benefit; '
                    'the sole authorized GPU0 is still occupied.', evidence=evidence)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=c['id'], evidence=ref(binding_path), stage=c['stage'], source_committed=False)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'record'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        (prepare if args.operation == 'prepare' else record)(args.token)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
