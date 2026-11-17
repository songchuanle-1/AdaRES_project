"""Prepare a scoped absolute-overlap-density input without touching live Y61."""

import argparse
import ast
import base64
from datetime import datetime, timedelta
import hashlib
import json
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from dispatch_prepared_candidates import snapshot
from check_ef_execution import validate_frozen
import closure
import experiment_first

CID = 'AdaRES-Y62'
PACKAGE = 'adares_y62_overlap_density'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / n for n in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def validate_cpu_result(result):
    assert result['complete'] and result['id'] == CID
    for key in ('implementation_test_passed', 'full_model_CPU_constructed',
                'configured_TRAIN_data_collated', 'released_backbone_loaded',
                'native_YAML_semantically_unchanged', 'full_decoder_CPU_executed',
                'decoder_anchor_all_stages_exact', 'decoder_backward_finite',
                'decoder_density_gradient_nonzero'):
        assert result[key], key
    assert result['parameters'] == 164629639 and result['additional_parameters'] == 2
    assert result['decoder_seven_calls'] == 7
    for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward',
                'native_sparse_forward', 'validation_used', 'admission'):
        assert result[key] is False, key
    controls = result['component_controls']
    assert len(controls) == 2 and {c['dtype'] for c in controls} == {'torch.float32', 'torch.float64'}
    for item in controls:
        assert item['focal_density_difference'] > .05
        assert all(value is True for key, value in item.items()
                   if key not in ('dtype', 'focal_density_difference'))


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(c['id'] == CID for c in work['candidates'])
    source = files()
    for path in source[:-1]:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
        if path.name not in ('__init__.py', 'dec.py'):
            assert path.read_bytes().replace(b'\r\n', b'\n').rstrip() == (path.parent.parent / PARENT / path.name).read_bytes().replace(b'\r\n', b'\n').rstrip()
    assert source[-1].read_bytes().replace(b'\r\n', b'\n').rstrip() == (source[-1].parent / (PARENT + '_res.yaml')).read_bytes().replace(b'\r\n', b'\n').rstrip()
    assert len(list((ROOT / 'AdaRES/configs_song').rglob(source[-1].name))) == 1
    encoded = lambda p: dict(base64=base64.b64encode(p.read_bytes()).decode(), sha256=ref(p)['sha256'])
    payload = dict(variant=dict(id=CID, package=PACKAGE,
        config=source[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={p.name: encoded(p) for p in source[:-1]}, config_data=encoded(source[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((source[2].parent.parent / PARENT / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    body = ROOT / 'scripts/probe_y62_cpu.py'
    program = ROOT / ('pipeline/checks/y62_cpu_' + token + '_program.py')
    assert not program.exists()
    content = 'PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8')
    ast.parse(content, feature_version=(3, 8))
    program.write_text(content, encoding='utf-8', newline='\n')
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='W2 row-normalized peer context and maximum overlap need not retain the total '
            'overlap strength of distinct competitors. A separate mean off-diagonal point-mass '
            'IoU density input may supply that missing scalar evidence to confidence. Utility is unmeasured.',
        matched_control='Unchanged W2 head plus density times a zero-initialized two-logit weight; '
            'no added RNG, thresholds, feature selection, loss/selector changes or density scale search. '
            'Keep all seven shared-head losses, released SPFormer, seed1999 and fixed30.',
        historical_scope='Not Y56 deduplication, Y57 peer second moments, Y47 full pairwise '
            'correspondence or Y50 higher-order joint mask incidence. No historical rejected ID reopened.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE,
        config=payload['variant']['config'],
        next_action='Run frozen CPU native-YAML/full-model/real TRAIN collate and float32/64 '
            'matched max/context but different distinct-mask density controls, finite gradient and '
            'seven-stage decoder integration. Native sparse two-step smoke remains mandatory on '
            'an exclusive authorized idle GPU0 before admission. Preserve live Y61 and paused Y57.',
        evidence=[ref(p) for p in source] + [ref(program), ref(body), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, program=ref(program), admitted=False)))


def record(token):
    result_path = ROOT / ('pipeline/analyses/y62_cpu_' + token + '.json')
    result = tick.load(result_path)
    validate_cpu_result(result)
    assert result['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    program_path = ROOT / ('pipeline/checks/y62_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(program_path.read_text(encoding='utf-8')).body[0].value)
    for path in files():
        entry = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == entry['sha256']
    server = monitor.load_servers(str(ROOT))['server-2']
    assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    tasks = monitor.load_queue(str(ROOT))['gpu_queues']['server-2']
    assert not any(t['status'] in ('waiting', 'claiming') for t in tasks)
    running = [t for t in tasks if t['status'] == 'running']
    assert len(running) == 1 and running[0]['id'] == 'AdaRES-Y61' and int(running[0]['gpu']) == 0
    paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold']
    live = snapshot(server, running, [])
    assert live['gpu_query_rc'] == live['compute_query_rc'] == 0
    assert live['tasks']['AdaRES-Y61']['process_live']
    assert str(running[0]['pid']) in live['compute']
    git = lambda *args: subprocess.check_output(['git', '-c', 'core.autocrlf=false', *args], cwd=ROOT)
    assert not git('diff', '--cached', '--name-only').strip()
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    assert candidate['stage'] == 'implementation' and not candidate.get('code_commit')
    paths = [p.relative_to(ROOT).as_posix() for p in files()]
    version = ROOT / ('pipeline/checks/y62_version_' + token + '.json')
    plan_path = ROOT / ('pipeline/checks/y62_smoke_' + token + '_plan.json')
    native_path = ROOT / ('pipeline/checks/y62_smoke_' + token + '_program.py')
    assert not any(p.exists() for p in (version, plan_path, native_path))
    body = ROOT / 'scripts/probe_y62_native_smoke.py'
    ast.parse(body.read_text(encoding='utf-8'), feature_version=(3, 8))
    parent = git('rev-parse', 'HEAD').decode().strip()
    git('add', '--', *paths)
    assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == set(paths)
    for path in paths:
        assert hashlib.sha256(git('show', ':' + path)).hexdigest() == ref(path)['sha256']
    git('commit', '--only', '-m', 'Add Y62 absolute overlap-density confidence evidence', '--', *paths)
    commit = git('rev-parse', 'HEAD').decode().strip()
    assert git('rev-parse', commit + '^').decode().strip() == parent
    assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).decode().splitlines()) == set(paths)
    assert not git('diff', '--cached', '--name-only').strip()
    frozen = dict(variant=payload['variant'], code_commit=commit,
        goal_sha256=payload['goal_sha256'], train_source_lf_sha256=payload['train_source_lf_sha256'])
    plan = dict(complete=True, id=CID, profile='experiment_first', config=candidate['config'],
        code_commit=commit, goal_sha256=frozen['goal_sha256'], native_executed=False,
        required_resource='Exclusive idle authorized server-2 GPU0; current lease/live compute/280W; preserve Y61 and Y57.',
        checks='Actual native YAML/TRAIN/tools.train two finite loss/backward/optimizer steps, '
            'seven shared calls per step, finite nonzero density-weight gradients, zero first-step '
            'anchor and second-step nonzero branch response, resource budget and frozen version.',
        planned_command='python -B scripts/run_experiment_first_probe.py y62_smoke ' +
            native_path.relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id 0')
    assert validate_frozen(dict(candidate, code_commit=commit), plan, frozen) == 7
    native_path.write_text('FROZEN=' + repr(frozen) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    plan['program'] = ref(native_path)
    tick.save(plan_path, plan)
    now = datetime.now().astimezone()
    tick.save(version, dict(complete=True, id=CID, observed_at=now.isoformat(), code_commit=commit,
        parent_commit=parent, exact_files=[ref(p) for p in files()], cpu_test=ref(result_path),
        native_smoke_plan=ref(plan_path), live=live, native_smoke_passed=False, admitted=False,
        authorization=experiment_first.effective(ROOT)['authorization'],
        production_sync=False, pushed=False, queue_changed=False, running_changed=False))
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    additions = [ref(result_path), ref(result_path.with_suffix('.log')), ref(version), ref(plan_path), ref(native_path), ref(body)]
    candidate['evidence'].extend(additions)
    candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        code_commit=commit, native_smoke_plan=ref(plan_path), required_server='server-2',
        blocker='Live Y61 occupies the sole authorized GPU0; do not share it or use GPU1.',
        occupation_evidence=ref(version), retry_after=(now + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Canonical GPU0 slot opens and fresh live compute/memory/280W checks show exclusive idle.',
        next_action='Preserve tested seven-file version and frozen plan. When Y61 has genuinely '
            'ended and been reconciled, run current-token native YAML/TRAIN two-step smoke on '
            'exclusive GPU0. Only actual PASS allows admission/gated dispatch; no CPU substitution '
            'or repeated routine approval, no Y57 recovery, no GPU1, no production sync while busy.')
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y62_cpu ' + program_path.relative_to(ROOT).as_posix() + ' --token ' + token + ' --seconds 600',
        observation='CPU native-YAML/full model/released pretrain/real TRAIN collate and float32/64 '
            'component plus full seven-stage synthetic decoder tests passed. GPU0 remains busy with '
            'Y61; no native sparse two-step test, admission, production sync or trained efficacy.', evidence=additions)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(version=ref(version), plan=ref(plan_path), code_commit=commit, scoped_files=7,
        native_smoke_executed=False, admitted=False, queue_changed=False), indent=2))


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
