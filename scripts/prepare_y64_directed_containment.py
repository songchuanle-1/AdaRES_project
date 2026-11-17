"""Prepare directed containment confidence without advancing queued Y63's HEAD."""

import argparse
import ast
import base64
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y64'
PACKAGE = 'adares_y64_directed_containment'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / n for n in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(c['id'] == CID for c in work['candidates'])
    target = files()
    assert not any(p.exists() for p in target)
    parent = target[0].parent.parent / PARENT
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    init = '        self.query_norm = nn.LayerNorm(d_model)'
    assert source.count(init) == 1
    source = source.replace(init, '        self.containment_weight = nn.Parameter(torch.zeros(d_model, 2))\n' + init)
    ret = '        return self.head(self.evidence_norm(evidence))'
    assert source.count(ret) == 1
    source = source.replace(ret,
        '        # Positive edges indicate that the focal mask contains its peer.\n'
        '        directed = (intersection / areas.unsqueeze(1).clamp_min(1e-6)\n'
        '                    - intersection / areas.unsqueeze(-1).clamp_min(1e-6))\n'
        '        directed = directed.masked_fill(diagonal, 0.0)\n'
        "        directed_context = torch.einsum('bij,bjd->bid', directed, normalized_query)\n"
        '        directed_context = directed_context / max(query.shape[1] - 1, 1)\n'
        '        return self.head(self.evidence_norm(evidence)) + directed_context @ self.containment_weight')
    target[0].parent.mkdir()
    for path in target[:-1]:
        data = source.encode('utf-8') if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            data = b'# AdaRES-Y64: directed containment peer confidence.\nfrom .ipdn import IPDN\n'
        path.write_bytes(data)
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    target[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())
    assert len(list((ROOT / 'AdaRES/configs_song').rglob(target[-1].name))) == 1
    encoded = lambda p: dict(base64=base64.b64encode(p.read_bytes()).decode(), sha256=ref(p)['sha256'])
    payload = dict(variant=dict(id=CID, package=PACKAGE,
        config=target[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={p.name: encoded(p) for p in target[:-1]}, config_data=encoded(target[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    program = ROOT / ('pipeline/checks/y64_cpu_' + token + '_program.py')
    assert not program.exists()
    body = ROOT / 'scripts/probe_y64_cpu.py'
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='W2 pools peers using symmetric soft point-mass IoU. A signed containment '
            'peer path may distinguish a focal mask containing a smaller peer from being '
            'contained by a larger peer, and make confidence fitting easier. Existing query '
            'features may implicitly encode size; missing information and efficacy are not presumed.',
        matched_control='Unchanged W2 plus zero-initialized 256x2 projection of signed peer '
            'context: sum_j [I_ij/A_j-I_ij/A_i] qnorm_j/(Q-1), self edges zero. '
            'All channels retained, no threshold, selector, new loss, epoch, seed or validation tuning; '
            'same seven shared confidence calls, released SPFormer and fixed30/seed1999.',
        historical_scope='Pairwise directed containment, not Y27 centroid geometry, Y50 higher-order '
            'coverage moments, Y57 peer second moments, Y62 symmetric overlap density or Y63 '
            'focal-peer bilinear products. Targeted historical mechanism check found no same '
            'rejected route; this is not a literature novelty claim.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE,
        config=payload['variant']['config'],
        next_action='Run frozen CPU native YAML/full model/released pretrain/real TRAIN collate, '
            'zero output/input-gradient/RNG anchor, literal directed containment and wrong-direction '
            'negative controls plus all seven decoder stages. Native two-step sparse TRAIN '
            'loss/backward/optimizer on exclusive authorized idle GPU0 remains mandatory before admission. '
            'Do not advance HEAD or sync new source while Y63 is waiting on its admitted version.',
        evidence=[ref(p) for p in target] + [ref(program), ref(body), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, program=ref(program), committed=False, admitted=False)))


def record(token):
    result_path = ROOT / ('pipeline/analyses/y64_cpu_' + token + '.json')
    result = tick.load(result_path)
    assert result['complete'] and result['id'] == CID
    for key in ('implementation_test_passed', 'full_model_CPU_constructed',
                'configured_TRAIN_data_collated', 'released_backbone_loaded',
                'native_YAML_semantically_unchanged', 'full_decoder_CPU_executed',
                'decoder_anchor_all_stages_exact', 'decoder_backward_finite',
                'decoder_containment_gradient_nonzero'):
        assert result[key], key
    assert result['parameters'] == 164630149 and result['additional_parameters'] == 512
    assert result['decoder_seven_calls'] == 7
    assert len(result['component_controls']) == 2
    for item in result['component_controls']:
        assert item['dtype'] in ('torch.float32', 'torch.float64')
        assert all(value is True for key, value in item.items() if key != 'dtype')
    assert {v['dtype'] for v in result['component_controls']} == {'torch.float32', 'torch.float64'}
    for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward',
                'native_sparse_forward', 'validation_used', 'admission'):
        assert result[key] is False, key
    assert result['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    program = ROOT / ('pipeline/checks/y64_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    for path in files():
        entry = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == entry['sha256']
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    assert head == '82fbd9fb1fc56da6afee5b145f59cfa533464707'
    manifest = ROOT / ('pipeline/checks/y64_cpu_version_' + token + '.json')
    assert not manifest.exists()
    tick.save(manifest, dict(complete=True, id=CID, observed_at=datetime.now().astimezone().isoformat(),
        parent_head=head, exact_files=[ref(p) for p in files()], cpu_test=ref(result_path),
        uncommitted=True, production_sync=False, native_smoke_passed=False, admitted=False,
        limitation='A CPU-tested source manifest, not a code-commit or native-training admission receipt.'))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    assert candidate['stage'] == 'implementation'
    evidence = [ref(result_path), ref(result_path.with_suffix('.log')), ref(manifest)]
    candidate['evidence'].extend(evidence)
    candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        required_server='server-2',
        blocker='Authorized server-2 GPU0 is occupied by foreign compute; server-1 GPU0 is running Y62 and Y63 is waiting. Preserve both and paused Y57/GPU1.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Exclusive authorized GPU0 becomes genuinely idle and its current-token identity/compute/memory/lease checks pass.',
        source_manifest=ref(manifest),
        next_action='Preserve CPU-tested uncommitted seven-file source and queued Y63 receipt. '
            'When an authorized slot opens, validate unchanged hashes and preserve Y63 version '
            'before saving the scoped commit/frozen native plan. Run at least two actual '
            'native YAML/TRAIN criterion/backward/optimizer steps with seven branch calls '
            'per step and finite nonzero containment-weight gradients. Freeze the same full '
            'post-training protocol as Y62/Y63; only native PASS permits receipt/admission '
            'and gated dispatch. No CPU substitution, GPU1 use or foreign-job interference.')
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y64_cpu ' + program.relative_to(ROOT).as_posix() + ' --token ' + token + ' --seconds 600',
        observation='CPU native YAML/model/pretrain/real TRAIN collate, float32/64 literal '
            'directed containment controls and synthetic seven-stage decoder backward passed; '
            'no CUDA/native sparse task steps, commit, sync, admission or efficacy claim.', evidence=evidence)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, manifest=ref(manifest), native_smoke_passed=False, committed=False, admitted=False)))


if __name__ == '__main__':
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
