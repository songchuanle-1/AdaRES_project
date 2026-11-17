"""Prepare overlap-weighted peer-state refinement feedback without changing production source."""
import argparse
import ast
import base64
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y66'
PACKAGE = 'adares_y66_peer_state_feedback'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / n for n in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


FEEDBACK = '''class PeerStateFeedback(nn.Module):
    """Feed overlap-weighted peer states into the next query refinement."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, query, pred_masks, batch_mask, point_mass):
        valid = (~batch_mask).unsqueeze(1).to(pred_masks.dtype)
        probabilities = pred_masks.sigmoid() * valid
        weighted = probabilities * point_mass.unsqueeze(1)
        area = weighted.sum(-1)
        intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
        union = area.unsqueeze(-1) + area.unsqueeze(1) - intersection
        overlap = intersection / union.clamp_min(1e-6)
        diagonal = torch.eye(overlap.shape[-1], dtype=torch.bool,
                             device=overlap.device).unsqueeze(0)
        overlap = overlap.masked_fill(diagonal, 0.0)
        competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
        context = torch.einsum('bij,bjd->bid', competition, query)
        return query + context @ self.weight


'''


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(c['id'] == CID for c in work['candidates'])
    target = files()
    assert not any(p.exists() for p in target)
    parent = target[0].parent.parent / PARENT
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    assert source.count('class DEC(nn.Module):') == 1
    source = source.replace('class DEC(nn.Module):', FEEDBACK + 'class DEC(nn.Module):')
    init = '        self.indi_norm = nn.LayerNorm(d_model)'
    assert source.count(init) == 1
    source = source.replace(init, init + '\n        self.peer_feedback = PeerStateFeedback(d_model)')
    marker = '        # multi-round\n\n        for i in range(self.num_layer):'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '\n            query = self.peer_feedback(query, pred_masks, batch_mask, point_mass)')
    target[0].parent.mkdir()
    for path in target[:-1]:
        data = source.encode('utf-8') if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            data = b'# AdaRES-Y66: peer-state feedback into query refinement.\nfrom .ipdn import IPDN\n'
        path.write_bytes(data)
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    target[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())
    assert len(list((ROOT / 'AdaRES/configs_song').rglob(target[-1].name))) == 1
    def encoded(path): return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])
    payload = dict(variant=dict(id=CID, package=PACKAGE,
        config=target[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={p.name: encoded(p) for p in target[:-1]}, config_data=encoded(target[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    program = ROOT / ('pipeline/checks/y66_cpu_' + token + '_program.py')
    assert not program.exists()
    body = ROOT / 'scripts/probe_y66_cpu.py'
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='W2 uses overlap-weighted peer states only to read out confidence, while query refinement '
            'does not explicitly consume that competition context. A zero-initialized learned projection '
            'of full peer states into the next refinement may make masks and confidence co-adapt. Existing '
            'self-attention can represent peer interactions; neither missing information nor efficacy is presumed.',
        matched_control='Unchanged W2 plus q_next_input=q+C(q,masks,masses)@W, where C is the row-normalized '
            'off-diagonal soft point-mass-IoU weighted mean of raw peer queries. Shared W[256,256] starts '
            'at exact zero without RNG. Six feedback calls before coarse query-to-instance attention, '
            'seven unchanged confidence losses, released SPFormer, fixed30/seed1999 and native YAML.',
        historical_scope='Targeted ledger scan: Y25 rejected changing the mask-to-cross-attention gate; this '
            'keeps that gate unchanged and adds a query-state residual. Y57 is peer second-moment '
            'confidence evidence, Y60 stage heads, and Y62/Y63/Y64 confidence readouts. Y65 shares the '
            'recurrent insertion location but uses centered scalar confidence times one vector; this '
            'uses full peer-state context through a 256x256 projection and changes subsequent refinement. '
            'No rejected/banned route is reopened and no literature-novelty claim is made.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        next_action='CPU-check exact YAML/full model/released backbone/real TRAIN collate, '
            'zero-state output/input-gradient/RNG anchors, scalar peer-context formula, query '
            'permutation/padding/mass-scale/superpoint-split/singleton controls, six feedback/seven indicator calls '
            'and synthetic decoder backward. A native sparse TRAIN two-step loss/backward/optimizer '
            'check on exclusive authorized GPU0 is still mandatory before admission.',
        evidence=[ref(p) for p in target] + [ref(program), ref(body), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, program=ref(program), committed=False, admitted=False)))


def record(token):
    result_path = ROOT / ('pipeline/analyses/y66_cpu_' + token + '.json')
    result = tick.load(result_path)
    assert result['complete'] and result['id'] == CID
    for key in ('implementation_test_passed', 'full_model_CPU_constructed',
                'configured_TRAIN_data_collated', 'released_backbone_loaded',
                'native_YAML_semantically_unchanged', 'decoder_zero_anchor_all_stages_exact',
                'decoder_backward_finite', 'decoder_feedback_gradient_nonzero',
                'nonzero_feedback_changes_refinement', 'decoder_input_gradients_anchor_close'):
        assert result[key], key
    assert result['parameters'] == 164695173 and result['additional_parameters'] == 65536
    assert result['decoder_feedback_calls'] == 6 and result['decoder_indicator_calls'] == 7
    assert len(result['component_controls']) == 2
    for item in result['component_controls']:
        assert all(value is True for key, value in item.items() if key != 'dtype')
    assert {v['dtype'] for v in result['component_controls']} == {'torch.float32', 'torch.float64'}
    for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward',
                'native_sparse_forward', 'validation_used', 'admission'):
        assert result[key] is False, key
    assert result['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    program = ROOT / ('pipeline/checks/y66_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    for path in files():
        entry = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == entry['sha256']
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    assert head == '9706a8c36bead4459668ab66a3d1abc39344ef8b'
    manifest = ROOT / ('pipeline/checks/y66_cpu_version_' + token + '.json')
    assert not manifest.exists()
    tick.save(manifest, dict(complete=True, id=CID, observed_at=datetime.now().astimezone().isoformat(),
        parent_head=head, exact_files=[ref(p) for p in files()], cpu_test=ref(result_path),
        uncommitted=True, production_sync=False, native_smoke_passed=False, admitted=False,
        limitation='CPU preparation only; not a commit, native GPU task test or efficacy result.'))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    assert candidate['stage'] == 'implementation'
    evidence = [ref(result_path), ref(result_path.with_suffix('.log')), ref(manifest)]
    candidate['evidence'].extend(evidence)
    candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        required_server='server-2',
        blocker='Authorized server-2 GPU0 is occupied by foreign compute; server-1 GPU0 runs Y64. '
                'Preserve existing runs and paused Y57/GPU1; native two-step GPU check is pending.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Exclusive authorized GPU0 becomes idle and current identity/source/data/compute/lease checks pass.',
        source_manifest=ref(manifest),
        next_action='Preserve CPU-tested seven-file manifest and running Y64 source. On exclusive authorized '
            'idle GPU0 save scoped version/native plan and run >=2 actual native YAML/TRAIN criterion/'
            'backward/optimizer steps. Verify six feedback/seven indicator calls, finite nonzero feedback '
            'projection gradients and actual subsequent-step response. Freeze original full post-training '
            'protocol before admission; CPU controls do not authorize queue/launch or prove efficacy.')
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y66_cpu ' + program.relative_to(ROOT).as_posix() + ' --token ' + token + ' --seconds 600',
        observation='CPU native YAML/full model/pretrain/real TRAIN collate and scalar peer-context/anchor/'
            'six-stage execution/backward controls passed; no GPU/native sparse criterion steps or efficacy claim.',
        evidence=evidence)
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
