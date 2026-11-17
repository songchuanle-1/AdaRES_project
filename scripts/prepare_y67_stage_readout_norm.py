"""Prepare one independent stage-readout-normalization experiment; never touch live runs."""
import argparse
import ast
import base64
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import experiment_first

CID = 'AdaRES-Y67'
PACKAGE = 'adares_y67_stage_readout_norm'
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
    assert not any(p.exists() for p in target) and not target[0].parent.exists()
    parent = target[0].parent.parent / PARENT
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    source = 'import copy\n' + source
    marker = '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '\n        self.refinement_out_norms = nn.ModuleList([\n'
        '            copy.deepcopy(self.out_norm) for _ in range(num_layer)\n        ])')
    call = '            pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)'
    assert source.count(call) == 1
    source = source.replace(call, call[:-1] + ', stage=i + 1)')
    old = '    def prediction_head(self, query, mask_feats, batch_mask):\n        query = self.out_norm(query)'
    new = '    def prediction_head(self, query, mask_feats, batch_mask, stage=0):\n' \
        '        if not isinstance(stage, int) or not 0 <= stage <= self.num_layer:\n' \
        '            raise ValueError("Invalid decoder prediction stage")\n' \
        '        norm = self.out_norm if stage == 0 else self.refinement_out_norms[stage - 1]\n' \
        '        query = norm(query)'
    assert source.count(old) == 1
    source = source.replace(old, new)
    target[0].parent.mkdir()
    for path in target[:-1]:
        raw = source.encode('utf-8') if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            raw = b'# AdaRES-Y67: independent prediction-stage affine normalization.\nfrom .ipdn import IPDN\n'
        path.write_bytes(raw)
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    target[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())
    assert len(list((ROOT / 'AdaRES/configs_song').rglob(target[-1].name))) == 1
    def encoded(p):
        return dict(base64=base64.b64encode(p.read_bytes()).decode(), sha256=ref(p)['sha256'])
    payload = dict(variant=dict(id=CID, package=PACKAGE,
        config=target[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={p.name: encoded(p) for p in target[:-1]}, config_data=encoded(target[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    program = ROOT / ('pipeline/checks/y67_cpu_' + token + '_program.py')
    assert not program.exists()
    body = ROOT / 'scripts/probe_y67_cpu.py'
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='One output LayerNorm affine transform is reused for initial and six refined mask/score '
            'readouts although their query distributions differ. Independent per-stage affine readout '
            'normalization may reduce shared mask/score-gradient interference. This is an architectural '
            'allocation test; existing stage layers can implicitly compensate and benefit is unknown.',
        matched_control='Unchanged W2 native YAML, seven shared confidence-indicator calls and all native '
            'losses. Copy the existing out_norm into six private refinement LayerNorms without RNG, '
            'retaining stage0 out_norm. Only prediction_head affine sharing changes; Q128/seed1999/'
            'fixed30/released SPFormer/selector/Boolean attention gate remain unchanged. +3072 parameters.',
        historical_scope='Targeted ledger scan found no out_norm untying experiment. Y60 unties the '
            'confidence MLP; this keeps that head shared and unties an already-consumed mask/score '
            'normalization. Y26 rejected reconnecting a discarded query-language norm; no discarded '
            'norm is reconnected. Y25 rejected a soft attention-gate change; native gate formula stays '
            'exact. Y65/Y66 recurrent residuals are absent. No historical rejected route is reopened '
            'and no missing-information, novelty or efficacy claim is made.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        next_action='Execute frozen CPU native YAML/full model/released backbone/real TRAIN collate '
            'and all-seven-stage RNG/output/input-gradient anchors, private-gradient isolation and '
            'synthetic decoder checks. Real native sparse TRAIN two-step loss/backward/optimizer '
            'on exclusive authorized GPU0 is still required before admission.',
        evidence=[ref(p) for p in target] + [ref(program), ref(body), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, program=ref(program), native_executed=False, admitted=False)))


def record(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    c = next(c for c in work['candidates'] if c['id'] == CID)
    assert c['stage'] == 'implementation' and not c.get('code_commit')
    result_path = ROOT / ('pipeline/analyses/y67_cpu_' + token + '.json')
    cpu = tick.load(result_path)
    for key in ('complete', 'implementation_test_passed', 'full_model_CPU_constructed',
                'configured_TRAIN_data_collated', 'released_backbone_loaded', 'native_YAML_semantically_unchanged',
                'decoder_initial_RNG_and_parent_weights_exact', 'decoder_all_stages_anchor_exact',
                'decoder_input_gradients_anchor_close', 'decoder_backward_finite',
                'decoder_all_norm_gradients_finite_nonzero', 'final_stage_intervention_changes_only_final_prediction',
                'restoration_negative_control_exact'):
        assert cpu[key] is True, key
    assert cpu['id'] == CID and cpu['parameters'] == 164632709 and cpu['additional_parameters'] == 3072
    assert cpu['decoder_norm_calls'] == [1] * 7 and cpu['decoder_indicator_calls'] == 7
    assert len(cpu['component_controls']) == 2
    assert all(v is True for row in cpu['component_controls'] for k, v in row.items() if k != 'dtype')
    assert not cpu['CUDA_initialized'] and not cpu['native_smoke_passed'] and not cpu['validation_used']
    program = ROOT / ('pipeline/checks/y67_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert cpu['goal_sha256'] == payload['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    for path in files():
        entry = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == entry['sha256']
    manifest = ROOT / ('pipeline/checks/y67_cpu_version_' + token + '.json')
    assert not manifest.exists()
    tick.save(manifest, dict(complete=True, id=CID, observed_at=datetime.now().astimezone().isoformat(),
        parent_head=closure.git(ROOT, 'rev-parse', 'HEAD'), exact_files=[ref(p) for p in files()],
        cpu_test=ref(result_path), uncommitted=True, native_smoke_passed=False,
        production_sync=False, admitted=False, limitation='CPU preparation only; no actual sparse task steps or efficacy result.'))
    c.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        required_server='server-2', resource_work_kind='native_smoke', source_manifest=ref(manifest),
        blocker='Authorized server-2 GPU0 runs Y65 and server-1 GPU0 runs Y66. Preserve exact live runs and paused Y57/GPU1; GPU native two-step test waits.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Fresh exclusive authorized idle GPU0 with no compute PID or canonical reservation.',
        next_action='Preserve exact CPU-tested seven files; save scoped version and native plan. On '
            'exclusive authorized idle GPU0 execute native YAML/TRAIN two finite loss/backward/optimizer '
            'steps with all seven private normalization paths, actual mask/score consumption and '
            'finite nonzero per-stage gradients/updates. Freeze original full post-training protocol '
            'before admission and same-tick dispatch; no CPU-based admission or efficacy claim.')
    additions = [ref(result_path), ref(result_path.with_suffix('.log')), ref(manifest)]
    c['evidence'].extend(additions)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y67_cpu ' + ref(program)['path'] + ' --token ' + token + ' --seconds 600',
        observation='Real CPU native YAML/full model/pretrain/TRAIN collate and seven-stage normalization/gradient/anchor controls passed; native sparse task steps and efficacy remain untested.', evidence=additions)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, manifest=ref(manifest), native_executed=False, admitted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'record'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        dict(prepare=prepare, record=record)[args.operation](args.token)
    finally:
        lock.close()
