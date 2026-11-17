import argparse
from datetime import datetime,timedelta
import ast,json
from prepare_y67_stage_readout_norm import ROOT,CID,files,lease,monitor,ref,tick,closure

def record(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    c = next(c for c in work['candidates'] if c['id'] == CID)
    assert c['stage'] == 'implementation' and not c.get('code_commit')
    result_path = ROOT / ('pipeline/analyses/y67_cpu_attempt2_' + token + '.json')
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
    program = ROOT / ('pipeline/checks/y67_cpu_' + token + '_attempt2_program.py')
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
        command='python -B scripts/run_experiment_first_probe.py y67_cpu_attempt2 ' + ref(program)['path'] + ' --token ' + token + ' --seconds 600',
        observation='Real CPU native YAML/full model/pretrain/TRAIN collate and seven-stage normalization/gradient/anchor controls passed; native sparse task steps and efficacy remain untested.', evidence=additions)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, manifest=ref(manifest), native_executed=False, admitted=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args()
    lease(a.token,180);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:record(a.token)
    finally:lock.close()
