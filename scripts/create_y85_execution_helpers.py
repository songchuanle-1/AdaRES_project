"""One-time specialization of the tested Y84 lifecycle; retain signed originals."""
import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def replace_function(source, name, replacement):
    nodes = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1, name
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + replacement.rstrip() + '\n' + ''.join(lines[node.end_lineno:])


VALIDATE = '''def validate_result(raw, candidate, frozen):
    assert raw['complete'] is True and raw['id'] == candidate['id'] == CID == frozen['variant']['id']
    assert raw['config'] == candidate['config'] == frozen['variant']['config']
    assert raw['code_commit'] == frozen['code_commit'] == Y85
    assert raw['goal_sha256'] == frozen['goal_sha256']
    assert raw['native_YAML_sha256'] == frozen['variant']['config_data']['sha256']
    assert raw['steps'] == raw['optimizer_steps'] == 2
    assert len(raw['losses']) == 2 and all(math.isfinite(value) for value in raw['losses'])
    assert raw['native_entry'] == 'tools/train.py::train' and raw['split'] == 'train'
    for key in ('native_data_loaded', 'backward_finite', 'intended_path_observed',
                'native_modes_verified', 'static_mask_feature_reuse_preserved',
                'original_hard_gate_algorithm_preserved', 'normalization_before_batching'):
        assert raw[key] is True
    assert raw['validation_used'] is False and raw['model_weights_saved'] is False
    assert raw['shared_head_calls'] == raw['query_norm_consumption_calls'] == 14
    assert raw['mask_feature_norm_calls'] == 2 and raw['original_coarse_gate_consumption_calls'] == 12
    assert len(raw['gradient_checks']) == len(raw['batches']) == 2
    from test_y85_native_smoke_plan import N
    N['check_mask_norm_rows'](raw['norm_paths'])
    for index, row in enumerate(raw['gradient_checks']):
        assert type(row['step']) is int and row['step'] == index
        assert row['all_finite'] is True and row['tensors'] > 0 and row['indicator_tensors'] == 10
        assert row['prediction_calls'] == row['indicator_calls'] == row['query_norm_calls'] == 7
        assert row['coarse_gate_calls'] == 6 and row['raw_and_normalized_gradients_finite'] is True
        for key in ('new_gradient_maxima', 'parameter_updates'):
            assert len(row[key]) == 2 and all(math.isfinite(value) and value > 0 for value in row[key])
    assert 0 < raw['peak_reserved_bytes'] < 24 * 2**30
    assert all(len(batch['scan_ids']) == 2 and batch['expressions'] > 0 and batch['points'] > 0
               for batch in raw['batches'])
    assert len({scene for batch in raw['batches'] for scene in batch['scan_ids']}) == 4
    assert raw['data_source']['dataset_class'] == 'ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'
    assert raw['data_source']['config_data_root'].rstrip('/') == '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data'
'''

ACTIVATE = '''def activate(token):
    server = monitor.load_servers(str(ROOT))[NAME]
    closure.validate_adares_server(ROOT, server)
    assert server['allowed_gpus'] == [0]
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == CID)
    assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
    assert not candidate.get('admission_receipt') and not candidate.get('native_smoke_result')
    unreserved(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), NAME, GPU)
    done = monitor.load_experiment(str(ROOT), 'AdaRES-Y83')
    assert done['status'] == 'done_complete' and str(done['gpu']) == '0'
    proof = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
    assert proof['completed_epochs'] == {phase: list(range(1, 31)) for phase in ('Train', 'Val')}
    current = live(server, GPU)
    idle(current, POWER)
    candidate.update(required_server=NAME, required_gpu=GPU,
                     native_smoke_plan=candidate['native_smoke_plan_server2_gpu0'])
    candidate, _, program, frozen = prepared(work)
    assert frozen['runtime_server'] == NAME
    authority = ref(GRANT)
    assert authority['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
    out = ROOT / ('pipeline/checks/y85_gpu0_activation_' + token + '.json')
    assert not out.exists()
    tick.save(out, dict(complete=True, token=token, id=CID, observation=current,
        active_plan=candidate['native_smoke_plan'], Y83_terminal=done['terminal_evidence'],
        latest_resource_authority=authority, retained_plan_grant=frozen['resource_authorization'],
        GPU1_use_authorized=False, native_executed=False, helper=ref(__file__)))
    candidate.update(stage='validation', next_action='Execute the original GPU0 native TRAIN two-step plan, then verify/admit/dispatch only on actual PASS.')
    for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition', 'occupation_evidence'):
        candidate.pop(key, None)
    candidate['evidence'].extend([ref(out), authority])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(activation=ref(out), program=ref(program), native_executed=False)))


def attach(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate, _, _, frozen = prepared(work)
    assert candidate['stage'] == 'validation' and not candidate.get('native_smoke_result')
    path = ROOT / ('pipeline/analyses/y85_gpu0_smoke_' + token + '.json')
    raw = tick.load(path)
    validate_result(raw, candidate, frozen)
    candidate['native_smoke_result'] = ref(path)
    candidate['evidence'].extend([ref(path), ref(path.with_suffix('.log')), ref(__file__)])
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='native_smoke', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y85_gpu0_smoke <original-frozen-program> --token ' + token + ' --seconds 600 --gpu --gpu-id 0',
        observation='Original native TRAIN two finite losses/backward/optimizer steps and shared mask normalization paths verified. Active output equivalence to W2 is not claimed.',
        evidence=[ref(path), ref(path.with_suffix('.log'))])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(result=ref(path), losses=raw['losses'], peak_reserved_bytes=raw['peak_reserved_bytes'])))
'''

GRADIENT_TEST = '''    def test_input_gradient_and_real_update_required(self):
        raw = copy.deepcopy(self.fixture)
        raw['gradient_checks'][1]['raw_and_normalized_gradients_finite'] = False
        with self.assertRaises(AssertionError):
            self.check(raw)
        for key in ('new_gradient_maxima', 'parameter_updates'):
            for value in (0., float('nan'), float('inf')):
                raw = copy.deepcopy(self.fixture)
                raw['gradient_checks'][1][key][1] = value
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    self.check(raw)
            raw = copy.deepcopy(self.fixture)
            raw['gradient_checks'][1][key].pop()
            with self.assertRaises(AssertionError):
                self.check(raw)
'''

PATH_TEST = '''    def test_original_initialization_identities_and_reference_required(self):
        for key in ('original_x_mask_output_identity', 'all_channels_and_original_epsilon',
                    'input_undetached', 'float64_LayerNorm_reference', 'shared_module_identity'):
            raw = copy.deepcopy(self.fixture)
            raw['norm_paths'][1][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(raw)
'''

NORM_TEST = '''    def test_zero_then_learned_response_required(self):
        for key in ('normalization_before_batching', 'static_mask_feature_reuse_preserved',
                    'original_hard_gate_algorithm_preserved'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.fixture, **{key: False}))
'''


def create():
    outputs = {}
    for filename in ('y84_deployment.py', 'advance_y84.py', 'test_y84_admission.py'):
        source = (SCRIPTS / filename).read_text(encoding='utf-8')
        text = source.replace('Y84', 'Y85').replace('y84', 'y85').replace('Y83', 'Y84').replace('y83', 'y84')
        text = text.replace('7bef2cd313b870583c970c56580be5a78097234d', 'aa9d8db3ea05432181ada752eaeeca27952aae27')
        if filename == 'y84_deployment.py':
            text = text.replace('adares_y85_visual_query_init', 'adares_y85_mask_feature_norm')
            text = text.replace('range(72, 85)', 'range(72, 86)')
            text = text.replace('previous.NAMES', 'previous.previous.NAMES')
            text = text.replace("{'AdaRES-Y84': ('server-2', Y84, 280.),\n                           'AdaRES-Y85': ('server-1', Y85, 350.)}",
                                "{'AdaRES-Y84': ('server-1', Y84, 350.),\n                           'AdaRES-Y85': ('server-2', Y85, 280.)}")
        elif filename == 'advance_y84.py':
            text = text.replace("NAME, GPU, POWER = 'server-1', 0, 350.", "NAME, GPU, POWER = 'server-2', 0, 280.")
            text = text.replace("GRANT = 'pipeline/authorizations/server1_gpu0_20261025.json'",
                                "GRANT = 'pipeline/authorizations/server2_gpu1_pause_20261104.json'\nFROZEN_GRANT = 'pipeline/authorizations/server2_gpu1_20261029.json'")
            text = text.replace('native_smoke_plan_server1', 'native_smoke_plan_server2_gpu0')
            text = text.replace("frozen['resource_authorization'] == ref(GRANT)", "frozen['resource_authorization'] == ref(FROZEN_GRANT)")
            text = replace_function(text, 'validate_result', VALIDATE)
            text = text.replace("    command = 'python -B scripts/y85_server1_alternative.py run --token ' + token",
                "    command = 'python -B scripts/run_experiment_first_probe.py y85_gpu0_smoke ' + ref(program)['path'] + ' --token ' + token + ' --seconds 600 --gpu --gpu-id 0'")
            text = text.replace('visual query initialization', 'shared visual mask-feature normalization')
            text = text.replace('and350W', 'and280W')
            text = text.replace("choices=('verify', 'admit', 'dispatch')", "choices=('activate', 'attach', 'verify', 'admit', 'dispatch')")
            text = text.replace("        dict(verify=verify, admit=admit, dispatch=dispatch)[args.operation](args.token)",
                                "        dict(activate=activate, attach=attach, verify=verify, admit=admit, dispatch=dispatch)[args.operation](args.token)")
            text = text.replace("\nif __name__ == '__main__':", '\n' + ACTIVATE + "\n\nif __name__ == '__main__':")
        else:
            text = text.replace('native_smoke_plan_server1', 'native_smoke_plan_server2_gpu0')
            text = text.replace('new_gradient_max=.1, parameter_update_max=.01',
                                'new_gradient_maxima=[.1, .2], parameter_updates=[.01, .02]')
            text = text.replace('both_query_and_scene_gradients_finite=True',
                                'raw_and_normalized_gradients_finite=True, coarse_gate_calls=6')
            text = text.replace('initialization_calls', 'mask_feature_norm_calls').replace('initialization_paths', 'norm_paths')
            text = text.replace("cpu['CPU_norm_paths']", "cpu['CPU_norm_paths']")
            text = text.replace("cpu['CPU_initialization_paths']", "cpu['CPU_norm_paths']")
            text = text.replace('static_mask_features_preserved', 'static_mask_feature_reuse_preserved')
            text = text.replace("mask_feature_norm_calls=2,", "mask_feature_norm_calls=2, original_coarse_gate_consumption_calls=12, normalization_before_batching=True,")
            text = text.replace("'/data1/songcl/Project/AdaRES_project/data/'",
                                "'/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/'")
            text = replace_function(text, 'test_input_gradient_and_real_update_required', GRADIENT_TEST)
            text = replace_function(text, 'test_original_initialization_identities_and_reference_required', PATH_TEST)
            text = replace_function(text, 'test_zero_then_learned_response_required', NORM_TEST)
            text = text.replace("'query_norm_consumption_calls', 'mask_feature_norm_calls')",
                                "'query_norm_consumption_calls', 'mask_feature_norm_calls', 'original_coarse_gate_consumption_calls')")
            text = text.replace("'indicator_calls', 'query_norm_calls', 'tensors')",
                                "'indicator_calls', 'query_norm_calls', 'coarse_gate_calls', 'tensors')")
            text = text.replace("('AdaRES-Y82', deploy.previous.Y82)", "('AdaRES-Y83', deploy.previous.Y83)")
            text = text.replace("advance.idle(dict(gpu='0,1,350', compute=''), 350.)", "advance.idle(dict(gpu='0,1,280', compute=''), 280.)")
            text = text.replace("[('nan,0,350', ''), ('0,0,390', ''), ('0,0,280', ''),\n                             ('0,2000,350', ''), ('0,0,350', '99')]",
                                "[('nan,0,280', ''), ('0,0,390', ''), ('0,0,350', ''),\n                             ('0,2000,280', ''), ('0,0,280', '99')]")
            text = text.replace("advance.idle(dict(gpu=gpu, compute=compute), 350.)", "advance.idle(dict(gpu=gpu, compute=compute), 280.)")
        ast.parse(text, feature_version=(3, 8))
        outputs[SCRIPTS / filename.replace('y84', 'y85')] = text
    assert all(not path.exists() for path in outputs)
    for path, source in outputs.items():
        path.write_text(source, encoding='utf-8', newline='\n')
    print('Created Y85 execution/source/admission guard helpers; original files retained.')


if __name__ == '__main__':
    create()
