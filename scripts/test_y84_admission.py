"""Schema fixtures reject false evidence; the native-record test requires real output."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
import advance_y84 as advance
import y84_deployment as deploy


class Admission(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = copy.deepcopy(next(candidate for candidate in work['candidates']
                                   if candidate['id'] == 'AdaRES-Y84'))
        plan = tick.load(ROOT / self.c['native_smoke_plan_server1']['path'])
        self.f = ast.literal_eval(ast.parse(
            (ROOT / plan['program']['path']).read_text(encoding='utf-8')).body[0].value)
        version = tick.load(ROOT / self.c['source_manifest']['path'])
        cpu = tick.load(ROOT / version['CPU_result']['path'])
        row = dict(step=0, all_finite=True, tensors=1, indicator_tensors=10,
            new_gradient_max=.1, parameter_update_max=.01, prediction_calls=7,
            indicator_calls=7, query_norm_calls=7, both_query_and_scene_gradients_finite=True)
        self.fixture = dict(complete=True, id='AdaRES-Y84', config=self.c['config'],
            code_commit=deploy.Y84, goal_sha256=self.f['goal_sha256'],
            native_YAML_sha256=self.f['variant']['config_data']['sha256'],
            native_entry='tools/train.py::train', split='train', steps=2, optimizer_steps=2,
            losses=[1., 2.], native_data_loaded=True, backward_finite=True,
            intended_path_observed=True, native_modes_verified=True, static_mask_features_preserved=True,
            original_hard_gate_algorithm_preserved=True, validation_used=False, model_weights_saved=False,
            shared_head_calls=14, query_norm_consumption_calls=14, initialization_calls=2,
            gradient_checks=[copy.deepcopy(row), dict(row, step=1)],
            batches=[dict(scan_ids=['a', 'b'], expressions=1, points=1),
                     dict(scan_ids=['c', 'd'], expressions=1, points=1)],
            initialization_paths=copy.deepcopy(cpu['CPU_initialization_paths']), peak_reserved_bytes=1024,
            data_source=dict(dataset_class='ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge',
                             config_data_root='/data1/songcl/Project/AdaRES_project/data/'))

    def check(self, raw=None):
        advance.validate_result(self.fixture if raw is None else raw, self.c, self.f)

    def test_schema_fixture_only(self):
        self.check()

    def test_actual_native_record_when_available(self):
        if not self.c.get('native_smoke_result'):
            self.skipTest('No actual native GPU record yet; schema fixtures do not confer PASS')
        raw = tick.load(ROOT / self.c['native_smoke_result']['path'])
        self.check(raw)
        self.assertEqual(raw['id'], 'AdaRES-Y84')

    def test_wrong_identity_version_yaml_or_goal(self):
        for key, value in [('id', 'AdaRES-Y82'), ('code_commit', '0' * 40),
                           ('config', 'other.yaml'), ('goal_sha256', '0' * 64),
                           ('native_YAML_sha256', '0' * 64)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.fixture, **{key: value}))

    def test_missing_or_nonfinite_losses(self):
        for losses in ([1.], [1., float('nan')], [float('inf'), 1.]):
            with self.assertRaises(AssertionError):
                self.check(dict(self.fixture, losses=losses))

    def test_native_entry_train_and_real_execution_required(self):
        for key, value in [('split', 'val'), ('native_entry', 'synthetic'),
                           ('native_data_loaded', False), ('backward_finite', False),
                           ('intended_path_observed', False), ('native_modes_verified', False),
                           ('validation_used', True), ('model_weights_saved', True)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.fixture, **{key: value}))

    def test_all_native_steps_and_shared_consumers_required(self):
        for key in ('steps', 'optimizer_steps', 'shared_head_calls',
                    'query_norm_consumption_calls', 'initialization_calls'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.fixture, **{key: 0}))

    def test_static_mask_and_original_gate_algorithm_required(self):
        for key in ('static_mask_features_preserved', 'original_hard_gate_algorithm_preserved'):
            with self.assertRaises(AssertionError):
                self.check(dict(self.fixture, **{key: False}))

    def test_original_initialization_identities_and_reference_required(self):
        for key in ('original_generated_query_identity', 'original_scene_visual_identity',
                    'original_padding_and_point_mass_identity', 'all_inputs_undetached',
                    'float64_weighted_context_reference'):
            raw = copy.deepcopy(self.fixture)
            raw['initialization_paths'][1][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(raw)

    def test_zero_then_learned_response_required(self):
        for index, key, value in [(0, 'weight_nonzero', True), (0, 'query_delta_max', .1),
                                 (1, 'weight_nonzero', False), (1, 'query_delta_max', 0.)]:
            raw = copy.deepcopy(self.fixture)
            raw['initialization_paths'][index][key] = value
            with self.assertRaises(AssertionError):
                self.check(raw)

    def test_two_ordered_initialization_rows_required(self):
        for rows in (self.fixture['initialization_paths'][:1],
                     list(reversed(self.fixture['initialization_paths']))):
            with self.assertRaises(AssertionError):
                self.check(dict(self.fixture, initialization_paths=rows))

    def test_input_gradient_and_real_update_required(self):
        for key, value in [('both_query_and_scene_gradients_finite', False),
                           ('new_gradient_max', 0.), ('parameter_update_max', 0.),
                           ('new_gradient_max', float('nan')), ('parameter_update_max', float('inf'))]:
            raw = copy.deepcopy(self.fixture)
            raw['gradient_checks'][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(raw)

    def test_per_step_shared_consumers_and_order_required(self):
        for key in ('indicator_tensors', 'prediction_calls', 'indicator_calls', 'query_norm_calls', 'tensors'):
            raw = copy.deepcopy(self.fixture)
            raw['gradient_checks'][1][key] = 0
            with self.assertRaises(AssertionError):
                self.check(raw)
        raw = copy.deepcopy(self.fixture)
        raw['gradient_checks'].reverse()
        with self.assertRaises(AssertionError):
            self.check(raw)

    def test_original_dataset_and_four_scenes_required(self):
        for key, value in [('config_data_root', '/wrong/data'), ('dataset_class', 'synthetic')]:
            raw = copy.deepcopy(self.fixture)
            raw['data_source'][key] = value
            with self.assertRaises(AssertionError):
                self.check(raw)
        raw = copy.deepcopy(self.fixture)
        raw['batches'][1] = copy.deepcopy(raw['batches'][0])
        with self.assertRaises(AssertionError):
            self.check(raw)

    def test_24GB_resource_bound_required(self):
        for peak in (0, 24 * 2**30, float('nan')):
            with self.assertRaises(AssertionError):
                self.check(dict(self.fixture, peak_reserved_bytes=peak))

    def test_foreign_compute_busy_or_changed_power_refused(self):
        advance.idle(dict(gpu='0,1,350', compute=''), 350.)
        for gpu, compute in [('nan,0,350', ''), ('0,0,390', ''), ('0,0,280', ''),
                             ('0,2000,350', ''), ('0,0,350', '99')]:
            with self.assertRaises(AssertionError):
                advance.idle(dict(gpu=gpu, compute=compute), 350.)

    def test_canonical_reservations_and_y57_pause_required(self):
        paused = dict(id='AdaRES-Y57', status='paused', gpu=1,
                      dispatch_hold=True, resume_requires_user_authorization=True)
        queue = dict(gpu_queues={'server-2': [paused], 'server-1': []})
        advance.unreserved(queue, {}, 'server-1', 0)
        for status in ('waiting', 'claiming', 'running'):
            value = copy.deepcopy(queue)
            value['gpu_queues']['server-1'].append(dict(id='foreign', status=status, gpu=0))
            with self.assertRaises(AssertionError):
                advance.unreserved(value, {}, 'server-1', 0)
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, {'server-1': {'0': 'foreign'}}, 'server-1', 0)
        queue['gpu_queues']['server-2'][0]['status'] = 'running'
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, {}, 'server-1', 0)

    def test_only_seven_exact_additions_allowed(self):
        rows = [['A', path] for path in deploy.paths('AdaRES-Y84')]
        deploy.validate_delta(rows, 'AdaRES-Y84')
        for bad in (rows + [['A', 'other.py']], [['M', rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, 'AdaRES-Y84')

    def test_additive_chain_preserves_original_y82_y83(self):
        for cid, commit in [('AdaRES-Y82', deploy.previous.Y82), ('AdaRES-Y83', deploy.Y83)]:
            self.assertIn(deploy.Y84, deploy.contract(commit, deploy.Y84, cid))

    def test_unknown_or_reversed_source_chain_refused(self):
        for commit, head in [(deploy.Y83, '0' * 40), ('0' * 40, deploy.Y84), (deploy.Y84, deploy.Y83)]:
            with self.assertRaises(AssertionError):
                deploy.contract(commit, head, 'AdaRES-Y83')


if __name__ == '__main__':
    unittest.main()
