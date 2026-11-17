"""Reject false Y86 evidence; fixtures do not count as native execution."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
import advance_y86 as advance
import y86_deployment as deploy


class Admission(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = copy.deepcopy(next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y86'))
        plan = tick.load(ROOT / self.c['native_smoke_plan_server2_gpu0']['path'])
        self.f = ast.literal_eval(ast.parse(
            (ROOT / plan['program']['path']).read_text(encoding='utf-8')).body[0].value)
        version = tick.load(ROOT / self.c['source_manifest']['path'])
        cpu = tick.load(ROOT / version['CPU_result']['path'])
        row = dict(step=0, all_finite=True, tensors=1, original_head_gradient_tensors=10,
            new_gradients=[.1, .2], parameter_updates=[.01, .02], prediction_calls=7,
            indicator_calls=7, query_norm_calls=7, shortcut_calls=7,
            original_query_and_mask_gradients_finite=True)
        self.fixture = dict(complete=True, id='AdaRES-Y86', config=self.c['config'],
            code_commit=deploy.Y86, goal_sha256=self.f['goal_sha256'],
            native_YAML_sha256=self.f['variant']['config_data']['sha256'],
            native_entry='tools/train.py::train', split='train', steps=2, optimizer_steps=2,
            losses=[1., 2.], native_data_loaded=True, backward_finite=True,
            intended_path_observed=True, native_modes_verified=True, static_mask_features_preserved=True,
            original_hard_gate_algorithm_preserved=True, original_max_overlap_formula_preserved=True,
            validation_used=False, model_weights_saved=False, shared_head_calls=14,
            query_norm_consumption_calls=14, shortcut_calls=14,
            gradient_checks=[copy.deepcopy(row), dict(row, step=1)],
            batches=[dict(scan_ids=['a', 'b'], expressions=1, points=1),
                     dict(scan_ids=['c', 'd'], expressions=1, points=1)],
            shortcut_paths=copy.deepcopy(cpu['CPU_shortcut_paths']), peak_reserved_bytes=1024,
            data_source=dict(dataset_class='ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge',
                             config_data_root='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/'))

    def check(self, raw=None):
        advance.validate_result(self.fixture if raw is None else raw, self.c, self.f)

    def test_schema_fixture_only(self):
        self.check()

    def test_actual_native_record_when_available(self):
        if not self.c.get('native_smoke_result'):
            self.skipTest('Actual native record is not available yet')
        self.check(tick.load(ROOT / self.c['native_smoke_result']['path']))

    def test_wrong_identity_version_yaml_or_goal(self):
        for key, value in [('id', 'AdaRES-Y85'), ('code_commit', '0' * 40),
                           ('config', 'other.yaml'), ('goal_sha256', '0' * 64),
                           ('native_YAML_sha256', '0' * 64)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.fixture, **{key: value}))

    def test_native_train_and_two_real_steps_required(self):
        for key, value in [('split', 'val'), ('native_entry', 'synthetic'), ('steps', 0),
                ('optimizer_steps', 1), ('native_data_loaded', False), ('backward_finite', False),
                ('intended_path_observed', False), ('native_modes_verified', False),
                ('validation_used', True), ('model_weights_saved', True)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.fixture, **{key: value}))

    def test_two_finite_losses_required(self):
        for losses in ([1.], [1., float('nan')], [float('inf'), 1.]):
            with self.assertRaises(AssertionError):
                self.check(dict(self.fixture, losses=losses))

    def test_original_masks_gates_and_overlap_formula_required(self):
        for key in ('static_mask_features_preserved', 'original_hard_gate_algorithm_preserved',
                    'original_max_overlap_formula_preserved'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.fixture, **{key: False}))

    def test_all_shared_consumers_required(self):
        for key in ('shared_head_calls', 'query_norm_consumption_calls', 'shortcut_calls'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.fixture, **{key: 0}))

    def test_existing_overlap_and_original_logit_identities_required(self):
        for key in ('original_logits_identity', 'original_all_peer_max_overlap_reference',
                    'no_new_overlap_or_detach', 'both_inputs_undetached', 'float64_scalar_projection_reference'):
            raw = copy.deepcopy(self.fixture)
            raw['shortcut_paths'][8][key] = False
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(raw)

    def test_zero_then_learned_response_required(self):
        for index, key, value in [(0, 'weight_nonzero', True), (3, 'confidence_delta_max', .1),
                                 (8, 'weight_nonzero', False), (10, 'confidence_delta_max', 0.)]:
            raw = copy.deepcopy(self.fixture)
            raw['shortcut_paths'][index][key] = value
            with self.assertRaises(AssertionError):
                self.check(raw)

    def test_exact_shortcut_rows_and_order_required(self):
        for rows in (self.fixture['shortcut_paths'][:-1], list(reversed(self.fixture['shortcut_paths']))):
            with self.assertRaises(AssertionError):
                self.check(dict(self.fixture, shortcut_paths=rows))

    def test_nonfinite_shortcut_response_refused(self):
        for value in (float('nan'), float('inf'), -1.):
            raw = copy.deepcopy(self.fixture)
            raw['shortcut_paths'][8]['confidence_delta_max'] = value
            with self.assertRaises(AssertionError):
                self.check(raw)

    def test_original_input_gradients_and_new_parameter_updates_required(self):
        raw = copy.deepcopy(self.fixture)
        raw['gradient_checks'][1]['original_query_and_mask_gradients_finite'] = False
        with self.assertRaises(AssertionError):
            self.check(raw)
        for key in ('new_gradients', 'parameter_updates'):
            for value in (0., float('nan'), float('inf')):
                raw = copy.deepcopy(self.fixture)
                raw['gradient_checks'][1][key][1] = value
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    self.check(raw)

    def test_per_step_shared_consumers_and_order_required(self):
        for key in ('original_head_gradient_tensors', 'prediction_calls', 'indicator_calls',
                    'query_norm_calls', 'shortcut_calls', 'tensors'):
            raw = copy.deepcopy(self.fixture)
            raw['gradient_checks'][1][key] = 0
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(raw)
        raw = copy.deepcopy(self.fixture)
        raw['gradient_checks'].reverse()
        with self.assertRaises(AssertionError):
            self.check(raw)

    def test_original_dataset_and_four_scenes_required(self):
        for key, value in [('config_data_root', '/wrong/data'), ('dataset_class', 'synthetic')]:
            raw = copy.deepcopy(self.fixture)
            raw['data_source'][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
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
        advance.idle(dict(gpu='0,1,280', compute=''), 280.)
        for gpu, compute in [('nan,0,280', ''), ('0,0,390', ''), ('0,0,350', ''),
                             ('0,2000,280', ''), ('0,0,280', '99')]:
            with self.assertRaises(AssertionError):
                advance.idle(dict(gpu=gpu, compute=compute), 280.)

    def test_reservations_and_Y57_pause_required(self):
        paused = dict(id='AdaRES-Y57', status='paused', gpu=1,
                      dispatch_hold=True, resume_requires_user_authorization=True)
        queue = dict(gpu_queues={'server-2': [paused]})
        advance.unreserved(queue, {}, 'server-2', 0)
        for status in ('waiting', 'claiming', 'running'):
            value = copy.deepcopy(queue)
            value['gpu_queues']['server-2'].append(dict(id='foreign', status=status, gpu=0))
            with self.assertRaises(AssertionError):
                advance.unreserved(value, {}, 'server-2', 0)
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, {'server-2': {'0': 'foreign'}}, 'server-2', 0)
        queue['gpu_queues']['server-2'][0]['status'] = 'running'
        with self.assertRaises(AssertionError):
            advance.unreserved(queue, {}, 'server-2', 0)

    def test_only_seven_tested_additions_allowed(self):
        rows = [['A', path] for path in deploy.paths('AdaRES-Y86')]
        deploy.validate_delta(rows, 'AdaRES-Y86')
        for bad in (rows + [['A', 'other.py']], [['M', rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, 'AdaRES-Y86')

    def test_additive_source_chain_preserves_original_production(self):
        for cid, commit in [('AdaRES-Y84', deploy.Y84), ('AdaRES-Y85', deploy.Y85)]:
            self.assertIn(deploy.Y86, deploy.contract(commit, deploy.Y86, cid))
        for commit, head in [(deploy.Y85, '0' * 40), ('0' * 40, deploy.Y86), (deploy.Y86, deploy.Y85)]:
            with self.assertRaises(AssertionError):
                deploy.contract(commit, head, 'AdaRES-Y84')


if __name__ == '__main__':
    unittest.main()
