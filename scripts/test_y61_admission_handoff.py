"""Synthetic schema fixtures are negative controls, never admission evidence."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import admit_dispatch_y61_gpu0 as handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.candidate = dict(id='AdaRES-Y61', config='candidate.yaml', code_commit='a' * 40)
        self.frozen = dict(code_commit='a' * 40, goal_sha256='b' * 64,
            variant=dict(config='candidate.yaml', config_data=dict(sha256='c' * 64)))
        check = dict(all_finite=True, indicator_tensors=10, indicator_calls=7,
            query_norm_input_requires_grad=[True, False] * 7,
            query_and_mask_input_gradients_finite=True, trainable_query_norm_gradient_nonzero=True)
        self.raw = dict(complete=True, id='AdaRES-Y61', config='candidate.yaml', code_commit='a' * 40,
            goal_sha256='b' * 64, native_YAML_sha256='c' * 64, steps=2, optimizer_steps=2,
            losses=[7.0, 8.0], native_entry='tools/train.py::train', split='train',
            native_data_loaded=True, backward_finite=True, intended_path_observed=True,
            validation_used=False, model_weights_saved=False, shared_head_calls=14,
            query_norm_input_requires_grad=[True, False] * 14,
            gradient_checks=[copy.deepcopy(check), copy.deepcopy(check)], peak_reserved_bytes=1024,
            batches=[dict(scan_ids=['a', 'b'], expressions=8, points=100)] * 2,
            data_source=dict(dataset_class='fixture_not_real', config_data_root='fixture_not_real'))

    def verify(self):
        handoff.validate_result(self.raw, self.candidate, self.frozen)

    def test_schema_fixture_only(self):
        self.verify()

    def test_nonfinite_loss_refused(self):
        self.raw['losses'][0] = float('nan')
        with self.assertRaises(AssertionError):
            self.verify()

    def test_one_step_refused(self):
        self.raw['optimizer_steps'] = 1
        with self.assertRaises(AssertionError):
            self.verify()

    def test_wrong_goal_refused(self):
        self.raw['goal_sha256'] = 'd' * 64
        with self.assertRaises(AssertionError):
            self.verify()

    def test_wrong_yaml_refused(self):
        self.raw['native_YAML_sha256'] = 'd' * 64
        with self.assertRaises(AssertionError):
            self.verify()

    def test_wrong_commit_refused(self):
        self.raw['code_commit'] = 'e' * 40
        with self.assertRaises(AssertionError):
            self.verify()

    def test_wrong_candidate_refused(self):
        self.raw['id'] = 'AdaRES-Y60'
        with self.assertRaises(AssertionError):
            self.verify()

    def test_unrun_native_refused(self):
        self.raw['native_data_loaded'] = False
        with self.assertRaises(AssertionError):
            self.verify()

    def test_live_peer_input_refused(self):
        self.raw['gradient_checks'][1]['query_norm_input_requires_grad'] = [True, True] * 7
        with self.assertRaises(AssertionError):
            self.verify()

    def test_missing_head_path_refused(self):
        self.raw['shared_head_calls'] = 12
        with self.assertRaises(AssertionError):
            self.verify()

    def test_validation_input_refused(self):
        self.raw['split'] = 'val'
        with self.assertRaises(AssertionError):
            self.verify()

    def test_running_task_stops_admit_before_artifact_writes(self):
        tasks = [dict(id='AdaRES-Y60', status='running')]
        with patch.object(handoff, 'exact_candidate', return_value=({}, self.candidate, {}, Path('unused'), self.frozen)), \
             patch.object(handoff, 'protected_queue', return_value=({}, {}, tasks)), \
             patch.object(handoff.tick, 'save') as save:
            with self.assertRaises(AssertionError):
                handoff.admit('fixture')
            save.assert_not_called()

    def test_busy_gpu_stops_dispatch_before_launcher(self):
        tasks = [dict(id='AdaRES-Y61', status='waiting')]
        with patch.object(handoff, 'protected_queue', return_value=({}, {}, tasks)), \
             patch.object(handoff.monitor, 'load_experiment', return_value=dict(status='done_complete')), \
             patch.object(handoff.closure, 'admission'), \
             patch.object(handoff, 'require_idle', side_effect=AssertionError('busy GPU0')), \
             patch.object(handoff.monitor, '_process_queue') as launch, \
             patch.object(handoff.tick, 'save') as save:
            with self.assertRaises(AssertionError):
                handoff.dispatch('fixture')
            launch.assert_not_called()
            save.assert_not_called()

    def test_missing_raw_smoke_refused_before_artifact_writes(self):
        self.candidate['stage'] = 'validation'

        def load(path):
            if Path(path).name == '.tick_model_view.json':
                return dict(queue=dict(candidate_budget=1))
            raise FileNotFoundError('No real native smoke result')

        def experiment(root, identity):
            return dict(status='done_complete') if identity == 'AdaRES-Y60' else None

        with patch.object(handoff, 'exact_candidate', return_value=({}, self.candidate, {}, Path('unused'), self.frozen)), \
             patch.object(handoff, 'protected_queue', return_value=({}, {}, [])), \
             patch.object(handoff.monitor, 'load_experiment', side_effect=experiment), \
             patch.object(handoff.tick, 'load', side_effect=load), \
             patch.object(handoff.tick, 'save') as save:
            with self.assertRaises(FileNotFoundError):
                handoff.admit('fixture')
            save.assert_not_called()

    def test_actual_occupied_inventory_refused_before_power_query(self):
        live = dict(gpu_query_rc=0, compute_query_rc=0,
            authorized_device_inventory='0, GPU-fixture, RTX3090, 50, 22744, 24576\n',
            compute='GPU-fixture, 123, 22000\n')
        with patch.object(handoff, 'snapshot', return_value=live), \
             patch.object(handoff.monitor, 'ssh') as ssh:
            with self.assertRaises(AssertionError):
                handoff.require_idle({})
            ssh.assert_not_called()


if __name__ == '__main__':
    unittest.main()
