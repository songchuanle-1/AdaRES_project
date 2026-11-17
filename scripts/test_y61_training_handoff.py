"""CPU guard fixtures are not observations of production training."""

import copy
import unittest

from capture_y61_training_handoff import validate_evaluation_state, validate_progress


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.root = '/project/AdaRES'
        self.task = dict(id='AdaRES-Y61', status='running', gpu=0, pid=42,
            config='configs/y61.yaml', launch_verified=True, fixed_epochs=30,
            min_epochs=30, patience=30)
        sample = dict(process_live=True, cwd=self.root, starttime='123',
            cmdline=['python', 'tools/train.py', self.task['config']],
            log_size_bytes=100, progress=['Train', '1', '30', '10', '4686'],
            tail_errors=[], device=dict(uuid='GPU-e0b11432-b7aa-2006-1903-dcc739f26146',
                index=0, power_limit_watts=280))
        second = copy.deepcopy(sample)
        second.update(log_size_bytes=200, progress=['Train', '1', '30', '20', '4686'])
        self.observation = dict(samples=[sample, second], compute=dict(rc=0,
            stdout='GPU-e0b11432-b7aa-2006-1903-dcc739f26146, 42, 20000\n'
                   'GPU-f740248b-4c9e-c8bb-b14b-99d0b9eb6c79, 99, 20000\n'),
            performance=dict(rc=0, stdout='SW Thermal Slowdown : Not Active'),
            kernel=dict(rc=0, stdout=''))

    def test_growth_with_foreign_gpu1(self):
        result = validate_progress(self.task, self.observation, self.root)
        self.assertTrue(result['native_progress_verified'])
        self.assertFalse(result['gpu1_compute_empty'])

    def test_wrong_gpu(self):
        self.task['gpu'] = 1
        with self.assertRaises(AssertionError):
            validate_progress(self.task, self.observation, self.root)

    def test_no_growth(self):
        self.observation['samples'][1] = copy.deepcopy(self.observation['samples'][0])
        with self.assertRaises(AssertionError):
            validate_progress(self.task, self.observation, self.root)

    def test_regression(self):
        self.observation['samples'][1]['progress'][3] = '9'
        with self.assertRaises(AssertionError):
            validate_progress(self.task, self.observation, self.root)

    def test_changed_process(self):
        self.observation['samples'][1]['starttime'] = '999'
        with self.assertRaises(AssertionError):
            validate_progress(self.task, self.observation, self.root)

    def test_wrong_cap(self):
        self.observation['samples'][1]['device']['power_limit_watts'] = 390
        with self.assertRaises(AssertionError):
            validate_progress(self.task, self.observation, self.root)

    def test_runtime_error(self):
        self.observation['samples'][1]['tail_errors'] = ['RuntimeError']
        with self.assertRaises(AssertionError):
            validate_progress(self.task, self.observation, self.root)

    def test_thermal_observation_not_suppressed(self):
        self.observation['performance']['stdout'] = 'SW Thermal Slowdown : Active'
        result = validate_progress(self.task, self.observation, self.root)
        self.assertTrue(result['thermal_slowdown_active'])
        self.assertFalse(result['hardware_root_cause_resolved'])

    def test_stale_evaluation_resource_field(self):
        validate_evaluation_state(dict(id='AdaRES-Y56', stage='validation',
            scientific_status='fixed30_complete_frozen_review_pending',
            blocker_kind='exclusive_gpu_slot'))

    def test_human_hold_not_cleared(self):
        with self.assertRaises(AssertionError):
            validate_evaluation_state(dict(id='AdaRES-Y56', stage='blocked',
                scientific_status='fixed30_complete_frozen_review_pending',
                blocker_kind='human_authorization'))


if __name__ == '__main__':
    unittest.main()
