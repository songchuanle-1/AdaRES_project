"""CPU controls for the observational single-card trial checker."""

import ast
import copy
import unittest

from check_gpu0_single_card_trial import REMOTE, evaluate


class TrialObservationTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(pid=13738, config='configs/unique.yaml')
        self.root = '/project/AdaRES'
        sample = dict(process_live=True, cwd=self.root, starttime='123',
            cmdline=['python', 'tools/train.py', self.task['config']],
            progress=['Train', '26', '30', '10', '4686'], log_size_bytes=100,
            device=dict(index=0, uuid='GPU-e0b11432-b7aa-2006-1903-dcc739f26146',
                        power_limit_watts=280), tail_errors=[])
        second = copy.deepcopy(sample)
        second.update(progress=['Train', '26', '30', '20', '4686'], log_size_bytes=200)
        self.observed = dict(samples=[sample, second],
            compute=dict(rc=0, stdout='GPU-e0b11432-b7aa-2006-1903-dcc739f26146, 13738, 22674\n'),
            performance=dict(rc=0, stdout='SW Thermal Slowdown : Not Active\nHW Thermal Slowdown : Not Active'),
            kernel=dict(rc=0, stdout=''))

    def check(self):
        return evaluate(self.observed, self.task, self.root)

    def test_healthy_and_remote_python38(self):
        ast.parse('TASK={}\n' + REMOTE, feature_version=(3, 8))
        self.assertTrue(self.check()['sample_healthy'])
        self.assertFalse(self.check()['hardware_root_cause_resolved'])

    def test_gpu1_occupied(self):
        self.observed['compute']['stdout'] += 'GPU-f740248b-4c9e-c8bb-b14b-99d0b9eb6c79, 999, 1\n'
        self.assertFalse(self.check()['gpu1_compute_empty'])

    def test_pid_reused(self):
        self.observed['samples'][1]['starttime'] = '456'
        self.assertFalse(self.check()['native_progress_verified'])

    def test_wrong_cwd(self):
        self.observed['samples'][1]['cwd'] = '/another/project'
        self.assertFalse(self.check()['process_identity_verified'])

    def test_unchanged_progress(self):
        self.observed['samples'][1]['progress'] = self.observed['samples'][0]['progress']
        self.assertFalse(self.check()['sample_healthy'])

    def test_thermal_active(self):
        self.observed['performance']['stdout'] = 'SW Thermal Slowdown : Active'
        self.assertTrue(self.check()['thermal_slowdown_active'])
        self.assertFalse(self.check()['sample_healthy'])

    def test_unknown_kernel_not_healthy(self):
        self.observed['kernel']['rc'] = 1
        self.assertIsNone(self.check()['kernel_xid_or_fallen_off_observed'])
        self.assertFalse(self.check()['sample_healthy'])

    def test_changed_power_cap(self):
        self.observed['samples'][1]['device']['power_limit_watts'] = 390
        self.assertFalse(self.check()['power_cap_preserved'])


if __name__ == '__main__':
    unittest.main()
