"""Negative guards for the exact additive marker contract; no training fixtures."""
import copy
import unittest

import observe_server2_y68_additive_phase as observer


class AdditiveGuards(unittest.TestCase):
    def setUp(self):
        self.edges = ((observer.ORIGINAL, observer.Y67_COMMIT, 'adares_y67_stage_readout_norm'),
                      (observer.Y67_COMMIT, observer.Y68_COMMIT, 'adares_y68_language_mask_modulation'))

    def test_both_exact_seven_file_additions(self):
        for base, target, package in self.edges:
            rows = [['A', path] for path in observer.expected_additions(package)]
            self.assertEqual(observer.validate_delta(base, target, rows), package)

    def test_unknown_descendant_or_reversed_edge_refused(self):
        for base, target in ((observer.Y68_COMMIT, 'a' * 40),
                             (observer.Y67_COMMIT, observer.ORIGINAL),
                             (observer.ORIGINAL, observer.Y68_COMMIT)):
            with self.assertRaises(AssertionError): observer.validate_delta(base, target, [])

    def test_missing_or_duplicate_path_refused(self):
        for base, target, package in self.edges:
            rows = [['A', path] for path in observer.expected_additions(package)]
            for changed in (rows[:-1], rows[:-1] + [rows[0]]):
                with self.assertRaises(AssertionError): observer.validate_delta(base, target, changed)

    def test_changed_keeper_or_extra_file_refused(self):
        for base, target, package in self.edges:
            rows = [['A', path] for path in observer.expected_additions(package)]
            for changed in (rows + [['M', 'AdaRES/tools/train.py']],
                            rows[:-1] + [['A', 'AdaRES/ipdn/model_paper5/adares_y65_confidence_feedback/dec.py']]):
                with self.assertRaises(AssertionError): observer.validate_delta(base, target, changed)

    def test_modified_deleted_or_renamed_candidate_refused(self):
        for base, target, package in self.edges:
            rows = [['A', path] for path in observer.expected_additions(package)]
            for status in ('M', 'D', 'R100'):
                changed = copy.deepcopy(rows); changed[0][0] = status
                with self.assertRaises(AssertionError): observer.validate_delta(base, target, changed)

    def task(self, cid):
        identity = observer.IDENTITIES[cid]
        return dict(id=cid, status='running', launch_verified=True, gpu=identity['gpu'],
            pid=identity['pid'], code_commit=identity['code_commit'],
            config='configs_song/model_paper5/' + identity['package'] + '_res.yaml')

    def test_exact_original_task_identities(self):
        for cid in observer.IDENTITIES:
            self.assertEqual(observer.validate_identity(self.task(cid)), observer.IDENTITIES[cid])

    def test_changed_gpu_pid_commit_config_refused(self):
        for cid in observer.IDENTITIES:
            for key, value in (('gpu', 9), ('pid', 1), ('code_commit', 'a' * 40), ('config', 'other.yaml')):
                task = self.task(cid); task[key] = value
                with self.assertRaises(AssertionError): observer.validate_identity(task)

    def test_paused_unverified_or_nonboolean_launch_refused(self):
        for cid in observer.IDENTITIES:
            for key, value in (('status', 'paused'), ('launch_verified', False), ('launch_verified', 1)):
                task = self.task(cid); task[key] = value
                with self.assertRaises(AssertionError): observer.validate_identity(task)

    def test_other_candidate_or_paused_y57_refused(self):
        for cid in ('AdaRES-Y57', 'AdaRES-Y68'):
            with self.assertRaises(KeyError): observer.validate_identity(dict(id=cid))

    def test_log_regression_or_truncation_refused(self):
        before = dict(latest=dict(phase='Train', epoch=24, total=30, step=10, count=4686), log_bytes=100)
        for step, size in ((9, 101), (11, 99)):
            after = copy.deepcopy(before); after['latest']['step'] = step; after['log_bytes'] = size
            with self.assertRaises(AssertionError): observer.phases.progressed(before, after)

    def test_static_counter_is_unconfirmed(self):
        sample = dict(latest=dict(phase='Train', epoch=24, total=30, step=10, count=4686), log_bytes=100)
        self.assertFalse(observer.phases.progressed(sample, copy.deepcopy(sample)))

    def test_remote_program_compiles_as_python38_and_remains_readonly(self):
        program = observer.remote_program(dict())
        self.assertIn("assert verify_runtime()==marker", program)
        self.assertIn("marker in P['allowed_markers']", program)
        self.assertNotIn("write_text", program)
        self.assertNotIn("os.kill", program)


if __name__ == '__main__':
    unittest.main()
