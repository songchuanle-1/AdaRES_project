"""No-CUDA controls for busy/idle/completed/user-held lifecycle checks."""

from datetime import datetime
import unittest

from recheck_gpu0_lifecycle_v4 import is_due, update_resource, work_kind


class LifecycleV4Tests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.fromisoformat('2026-10-23T20:54:25+08:00')

    def test_actual_idle_collector_shape_has_no_retry(self):
        c = dict(id='AdaRES-Y60', stage='validation', native_smoke_plan={'path': 'plan'})
        self.assertTrue(is_due(c, self.now))
        self.assertEqual(work_kind(c, {}), 'native_smoke')

    def test_completed_y59_is_evaluation_even_with_old_smoke_plan(self):
        c = dict(id='AdaRES-Y59', stage='validation', admission_receipt='old',
            scientific_status='fixed30_complete_frozen_review_pending', post_training_freeze={'path': 'freeze'},
            native_smoke_plan={'path': 'old'})
        self.assertTrue(is_due(c, self.now))
        self.assertEqual(work_kind(c, {'status': 'done_complete'}), 'frozen_evaluation')

    def test_completion_requires_canonical_evidence(self):
        c = dict(id='AdaRES-Y59', stage='validation',
                 scientific_status='fixed30_complete_frozen_review_pending', post_training_freeze={'path': 'freeze'})
        with self.assertRaises(AssertionError):
            work_kind(c, {'status': 'running'})

    def test_live_y60_is_never_returned_to_smoke(self):
        self.assertIsNone(work_kind(dict(id='AdaRES-Y60', stage='training', admission_receipt='receipt'), {}))

    def test_paused_y57_remains_human_held(self):
        self.assertIsNone(work_kind(dict(id='AdaRES-Y57', stage='blocked', blocker_kind='human_authorization'), {}))

    def test_future_retry_is_not_due(self):
        self.assertFalse(is_due(dict(stage='blocked', retry_after='2026-10-23T21:54:25+08:00'), self.now))

    def test_busy_evaluation_never_restarts_training(self):
        original = dict(stage='validation', post_training_freeze={'path': 'untouched'})
        out = update_resource(original, 'frozen_evaluation', [], self.now, ['AdaRES-Y60'])
        self.assertEqual(out['stage'], 'blocked')
        self.assertIn('AdaRES-Y60', out['blocker'])
        self.assertIn('No training restart', out['next_action'])
        self.assertEqual(out['post_training_freeze'], original['post_training_freeze'])
        self.assertEqual(original['stage'], 'validation')

    def test_prepared_y62_y63_are_native_work(self):
        for cid in ('AdaRES-Y62', 'AdaRES-Y63'):
            c = dict(id=cid, stage='blocked', blocker_kind='exclusive_gpu_slot', native_smoke_plan={'path': 'plan'})
            self.assertEqual(work_kind(c, {}), 'native_smoke')

    def test_unregistered_busy_process_is_reported(self):
        out = update_resource(dict(stage='validation'), 'native_smoke', [], self.now,
                              ['unregistered GPU0 compute PID 13813'])
        self.assertEqual(out['stage'], 'blocked')
        self.assertIn('13813', out['blocker'])

    def test_idle_removes_retry_and_blocker(self):
        out = update_resource(dict(stage='blocked', retry_after='old', blocker_kind='exclusive_gpu_slot'),
                              'native_smoke', [0], self.now, [])
        self.assertEqual(out['stage'], 'validation')
        self.assertNotIn('retry_after', out)
        self.assertTrue(is_due(out, self.now))


if __name__ == '__main__':
    unittest.main()
