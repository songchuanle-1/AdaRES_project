"""CPU-only regression checks for lifecycle-specific resource handoffs."""

from copy import deepcopy
import unittest

from recheck_gpu0_lifecycle import disposition, role


class LifecycleTests(unittest.TestCase):
    def completed(self):
        return dict(id='AdaRES-Y58', stage='blocked', blocked_from_stage='validation',
                    blocker_kind='exclusive_gpu_slot',
                    scientific_status='fixed30_complete_frozen_review_pending',
                    post_training_freeze={'path': 'frozen.json', 'sha256': 'untouched'},
                    native_smoke_plan={'path': 'old_training_plan.json'},
                    evidence=[{'path': 'original.json', 'sha256': 'untouched'}])

    def resource(self, idle=False):
        return dict(observed_at='2026-10-23T00:51:39+08:00', idle_authorized_gpus=[0] if idle else [],
                    training=[dict(id='AdaRES-Y59', process_verified=True)])

    def test_completed_with_old_smoke_plan_still_requires_evaluation(self):
        c = self.completed()
        self.assertEqual(role(c, {'status': 'done_complete'}), 'frozen_evaluation')

    def test_completion_needs_canonical_terminal_state(self):
        with self.assertRaises(AssertionError):
            role(self.completed(), {'status': 'running'})

    def test_user_pause_is_never_released_by_resource_check(self):
        self.assertIsNone(role(dict(id='AdaRES-Y57', blocker_kind='human_authorization'), {}))

    def test_busy_completed_work_preserves_evidence_and_protocol(self):
        c = self.completed()
        old = deepcopy(c)
        updated = disposition(c, 'frozen_evaluation', self.resource(), {'path': 'resource.json'})
        self.assertEqual(c, old)
        self.assertEqual(updated['post_training_freeze'], old['post_training_freeze'])
        self.assertEqual(updated['evidence'], old['evidence'])
        self.assertIn('Do not restart training', updated['next_action'])
        self.assertIn('AdaRES-Y59', updated['blocker'])
        self.assertNotIn('Y56', updated['blocker'])

    def test_idle_completed_work_never_reenters_training_admission(self):
        updated = disposition(self.completed(), 'frozen_evaluation', self.resource(True), {})
        self.assertEqual(updated['resource_work_kind'], 'frozen_evaluation')
        self.assertNotIn('blocker_kind', updated)
        self.assertIn('registered full frozen evaluation', updated['next_action'])

    def test_uncommitted_preparation_stays_distinct_from_native_smoke(self):
        c = dict(id='AdaRES-Y61', blocker_kind='exclusive_gpu_slot', decoder_integration_check={"path": "cpu"})
        self.assertEqual(role(c, {}), 'version_then_native_smoke')
        updated = disposition(c, 'version_then_native_smoke', self.resource(), {})
        self.assertEqual(updated['blocked_from_stage'], 'implementation')
        self.assertIn('already running, not waiting', updated['next_action'])

    def test_committed_y60_keeps_current_smoke_priority(self):
        c = dict(id='AdaRES-Y60', blocker_kind='exclusive_gpu_slot', native_smoke_plan={'path': 'plan'})
        self.assertEqual(role(c, {}), 'native_smoke')
        updated = disposition(c, 'native_smoke', self.resource(), {})
        self.assertEqual(updated['native_smoke_plan'], c['native_smoke_plan'])
        self.assertIn('already committed Y60', updated['next_action'])


if __name__ == '__main__':
    unittest.main()
