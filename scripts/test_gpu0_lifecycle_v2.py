"""Lifecycle v1 regression suite plus the collector's idle-slot transition."""

import unittest

from test_gpu0_lifecycle import LifecycleTests
from recheck_gpu0_lifecycle_v2 import role


class IdleTransitionTests(unittest.TestCase):
    def test_idle_collector_output_is_pending_native_smoke(self):
        candidate = dict(id='AdaRES-Y60', stage='validation', native_smoke_plan={'path': 'plan'})
        self.assertEqual(role(candidate, {}), 'native_smoke')
        self.assertNotIn('blocker_kind', candidate)

    def test_user_hold_not_reclassified_even_if_validation(self):
        candidate = dict(id='AdaRES-Y60', stage='validation', blocker_kind='human_authorization')
        self.assertIsNone(role(candidate, {}))

    def test_completed_state_not_reclassified_as_unrun_smoke(self):
        candidate = LifecycleTests().completed()
        self.assertEqual(role(candidate, {'status': 'done_complete'}), 'frozen_evaluation')


if __name__ == '__main__':
    unittest.main()
