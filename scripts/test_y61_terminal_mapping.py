"""Guard the exact stale-slot repair without touching another slot or paused job."""

import unittest
from finalize_y61_fixed30 import reconciled_mapping


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.running = {'server-2': {'0': 'AdaRES-Y61', '1': 'untouched'}, 'other': {'0': 'untouched'}}
        self.tasks = [dict(id='AdaRES-Y57', status='paused', gpu=1, dispatch_hold='user pause')]
        self.experiment = dict(id='AdaRES-Y61', status='done_complete')

    def test_only_exact_stale_slot_removed(self):
        result = reconciled_mapping(self.running, self.tasks, self.experiment)
        self.assertEqual(result, {'server-2': {'1': 'untouched'}, 'other': {'0': 'untouched'}})
        self.assertEqual(self.running['server-2']['0'], 'AdaRES-Y61')

    def test_not_completed_refused(self):
        self.experiment['status'] = 'running'
        with self.assertRaises(AssertionError):
            reconciled_mapping(self.running, self.tasks, self.experiment)

    def test_queued_y61_refused(self):
        self.tasks.append(dict(id='AdaRES-Y61', status='running', gpu=0))
        with self.assertRaises(AssertionError):
            reconciled_mapping(self.running, self.tasks, self.experiment)

    def test_new_slot_owner_refused(self):
        self.running['server-2']['0'] = 'AdaRES-Y62'
        with self.assertRaises(AssertionError):
            reconciled_mapping(self.running, self.tasks, self.experiment)

    def test_other_active_task_refused(self):
        self.tasks.append(dict(id='AdaRES-Y62', status='claiming', gpu=0))
        with self.assertRaises(AssertionError):
            reconciled_mapping(self.running, self.tasks, self.experiment)

    def test_y57_pause_change_refused(self):
        self.tasks[0]['status'] = 'running'
        with self.assertRaises(AssertionError):
            reconciled_mapping(self.running, self.tasks, self.experiment)


if __name__ == '__main__':
    unittest.main()
