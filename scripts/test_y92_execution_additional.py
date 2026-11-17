"""Exercise current activation identities without GPU use or native-result fixtures."""
import ast
import copy
import unittest
from unittest.mock import patch
from prepare_y60_stage_heads import ROOT, monitor, ref, tick
import advance_y92 as advance
import y92_deployment as deploy
import test_y92_execution_guards as original


class ExecutionGuards(original.ExecutionGuards):
    def test_chosen_server1_payload_without_mutating_actual_blocked_candidate(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        before = copy.deepcopy(work)
        chosen = copy.deepcopy(work)
        candidate = next(row for row in chosen['candidates'] if row['id'] == 'AdaRES-Y92')
        candidate.update(required_server='server-1', required_gpu=0,
                         native_smoke_plan=candidate['native_smoke_plan_server1'])
        actual, plan, program, frozen = advance.prepared(chosen)
        self.assertEqual(actual['code_commit'], deploy.Y92)
        self.assertEqual(frozen['runtime_server'], 'server-1')
        self.assertEqual(frozen['runtime_power_limit_watts'], 350.)
        self.assertEqual(plan['program'], ref(program))
        self.assertEqual(tick.load(ROOT / 'pipeline/research_work.json'), before)

    def test_actual_current_owners_are_still_reserved(self):
        queue = monitor.load_queue(str(ROOT))
        running = monitor.load_running(str(ROOT))
        self.assertEqual(running, {'server-1': {'0': 'AdaRES-Y90'}, 'server-2': {'0': 'AdaRES-Y91'}})
        for name, cid in [('server-1', 'AdaRES-Y90'), ('server-2', 'AdaRES-Y91')]:
            task = next(row for row in queue['gpu_queues'][name] if row['id'] == cid)
            self.assertEqual(task['status'], 'running')
            self.assertTrue(task['launch_verified'])
            with self.assertRaises(AssertionError):
                advance.unreserved(queue, running, name, 0)

    def test_native_result_is_missing_and_cannot_reach_verify(self):
        candidate = self.candidate()
        self.assertFalse(candidate.get('native_smoke_result'))
        self.assertFalse(candidate.get('admission_receipt'))
        self.assertFalse((ROOT / 'pipeline/admissions/AdaRES-Y92.json').exists())
        with patch.object(advance, 'prepared', return_value=(candidate, None, None, None)):
            with self.assertRaises(AssertionError):
                advance.verify('fixture_no_native')

    def test_actual_admission_suite_requires_native_result_before_checking(self):
        import test_y92_admission
        case = test_y92_admission.Admission('test_actual_native_result_and_chosen_runtime')
        with self.assertRaisesRegex(AssertionError, 'Actual Y92 native evidence'):
            case.setUp()
        for name in ('advance_y92.py', 'y92_deployment.py', 'test_y92_admission.py'):
            ast.parse((ROOT / 'scripts' / name).read_text(encoding='utf-8'), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
