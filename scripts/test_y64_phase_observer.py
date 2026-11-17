"""Detect real Train/Val progress while refusing stalls, regressions and wrong counters."""
import unittest
from observe_y64_server1_phase import parse_phase, progressed


def sample(phase='Train', epoch=24, step=4686, size=100):
    return dict(latest=dict(phase=phase, epoch=epoch, total=30, step=step,
                           count=4686 if phase == 'Train' else 363), log_bytes=size)


class PhaseObserverGuards(unittest.TestCase):
    def test_train_counter_advances(self):
        self.assertTrue(progressed(sample(step=10), sample(step=18, size=200)))

    def test_validation_only_tail_parses_and_progresses(self):
        row = parse_phase('Val[24/30]:  42%|progress| 153/363 [01:04<01:07]\r')
        self.assertEqual(row, sample('Val', step=153)['latest'])
        self.assertTrue(progressed(sample('Val', step=153), sample('Val', step=165, size=200)))

    def test_last_logged_phase_selected(self):
        text = 'Train[24/30]: 100%|x| 4686/4686 [done]\rVal[24/30]: 1%|x| 5/363 [step]\r'
        self.assertEqual(parse_phase(text), sample('Val', step=5)['latest'])

    def test_train_to_val_and_next_epoch(self):
        self.assertTrue(progressed(sample(), sample('Val', step=1, size=200)))
        self.assertTrue(progressed(sample('Val', step=363), sample(epoch=25, step=1, size=200)))

    def test_unchanged_counters_or_no_log_growth_unconfirmed(self):
        self.assertFalse(progressed(sample(step=10), sample(step=10, size=200)))
        self.assertFalse(progressed(sample(step=10), sample(step=11)))

    def test_backward_phase_or_truncated_log_refused(self):
        for before, after in ((sample('Val', step=1), sample(size=200)),
                              (sample(epoch=25, step=10), sample(epoch=24, size=200)),
                              (sample(step=11), sample(step=10, size=200)),
                              (sample(step=10), sample(step=11, size=99))):
            with self.assertRaises(AssertionError): progressed(before, after)

    def test_wrong_schedule_or_denominator_refused(self):
        for text in ('Val[24/40]: 1%|x| 5/363 [x]', 'Train[24/30]: 1%|x| 10/363 [x]',
                     'Val[31/30]: 1%|x| 5/363 [x]', 'Val[24/30]: 1%|x| 364/363 [x]'):
            with self.assertRaises(AssertionError): parse_phase(text)

    def test_unrelated_or_missing_native_counter_refused(self):
        for text in ('', 'some-process 10/30 [x]', 'Train[24/30] broken', 'Val[24/30]: metrics only'):
            with self.assertRaises(AssertionError): parse_phase(text)


if __name__ == '__main__':
    unittest.main()
