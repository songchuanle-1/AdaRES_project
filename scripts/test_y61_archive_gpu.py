"""Guard a single metadata correction and preserve unrelated ledger content."""

import unittest
from reconcile_y61_archive_gpu import corrected_experiment, corrected_ledger, corrected_analysis


class GPURepairTests(unittest.TestCase):
    def setUp(self):
        self.exp = dict(id='AdaRES-Y61', status='done_complete', server='server-2', gpu='9',
                        history=[dict(action='launched', gpu='0')], preserved='value')
        self.csv = 'id,status,gpus,notes\r\nOther,done_complete,1,"a,b"\r\nAdaRES-Y61,done_complete,9,kept\r\n'

    def test_only_experiment_gpu_changes(self):
        result = corrected_experiment(self.exp, 9)
        self.assertEqual(result['gpu'], '0')
        self.assertEqual(result['preserved'], 'value')
        self.assertEqual(self.exp['gpu'], '9')

    def test_wrong_id_refused(self):
        self.exp['id'] = 'AdaRES-Y60'
        with self.assertRaises(AssertionError):
            corrected_experiment(self.exp, 9)

    def test_unfinished_refused(self):
        self.exp['status'] = 'running'
        with self.assertRaises(AssertionError):
            corrected_experiment(self.exp, 9)

    def test_launch_proof_required(self):
        self.exp['history'][0]['gpu'] = '1'
        with self.assertRaises(AssertionError):
            corrected_experiment(self.exp, 9)

    def test_only_target_csv_cell_changes(self):
        self.assertEqual(corrected_ledger(self.csv, 9), self.csv.replace('complete,9,kept', 'complete,0,kept'))

    def test_duplicate_csv_refused(self):
        with self.assertRaises(AssertionError):
            corrected_ledger(self.csv + 'AdaRES-Y61,done_complete,9,kept\r\n', 9)

    def test_changed_gpu_refused(self):
        with self.assertRaises(AssertionError):
            corrected_ledger(self.csv.replace('complete,9,kept', 'complete,1,kept'), 9)

    def test_only_analysis_gpu_line_changes(self):
        before = '  - ID: AdaRES-Y61\r\n  - Server: server-2 | GPU: 9\r\n  - State: done_complete\r\n'
        self.assertEqual(corrected_analysis(before, 9), before.replace('GPU: 9', 'GPU: 0'))


if __name__ == '__main__':
    unittest.main()
