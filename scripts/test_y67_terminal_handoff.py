"""CPU lifecycle guards; fixtures are not actual completion evidence."""
import copy
import unittest
import finalize_y67_fixed30 as archive


class TerminalHandoff(unittest.TestCase):
    def setUp(self):
        self.task = dict(id='AdaRES-Y67', status='running', launch_verified=True, gpu='1',
            server='server-2', pid=100811, config='candidate.yaml', code_commit='a' * 40)
        self.candidate = dict(id='AdaRES-Y67', stage='training', config='candidate.yaml', code_commit='a' * 40)
        self.paused = dict(id='AdaRES-Y57', status='paused', dispatch_hold='user pause', resume_requires_user_authorization=True)
        self.payload = dict(id='AdaRES-Y67', work='/project/checkpoints', basename='candidate',
            checkpoint_directory='20261027_005030', log='/project/code/logs/run.log')
        self.proof = dict(complete=True, id='AdaRES-Y67', process_absent=True, CUDA_initialized=False,
            log=self.payload['log'], log_sha256='b' * 64, full_frozen_evaluation_executed=False,
            checkpoints=[dict(path='/project/checkpoints/candidate/20261027_005030/' + name,
                epoch=epoch, scheduler_last_epoch=epoch, optimizer_state_count=544, best_miou=.5,
                model_finite=True, optimizer_finite=True, sha256='c' * 64)
                for name, epoch in [('last.pth', 30), ('best.pth', 21)]])

    def verify(self):
        archive.validate_proof(self.proof, self.payload)

    def test_valid_schema_only(self):
        archive.validate_task(self.task, self.candidate, self.paused)
        self.verify()

    def test_wrong_id_gpu_server_pid_or_status_refused(self):
        for key, value in [('id', 'AdaRES-Y62'), ('gpu', '0'), ('server', 'server-1'),
                           ('pid', 1), ('status', 'done_complete')]:
            task = dict(self.task, **{key: value})
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.validate_task(task, self.candidate, self.paused)

    def test_changed_pause_refused(self):
        self.paused['status'] = 'running'
        with self.assertRaises(AssertionError):
            archive.validate_task(self.task, self.candidate, self.paused)

    def test_incomplete_last_epoch_refused(self):
        self.proof['checkpoints'][0].update(epoch=29, scheduler_last_epoch=29)
        with self.assertRaises(AssertionError): self.verify()

    def test_reset_scheduler_refused(self):
        self.proof['checkpoints'][0]['scheduler_last_epoch'] = 0
        with self.assertRaises(AssertionError): self.verify()

    def test_empty_optimizer_refused(self):
        self.proof['checkpoints'][0]['optimizer_state_count'] = 0
        with self.assertRaises(AssertionError): self.verify()

    def test_wrong_checkpoint_lineage_refused(self):
        self.proof['checkpoints'][0]['path'] = '/other/last.pth'
        with self.assertRaises(AssertionError): self.verify()

    def test_nonfinite_model_optimizer_or_metric_refused(self):
        for field, value in [('model_finite', False), ('optimizer_finite', False), ('best_miou', float('nan'))]:
            before = copy.deepcopy(self.proof)
            self.proof['checkpoints'][0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError): self.verify()
            self.proof = before

    def test_live_process_not_terminal_refused(self):
        self.proof['process_absent'] = False
        with self.assertRaises(AssertionError): self.verify()

    def test_only_exact_mapping_removed_without_aliasing(self):
        before = {'server-2': {'1': 'AdaRES-Y67', '0': 'untouched'}, 'server-1': {'0': 'untouched'}}
        original = copy.deepcopy(before)
        after = archive.expected_mapping(before)
        self.assertEqual(after, {'server-2': {'0': 'untouched'}, 'server-1': {'0': 'untouched'}})
        self.assertEqual(before, original)

    def test_new_mapping_owner_or_duplicate_refused(self):
        for value in ({'server-2': {'1': 'foreign'}},
                      {'server-2': {'1': 'AdaRES-Y67'}, 'server-1': {'1': 'AdaRES-Y67'}}):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                archive.expected_mapping(value)

    def test_empty_server_removed_other_server_preserved(self):
        self.assertEqual(archive.expected_mapping({'server-2': {'1': 'AdaRES-Y67'}, 'server-1': {}}),
                         {'server-1': {}})


if __name__ == '__main__':
    unittest.main()
