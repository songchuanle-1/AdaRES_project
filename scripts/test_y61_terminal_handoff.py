"""CPU-only archival guards; fixtures are not terminal or checkpoint evidence."""

import unittest
from unittest.mock import patch

import finalize_y61_fixed30 as archive


class TerminalHandoffTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(id='AdaRES-Y61', status='running', launch_verified=True, gpu=0,
                         pid=78290, config='candidate.yaml', code_commit='a' * 40,
                         log='/project/code/logs/run.log')
        self.candidate = dict(id='AdaRES-Y61', stage='training', config='candidate.yaml',
                              code_commit='a' * 40)
        self.paused = dict(id='AdaRES-Y57', status='paused', dispatch_hold=True)
        self.payload = dict(id='AdaRES-Y61', work='/project/checkpoints', basename='candidate',
                            checkpoint_directory='20261023_205330', log='/project/code/logs/run.log')
        self.proof = dict(complete=True, id='AdaRES-Y61', process_absent=True, CUDA_initialized=False,
            log=self.payload['log'], log_sha256='b' * 64, full_frozen_evaluation_executed=False,
            checkpoints=[dict(path='/project/checkpoints/candidate/20261023_205330/' + name,
                epoch=epoch, scheduler_last_epoch=epoch, optimizer_state_count=599,
                best_miou=0.5, model_finite=True, optimizer_finite=True, sha256='c' * 64) for name, epoch in [('last.pth', 30), ('best.pth', 20)]])

    def verify(self):
        archive.validate_proof(self.proof, self.payload)

    def test_synthetic_schema_only(self):
        archive.validate_task(self.task, self.candidate, self.paused)
        self.verify()

    def test_wrong_experiment_refused(self):
        self.task['id'] = 'AdaRES-Y59'
        with self.assertRaises(AssertionError):
            archive.validate_task(self.task, self.candidate, self.paused)

    def test_gpu1_refused(self):
        self.task['gpu'] = 1
        with self.assertRaises(AssertionError):
            archive.validate_task(self.task, self.candidate, self.paused)

    def test_changed_y57_pause_refused(self):
        self.paused['status'] = 'running'
        with self.assertRaises(AssertionError):
            archive.validate_task(self.task, self.candidate, self.paused)

    def test_incomplete_epoch_refused(self):
        self.proof['checkpoints'][0].update(epoch=29, scheduler_last_epoch=29)
        with self.assertRaises(AssertionError):
            self.verify()

    def test_reset_scheduler_refused(self):
        self.proof['checkpoints'][0]['scheduler_last_epoch'] = 0
        with self.assertRaises(AssertionError):
            self.verify()

    def test_empty_optimizer_refused(self):
        self.proof['checkpoints'][0]['optimizer_state_count'] = 0
        with self.assertRaises(AssertionError):
            self.verify()

    def test_wrong_checkpoint_lineage_refused(self):
        self.proof['checkpoints'][0]['path'] = '/project/checkpoints/another/20261023_205330/last.pth'
        with self.assertRaises(AssertionError):
            self.verify()

    def test_nonfinite_best_refused(self):
        self.proof['checkpoints'][0]['best_miou'] = float('nan')
        with self.assertRaises(AssertionError):
            self.verify()

    def test_nonfinite_model_refused(self):
        self.proof['checkpoints'][0]['model_finite'] = False
        with self.assertRaises(AssertionError):
            self.verify()

    def test_nonfinite_optimizer_refused(self):
        self.proof['checkpoints'][1]['optimizer_finite'] = False
        with self.assertRaises(AssertionError):
            self.verify()

    def test_absent_process_not_claimed_refused(self):
        self.proof['process_absent'] = False
        with self.assertRaises(AssertionError):
            self.verify()

    def test_live_process_stops_before_checkpoint_load_or_archive(self):
        self.candidate['terminal_handoff_plan'] = dict(path='fixture', sha256='fixture')
        handoff = dict(helper=archive.ref(archive.__file__), readonly_checkpoint_verifier={},
            code_commit='a' * 40, payload=dict(self.task, root='/code', work='/checkpoints'))
        server = dict(remote_code_root='/code', work_dir='/checkpoints')
        before = dict(gpu_query_rc=0, compute_query_rc=0,
                      tasks={'AdaRES-Y61': dict(process_live=True)})
        with patch.object(archive, 'current', return_value=(server, {}, self.task, {}, self.candidate)), \
             patch.object(archive.closure, 'artifact', return_value='fixture'), \
             patch.object(archive.tick, 'load', return_value=handoff), \
             patch.object(archive, 'snapshot', return_value=before), \
             patch.object(archive.subprocess, 'run') as remote, \
             patch.object(archive.monitor, 'finalize_task') as finalize, \
             patch.object(archive.tick, 'save') as save:
            with self.assertRaises(AssertionError):
                archive.finalize('fixture')
            remote.assert_not_called()
            finalize.assert_not_called()
            save.assert_not_called()


if __name__ == '__main__':
    unittest.main()
