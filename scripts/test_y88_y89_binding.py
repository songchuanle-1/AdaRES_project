"""Additive source guards; these tests never load Y88 checkpoints or stop training."""
import copy
import unittest
from prepare_y60_stage_heads import ROOT, monitor
import y88_y89_terminal_contract as binding
import y89_deployment as deploy


class AdditiveBinding(unittest.TestCase):
    def task_and_server(self):
        task = dict(binding.SPEC, status='running', launch_verified=True, gpu=0)
        server = monitor.load_servers(str(ROOT))['server-1']
        return task, server

    def test_original_run_and_checkpoint_directory(self):
        self.assertEqual(binding.SPEC['pid'], 3527278)
        self.assertEqual(binding.SPEC['code_commit'], deploy.Y88)
        self.assertEqual(binding.CHECKPOINT_DIRECTORY, '20261106_053553')
        self.assertEqual(binding.SPEC['admission_sha256'],
                         '39f9a71debe48314d479b42b604565cf557ccbe40e5582e949a66959587b3572')
        self.assertEqual(binding.SPEC['freeze_sha256'],
                         '93bf51b65a580cbdd5a7ade69fae370d8d0e6e6db5ab665645892c775d84aa81')

    def test_only_explicit_original_and_tested_additive_markers(self):
        for marker in (deploy.Y88, deploy.Y89):
            binding.validate_marker(marker, 'AdaRES-Y88')
        for marker in (deploy.Y87, '0' * 40):
            with self.assertRaises(AssertionError):
                binding.validate_marker(marker, 'AdaRES-Y88')
        with self.assertRaises(AssertionError):
            binding.validate_marker(deploy.Y89, 'AdaRES-Y89')

    def test_unchanged_original_source_and_exact_new_additions(self):
        deployments = deploy.contract(deploy.Y88, deploy.Y89, 'AdaRES-Y88')
        self.assertEqual(set(deployments), {deploy.Y88, deploy.Y89})
        self.assertEqual(len(deployments[deploy.Y89]['source_files']), 6)
        self.assertEqual(len(deployments[deploy.Y89]['configs']), 1)

    def test_actual_current_additive_contract(self):
        task, server = self.task_and_server()
        value = binding.build_contract(task, server)
        self.assertEqual(set(value['allowed_markers']), {deploy.Y88, deploy.Y89})

    def test_original_run_identity_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in [('pid', 1), ('gpu', 1), ('server', 'server-2'),
                           ('code_commit', deploy.Y89), ('log', '/wrong/run.log')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                binding.build_contract(dict(task, **{key: value}), server)

    def test_original_runtime_and_mappings_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in [('python', '/wrong/python'), ('runtime_library_path', '/wrong/lib'),
                ('allowed_gpus', [0, 1]), ('runtime_path_mappings', [])]:
            changed = copy.deepcopy(server)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                binding.build_contract(task, changed)


if __name__ == '__main__':
    unittest.main()
