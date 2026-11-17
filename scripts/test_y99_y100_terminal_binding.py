"""Check original Y99 plus actual native Y100; no synthetic success artifacts."""
import copy
import unittest
from prepare_y60_stage_heads import ROOT, monitor, tick
import y99_y100_terminal_contract as binding
import y100_deployment as deploy

class AdditiveBinding(unittest.TestCase):

    def task_and_server(self):
        return (dict(binding.SPEC, status='running', launch_verified=True, gpu=0), monitor.load_servers(str(ROOT))['server-1'])

    def test_original_run_directory_and_signed_receipts(self):
        self.assertEqual(binding.SPEC['pid'], 2980836)
        self.assertEqual(binding.SPEC['code_commit'], deploy.Y99)
        self.assertEqual(binding.CHECKPOINT_DIRECTORY, '20261111_180638')
        self.assertEqual(binding.SPEC['admission_sha256'], '4d87ad8b3a6efe78be8dc9e8214bba9226160a22b649f64f8e1dd26d73988ede')
        self.assertEqual(binding.SPEC['freeze_sha256'], 'fd07c0536891d905a7cd89e2b4211b32ed31a532b7b70ac9d71a8d7a66e97dbc')

    def test_only_original_and_actual_native_markers(self):
        for marker in (deploy.Y99, deploy.Y100):
            binding.validate_marker(marker, 'AdaRES-Y99')
        for marker in (deploy.Y98, '0' * 40):
            with self.assertRaises(AssertionError):
                binding.validate_marker(marker, 'AdaRES-Y99')
        with self.assertRaises(AssertionError):
            binding.validate_marker(deploy.Y100, 'AdaRES-Y100')

    def test_exact_additions_and_unchanged_original_source(self):
        values = deploy.contract(deploy.Y99, deploy.Y100, 'AdaRES-Y99')
        self.assertEqual(list(values), [deploy.Y99, deploy.Y100])
        self.assertEqual(len(values[deploy.Y100]['source_files']), 6)
        self.assertEqual(len(values[deploy.Y100]['configs']), 1)
        rows = [row.split('\t') for row in deploy.git('diff', '--no-renames', '--name-status', deploy.Y99, deploy.Y100).decode().splitlines()]
        deploy.validate_delta(rows, 'AdaRES-Y100')

    def test_actual_native_bound_full_contract(self):
        task, server = self.task_and_server()
        self.assertEqual(binding.build_contract(task, server)['allowed_markers'], [deploy.Y99, deploy.Y100])

    def test_original_owner_identity_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in (('id', 'AdaRES-Y100'), ('pid', 1), ('gpu', 1), ('server', 'server-2'), ('code_commit', deploy.Y100), ('config', 'other.yaml'), ('log', '/wrong/run.log'), ('status', 'done_complete'), ('launch_verified', False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                binding.build_contract(dict(task, **{key: value}), server)

    def test_private_runtime_and_exact_mappings_cannot_change(self):
        task, server = self.task_and_server()
        for key, value in (('python', '/wrong/python'), ('allowed_gpus', [0, 1]), ('work_dir', '/wrong/checkpoints'), ('expected_hostname', 'foreign'), ('remote_code_root', '/wrong/code'), ('runtime_path_mappings', []), ('runtime_library_path', '/wrong/cuda')):
            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):
                binding.build_contract(task, dict(server, **{key: value}))

    def test_actual_native_evidence_mandatory(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y100'))
        self.assertEqual(binding.native_evidence(work), candidate['native_smoke_result'])
        changed = copy.deepcopy(work)
        next((row for row in changed['candidates'] if row['id'] == 'AdaRES-Y100')).pop('native_smoke_result')
        with self.assertRaisesRegex(AssertionError, 'Actual additive Y100 native evidence'):
            binding.native_evidence(changed)

    def test_foreign_modified_missing_additions_refused(self):
        rows = [['A', path] for path in deploy.paths('AdaRES-Y100')]
        for bad in (rows[:-1], rows + [['A', 'other.py']], [['M', rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, 'AdaRES-Y100')
if __name__ == '__main__':
    unittest.main()
