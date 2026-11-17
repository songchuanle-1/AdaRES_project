"""Actual native evidence and original Y98 lifecycle compatibility."""
import copy
import unittest
from prepare_y60_stage_heads import ROOT, monitor, tick
import y98_y99_terminal_contract as binding
import y99_deployment as deploy

class AdditiveBinding(unittest.TestCase):

    def task_and_server(self):
        return (dict(binding.SPEC, status='running', launch_verified=True, gpu=0), monitor.load_servers(str(ROOT))['server-2'])

    def test_original_identity_and_directory(self):
        self.assertEqual(binding.SPEC['pid'], 25962)
        self.assertEqual(binding.CHECKPOINT_DIRECTORY, '20261110_080627')
        self.assertEqual(binding.SPEC['admission_sha256'], 'dc1a8a6dd3afde694d255cceca30b1b16009e8249390aa706632e2a567136a62')

    def test_only_original_and_native_tested_markers(self):
        for marker in (deploy.Y95, deploy.Y96):
            binding.validate_marker(marker, 'AdaRES-Y98')
        for marker, cid in ((deploy.Y94, 'AdaRES-Y98'), ('0' * 40, 'AdaRES-Y98'), (deploy.Y96, 'AdaRES-Y99')):
            with self.assertRaises(AssertionError):
                binding.validate_marker(marker, cid)

    def test_exact_seven_additions_and_preserved_original(self):
        result = deploy.contract(deploy.Y95, deploy.Y96, 'AdaRES-Y98')
        self.assertEqual(list(result), [deploy.Y95, deploy.Y96])
        self.assertEqual(len(result[deploy.Y96]['source_files']), 6)
        self.assertEqual(len(result[deploy.Y96]['configs']), 1)

    def test_actual_native_bound_contract(self):
        task, server = self.task_and_server()
        self.assertEqual(binding.build_contract(task, server)['allowed_markers'], [deploy.Y95, deploy.Y96])

    def test_owner_changes_refused(self):
        task, server = self.task_and_server()
        for key, value in (('id', 'AdaRES-Y99'), ('pid', 1), ('gpu', 1), ('server', 'server-1'), ('code_commit', deploy.Y96), ('config', 'other.yaml'), ('log', '/wrong/run.log'), ('status', 'done_complete'), ('launch_verified', False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                binding.build_contract(dict(task, **{key: value}), server)

    def test_runtime_changes_refused(self):
        task, server = self.task_and_server()
        for key, value in (('python', '/wrong/python'), ('allowed_gpus', [0, 1]), ('work_dir', '/wrong/checkpoints'), ('expected_hostname', 'foreign'), ('remote_code_root', '/wrong/code'), ('runtime_path_mappings', [{'source': '/a', 'target': '/b'}])):
            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):
                binding.build_contract(task, dict(server, **{key: value}))

    def test_actual_native_evidence_required(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y99'))
        self.assertEqual(binding.native_evidence(work), candidate['native_smoke_result'])
        changed = copy.deepcopy(work)
        next((row for row in changed['candidates'] if row['id'] == 'AdaRES-Y99')).pop('native_smoke_result')
        with self.assertRaisesRegex(AssertionError, 'Actual additive Y99'):
            binding.native_evidence(changed)

    def test_missing_foreign_modified_paths_refused(self):
        rows = [['A', path] for path in deploy.paths('AdaRES-Y99')]
        deploy.validate_delta(rows, 'AdaRES-Y99')
        for bad in (rows[:-1], rows + [['A', 'other.py']], [['M', rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, 'AdaRES-Y99')
