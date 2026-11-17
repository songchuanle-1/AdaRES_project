"""CPU lifecycle guards; fixtures are not actual completion evidence."""
import copy
import unittest
import finalize_y99_fixed30 as archive

class TerminalHandoff(unittest.TestCase):

    def setUp(self):
        self.task = dict(id='AdaRES-Y99', status='running', launch_verified=True, gpu='0', server='server-1', pid=2980836, config='configs_song/model_paper5/adares_y99_spatial_mask_skip_res.yaml', code_commit='53f5b028bdf24e8b41dd934931d1fedc99843d8d')
        self.candidate = dict(id='AdaRES-Y99', stage='training', config='configs_song/model_paper5/adares_y99_spatial_mask_skip_res.yaml', code_commit='53f5b028bdf24e8b41dd934931d1fedc99843d8d')
        self.paused = dict(id='AdaRES-Y57', status='paused', dispatch_hold='user pause', resume_requires_user_authorization=True)
        self.payload = dict(id='AdaRES-Y99', work='/project/checkpoints', basename='candidate', checkpoint_directory='20261111_180638', log='/project/code/logs/run.log')
        self.proof = dict(complete=True, id='AdaRES-Y99', process_absent=True, CUDA_initialized=False, log=self.payload['log'], log_sha256='b' * 64, full_frozen_evaluation_executed=False, completed_epochs={p: list(range(1, 31)) for p in ('Train', 'Val')}, checkpoints=[dict(path='/project/checkpoints/candidate/20261111_180638/' + name, epoch=epoch, scheduler_last_epoch=epoch, optimizer_state_count=544, best_miou=0.5, model_finite=True, optimizer_finite=True, sha256='c' * 64) for name, epoch in [('last.pth', 30), ('best.pth', 21)]])

    def verify(self):
        archive.validate_proof(self.proof, self.payload)

    def test_valid_schema_only(self):
        archive.validate_task(self.task, self.candidate, self.paused)
        self.verify()

    def test_wrong_id_gpu_server_pid_or_status_refused(self):
        for key, value in [('id', 'AdaRES-Y62'), ('gpu', '1'), ('server', 'server-2'), ('pid', 1), ('status', 'done_complete')]:
            task = dict(self.task, **{key: value})
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.validate_task(task, self.candidate, self.paused)

    def test_changed_pause_refused(self):
        self.paused['status'] = 'running'
        with self.assertRaises(AssertionError):
            archive.validate_task(self.task, self.candidate, self.paused)

    def test_incomplete_last_epoch_refused(self):
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
        self.proof['checkpoints'][0]['path'] = '/other/last.pth'
        with self.assertRaises(AssertionError):
            self.verify()

    def test_nonfinite_model_optimizer_or_metric_refused(self):
        for field, value in [('model_finite', False), ('optimizer_finite', False), ('best_miou', float('nan'))]:
            before = copy.deepcopy(self.proof)
            self.proof['checkpoints'][0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.verify()
            self.proof = before

    def test_live_process_not_terminal_refused(self):
        self.proof['process_absent'] = False
        with self.assertRaises(AssertionError):
            self.verify()

    def test_only_exact_mapping_removed_without_aliasing(self):
        before = {'server-1': {'0': 'AdaRES-Y99', '1': 'untouched'}, 'server-2': {'1': 'untouched'}}
        original = copy.deepcopy(before)
        after = archive.expected_mapping(before)
        self.assertEqual(after, {'server-1': {'1': 'untouched'}, 'server-2': {'1': 'untouched'}})
        self.assertEqual(before, original)

    def test_new_mapping_owner_or_duplicate_refused(self):
        for value in ({'server-1': {'0': 'foreign'}}, {'server-1': {'0': 'AdaRES-Y99'}, 'server-2': {'0': 'AdaRES-Y99'}}):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                archive.expected_mapping(value)

    def test_empty_server_removed_other_server_preserved(self):
        self.assertEqual(archive.expected_mapping({'server-1': {'0': 'AdaRES-Y99'}, 'server-2': {}}), {'server-2': {}})

    def test_changed_code_or_config_refused(self):
        for key, value in [('code_commit', 'a' * 40), ('config', 'other.yaml')]:
            t = dict(self.task, **{key: value})
            c = dict(self.candidate, **{key: value})
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.validate_task(t, c, self.paused)

    def test_missing_native_epoch_refused(self):
        self.proof['completed_epochs']['Train'].remove(13)
        with self.assertRaises(AssertionError):
            self.verify()

    def test_mismatched_best_metric_refused(self):
        self.proof['checkpoints'][0]['best_miou'] = 0.6
        with self.assertRaises(AssertionError):
            self.verify()

    def test_unknown_or_wrong_resource_refused(self):
        base = dict(complete=True, process_absent=True, hostname='dl02-MS-7B73', gpu='0, 0, 350', compute='')
        for field, value in [('gpu', 'nan, 0, 350'), ('gpu', '0, 0, 390'), ('hostname', 'other'), ('complete', False)]:
            with self.subTest(field=field, value=value), self.assertRaises(AssertionError):
                archive.validate_resource(dict(base, **{field: value}))

    def test_foreign_compute_preserved_as_busy(self):
        base = dict(complete=True, process_absent=True, hostname='dl02-MS-7B73', gpu='0, 0, 350', compute='')
        self.assertTrue(archive.validate_resource(base))
        self.assertFalse(archive.validate_resource(dict(base, compute='999')))
        self.assertFalse(archive.validate_resource(dict(base, gpu='2300, 60, 350')))

    def test_remote_live_pid_refused_before_gpu_or_checkpoint_reads(self):
        from unittest.mock import patch
        with patch('subprocess.check_output', return_value=b'dl02-MS-7B73') as query, patch('pathlib.Path.exists', return_value=True):
            with self.assertRaises(AssertionError):
                exec(archive.RESOURCE, {'P': {'pid': 2980836}})
            self.assertEqual(query.call_args_list[0].args[0], ['hostname'])
            self.assertEqual(query.call_count, 1)

    def test_exact_registered_production_markers_only(self):
        from y99_terminal_contract import Y99, validate_marker
        for value in (Y99,):
            validate_marker(value, 'AdaRES-Y99')
        with self.assertRaises(AssertionError):
            validate_marker('a' * 40, 'AdaRES-Y99')

    def test_additive_edges_reject_foreign_or_modified_paths(self):
        from y99_terminal_contract import validate_delta, paths
        for cid in ('AdaRES-Y99',):
            rows = [['A', path] for path in paths(cid)]
            validate_delta(rows, cid)
            for bad in (rows + [['A', 'other.py']], [['M', rows[0][1]]] + rows[1:]):
                with self.assertRaises(AssertionError):
                    validate_delta(bad, cid)

    def test_archived_resource_follows_original_server1_gpu0(self):
        result = dict(id=self.task['id'], status='done_complete', gpu='0')
        archive.validate_archived_experiment(result, self.task)
        for key, value in [('gpu', '1'), ('id', 'foreign'), ('status', 'running')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.validate_archived_experiment(dict(result, **{key: value}), self.task)

    def test_evaluation_resource_follows_original_server1_gpu0(self):
        self.assertEqual(archive.evaluation_resource(self.task), dict(required_server='server-1', required_gpu=0))
        for key, value in [('gpu', '1'), ('server', 'server-2'), ('id', 'foreign')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.evaluation_resource(dict(self.task, **{key: value}))

    def test_declared_original_location_required(self):
        location = dict(complete=True, checkpoint_loaded=False, remote_mutations=False, log=self.payload['log'], checkpoint_directory='20261111_180638', declared_last_path='/project/checkpoints/candidate/20261111_180638/last.pth')
        archive.validate_location(location, self.payload)
        for key, value in [('checkpoint_directory', '20261101_000000'), ('declared_last_path', '/other/20261111_180638/last.pth'), ('checkpoint_loaded', True), ('remote_mutations', True), ('log', '/other/run.log')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.validate_location(dict(location, **{key: value}), self.payload)

    def test_other_timestamp_cannot_substitute_original_checkpoints(self):
        self.payload['checkpoint_directory'] = '20261101_000000'
        for row in self.proof['checkpoints']:
            row['path'] = row['path'].replace('20261111_180638', '20261101_000000')
        with self.assertRaises(AssertionError):
            self.verify()

    def test_terminal_source_identity_precedes_checkpoint_load(self):
        self.assertIn(archive.IDENTITY, archive.REMOTE)
        self.assertLess(archive.REMOTE.index(archive.IDENTITY), archive.REMOTE.index('import torch'))
        self.assertIn('assert all(value==list(range(1,31))', archive.REMOTE)

class CurrentBinding(unittest.TestCase):

    def test_current_original_task_and_actual_declared_location(self):
        from y99_terminal_contract import SPEC, Y99, build_contract
        server, queue, task, work, candidate = archive.current()
        self.assertEqual(build_contract(task, server)['allowed_markers'], [Y99])
        self.assertEqual(task['pid'], 2980836)
        self.assertEqual(candidate['stage'], 'training')
        self.assertEqual(SPEC['log'], task['log'])
        self.assertTrue(SPEC['checkpoint_directory'])
        self.assertEqual(SPEC['admission_sha256'], '4d87ad8b3a6efe78be8dc9e8214bba9226160a22b649f64f8e1dd26d73988ede')
        self.assertEqual(SPEC['freeze_sha256'], 'fd07c0536891d905a7cd89e2b4211b32ed31a532b7b70ac9d71a8d7a66e97dbc')

    def test_actual_GPU1_pause_and_goal_unchanged(self):
        from prepare_y60_stage_heads import ROOT, ref, monitor
        servers = monitor.load_servers(str(ROOT))
        self.assertEqual(servers['server-1']['allowed_gpus'], [0])
        self.assertEqual(servers['server-2']['allowed_gpus'], [0])
        self.assertEqual(servers['server-6']['allowed_gpus'], [])
        self.assertEqual(ref('pipeline/goal.json')['sha256'], '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a')
        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'], '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')

    def test_old_marker_and_wrong_candidate_refused(self):
        from y99_terminal_contract import Y99, validate_marker
        for marker, cid in [('59baf94e9d3e406afa13b2ddc96968cdcc976cae', 'AdaRES-Y99'), (Y99, 'AdaRES-Y91'), ('a' * 40, 'AdaRES-Y99')]:
            with self.subTest(marker=marker, cid=cid), self.assertRaises(AssertionError):
                validate_marker(marker, cid)

    def test_actual_runtime_mapping_and_owner_cannot_change(self):
        from y99_terminal_contract import build_contract
        server, queue, task, work, candidate = archive.current()
        for key, value in [('pid', 1305468), ('server', 'server-2'), ('gpu', '1')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                build_contract(dict(task, **{key: value}), server)
        for key, value in [('python', '/usr/bin/python'), ('expected_hostname', 'foreign'), ('allowed_gpus', [0, 1]), ('runtime_path_mappings', []), ('runtime_library_path', '/wrong/cuda')]:
            with self.subTest(key=key), self.assertRaises((AssertionError, ValueError)):
                build_contract(task, dict(server, **{key: value}))

    def test_actual_native_and_full_parent_protocol_retained(self):
        import copy
        from prepare_y60_stage_heads import ROOT, tick
        from advance_y99_v2 import prepared
        from y99_contract import validate_result
        import closure
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate, _, _, frozen = prepared(work)
        raw = tick.load(closure.artifact(ROOT, candidate['native_smoke_result']))
        validate_result(raw, candidate, frozen)
        current = tick.load(closure.artifact(ROOT, candidate['post_training_freeze']))
        parent_candidate = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y96'))
        parent = tick.load(closure.artifact(ROOT, parent_candidate['post_training_freeze']))
        restored = copy.deepcopy(current['evaluation'])
        restored['methods'][0] = 'AdaRES-Y96'
        self.assertEqual(restored, parent['evaluation'])
        self.assertEqual(current['training'], parent['training'])
if __name__ == '__main__':
    unittest.main()
