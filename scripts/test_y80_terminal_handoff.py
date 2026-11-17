"""CPU lifecycle guards; fixtures are not actual completion evidence."""
import copy
import unittest
import finalize_y80_fixed30 as archive


class TerminalHandoff(unittest.TestCase):
    def setUp(self):
        self.task = dict(id='AdaRES-Y80', status='running', launch_verified=True, gpu='0',
            server='server-2', pid=42226, config='configs_song/model_paper5/adares_y80_query_visual_feedback_res.yaml', code_commit='4e68cd84d56d890222adc2bec8ab59cb0c8715b6')
        self.candidate = dict(id='AdaRES-Y80', stage='training', config='configs_song/model_paper5/adares_y80_query_visual_feedback_res.yaml', code_commit='4e68cd84d56d890222adc2bec8ab59cb0c8715b6')
        self.paused = dict(id='AdaRES-Y57', status='paused', dispatch_hold='user pause', resume_requires_user_authorization=True)
        self.payload = dict(id='AdaRES-Y80', work='/project/checkpoints', basename='candidate',
            checkpoint_directory='20261102_175917', log='/project/code/logs/run.log')
        self.proof = dict(complete=True, id='AdaRES-Y80', process_absent=True, CUDA_initialized=False,
            log=self.payload['log'], log_sha256='b' * 64, full_frozen_evaluation_executed=False,
            completed_epochs={p:list(range(1,31)) for p in ('Train','Val')},
            checkpoints=[dict(path='/project/checkpoints/candidate/20261102_175917/' + name,
                epoch=epoch, scheduler_last_epoch=epoch, optimizer_state_count=544, best_miou=.5,
                model_finite=True, optimizer_finite=True, sha256='c' * 64)
                for name, epoch in [('last.pth', 30), ('best.pth', 21)]])

    def verify(self):
        archive.validate_proof(self.proof, self.payload)

    def test_valid_schema_only(self):
        archive.validate_task(self.task, self.candidate, self.paused)
        self.verify()

    def test_wrong_id_gpu_server_pid_or_status_refused(self):
        for key, value in [('id', 'AdaRES-Y62'), ('gpu', '1'), ('server', 'server-1'),
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
        before={'server-2':{'0':'AdaRES-Y80','1':'AdaRES-Y81'},'server-1':{'0':'AdaRES-Y82'}}
        original=copy.deepcopy(before);after=archive.expected_mapping(before)
        self.assertEqual(after,{'server-2':{'1':'AdaRES-Y81'},'server-1':{'0':'AdaRES-Y82'}})
        self.assertEqual(before,original)

    def test_new_mapping_owner_or_duplicate_refused(self):
        for before in ({'server-2':{'0':'foreign'}},{'server-2':{'0':'AdaRES-Y80'},'server-1':{'0':'AdaRES-Y80'}}):
            with self.assertRaises(AssertionError):archive.expected_mapping(before)

    def test_empty_server_removed_other_server_preserved(self):
        self.assertEqual(archive.expected_mapping({'server-2':{'0':'AdaRES-Y80'},'server-1':{}}),{'server-1':{}})


    def test_changed_code_or_config_refused(self):
        for key,value in [('code_commit','a'*40),('config','other.yaml')]:
            t=dict(self.task,**{key:value});c=dict(self.candidate,**{key:value})
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.validate_task(t,c,self.paused)

    def test_missing_native_epoch_refused(self):
        self.proof['completed_epochs']['Train'].remove(13)
        with self.assertRaises(AssertionError): self.verify()

    def test_mismatched_best_metric_refused(self):
        self.proof['checkpoints'][0]['best_miou']=.6
        with self.assertRaises(AssertionError): self.verify()

    def test_unknown_or_wrong_resource_refused(self):
        base=dict(complete=True,process_absent=True,hostname='dl01-MS-7C60',gpu='0, 0, 280',compute='')
        for field,value in [('gpu','nan, 0, 350'),('gpu','0, 0, 390'),('hostname','other'),('complete',False)]:
            with self.subTest(field=field,value=value), self.assertRaises(AssertionError):
                archive.validate_resource(dict(base,**{field:value}))

    def test_foreign_compute_preserved_as_busy(self):
        base=dict(complete=True,process_absent=True,hostname='dl01-MS-7C60',gpu='0, 0, 280',compute='')
        self.assertTrue(archive.validate_resource(base))
        self.assertFalse(archive.validate_resource(dict(base,compute='999')))
        self.assertFalse(archive.validate_resource(dict(base,gpu='2300, 60, 280')))


    def test_remote_live_pid_refused_before_gpu_or_checkpoint_reads(self):
        from unittest.mock import patch
        with patch('subprocess.check_output',return_value=b'dl01-MS-7C60') as query, patch('pathlib.Path.exists',return_value=True):
            with self.assertRaises(AssertionError): exec(archive.RESOURCE,{'P':{'pid':42226}})
            self.assertEqual(query.call_args_list[0].args[0],['hostname'])
            self.assertEqual(query.call_count,1)


    def test_exact_registered_production_markers_only(self):
        from y80_terminal_contract import Y80,Y81,Y82,validate_marker
        for value in (Y80,Y81,Y82):validate_marker(value,'AdaRES-Y80')
        with self.assertRaises(AssertionError):validate_marker('a'*40,'AdaRES-Y80')

    def test_additive_edges_reject_foreign_or_modified_paths(self):
        from y80_terminal_contract import validate_delta,paths
        for cid in ('AdaRES-Y81','AdaRES-Y82'):
            rows=[['A',path] for path in paths(cid)];validate_delta(rows,cid)
            for bad in (rows+[['A','other.py']],[['M',rows[0][1]]]+rows[1:]):
                with self.assertRaises(AssertionError):validate_delta(bad,cid)

    def test_archived_resource_follows_original_server2_gpu0(self):
        result = dict(id=self.task['id'], status='done_complete', gpu='0')
        archive.validate_archived_experiment(result, self.task)
        for key, value in [('gpu', '1'), ('id', 'foreign'), ('status', 'running')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.validate_archived_experiment(dict(result, **{key: value}), self.task)

    def test_evaluation_resource_follows_original_server2_gpu0(self):
        self.assertEqual(archive.evaluation_resource(self.task), dict(required_server='server-2', required_gpu=0))
        for key, value in [('gpu', '1'), ('server', 'server-1'), ('id', 'foreign')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.evaluation_resource(dict(self.task, **{key: value}))

    def test_declared_original_location_required(self):
        location = dict(complete=True, checkpoint_loaded=False, remote_mutations=False,
            log=self.payload['log'], checkpoint_directory='20261102_175917',
            declared_last_path='/project/checkpoints/candidate/20261102_175917/last.pth')
        archive.validate_location(location, self.payload)
        for key, value in [('checkpoint_directory', '20261101_000000'),
                ('declared_last_path', '/other/20261102_175917/last.pth'),
                ('checkpoint_loaded', True), ('remote_mutations', True), ('log', '/other/run.log')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.validate_location(dict(location, **{key: value}), self.payload)

    def test_other_timestamp_cannot_substitute_original_checkpoints(self):
        self.payload['checkpoint_directory'] = '20261101_000000'
        for row in self.proof['checkpoints']:
            row['path'] = row['path'].replace('20261102_175917', '20261101_000000')
        with self.assertRaises(AssertionError): self.verify()

    def test_terminal_source_identity_precedes_checkpoint_load(self):
        self.assertIn(archive.IDENTITY, archive.REMOTE)
        self.assertLess(archive.REMOTE.index(archive.IDENTITY), archive.REMOTE.index('import torch'))
        self.assertIn("assert all(value==list(range(1,31))", archive.REMOTE)


if __name__ == '__main__':
    unittest.main()
