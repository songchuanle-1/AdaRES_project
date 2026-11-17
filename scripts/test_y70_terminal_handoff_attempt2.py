"""CPU lifecycle guards; fixtures are not actual completion evidence."""
import copy
import unittest
import finalize_y70_fixed30_attempt2 as archive


class TerminalHandoff(unittest.TestCase):
    def setUp(self):
        self.task = dict(id='AdaRES-Y70', status='running', launch_verified=True, gpu='1',
            server='server-2', pid=52561, config='configs_song/model_paper5/adares_y70_query_scale_fusion_res.yaml', code_commit='39bdb8d9783512e70cdc2763e9a6097ab0f67a0f')
        self.candidate = dict(id='AdaRES-Y70', stage='training', config='configs_song/model_paper5/adares_y70_query_scale_fusion_res.yaml', code_commit='39bdb8d9783512e70cdc2763e9a6097ab0f67a0f')
        self.paused = dict(id='AdaRES-Y57', status='paused', dispatch_hold='user pause', resume_requires_user_authorization=True)
        self.payload = dict(id='AdaRES-Y70', work='/project/checkpoints', basename='candidate',
            checkpoint_directory='20261027_005030', log='/project/code/logs/run.log')
        self.proof = dict(complete=True, id='AdaRES-Y70', process_absent=True, CUDA_initialized=False,
            log=self.payload['log'], log_sha256='b' * 64, full_frozen_evaluation_executed=False,
            completed_epochs={p:list(range(1,31)) for p in ('Train','Val')},
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
        before = {'server-2': {'1': 'AdaRES-Y70', '0': 'untouched'}, 'server-1': {'0': 'untouched'}}
        original = copy.deepcopy(before)
        after = archive.expected_mapping(before)
        self.assertEqual(after, {'server-2': {'0': 'untouched'}, 'server-1': {'0': 'untouched'}})
        self.assertEqual(before, original)

    def test_new_mapping_owner_or_duplicate_refused(self):
        for value in ({'server-2': {'1': 'foreign'}},
                      {'server-2': {'1': 'AdaRES-Y70'}, 'server-1': {'0': 'AdaRES-Y70'}}):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                archive.expected_mapping(value)

    def test_empty_server_removed_other_server_preserved(self):
        self.assertEqual(archive.expected_mapping({'server-2': {'1': 'AdaRES-Y70'}, 'server-1': {}}),
                         {'server-1': {}})


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
        base=dict(complete=True,process_absent=True,hostname='dl01-MS-7C60',gpu='0, 0, 390',compute='')
        for field,value in [('gpu','nan, 0, 390'),('gpu','0, 0, 280'),('hostname','other'),('complete',False)]:
            with self.subTest(field=field,value=value), self.assertRaises(AssertionError):
                archive.validate_resource(dict(base,**{field:value}))

    def test_foreign_compute_preserved_as_busy(self):
        base=dict(complete=True,process_absent=True,hostname='dl01-MS-7C60',gpu='0, 0, 390',compute='')
        self.assertTrue(archive.validate_resource(base))
        self.assertFalse(archive.validate_resource(dict(base,compute='999')))
        self.assertFalse(archive.validate_resource(dict(base,gpu='2300, 60, 390')))


    def test_remote_live_pid_refused_before_gpu_or_checkpoint_reads(self):
        from unittest.mock import patch
        with patch('subprocess.check_output',return_value=b'dl01-MS-7C60') as query, patch('pathlib.Path.exists',return_value=True):
            with self.assertRaises(AssertionError): exec(archive.RESOURCE,{'P':{'pid':52561}})
            self.assertEqual(query.call_args_list[0].args[0],['hostname'])
            self.assertEqual(query.call_count,1)


    def test_production_marker_only_exact_two_versions(self):
        from y70_terminal_contract import validate_marker, OLD, NEW
        validate_marker(OLD,'AdaRES-Y70');validate_marker(NEW,'AdaRES-Y70')
        with self.assertRaises(AssertionError):validate_marker('a'*40,'AdaRES-Y70')

    def test_deployment_refuses_unrelated_or_modified_file(self):
        from y70_terminal_contract import validate_additive_delta,NAMES
        package='adares_y71_visual_product_fusion'
        rows=[['A','AdaRES/ipdn/model_paper5/'+package+'/'+n] for n in NAMES]
        rows.append(['A','AdaRES/configs_song/model_paper5/'+package+'_res.yaml'])
        validate_additive_delta(rows)
        for changed in (rows+[['A','other.py']],[['M',rows[0][1]]]+rows[1:]):
            with self.assertRaises(AssertionError):validate_additive_delta(changed)


if __name__ == '__main__':
    unittest.main()
