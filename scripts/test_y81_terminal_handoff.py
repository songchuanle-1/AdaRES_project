"""CPU lifecycle rejection tests; fixtures do not prove actual completion."""
import copy
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import finalize_y81_fixed30 as archive


class TerminalHandoff(unittest.TestCase):
    def setUp(self):
        self.task = dict(id='AdaRES-Y81', status='running', launch_verified=True, gpu='1',
            server='server-2', pid=120624,
            config='configs_song/model_paper5/adares_y81_query_mask_refinement_res.yaml',
            code_commit='4157937d0c3cf344a44394ce15c95349aa94d50f')
        self.candidate = dict(id=self.task['id'], stage='training', config=self.task['config'], code_commit=self.task['code_commit'])
        self.paused = dict(id='AdaRES-Y57', status='paused', dispatch_hold='user pause', resume_requires_user_authorization=True)
        self.payload = dict(id='AdaRES-Y81', work='/project/checkpoints', basename='candidate',
            checkpoint_directory='20261102_220631', log='/project/code/logs/run.log')
        self.proof = dict(complete=True, id='AdaRES-Y81', process_absent=True, CUDA_initialized=False,
            log=self.payload['log'], log_sha256='b'*64, full_frozen_evaluation_executed=False,
            completed_epochs={p:list(range(1,31)) for p in ('Train','Val')},
            checkpoints=[dict(path='/project/checkpoints/candidate/20261102_220631/'+name,
                epoch=epoch, scheduler_last_epoch=epoch, optimizer_state_count=544,
                best_miou=.5, model_finite=True, optimizer_finite=True, sha256='c'*64)
                for name,epoch in [('last.pth',30),('best.pth',22)]])
        self.resource = dict(complete=True, process_absent=True, same_config_process_absent=True,
            hostname='dl01-MS-7C60', GPU1_use_authorized=False, GPU_queries_executed=False, remote_mutations=False)

    def verify(self): archive.validate_proof(self.proof,self.payload)

    def test_valid_schema_only(self):
        archive.validate_task(self.task,self.candidate,self.paused);self.verify()

    def test_wrong_identity_gpu_server_pid_status_refused(self):
        for key,value in [('id','foreign'),('gpu','0'),('server','server-1'),('pid',1),('status','paused')]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                archive.validate_task(dict(self.task,**{key:value}),self.candidate,self.paused)

    def test_changed_y57_pause_refused(self):
        for key,value in [('status','running'),('dispatch_hold',False),('resume_requires_user_authorization',False)]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                archive.validate_task(self.task,self.candidate,dict(self.paused,**{key:value}))

    def test_incomplete_last_refused(self):
        self.proof['checkpoints'][0].update(epoch=29,scheduler_last_epoch=29)
        with self.assertRaises(AssertionError):self.verify()

    def test_scheduler_reset_refused(self):
        self.proof['checkpoints'][0]['scheduler_last_epoch']=0
        with self.assertRaises(AssertionError):self.verify()

    def test_empty_optimizer_refused(self):
        self.proof['checkpoints'][0]['optimizer_state_count']=0
        with self.assertRaises(AssertionError):self.verify()

    def test_wrong_checkpoint_lineage_refused(self):
        self.proof['checkpoints'][0]['path']='/other/last.pth'
        with self.assertRaises(AssertionError):self.verify()

    def test_nonfinite_model_optimizer_metric_refused(self):
        for key,value in [('model_finite',False),('optimizer_finite',False),('best_miou',float('nan'))]:
            previous=copy.deepcopy(self.proof);self.proof['checkpoints'][0][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):self.verify()
            self.proof=previous

    def test_live_process_refused(self):
        self.proof['process_absent']=False
        with self.assertRaises(AssertionError):self.verify()

    def test_only_gpu1_mapping_removed_without_aliasing(self):
        before={'server-2':{'0':'AdaRES-Y83','1':'AdaRES-Y81'},'server-1':{'0':'AdaRES-Y82'}}
        original=copy.deepcopy(before)
        self.assertEqual(archive.expected_mapping(before),{'server-2':{'0':'AdaRES-Y83'},'server-1':{'0':'AdaRES-Y82'}})
        self.assertEqual(before,original)

    def test_new_or_duplicate_mapping_owner_refused(self):
        for before in ({'server-2':{'1':'foreign'}},{'server-2':{'1':'AdaRES-Y81'},'server-1':{'0':'AdaRES-Y81'}}):
            with self.assertRaises(AssertionError):archive.expected_mapping(before)

    def test_empty_server_removed_other_server_preserved(self):
        self.assertEqual(archive.expected_mapping({'server-2':{'1':'AdaRES-Y81'},'server-1':{}}),{'server-1':{}})

    def test_changed_source_or_config_refused(self):
        for key,value in [('code_commit','a'*40),('config','other.yaml')]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                archive.validate_task(dict(self.task,**{key:value}),dict(self.candidate,**{key:value}),self.paused)

    def test_missing_native_epoch_refused(self):
        self.proof['completed_epochs']['Train'].remove(13)
        with self.assertRaises(AssertionError):self.verify()

    def test_best_metric_disagreement_refused(self):
        self.proof['checkpoints'][0]['best_miou']=.6
        with self.assertRaises(AssertionError):self.verify()

    def test_unknown_resource_or_unpaused_gpu_refused(self):
        for key,value in [('hostname','other'),('GPU1_use_authorized',True),('GPU_queries_executed',True),('remote_mutations',True),('process_absent',False),('same_config_process_absent',False)]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                archive.validate_resource(dict(self.resource,**{key:value}))

    def test_pause_never_returns_resource_permission(self):
        self.assertFalse(archive.validate_resource(self.resource))

    def test_live_pid_refused_before_process_or_checkpoint_queries(self):
        with patch('subprocess.check_output',return_value=b'dl01-MS-7C60') as query,patch('pathlib.Path.exists',return_value=True):
            with self.assertRaises(AssertionError):exec(archive.RESOURCE,{'P':{'pid':120624,'config':'test'}})
            self.assertEqual(query.call_count,1)

    def test_exact_additive_markers_only(self):
        from y81_terminal_contract import Y81,Y82,Y83,validate_marker
        for marker in (Y81,Y82,Y83):validate_marker(marker,'AdaRES-Y81')
        with self.assertRaises(AssertionError):validate_marker('a'*40,'AdaRES-Y81')

    def test_additive_edges_reject_changes_and_foreign_paths(self):
        from y81_terminal_contract import validate_delta,paths
        for cid in ('AdaRES-Y82','AdaRES-Y83'):
            rows=[['A',p] for p in paths(cid)];validate_delta(rows,cid)
            for bad in (rows+[['A','foreign.py']],[['M',rows[0][1]]]+rows[1:]):
                with self.assertRaises(AssertionError):validate_delta(bad,cid)

    def test_archived_identity_preserves_original_gpu1(self):
        result=dict(id=self.task['id'],status='done_complete',gpu='1')
        archive.validate_archived_experiment(result,self.task)
        for key,value in [('gpu','0'),('id','foreign'),('status','running')]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                archive.validate_archived_experiment(dict(result,**{key:value}),self.task)

    def test_evaluation_resource_cannot_move_to_gpu0(self):
        self.assertEqual(archive.evaluation_resource(self.task),dict(required_server='server-2',required_gpu=1))
        for key,value in [('gpu','0'),('server','server-1'),('id','foreign')]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                archive.evaluation_resource(dict(self.task,**{key:value}))

    def test_original_location_required(self):
        value=dict(complete=True,checkpoint_loaded=False,remote_mutations=False,log=self.payload['log'],
            checkpoint_directory='20261102_220631',declared_last_path='/project/checkpoints/candidate/20261102_220631/last.pth')
        archive.validate_location(value,self.payload)
        for key,bad in [('checkpoint_directory','20261101_000000'),('declared_last_path','/other/last.pth'),('checkpoint_loaded',True),('remote_mutations',True),('log','/other/run.log')]:
            with self.subTest(key=key),self.assertRaises(AssertionError):archive.validate_location(dict(value,**{key:bad}),self.payload)

    def test_wrong_timestamp_cannot_substitute_checkpoints(self):
        self.payload['checkpoint_directory']='20261101_000000'
        with self.assertRaises(AssertionError):self.verify()

    def test_cpu_verification_keeps_identity_before_checkpoint_load(self):
        self.assertIn(archive.IDENTITY,archive.REMOTE)
        self.assertLess(archive.REMOTE.index(archive.IDENTITY),archive.REMOTE.index('import torch'))
        self.assertIn("assert all(value==list(range(1,31))",archive.REMOTE)
        self.assertNotIn('nvidia-smi',archive.RESOURCE)
        with patch('subprocess.check_output',side_effect=[b'dl01-MS-7C60',b'']) as query,patch('pathlib.Path.exists',return_value=False),redirect_stdout(io.StringIO()):
            exec(archive.RESOURCE,{'P':{'pid':120624,'config':'test'}})
            self.assertEqual(query.call_count,2)


if __name__=='__main__':unittest.main()
