"""Validate frozen runtime alternatives without running GPU tests."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick, monitor
from check_ef_execution import validate_frozen
from prepare_y69_server1_alternative import ADAPT
from run_y90_native import runtime, validate_terminal


class RuntimePlan(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT/'pipeline/research_work.json')
        self.c = next(candidate for candidate in work['candidates'] if candidate['id']=='AdaRES-Y90')
        self.servers = monitor.load_servers(str(ROOT))
        self.values = {}
        for name,key in (('server-1','native_smoke_plan_server1'),('server-2','native_smoke_plan_server2_gpu0')):
            plan = tick.load(ROOT/self.c[key]['path'])
            source = (ROOT/plan['program']['path']).read_text(encoding='utf-8')
            self.values[name] = plan,ast.literal_eval(ast.parse(source).body[0].value),source.partition('\n')[2]

    def test_seven_tested_files_for_both_frozen_plans(self):
        for name,(plan,frozen,_) in self.values.items():
            self.assertEqual(validate_frozen(self.c,plan,frozen),7)
            runtime(name,self.c,frozen,self.servers[name])

    def test_scientific_payload_identical(self):
        a,b = [value[1] for value in self.values.values()]
        for key in ('variant','code_commit','goal_sha256','train_source_lf_sha256'):
            self.assertEqual(a[key],b[key])

    def test_only_configured_path_adaptation_and_hostname(self):
        first=self.values['server-1'][2]
        second=self.values['server-2'][2]
        self.assertEqual(first.replace(ADAPT,'').replace("== 'dl02-MS-7B73'","== 'dl01-MS-7C60'"),second)

    def test_GPU1_changed_power_and_wrong_host_refused(self):
        for name,(_,frozen,_) in self.values.items():
            for key,value in (('runtime_physical_gpu',1),('runtime_power_limit_watts',390.),('runtime_server','server-6')):
                with self.subTest(name=name,key=key),self.assertRaises(AssertionError):
                    runtime(name,self.c,dict(frozen,**{key:value}),self.servers[name])
            with self.assertRaises(AssertionError):
                runtime(name,self.c,frozen,dict(self.servers[name],expected_hostname='foreign'))

    def test_stale_goal_and_tampered_bytes_refused(self):
        for plan,frozen,_ in self.values.values():
            for field in ('goal','source'):
                value=copy.deepcopy(frozen)
                if field=='goal':value['goal_sha256']='0'*64
                else:value['variant']['python_files']['dec.py']['base64']='Y2hhbmdlZA=='
                with self.assertRaises(AssertionError):
                    validate_frozen(self.c,plan,value)

    def test_live_or_incomplete_owner_never_unblocks(self):
        proof=dict(id='AdaRES-Y88',complete=True,process_absent=True,CUDA_initialized=False,
            completed_epochs={phase:list(range(1,31)) for phase in ('Train','Val')},
            checkpoints=[dict(epoch=epoch,scheduler_last_epoch=epoch,optimizer_state_count=1,
                model_finite=True,optimizer_finite=True) for epoch in (30,21)])
        validate_terminal(proof,'server-1')
        for key,value in (('process_absent',False),('CUDA_initialized',True),('id','foreign')):
            with self.assertRaises(AssertionError):
                validate_terminal(dict(proof,**{key:value}),'server-1')
        proof['checkpoints'][0]['epoch']=29
        with self.assertRaises(AssertionError):
            validate_terminal(proof,'server-1')


if __name__=='__main__':
    unittest.main()
