"""Reject missing or altered native Y83 evidence before source selection/admission."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick
import advance_y83 as advance
import y83_deployment as deploy


class Admission(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT / 'pipeline/research_work.json')
        self.c = copy.deepcopy(next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y83'))
        plan = tick.load(ROOT / self.c['native_smoke_plan_server2_gpu0']['path'])
        self.f = ast.literal_eval(ast.parse((ROOT / plan['program']['path']).read_text(encoding='utf-8')).body[0].value)
        self.raw = tick.load(ROOT / self.c['native_smoke_result']['path'])

    def check(self, raw=None):
        advance.validate_result(self.raw if raw is None else raw, self.c, self.f)

    def test_actual_recorded_native_result(self):
        self.check()

    def test_wrong_identity_version_config_goal_or_yaml(self):
        for key, value in [('id', 'AdaRES-Y77'), ('code_commit', '0'*40), ('config', 'other.yaml'), ('goal_sha256', '0'*64), ('native_YAML_sha256', '0'*64)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_missing_or_nonfinite_losses(self):
        for value in ([float('nan'), 1.], [1., float('inf')], [1.]):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, losses=value))

    def test_synthetic_validation_or_missing_native_execution(self):
        for key, value in [('split', 'val'), ('native_data_loaded', False), ('backward_finite', False), ('intended_path_observed', False), ('native_modes_verified', False), ('validation_used', True), ('model_weights_saved', True)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: value}))

    def test_all_native_steps_and_consumers_required(self):
        for key in ('steps','optimizer_steps','shared_head_calls','query_norm_consumption_calls','private_norm_calls','ffn_consumption_calls','original_coarse_gate_consumption_calls'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: 0}))

    def test_original_hard_gate_initial_prediction_and_static_mask_required(self):
        for key in ('static_mask_features_preserved', 'original_hard_gate_algorithm_preserved', 'original_initial_prediction_preserved'):
            with self.assertRaises(AssertionError):
                self.check(dict(self.raw, **{key: False}))

    def test_original_fusion_input_identity_private_norm_epsilon_reference_required(self):
        for key in ('original_self_identity','original_language_identity','sum_input_exact','both_inputs_undetached','private_module_identity','original_epsilon','float64_LayerNorm_reference'):
            raw=copy.deepcopy(self.raw);raw['norm_paths'][7][key]=False
            with self.subTest(key=key),self.assertRaises(AssertionError):self.check(raw)

    def test_disjoint_norm_parameters_and_original_epsilon_required(self):
        for key in ('norm_parameters_disjoint','original_epsilon_preserved'):
            with self.assertRaises(AssertionError):self.check(dict(self.raw,**{key:False}))
        for raw in (dict(self.raw,norm_paths=self.raw['norm_paths'][:6]),dict(self.raw,norm_paths=list(reversed(self.raw['norm_paths'])))):
            with self.assertRaises(AssertionError):self.check(raw)

    def test_all_twelve_private_gradients_and_updates_required(self):
        raw=copy.deepcopy(self.raw);raw['gradient_checks'][1]['both_input_gradients_finite']=False
        with self.assertRaises(AssertionError):self.check(raw)
        for key in ('norm_gradient_maxima','norm_parameter_updates'):
            for value in (0.,float('nan'),float('inf')):
                raw=copy.deepcopy(self.raw);raw['gradient_checks'][1][key][5]=value
                with self.assertRaises(AssertionError):self.check(raw)
            raw=copy.deepcopy(self.raw);raw['gradient_checks'][1][key].pop()
            with self.assertRaises(AssertionError):self.check(raw)

    def test_per_step_original_consumers_required(self):
        for key in ('indicator_tensors','indicator_calls','private_norm_calls','ffn_consumption_calls','original_coarse_gate_consumption_calls','query_norm_consumption_calls','norm_gradient_tensors'):
            raw = copy.deepcopy(self.raw)
            raw['gradient_checks'][1][key] = 0
            with self.assertRaises(AssertionError):
                self.check(raw)

    def test_original_dataset_and_four_scenes_required(self):
        raw = copy.deepcopy(self.raw)
        raw['data_source']['config_data_root'] = '/wrong/data'
        with self.assertRaises(AssertionError):
            self.check(raw)
        raw = copy.deepcopy(self.raw)
        raw['batches'][1] = copy.deepcopy(raw['batches'][0])
        with self.assertRaises(AssertionError):
            self.check(raw)

    def test_unknown_foreign_or_changed_power_refused(self):
        advance.idle(dict(gpu='0,1,280',compute=''),280.)
        for gpu,compute in [('nan,0,280',''),('0,0,390',''),('0,0,350',''),('0,2000,280',''),('0,0,280','99')]:
            with self.assertRaises(AssertionError):advance.idle(dict(gpu=gpu,compute=compute),280.)

    def test_slot_ownership_and_queued_work_refused(self):
        pause=dict(id='AdaRES-Y57',status='paused',gpu=1,dispatch_hold=True,resume_requires_user_authorization=True)
        queue=dict(gpu_queues={'server-2':[pause,dict(id='AdaRES-Y81',status='running',gpu=1)],'server-1':[]})
        advance.unreserved(queue,{'server-2':{'1':'AdaRES-Y81'}},'server-2',0)
        for status in ('waiting','claiming','running'):
            value=copy.deepcopy(queue);value['gpu_queues']['server-2'].append(dict(id='foreign',status=status,gpu=0))
            with self.assertRaises(AssertionError):advance.unreserved(value,{},'server-2',0)
        with self.assertRaises(AssertionError):advance.unreserved(queue,{'server-2':{'0':'foreign'}},'server-2',0)

    def test_y57_pause_required(self):
        pause = dict(id='AdaRES-Y57', status='running', dispatch_hold=True, resume_requires_user_authorization=True)
        with self.assertRaises(AssertionError):
            advance.unreserved(dict(gpu_queues={'server-2': [pause]}), {}, 'server-2', 1)

    def test_exact_chain_preserves_both_original_running_models(self):
        for cid,commit in [('AdaRES-Y81',deploy.Y81),('AdaRES-Y82',deploy.Y82)]:
            self.assertIn(deploy.Y83,deploy.contract(commit,deploy.Y83,cid))

    def test_unknown_or_reversed_chain_refused(self):
        for commit, head in [(deploy.Y77, '0'*40), ('0'*40, deploy.Y83), (deploy.Y83, deploy.Y77)]:
            with self.assertRaises(AssertionError):
                deploy.contract(commit, head, 'AdaRES-Y77')

    def test_only_seven_exact_additions_allowed(self):
        rows = [['A', path] for path in deploy.paths('AdaRES-Y83')]
        deploy.validate_delta(rows, 'AdaRES-Y83')
        for bad in (rows + [['A', 'other.py']], [['M', rows[0][1]]] + rows[1:]):
            with self.assertRaises(AssertionError):
                deploy.validate_delta(bad, 'AdaRES-Y83')


if __name__ == '__main__':
    unittest.main()
