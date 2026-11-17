"""Guard actual geometry-mask CPU evidence, its source, and pending native body."""
import ast,copy,unittest
from prepare_y60_stage_heads import ROOT,tick,ref
from prepare_y102_geometry_mask_skip import files,EDITS,PARENT,NAMES
from y102_contract import validate_cpu,validate_result

class Preparation(unittest.TestCase):
    def setUp(self):
        self.token=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')['token']
        self.cpu=tick.load(ROOT/('pipeline/analyses/y102_cpu_'+self.token+'.json'))
        self.body=(ROOT/('pipeline/checks/y102_native_'+self.token+'_body.py')).read_text(encoding='utf-8')
    def test_actual_CPU(self):validate_cpu(self.cpu)
    def test_identity_and_counts(self):
        for k,x in (('id','AdaRES-Y77'),('parameters',1),('additional_parameters',0)):
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(dict(self.cpu,**{k:x}))
    def test_independent_geometry_and_no_RNG_controls(self):
        for k in ('no_RNG','independent_scalar_all_scene_centroid_projection','coordinate_translation_and_positive_mass_scale_invariant','padded_NaN_positions_and_masses_ignored','scene_isolation_and_coordinate_mass_interventions','padded_coordinate_and_mass_gradients_zero'):
            v=copy.deepcopy(self.cpu);v['component_controls'][0][k]=False
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(v)
    def test_anchors_bypass_and_consumers(self):
        for k in ('decoder_all_stages_anchor_exact','decoder_input_gradients_anchor_close','exact_bypass_restoration','original_geometry_mass_and_static_mask_consumers_verified','all_seven_static_mask_consumers_verified'):
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(dict(self.cpu,**{k:False}))
    def test_all_original_gradients_and_updates(self):
        for k,x in (('parameter_update_max',0.),('new_gradient_max',float('nan')),('original_visual_parameter_tensors',0),('original_base_adapted_and_visual_gradients_finite',False),('frozen_text_encoder_preserved',False),('geometry_mask_calls',0),('decoder_input_gradients_finite',False)):
            v=copy.deepcopy(self.cpu);v['CPU_decoder_steps'][1][k]=x
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(v)
    def test_original_geometry_mass_padding_and_mask_reuse(self):
        for k,x in (('step',0),('weight_nonzero',False),('mask_feature_delta_max',0.),('original_full3_coordinates_and_point_masses',False),('original_unbatched_mask_and_batching',False),('original_padding_rows_preserved',False),('all_seven_predictions_reuse_same_features',False)):
            v=copy.deepcopy(self.cpu);v['CPU_geometry_mask_paths'][1][k]=x
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(v)
    def test_CPU_cannot_impersonate_native(self):
        for k in ('CUDA_initialized','native_smoke_passed','native_task_backward','native_sparse_forward','validation_used','admission'):
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(dict(self.cpu,**{k:True}))
        with self.assertRaises((AssertionError,KeyError)):validate_result(self.cpu,{'id':'AdaRES-Y102'},{'variant':{'id':'AdaRES-Y102'}})
    def test_exact_independent_W2_edits(self):
        parent=ROOT/'AdaRES/ipdn/model_paper5'/PARENT;s=(parent/'dec.py').read_text(encoding='utf-8')
        for a,b in EDITS:self.assertEqual(s.count(a),1);s=s.replace(a,b)
        self.assertEqual(s,(files()[0].parent/'dec.py').read_text(encoding='utf-8'))
        for n in NAMES:
            if n not in ('__init__.py','dec.py'):self.assertEqual((parent/n).read_bytes(),(files()[0].parent/n).read_bytes())
        self.assertEqual(files()[-1].read_bytes(),(ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    def test_same_CPU_callbacks_in_native(self):
        core=(ROOT/'scripts/y102_observer_core.py').read_text(encoding='utf-8');cpu=(ROOT/('pipeline/checks/y102_cpu_'+self.token+'_program.py')).read_text(encoding='utf-8')
        self.assertEqual(cpu.count(core),1);self.assertEqual(self.body.count(core),1);ast.parse(self.body,feature_version=(3,8))
        for t in ('native.train(1, model, FirstTwoBatches()','optimizer.step = checked_step',"state['raw_coordinates_expected'] = args[5]",'geometry_mask_calls=2'):self.assertIn(t,self.body)
        self.assertIn("assert torch.equal(parts(changed)[0]['scores'], parts(baseline)[0]['scores'])",cpu)
    def test_original_runs_and_earlier_holds(self):
        c={r['id']:r for r in tick.load(ROOT/'pipeline/research_work.json')['candidates']}
        self.assertFalse(c['AdaRES-Y102'].get('native_smoke_result'));self.assertFalse(c['AdaRES-Y102'].get('admission_receipt'))
        for cid in ('AdaRES-Y99','AdaRES-Y100'):self.assertEqual(c[cid]['stage'],'training')
        self.assertEqual(c['AdaRES-Y101']['blocker_kind'],'exclusive_gpu_slot');self.assertTrue(c['AdaRES-Y101']['implementation_test_passed'])
        self.assertEqual(c['AdaRES-Y97']['blocker_kind'],'numerical_control_failure');self.assertEqual(c['AdaRES-Y97']['revision_history'][0]['result'],'failed')
        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'],'3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')
