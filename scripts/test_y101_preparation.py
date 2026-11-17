"""Guard actual Y101 CPU observations and exact independently implemented files."""
import ast,copy,unittest
from prepare_y60_stage_heads import ROOT,tick,ref
from prepare_y101_query_ffn_skip import files,EDITS,PARENT,NAMES
from y101_contract import validate_cpu,validate_result


class Preparation(unittest.TestCase):
    def setUp(self):
        self.token=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')['token']
        self.cpu=tick.load(ROOT/('pipeline/analyses/y101_cpu_'+self.token+'.json'))
        self.body=(ROOT/('pipeline/checks/y101_native_'+self.token+'_body.py')).read_text(encoding='utf-8')
    def test_actual_CPU_evidence(self):
        validate_cpu(self.cpu)
    def test_identity_and_parameter_counts(self):
        for k,x in (('id','AdaRES-Y96'),('parameters',1),('additional_parameters',0)):
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(dict(self.cpu,**{k:x}))
    def test_scalar_locality_permutation_and_no_RNG(self):
        for k in ('no_RNG','independent_scalar_full_query_projection','joint_slot_batch_channel_permutations','current_query_intervention_is_slot_local'):
            v=copy.deepcopy(self.cpu);v['component_controls'][0][k]=False
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(v)
    def test_initial_and_all_stage_anchors_bypass(self):
        for k in ('decoder_all_stages_anchor_exact','decoder_input_gradients_anchor_close','all_six_complete_original_query_FFNs_verified','initial_readout_exact_and_refined_masks_change','exact_bypass_restoration'):
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(dict(self.cpu,**{k:False}))
    def test_all_original_FFN_input_and_output_gradients(self):
        for k,x in (('parameter_update_max',0.),('new_gradient_max',float('nan')),('original_query_FFN_parameter_tensors',0),('full_input_base_adapted_and_original_FFN_gradients_finite',False),('original_shared_fusion_norm_gradients_finite',False),('static_mask_gradients_finite',False),('query_FFN_skip_calls',5),('decoder_input_gradients_finite',False)):
            v=copy.deepcopy(self.cpu);v['CPU_decoder_steps'][1][k]=x
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(v)
    def test_all_twelve_consumers_required(self):
        for k,x in (('stage',0),('step',0),('weight_nonzero',False),('query_delta_max',0.),('original_full256_FFN_input_identity',False),('complete_original_FFN_output_identity',False),('prediction_consumes_adapted_query',False),('original_static_mask_preserved',False)):
            v=copy.deepcopy(self.cpu);v['CPU_query_FFN_paths'][-1][k]=x
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(v)
        v=copy.deepcopy(self.cpu);v['CPU_query_FFN_paths'].pop()
        with self.assertRaises(AssertionError):validate_cpu(v)
    def test_CPU_cannot_impersonate_native(self):
        for k in ('CUDA_initialized','native_smoke_passed','native_task_backward','native_sparse_forward','validation_used','admission'):
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_cpu(dict(self.cpu,**{k:True}))
        with self.assertRaises((AssertionError,KeyError)):validate_result(self.cpu,{'id':'AdaRES-Y101'},{'variant':{'id':'AdaRES-Y101'}})
    def test_exact_independent_W2_edits(self):
        parent=ROOT/'AdaRES/ipdn/model_paper5'/PARENT;source=(parent/'dec.py').read_text(encoding='utf-8')
        for a,b in EDITS:self.assertEqual(source.count(a),1);source=source.replace(a,b)
        self.assertEqual(source,(files()[0].parent/'dec.py').read_text(encoding='utf-8'))
        for n in NAMES:
            if n not in ('__init__.py','dec.py'):self.assertEqual((parent/n).read_bytes(),(files()[0].parent/n).read_bytes())
        self.assertEqual(files()[-1].read_bytes(),(ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    def test_same_CPU_executed_callbacks_for_native(self):
        core=(ROOT/'scripts/y101_observer_core.py').read_text(encoding='utf-8')
        cpu=(ROOT/('pipeline/checks/y101_cpu_'+self.token+'_program.py')).read_text(encoding='utf-8')
        self.assertEqual(cpu.count(core),1);self.assertEqual(self.body.count(core),1)
        ast.parse(self.body,feature_version=(3,8))
        for t in ('native.train(1, model, FirstTwoBatches()','optimizer.step = checked_step','original_query_FFN_calls=12'):
            self.assertIn(t,self.body)
    def test_original_runs_pauses_and_negative_evidence_preserved(self):
        c={r['id']:r for r in tick.load(ROOT/'pipeline/research_work.json')['candidates']}
        self.assertFalse(c['AdaRES-Y101'].get('native_smoke_result'));self.assertFalse(c['AdaRES-Y101'].get('admission_receipt'))
        for cid in ('AdaRES-Y99','AdaRES-Y100'):self.assertEqual(c[cid]['stage'],'training')
        self.assertEqual(c['AdaRES-Y97']['blocker_kind'],'numerical_control_failure')
        self.assertEqual(c['AdaRES-Y97']['revision_history'][0]['result'],'failed')
        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'],'3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')
