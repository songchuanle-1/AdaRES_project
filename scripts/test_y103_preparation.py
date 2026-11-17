"""Guard actual Y103 CPU evidence, its exact source, and unexecuted native plans."""
import ast
import copy
import unittest
from prepare_y60_stage_heads import ROOT, tick, ref
from prepare_y103_spatial_input_skip import files, EDITS, PARENT, NAMES
from y103_contract import validate_cpu, validate_result

class Preparation(unittest.TestCase):

    def setUp(self):
        self.token = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')['token']
        self.cpu = tick.load(ROOT / ('pipeline/analyses/y103_cpu_' + self.token + '.json'))
        self.body = (ROOT / ('pipeline/checks/y103_native_' + self.token + '_body.py')).read_text(encoding='utf-8')

    def test_actual_CPU_evidence(self):
        validate_cpu(self.cpu)

    def test_identity_and_counts(self):
        for k, v in (('id', 'AdaRES-Y98'), ('parameters', 1), ('additional_parameters', 0)):
            with self.subTest(k=k), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{k: v}))

    def test_component_controls_required(self):
        for k in ('no_RNG', 'independent_scalar_full_appearance_projection', 'row_input_output_channel_permutations', 'appearance_intervention_is_row_local', 'appearance_sign_reversal'):
            v = copy.deepcopy(self.cpu)
            v['component_controls'][0][k] = False
            with self.subTest(k=k), self.assertRaises(AssertionError):
                validate_cpu(v)

    def test_anchor_and_bypass_required(self):
        for k in ('decoder_all_stages_anchor_exact', 'decoder_input_gradients_anchor_close', 'exact_bypass_restoration', 'original_spatial_projection_and_fusion_consumers_verified', 'all_seven_original_static_masks_verified'):
            with self.subTest(k=k), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{k: False}))

    def test_all_actual_gradient_consumers(self):
        for k, x in (('parameter_update_max', 0.0), ('new_gradient_max', float('nan')), ('decoder_input_gradients_finite', False), ('original_visual_parameter_tensors', 0), ('base_adapted_fusion_and_mask_gradients_finite', False), ('frozen_text_encoder_preserved', False), ('first_language_memory_consumers', 0)):
            v = copy.deepcopy(self.cpu)
            v['CPU_decoder_steps'][1][k] = x
            with self.subTest(k=k), self.assertRaises(AssertionError):
                validate_cpu(v)

    def test_full_raw_and_projection_fusion_static_mask_identity(self):
        for k, x in (('step', 0), ('weight_nonzero', False), ('spatial_feature_delta_max', 0.0), ('original_full1024_appearance_identity', False), ('original_full32_spatial_identity', False), ('original_spatial_projection_identity', False), ('original_sum_norm_consumes_skip', False), ('original_language_memory_consumes_skip', False), ('all_seven_original_static_masks_preserved', False)):
            v = copy.deepcopy(self.cpu)
            v['CPU_spatial_input_paths'][1][k] = x
            with self.subTest(k=k), self.assertRaises(AssertionError):
                validate_cpu(v)

    def test_CPU_cannot_impersonate_native(self):
        for k in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward', 'native_sparse_forward', 'validation_used', 'admission'):
            with self.subTest(k=k), self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu, **{k: True}))
        with self.assertRaises((KeyError, AssertionError)):
            validate_result(self.cpu, {'id': 'AdaRES-Y103'}, {'variant': {'id': 'AdaRES-Y103'}})

    def test_exact_W2_independent_edits_only(self):
        parent = ROOT / 'AdaRES/ipdn/model_paper5' / PARENT
        source = (parent / 'dec.py').read_text(encoding='utf-8')
        for old, new in EDITS:
            self.assertEqual(source.count(old), 1)
            source = source.replace(old, new)
        self.assertEqual(source, (files()[0].parent / 'dec.py').read_text(encoding='utf-8'))
        for n in NAMES:
            if n not in ('__init__.py', 'dec.py'):
                self.assertEqual((parent / n).read_bytes(), (files()[0].parent / n).read_bytes())
        self.assertEqual(files()[-1].read_bytes(), (ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())

    def test_same_CPU_executed_callbacks_in_native(self):
        core = (ROOT / 'scripts/y103_observer_core.py').read_text(encoding='utf-8')
        cpu = (ROOT / ('pipeline/checks/y103_cpu_' + self.token + '_program.py')).read_text(encoding='utf-8')
        self.assertEqual(cpu.count(core), 1)
        self.assertEqual(self.body.count(core), 1)
        ast.parse(self.body, feature_version=(3, 8))
        for t in ('native.train(1, model, FirstTwoBatches()', 'optimizer.step = checked_step', "state['raw_spatial_expected'] = args[0]", 'original_sum_norm_calls=2'):
            self.assertIn(t, self.body)

    def test_active_runs_pauses_and_negative_evidence_preserved(self):
        w = tick.load(ROOT / 'pipeline/research_work.json')
        c = {r['id']: r for r in w['candidates']}
        self.assertFalse(c['AdaRES-Y103'].get('native_smoke_result'))
        self.assertFalse(c['AdaRES-Y103'].get('admission_receipt'))
        self.assertEqual(c['AdaRES-Y101']['stage'], 'training')
        self.assertEqual(c['AdaRES-Y100']['stage'], 'training')
        self.assertEqual(c['AdaRES-Y97']['blocker_kind'], 'numerical_control_failure')
        self.assertEqual(c['AdaRES-Y97']['revision_history'][0]['result'], 'failed')
        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'], '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')
