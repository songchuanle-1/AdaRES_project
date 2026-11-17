"""Actual CPU, causal-path, exact-source and negative preparation controls."""
import ast
import copy
import unittest
from prepare_y90_query_peer_affinity import ROOT,PARENT,EDITS,files
from prepare_y60_stage_heads import tick
from y90_contract import validate_cpu,validate_gradients
from y90_observer_core import check_response_rows


class Preparation(unittest.TestCase):
    def setUp(self):
        work = tick.load(ROOT/'pipeline/research_work.json')
        self.c = next(candidate for candidate in work['candidates'] if candidate['id']=='AdaRES-Y90')
        self.cpu = tick.load(ROOT/self.c['CPU_result']['path'])

    def test_actual_CPU_evidence(self):
        validate_cpu(self.cpu)

    def test_CPU_cannot_claim_native_or_admission(self):
        for key in ('CUDA_initialized','native_smoke_passed','native_task_backward','native_sparse_forward','validation_used','admission'):
            with self.subTest(key=key),self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu,**{key:True}))

    def test_anchors_original_overlap_and_causal_bypass_required(self):
        for key in ('decoder_all_stages_anchor_exact','decoder_input_gradients_anchor_close',
                    'original_competition_overlap_and_max_preserved','bypass_restoration_exact',
                    'active_affinity_changes_confidence_only','original_queries_masks_scores_and_hard_gate_preserved'):
            with self.subTest(key=key),self.assertRaises(AssertionError):
                validate_cpu(dict(self.cpu,**{key:False}))

    def test_original_inputs_and_context_consumer_required(self):
        for key in ('original_query_identity','original_mask_identity','original_mass_identity',
                    'normalized_query_input_undetached','float64_bilinear_gate_reference',
                    'original_overlap_and_max_preserved','weighted_peer_context_consumed'):
            rows = copy.deepcopy(self.cpu['CPU_affinity_paths'])
            rows[8][key] = False
            with self.subTest(key=key),self.assertRaises(AssertionError):
                check_response_rows(rows)

    def test_zero_and_learned_response_required(self):
        for index,key,value in ((0,'weight_nonzero',True),(0,'gate_delta_max',.1),
                               (8,'weight_nonzero',False),(8,'gate_delta_max',0.),
                               (8,'context_delta_max',float('nan'))):
            rows = copy.deepcopy(self.cpu['CPU_affinity_paths'])
            rows[index][key] = value
            with self.subTest(key=key),self.assertRaises(AssertionError):
                check_response_rows(rows)
        rows = copy.deepcopy(self.cpu['CPU_affinity_paths'])
        for row in rows[7:]:
            row['context_delta_max'] = 0.
        with self.assertRaises(AssertionError):
            check_response_rows(rows)

    def test_individual_constant_gate_cancellation_not_fabricated_failure(self):
        rows = copy.deepcopy(self.cpu['CPU_affinity_paths'])
        rows[-1]['context_delta_max'] = 0.
        check_response_rows(rows)
        for rows in (rows[:-1],list(reversed(rows))):
            with self.assertRaises(AssertionError):
                check_response_rows(rows)

    def test_actual_gradients_updates_and_original_counts_required(self):
        for key,value in (('new_gradient_max',0.),('parameter_update_max',float('inf')),
                          ('original_head_parameter_tensors',0),('input_gradients_finite',False),
                          ('weighted_context_consumers',0),('coarse_gate_consumers',0)):
            rows = copy.deepcopy(self.cpu['CPU_decoder_steps'])
            rows[1][key] = value
            with self.subTest(key=key),self.assertRaises(AssertionError):
                validate_gradients(rows)

    def test_literal_CPU_native_callbacks_identical(self):
        core = (ROOT/self.c['observer_core']['path']).read_text(encoding='utf-8')
        expected = {node.name:ast.dump(node,include_attributes=False)
                    for node in ast.parse(core).body if isinstance(node,ast.FunctionDef)}
        self.assertEqual(len(expected),11)
        for key in ('CPU_program','prepared_native_body'):
            source = (ROOT/self.c[key]['path']).read_text(encoding='utf-8')
            found = {node.name:ast.dump(node,include_attributes=False)
                     for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in expected}
            self.assertEqual(found,expected)
        self.assertIn('affinity.weight',source)
        self.assertNotIn('projection.weight',source)
        for field in ('peer_gate_calls=14','weighted_context_consumers=14','original_evidence_norm_calls=14'):
            self.assertIn(field,source)

    def test_three_edit_reversal_restores_original_W2(self):
        source = files()[2].read_text(encoding='utf-8')
        for old,new in reversed(EDITS):
            self.assertEqual(source.count(new),1)
            source = source.replace(new,old)
        self.assertEqual(source,(files()[2].parent.parent/PARENT/'dec.py').read_text(encoding='utf-8'))

    def test_other_native_sources_and_YAML_unchanged(self):
        for path in files()[1:-1]:
            if path.name!='dec.py':
                self.assertEqual(path.read_bytes(),(path.parent.parent/PARENT/path.name).read_bytes())
        parent = ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')
        self.assertEqual(files()[-1].read_bytes(),parent.read_bytes())


if __name__=='__main__':
    unittest.main()
