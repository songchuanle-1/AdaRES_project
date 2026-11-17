"""Reject false gate observations; fixtures do not establish native execution."""
import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).with_name('probe_y74_native_smoke.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE,feature_version=(3,8));N={}
FUNCTIONS=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_native_inputs','check_response_rows')]
assert len(FUNCTIONS)==3
exec(compile(ast.Module(body=FUNCTIONS,type_ignores=[]),'<language-update-gate-guards>','exec'),N)

class Guards(unittest.TestCase):
    def rows(self):
        return [dict(stage=i%6,weight_nonzero=i>=6,expressions=8,queries=128,
            reference_expression_rows=[0,7],reference_queries=[0,127],reference_channels=[0,127,255],
            reference_error_max=1e-7,gate_delta_max=.001 if i>=6 else 0.,language_update_abs_max=3.,
            native_qla_output_identity_verified=True,incoming_visual_query_identity_verified=True,both_inputs_undetached_verified=True,
            qqa_consumption_verified=True,static_mask_features_preserved=True,padding_identity_verified=True) for i in range(12)]
    def test_valid_schema_only(self):
        N['check_path_calls'](list(range(6)),list(range(6)),list(range(6)),list(range(7)),7,[True]*7)
        N['check_response_rows'](self.rows())
    def test_exact_native_output_and_incoming_query_objects(self):
        a,b=object(),object();args=[a,a,b,b];N['check_native_inputs'](*args)
        for i in range(4):
            bad=list(args);bad[i]=object()
            with self.assertRaises(AssertionError):N['check_native_inputs'](*bad)
    def test_missing_extra_or_reordered_stages(self):
        for i,count in ((0,6),(1,6),(2,6),(3,7)):
            for bad in (list(range(count-1)),list(range(count+1)),list(reversed(range(count))),list(map(float,range(count)))):
                args=[list(range(6)),list(range(6)),list(range(6)),list(range(7)),7,[True]*7];args[i]=bad
                with self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_missing_detached_or_nonboolean_head_calls(self):
        for count,norms in ((6,[True]*7),(8,[True]*7),(7.,[True]*7),(7,[False]*7),(7,[1]*7),(7,[True]*6)):
            with self.assertRaises(AssertionError):N['check_path_calls'](list(range(6)),list(range(6)),list(range(6)),list(range(7)),count,norms)
    def test_missing_second_step_or_extra_rows(self):
        for rows in (self.rows()[:6],self.rows()+self.rows()[:1]):
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonzero_initial_gate_refused(self):
        for k,v in (('weight_nonzero',True),('gate_delta_max',.1)):
            rows=self.rows();rows[0][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_missing_second_weight_update_or_language_increment(self):
        for k,v in (('weight_nonzero',False),('gate_delta_max',0.),('language_update_abs_max',0.)):
            rows=self.rows();rows[7][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_detached_bypassed_or_wrong_mask_paths(self):
        for k in ('native_qla_output_identity_verified','incoming_visual_query_identity_verified','both_inputs_undetached_verified','qqa_consumption_verified','static_mask_features_preserved','padding_identity_verified'):
            for v in (False,1):
                rows=self.rows();rows[7][k]=v
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_wrong_reference_rows_queries_channels_or_stage(self):
        for k,v in (('reference_expression_rows',[0,6]),('reference_queries',[0,126]),('reference_channels',[0,127,254]),('stage',0),('expressions',8.),('queries',128.),('weight_nonzero',1)):
            rows=self.rows();rows[7][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonfinite_or_negative_measurements(self):
        for k in ('reference_error_max','gate_delta_max','language_update_abs_max'):
            for v in (float('nan'),float('inf'),-.1):
                rows=self.rows();rows[8][k]=v
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_native_train_signature_unchanged(self):
        calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native' and n.func.attr=='train']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(n) for n in calls[0].args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])

if __name__=='__main__':unittest.main()
