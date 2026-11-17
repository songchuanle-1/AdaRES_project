"""Reject incomplete stage-conditioning evidence; fixtures are not GPU execution."""
import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).with_name('probe_y76_native_smoke.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE,feature_version=(3,8));N={}
FUNCTIONS=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_native_inputs','check_response_rows')]
assert len(FUNCTIONS)==3
exec(compile(ast.Module(body=FUNCTIONS,type_ignores=[]),'<stage-conditioning-guards>','exec'),N)

class Guards(unittest.TestCase):
    def rows(self):
        return [dict(stage=i%7,offset_nonzero=i>=7,expressions=8,queries=128,reference_expression_rows=[0,7],reference_queries=[0,127],reference_channels=[0,127,255],
            reference_error_max=1e-7,conditioning_delta_max=.001 if i>=7 else 0.,offset_abs_max=.001 if i>=7 else 0.,
            original_head_query_identity_verified=True,prediction_query_identity_verified=True,query_undetached_verified=True,query_norm_consumption_verified=True,
            static_mask_features_preserved=True,padding_identity_verified=True,original_mask_identity_verified=True) for i in range(14)]
    def test_valid_schema_only(self):
        N['check_path_calls'](*([list(range(7))]*4),[True]*7);N['check_response_rows'](self.rows())
    def test_missing_extra_reordered_or_float_stage_paths(self):
        for i in range(4):
            for bad in (list(range(6)),list(range(8)),list(reversed(range(7))),list(map(float,range(7)))):
                args=[list(range(7)) for _ in range(4)]+[[True]*7];args[i]=bad
                with self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_missing_second_step_or_extra_rows(self):
        for v in (self.rows()[:7],self.rows()+self.rows()[:1]):
            with self.assertRaises(AssertionError):N['check_response_rows'](v)
    def test_native_query_and_stage_identity_required(self):
        a=object();N['check_native_inputs'](a,a,a,3,3)
        for args in ((object(),a,a,3,3),(a,object(),a,3,3),(a,a,object(),3,3),(a,a,a,2,3),(a,a,a,3.,3),(a,a,a,True,1)):
            with self.assertRaises(AssertionError):N['check_native_inputs'](*args)
    def test_detached_bypassed_wrong_mask_or_norm_refused(self):
        for k in ('original_head_query_identity_verified','prediction_query_identity_verified','query_undetached_verified','query_norm_consumption_verified','static_mask_features_preserved','padding_identity_verified','original_mask_identity_verified'):
            for v in (False,1):
                rows=self.rows();rows[8][k]=v
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonboolean_offset_or_wrong_stage_refused(self):
        for k,v in (('offset_nonzero',1),('stage',1.),('stage',6)):
            rows=self.rows();rows[8][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_initial_nonzero_or_missing_next_step_update(self):
        for index,k,v in ((0,'offset_nonzero',True),(0,'conditioning_delta_max',.1),(0,'offset_abs_max',.1),(8,'offset_nonzero',False),(8,'conditioning_delta_max',0.),(8,'offset_abs_max',0.)):
            rows=self.rows();rows[index][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_wrong_reference_correspondence(self):
        for k,v in (('reference_expression_rows',[0,6]),('reference_queries',[0,126]),('reference_channels',[0,127,254]),('expressions',8.),('queries',128.)):
            rows=self.rows();rows[8][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonfinite_or_negative_measurement(self):
        for k in ('reference_error_max','conditioning_delta_max','offset_abs_max'):
            for v in (float('nan'),float('inf'),-.1):
                rows=self.rows();rows[8][k]=v
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_detached_or_missing_original_norm_paths(self):
        for v in ([True]*6,[False]*7,[1]*7):
            with self.assertRaises(AssertionError):N['check_path_calls'](*([list(range(7))]*4),v)
    def test_native_train_signature_unchanged(self):
        calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native' and n.func.attr=='train']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(n) for n in calls[0].args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])

if __name__=='__main__':unittest.main()
