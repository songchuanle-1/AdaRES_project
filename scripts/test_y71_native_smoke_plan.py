"""Refuse missing, detached, bypassed or inconsistent native visual interaction evidence."""
import ast
from pathlib import Path
import unittest
S=Path(__file__).with_name('probe_y71_native_smoke.py').read_text(encoding='utf-8');T=ast.parse(S,feature_version=(3,8));N={}
F=[n for n in T.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_response_rows')]
assert len(F)==2
exec(compile(ast.Module(body=F,type_ignores=[]),'<visual-product-guards>','exec'),N)
class Guards(unittest.TestCase):
    def rows(self):
        return [dict(weight_nonzero=bool(i),native_projection_identity_verified=True,normalization_consumption_verified=True,
            feature_rows=100,reference_feature_rows=[0,50,99],reference_channels=[0,127,255],reference_error_max=1e-7,
            fusion_delta_max=.01 if i else 0.,spatial_abs_max=3.,appearance_abs_max=4.) for i in range(2)]
    def test_valid_schema_only(self):N['check_path_calls'](1,1,7,[True]*7);N['check_response_rows'](self.rows())
    def test_missing_extra_or_noninteger_path(self):
        for f,c in ((0,1),(2,1),(1,0),(1,2),(True,1),(1,True)):
            with self.assertRaises(AssertionError):N['check_path_calls'](f,c,7,[True]*7)
    def test_missing_detached_or_nonboolean_indicator_path(self):
        for count,norms in ((6,[True]*7),(8,[True]*7),(7,[False]*7),(7,[1]*7),(7,[True]*6)):
            with self.assertRaises(AssertionError):N['check_path_calls'](1,1,count,norms)
    def test_missing_second_step_or_extra_measurement(self):
        for rows in (self.rows()[:1],self.rows()+self.rows()[:1]):
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonzero_initial_anchor(self):
        for key,value in (('weight_nonzero',True),('fusion_delta_max',.1)):
            rows=self.rows();rows[0][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_missing_second_weight_response_or_factor(self):
        for key,value in (('weight_nonzero',False),('fusion_delta_max',0.),('spatial_abs_max',0.),('appearance_abs_max',0.)):
            rows=self.rows();rows[1][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonfinite_or_negative_measurement(self):
        for key in ('reference_error_max','fusion_delta_max','spatial_abs_max','appearance_abs_max'):
            for value in (float('nan'),float('inf'),-.1):
                rows=self.rows();rows[1][key]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_bypassed_wrong_identity_or_nonboolean_flags(self):
        for key in ('native_projection_identity_verified','normalization_consumption_verified'):
            for value in (False,1):
                rows=self.rows();rows[1][key]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
        rows=self.rows();rows[1]['weight_nonzero']=1
        with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_wrong_reference_rows_channels_or_count(self):
        for key,value in (('reference_feature_rows',[0,49,99]),('feature_rows',100.),('reference_channels',[0,127,254])):
            rows=self.rows();rows[1][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_native_train_signature_unchanged(self):
        calls=[n for n in ast.walk(T) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native' and n.func.attr=='train']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(n) for n in calls[0].args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])
if __name__=='__main__':unittest.main()
