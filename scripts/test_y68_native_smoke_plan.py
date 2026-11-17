"""Refuse unexecuted, detached, bypassed or inconsistent native modulation evidence."""
import ast
from pathlib import Path
import unittest
S=Path(__file__).with_name('probe_y68_native_smoke.py').read_text(encoding='utf-8');T=ast.parse(S,feature_version=(3,8));N={}
F=[n for n in T.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_language_identity','check_response_rows')]
assert len(F)==3
exec(compile(ast.Module(body=F,type_ignores=[]),'<language-modulation-guards>','exec'),N)
class Guards(unittest.TestCase):
    def rows(self):
        return [dict(projection_nonzero=bool(i),native_input_correspondence_verified=True,consumed_by_all_predictions=True,
            reference_expression_rows=[0,3],expressions=4,valid_token_counts=[7,8,9,10],sentence_max=.2,feature_delta_max=.01 if i else 0.,reference_error_max=1e-7) for i in range(2)]
    def test_valid_schema_only(self):
        N['check_path_calls']([True]*7,7,1,7);N['check_response_rows'](self.rows());x=object();N['check_language_identity'](x,x)
    def test_missing_extra_detached_or_nonboolean_path(self):
        for args in (([True]*6,6,1,7),([True]*8,8,1,7),([False]*7,7,1,7),([1]*7,7,1,7),([True]*7,7,0,7),([True]*7,7,2,7),([True]*7,7,1,6)):
            with self.subTest(args=args),self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_wrong_native_language_tensor_identity(self):
        with self.assertRaises(AssertionError):N['check_language_identity'](object(),object())
    def test_missing_or_extra_second_step(self):
        for rows in (self.rows()[:1],self.rows()+self.rows()[:1]):
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonzero_initial_anchor(self):
        for key,value in (('projection_nonzero',True),('feature_delta_max',.1)):
            rows=self.rows();rows[0][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_missing_second_projection_response_or_context(self):
        for key,value in (('projection_nonzero',False),('feature_delta_max',0.),('sentence_max',0.)):
            rows=self.rows();rows[1][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonfinite_or_negative_measurements(self):
        for key in ('sentence_max','feature_delta_max','reference_error_max'):
            for value in (float('nan'),float('inf'),-.1):
                rows=self.rows();rows[1][key]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_bypassed_wrong_correspondence_or_nonboolean_flags(self):
        for key in ('native_input_correspondence_verified','consumed_by_all_predictions','projection_nonzero'):
            for value in (False,1):
                rows=self.rows();rows[1][key]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_wrong_reference_rows_or_expression_count(self):
        for key,value in (('reference_expression_rows',[0,2]),('expressions',3),('expressions',4.0)):
            rows=self.rows();rows[1][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_invalid_or_missing_valid_token_counts(self):
        for values in ([],[7,8,9],[0,8,9,10],[True,8,9,10]):
            rows=self.rows();rows[1]['valid_token_counts']=values
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
if __name__=='__main__':unittest.main()
