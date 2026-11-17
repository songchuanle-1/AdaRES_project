"""Reject missing native stage execution, gradient or optimizer updates."""
import ast
from pathlib import Path
import unittest
S=Path(__file__).with_name('probe_y67_native_smoke.py').read_text(encoding='utf-8')
T=ast.parse(S,feature_version=(3,8));N={}
F=[n for n in T.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_gradient_rows')]
assert len(F)==2
exec(compile(ast.Module(body=F,type_ignores=[]),'<native-normalization-guards>','exec'),N)
class Guards(unittest.TestCase):
    def rows(self):
        return [dict(stage=i,parameter_tensors=2,all_finite=True,gradient_max=.001,parameter_update_max=.0001) for i in range(7)]
    def test_valid_schema_only(self):
        N['check_path_calls'](list(range(7)),7,7,7);N['check_gradient_rows'](self.rows())
    def test_missing_reordered_duplicate_or_noninteger_path(self):
        for c in (list(range(6)),list(range(8)),list(reversed(range(7))),[0,1,2,3,4,5,5],[False,1,2,3,4,5,6]):
            with self.subTest(c=c),self.assertRaises(AssertionError):N['check_path_calls'](c,7,7,7)
    def test_missing_mask_score_or_indicator_consumption(self):
        for c in ((6,7,7),(7,6,7),(7,7,6),(8,7,7)):
            with self.subTest(c=c),self.assertRaises(AssertionError):N['check_path_calls'](list(range(7)),*c)
    def test_missing_or_extra_gradient_stage(self):
        for r in (self.rows()[:6],self.rows()+self.rows()[:1]):
            with self.assertRaises(AssertionError):N['check_gradient_rows'](r)
    def test_wrong_or_boolean_stage(self):
        for v in (1,False,'0'):
            r=self.rows();r[0]['stage']=v
            with self.assertRaises(AssertionError):N['check_gradient_rows'](r)
    def test_missing_affine_gradient(self):
        r=self.rows();r[3]['parameter_tensors']=1
        with self.assertRaises(AssertionError):N['check_gradient_rows'](r)
    def test_zero_negative_or_nonfinite_gradient_or_update(self):
        for field in ('gradient_max','parameter_update_max'):
            for v in (0.,-.1,float('nan'),float('inf')):
                r=self.rows();r[6][field]=v
                with self.subTest(field=field,v=v),self.assertRaises(AssertionError):N['check_gradient_rows'](r)
    def test_failed_or_nonboolean_finiteness(self):
        for v in (False,1,None):
            r=self.rows();r[6]['all_finite']=v
            with self.assertRaises(AssertionError):N['check_gradient_rows'](r)
if __name__=='__main__':unittest.main()
