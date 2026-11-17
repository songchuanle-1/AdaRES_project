"""Reject detached, bypassed or malformed native geometry observations."""
import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).with_name('probe_y69_native_smoke.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE,feature_version=(3,8));N={}
FUNCTIONS=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_response_rows')]
assert len(FUNCTIONS)==2
exec(compile(ast.Module(body=FUNCTIONS,type_ignores=[]),'<geometry-native-guards>','exec'),N)


class Guards(unittest.TestCase):
    def rows(self):
        return [dict(weight_nonzero=bool(i),bias_max=.01 if i else 0.,reference_error_max=1e-7,
            native_positions_verified=True,hard_masks_verified=True,consumed_by_all_coarse_attentions=True,
            expressions=8,heads=8,queries=128,reference_expression_rows=[0,7],
            reference_head_rows=[0,7],reference_query_rows=[0,127]) for i in range(2)]

    def test_valid_schema_only(self):
        N['check_path_calls'](list(range(6)),7,1,6);N['check_response_rows'](self.rows())

    def test_missing_extra_reordered_or_boolean_stage(self):
        for stages in (list(range(5)),list(range(7)),[0]*6,list(reversed(range(6))),[False,1,2,3,4,5]):
            with self.assertRaises(AssertionError):N['check_path_calls'](stages,7,1,6)

    def test_bypassed_indicators_geometry_or_attention(self):
        for args in ((list(range(6)),6,1,6),(list(range(6)),7,0,6),(list(range(6)),7,1,5),(list(range(6)),7,True,6)):
            with self.assertRaises(AssertionError):N['check_path_calls'](*args)

    def test_missing_or_extra_response(self):
        for rows in (self.rows()[:1],self.rows()+self.rows()[:1]):
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_nonzero_initial_anchor(self):
        for k,v in (('weight_nonzero',True),('bias_max',.01)):
            rows=self.rows();rows[0][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_missing_second_step_response(self):
        for k,v in (('weight_nonzero',False),('bias_max',0.)):
            rows=self.rows();rows[1][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_wrong_or_bypassed_native_positions_and_masks(self):
        for k in ('native_positions_verified','hard_masks_verified','consumed_by_all_coarse_attentions'):
            for value in (False,1):
                rows=self.rows();rows[1][k]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_nonfinite_or_negative_measurements(self):
        for k in ('bias_max','reference_error_max'):
            for value in (float('nan'),float('inf'),-.1):
                rows=self.rows();rows[1][k]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_wrong_reference_rows_or_native_dimensions(self):
        for k,v in (('reference_expression_rows',[0,6]),('reference_head_rows',[1,7]),
                    ('reference_query_rows',[0,126]),('heads',1),('queries',127),('expressions',8.0)):
            rows=self.rows();rows[1][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_nonboolean_parameter_response_flag(self):
        rows=self.rows();rows[1]['weight_nonzero']=1
        with self.assertRaises(AssertionError):N['check_response_rows'](rows)


if __name__=='__main__':unittest.main()
