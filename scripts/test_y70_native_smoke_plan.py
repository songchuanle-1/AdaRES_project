"""Reject missing, bypassed, reordered and fabricated scale-fusion evidence."""
import ast
import copy
from pathlib import Path
import unittest

SOURCE=Path(__file__).with_name('probe_y70_native_smoke.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE,feature_version=(3,8));N={}
FUNCTIONS=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_response_rows')]
assert len(FUNCTIONS)==2
exec(compile(ast.Module(body=FUNCTIONS,type_ignores=[]),'<scale-fusion-guards>','exec'),N)

class Guards(unittest.TestCase):
    def rows(self):
        return [dict(stage=i%6,weight_nonzero=i>=6,native_branch_identity_verified=True,
            downstream_consumption_verified=True,expressions=8,reference_expression_rows=[0,7],
            reference_queries=[0,127],reference_channels=[0,127,255],reference_error_max=1e-7,
            fusion_delta_max=.001 if i>=6 else 0.,query_abs_max=4.) for i in range(12)]
    def test_valid_structure_only(self):
        N['check_path_calls'](list(range(6)),list(range(6)),7,[True]*7)
        N['check_response_rows'](self.rows())
    def test_missing_extra_reordered_or_noninteger_stage(self):
        for stages in (list(range(5)),list(range(7)),list(reversed(range(6))),[float(i) for i in range(6)]):
            for index in (0,1):
                args=[list(range(6)),list(range(6)),7,[True]*7];args[index]=stages
                with self.subTest(args=args),self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_detached_missing_or_nonboolean_indicator_paths(self):
        for count,norms in ((6,[True]*7),(8,[True]*7),(7,[False]*7),(7,[1]*7),(7,[True]*6)):
            with self.assertRaises(AssertionError):N['check_path_calls'](list(range(6)),list(range(6)),count,norms)
    def test_missing_second_optimizer_step(self):
        for rows in (self.rows()[:6],self.rows()+self.rows()[:1]):
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonzero_initial_anchor(self):
        for key,value in (('weight_nonzero',True),('fusion_delta_max',.1)):
            rows=self.rows();rows[0][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_no_second_step_weight_or_response(self):
        for key,value in (('weight_nonzero',False),('fusion_delta_max',0.)):
            rows=self.rows();rows[6][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonfinite_or_negative_measurement(self):
        for key in ('reference_error_max','fusion_delta_max','query_abs_max'):
            for value in (float('nan'),float('inf'),-.1):
                rows=self.rows();rows[8][key]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_bypass_wrong_identity_and_nonboolean_flags(self):
        for key in ('native_branch_identity_verified','downstream_consumption_verified'):
            for value in (False,1):
                rows=self.rows();rows[7][key]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
        rows=self.rows();rows[7]['weight_nonzero']=1
        with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_wrong_reference_expressions_queries_channels(self):
        for key,value in (('reference_expression_rows',[0,6]),('expressions',8.0),('reference_queries',[0,126]),('reference_channels',[0,127,254]),('stage',0)):
            rows=self.rows();rows[7][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_native_train_call_has_original_signature(self):
        calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native' and n.func.attr=='train']
        self.assertEqual(len(calls),1)
        args=calls[0].args
        self.assertEqual(len(args),8)
        self.assertEqual([ast.unparse(n) for n in args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])

if __name__=='__main__':unittest.main()
