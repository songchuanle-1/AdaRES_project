"""Native initialization evidence guards; schema fixtures are not native PASS."""
import ast
from pathlib import Path
import unittest
SOURCE=Path(__file__).with_name('probe_y75_native_smoke.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE,feature_version=(3,8));N={}
functions=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_native_inputs','check_response_rows')]
assert len(functions)==3
exec(compile(ast.Module(body=functions,type_ignores=[]),'<initial-query-guards>','exec'),N)
class Guards(unittest.TestCase):
    def rows(self):
        return [dict(weight_nonzero=bool(i),expressions=8,queries=128,tokens=9,reference_expression_rows=[0,7],reference_queries=[0,127],reference_channels=[0,127,255],reference_valid_token_counts=[7,9],
            reference_error_max=1e-7,initialization_delta_max=.001 if i else 0.,context_abs_max=3.,generated_query_identity_verified=True,projected_language_identity_verified=True,
            sampling_padding_identity_verified=True,native_padding_values_verified=True,both_inputs_undetached_verified=True,initial_prediction_consumption_verified=True,static_mask_features_preserved=True,padding_identity_verified=True) for i in range(2)]
    def test_valid_schema_only(self):
        N['check_path_calls'](1,1,1,1,list(range(7)),7,[True]*7);N['check_response_rows'](self.rows())
    def test_exact_generated_language_padding_objects(self):
        a,b,c=object(),object(),object();args=[a,a,b,b,c,c];N['check_native_inputs'](*args)
        for i in range(6):
            bad=list(args);bad[i]=object()
            with self.assertRaises(AssertionError):N['check_native_inputs'](*bad)
    def test_missing_extra_noninteger_path_counts(self):
        for i in range(4):
            for bad in (0,2,1.,True):
                args=[1,1,1,1,list(range(7)),7,[True]*7];args[i]=bad
                with self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_bad_predictions_head_count_or_detached_norm(self):
        for predictions,count,norms in [(list(range(6)),7,[True]*7),(list(reversed(range(7))),7,[True]*7),(list(map(float,range(7))),7,[True]*7),(list(range(7)),6,[True]*7),(list(range(7)),7.,[True]*7),(list(range(7)),7,[False]*7),(list(range(7)),7,[1]*7)]:
            with self.assertRaises(AssertionError):N['check_path_calls'](1,1,1,1,predictions,count,norms)
    def test_missing_or_extra_step(self):
        for rows in (self.rows()[:1],self.rows()+self.rows()[:1]):
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_initial_nonzero_response_refused(self):
        for k,v in (('weight_nonzero',True),('initialization_delta_max',.01)):
            rows=self.rows();rows[0][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_missing_next_step_weight_or_response(self):
        for k,v in (('weight_nonzero',False),('initialization_delta_max',0.),('context_abs_max',0.)):
            rows=self.rows();rows[1][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_detached_bypassed_or_wrong_padding_consumption(self):
        for k in ('generated_query_identity_verified','projected_language_identity_verified','sampling_padding_identity_verified','native_padding_values_verified','both_inputs_undetached_verified','initial_prediction_consumption_verified','static_mask_features_preserved','padding_identity_verified'):
            for v in (False,1):
                rows=self.rows();rows[1][k]=v
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_invalid_reference_tokens_queries_channels(self):
        for k,v in (('reference_expression_rows',[0,6]),('reference_queries',[0,126]),('reference_channels',[0,127,254]),('reference_valid_token_counts',[0,9]),('reference_valid_token_counts',[7,10]),('reference_valid_token_counts',[7.,9]),('expressions',8.),('queries',128.),('tokens',9.),('weight_nonzero',1)):
            rows=self.rows();rows[1][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonfinite_negative_measurements(self):
        for k in ('reference_error_max','initialization_delta_max','context_abs_max'):
            for v in (float('nan'),float('inf'),-.1):
                rows=self.rows();rows[1][k]=v
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_native_train_signature_unchanged(self):
        calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native' and n.func.attr=='train']
        self.assertEqual(len(calls),1);self.assertEqual([ast.unparse(n) for n in calls[0].args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])
if __name__=='__main__':unittest.main()
