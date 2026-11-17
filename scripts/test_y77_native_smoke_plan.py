"""Reject false geometry path evidence; fixtures are not a native GPU PASS."""
import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).with_name('probe_y77_native_smoke.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE,feature_version=(3,8));N={}
FUNCTIONS=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_native_inputs','check_response_rows')]
assert len(FUNCTIONS)==3
exec(compile(ast.Module(body=FUNCTIONS,type_ignores=[]),'<geometry-query-guards>','exec'),N)

class Guards(unittest.TestCase):
    def args(self):return [1,1,1,list(range(7)),7,[True]*7,4,12]
    def rows(self):
        return [dict(weight_nonzero=bool(i),expressions=8,queries=128,reference_expression_rows=[0,7],reference_queries=[0,127],reference_channels=[0,127,255],
            reference_error_max=1e-7,initialization_delta_max=.001 if i else 0.,relative_position_abs_max=3.,point_mass_total_min=100.,
            generated_query_identity_verified=True,original_scene_position_identity_verified=True,original_padding_identity_verified=True,
            original_mass_storage_and_values_verified=True,native_sampled_coordinates_verified=True,original_SCG_query_position_consumption_verified=True,
            query_inputs_undetached_verified=True,initial_prediction_consumption_verified=True,static_mask_features_preserved=True,padding_identity_verified=True) for i in range(2)]
    def test_valid_schema_only(self):N['check_path_calls'](*self.args());N['check_response_rows'](self.rows())
    def test_missing_extra_or_float_call_counts(self):
        for index,correct in ((0,1),(1,1),(2,1),(4,7),(6,4),(7,12)):
            for v in (correct-1,correct+1,float(correct)):
                args=self.args();args[index]=v
                with self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_reordered_or_missing_prediction_stages(self):
        for v in (list(range(6)),list(range(8)),list(reversed(range(7))),list(map(float,range(7)))):
            args=self.args();args[3]=v
            with self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_missing_or_detached_original_norms(self):
        for v in ([True]*6,[False]*7,[1]*7):
            args=self.args();args[5]=v
            with self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_original_query_scene_and_padding_objects_required(self):
        q,s,p=object(),object(),object();args=[q,q,s,s,p,p,123,123,True];N['check_native_inputs'](*args)
        for i in range(6):
            bad=list(args);bad[i]=object()
            with self.assertRaises(AssertionError):N['check_native_inputs'](*bad)
    def test_mass_storage_or_correspondence_refused(self):
        q,s,p=object(),object(),object()
        for tail in ((124,123,True),(0,0,True),(123.,123,True),(123,123,False),(123,123,1)):
            with self.assertRaises(AssertionError):N['check_native_inputs'](q,q,s,s,p,p,*tail)
    def test_missing_second_step_or_extra_rows(self):
        for rows in (self.rows()[:1],self.rows()+self.rows()[:1]):
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_wrong_initial_or_missing_next_response(self):
        for index,k,v in ((0,'weight_nonzero',True),(0,'initialization_delta_max',.1),(1,'weight_nonzero',False),(1,'initialization_delta_max',0.),(1,'relative_position_abs_max',0.)):
            rows=self.rows();rows[index][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_bypassed_detached_wrong_geometry_or_consumers_refused(self):
        for k in ('generated_query_identity_verified','original_scene_position_identity_verified','original_padding_identity_verified','original_mass_storage_and_values_verified','native_sampled_coordinates_verified','original_SCG_query_position_consumption_verified','query_inputs_undetached_verified','initial_prediction_consumption_verified','static_mask_features_preserved','padding_identity_verified'):
            for v in (False,1):
                rows=self.rows();rows[1][k]=v
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_bad_reference_or_nonfinite_and_empty_mass_refused(self):
        for k,v in (('reference_expression_rows',[0,6]),('reference_queries',[0,126]),('reference_channels',[0,127,254]),('queries',128.),('weight_nonzero',1),('point_mass_total_min',0.)):
            rows=self.rows();rows[1][k]=v
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
        for k in ('reference_error_max','initialization_delta_max','relative_position_abs_max','point_mass_total_min'):
            for v in (float('nan'),float('inf'),-.1):
                rows=self.rows();rows[1][k]=v
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_native_train_signature_unchanged(self):
        calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native' and n.func.attr=='train']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(n) for n in calls[0].args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])

if __name__=='__main__':unittest.main()
