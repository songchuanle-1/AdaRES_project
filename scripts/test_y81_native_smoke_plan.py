"""Native Y81 evidence rejection controls; schema fixtures are not GPU execution."""
import ast
import copy
from pathlib import Path
import unittest

SOURCE=Path(__file__).with_name('probe_y81_native_smoke.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE,feature_version=(3,8));N={}
functions=[node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name in ('check_path_calls','check_response_rows')]
assert len(functions)==2
exec(compile(ast.Module(body=functions,type_ignores=[]),'<Y81-native-evidence-guards>','exec'),N)


class Guards(unittest.TestCase):
    def rows(self):
        return [dict(step=index//6,stage=index%6+1,weight_nonzero=index>=6,mask_feature_delta_max=.01 if index>=6 else 0.,
            original_static_mask_base_identity=True,current_refined_query_identity=True,original_padding_identity=True,
            both_inputs_undetached=True,padded_output_identity=True,float64_attention_reference_checked=True) for index in range(12)]

    def test_valid_schema_only(self):
        N['check_path_calls'](7,6,list(range(1,7)),6,list(range(7)),[True]*7);N['check_response_rows'](self.rows())

    def test_missing_counts_or_wrong_types(self):
        for index in (0,1,3):
            for value in (0,7.0,True):
                args=[7,6,list(range(1,7)),6,list(range(7)),[True]*7];args[index]=value
                with self.assertRaises(AssertionError):N['check_path_calls'](*args)

    def test_missing_or_reordered_consumers(self):
        for index in (2,4):
            for value in ([],list(reversed(range(7))),[1.]*6):
                args=[7,6,list(range(1,7)),6,list(range(7)),[True]*7];args[index]=value
                with self.assertRaises(AssertionError):N['check_path_calls'](*args)

    def test_wrong_shared_norm_consumption(self):
        for value in ([True]*6,[1]*7,[False]*7):
            with self.assertRaises(AssertionError):N['check_path_calls'](7,6,list(range(1,7)),6,list(range(7)),value)

    def test_missing_extra_or_reordered_rows(self):
        for rows in (self.rows()[:6],self.rows()+self.rows()[:1],list(reversed(self.rows()))):
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_wrong_stage_or_step(self):
        for key,value in [('step',0),('step',1.),('stage',1),('stage',2.)]:
            rows=self.rows();rows[7][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_detached_or_wrong_source_identities(self):
        for key in ('original_static_mask_base_identity','current_refined_query_identity','original_padding_identity','both_inputs_undetached','padded_output_identity','float64_attention_reference_checked'):
            for value in (False,1):
                rows=self.rows();rows[7][key]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_zero_then_nonzero_response(self):
        for index,key,value in [(0,'weight_nonzero',True),(0,'mask_feature_delta_max',.1),(7,'weight_nonzero',False),(7,'weight_nonzero',1),(7,'mask_feature_delta_max',0.)]:
            rows=self.rows();rows[index][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_nonfinite_or_negative_delta(self):
        for value in (float('nan'),float('inf'),-.1):
            rows=self.rows();rows[7]['mask_feature_delta_max']=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_actual_CPU_core_retained_exact(self):
        cpu=ast.parse(Path(__file__).with_name('probe_y81_cpu.py').read_text(encoding='utf-8'))
        names=('checked_prediction','capture_refinement','before_coarse','before_indicator','capture_query_norm')
        old={node.name:ast.dump(node,include_attributes=False) for node in cpu.body if isinstance(node,ast.FunctionDef) and node.name in names}
        new={node.name:ast.dump(node,include_attributes=False) for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name in names}
        self.assertEqual(old,new);self.assertEqual(len(new),5)

    def test_native_train_signature_unchanged(self):
        calls=[node for node in ast.walk(TREE) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='native' and node.func.attr=='train']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(node) for node in calls[0].args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])


if __name__=='__main__':unittest.main()
