"""Reject incomplete query-to-visual native evidence; fixtures are not GPU PASS."""
import ast,copy,unittest
from pathlib import Path
SOURCE=Path(__file__).with_name('probe_y80_native_smoke.py').read_text(encoding='utf8');TREE=ast.parse(SOURCE,feature_version=(3,8));N={}
FUNCTIONS=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_response_rows')];assert len(FUNCTIONS)==2
exec(compile(ast.Module(body=FUNCTIONS,type_ignores=[]),'<Y80-native-evidence-guards>','exec'),N)
class Guards(unittest.TestCase):
    def args(self):return [7,6,6,list(range(6)),list(range(6)),list(range(7)),[True]*7]
    def rows(self):
        return [dict(step=i//6,stage=i%6,weight_nonzero=i>=6,feedback_delta_max=.01 if i>=6 else 0.,current_query_identity=True,original_visual_ffn_identity=True,
            original_visual_padding_identity=True,both_inputs_undetached=True,padded_output_identity=True,static_mask_features_preserved=True,float64_attention_reference_checked=True) for i in range(12)]
    def test_valid_schema_only(self):N['check_path_calls'](*self.args());N['check_response_rows'](self.rows())
    def test_missing_extra_or_float_path_counts(self):
        for i,correct in enumerate((7,6,6)):
            for bad in (0,correct+1,float(correct),True):
                args=self.args();args[i]=bad
                with self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_missing_reordered_or_noninteger_consumers(self):
        for i,count in ((3,6),(4,6),(5,7)):
            for bad in (list(range(count-1)),list(reversed(range(count))),list(map(float,range(count)))):
                args=self.args();args[i]=bad
                with self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_detached_missing_or_nonboolean_norms(self):
        for bad in ([True]*6,[False]*7,[1]*7):
            args=self.args();args[6]=bad
            with self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_missing_extra_or_reordered_rows(self):
        for bad in (self.rows()[:6],self.rows()+self.rows()[:1],list(reversed(self.rows()))):
            with self.assertRaises(AssertionError):N['check_response_rows'](bad)
    def test_wrong_stage_step_or_types(self):
        for key,value in (('step',0.0),('stage',1.0),('step',0),('stage',5)):
            rows=self.rows();rows[7][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_detached_wrong_query_visual_padding_or_bypass_refused(self):
        for key in ('current_query_identity','original_visual_ffn_identity','original_visual_padding_identity','both_inputs_undetached','padded_output_identity','static_mask_features_preserved','float64_attention_reference_checked'):
            for value in (False,1):
                rows=self.rows();rows[7][key]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonzero_first_or_missing_next_response(self):
        for index,key,value in ((0,'weight_nonzero',True),(0,'feedback_delta_max',.1),(7,'weight_nonzero',False),(7,'feedback_delta_max',0.),(7,'weight_nonzero',1)):
            rows=self.rows();rows[index][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonfinite_or_negative_delta(self):
        for value in (float('nan'),float('inf'),-.1):
            rows=self.rows();rows[7]['feedback_delta_max']=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_exact_actual_CPU_callback_core_retained(self):
        cpu=Path(__file__).with_name('probe_y80_cpu.py').read_text(encoding='utf8');tree=ast.parse(cpu)
        names=('checked_prediction','capture_visual_ffn','capture_feedback','before_qia','before_lia','before_indicator','capture_query_norm')
        old={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
        new={n.name:ast.dump(n,include_attributes=False) for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in names}
        self.assertEqual(old,new);self.assertEqual(len(new),7)
    def test_native_train_signature_unchanged(self):
        calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native' and n.func.attr=='train']
        self.assertEqual(len(calls),1);self.assertEqual([ast.unparse(n) for n in calls[0].args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])
if __name__=='__main__':unittest.main()
