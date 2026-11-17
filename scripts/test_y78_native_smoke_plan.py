"""Reject missing/false Y78 native evidence; fixtures are not GPU execution."""
import ast,copy,unittest
from pathlib import Path
SOURCE=Path(__file__).with_name('probe_y78_native_smoke.py').read_text(encoding='utf8');TREE=ast.parse(SOURCE,feature_version=(3,8));N={}
FUNCTIONS=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_response_rows')]
assert len(FUNCTIONS)==2
exec(compile(ast.Module(body=FUNCTIONS,type_ignores=[]),'<Y78-native-evidence-guards>','exec'),N)
class Guards(unittest.TestCase):
    def rows(self):
        return [dict(step=i//6,stage=i%6+1,weight_nonzero=i>=6,raw_query_previous_padding_identities=True,previous_undetached=True,
            static_mask_features_preserved=True,independent_formula_checked=True,residual_delta_max=.01 if i>=6 else 0.) for i in range(12)]
    def test_valid_schema_only(self):N['check_path_calls'](7,7,7,6,7,6,[True]*7);N['check_response_rows'](self.rows())
    def test_missing_extra_paths(self):
        for i in range(6):
            for wrong in (0,8,7.0,True):
                args=[7,7,7,6,7,6,[True]*7];args[i]=wrong
                if type(wrong) is int and wrong==args[i] and wrong in (7,6):continue
                with self.assertRaises(AssertionError):N['check_path_calls'](*args)
    def test_missing_detached_or_nonboolean_norms(self):
        for bad in ([True]*6,[1]*7,[False]*7):
            with self.assertRaises(AssertionError):N['check_path_calls'](7,7,7,6,7,6,bad)
    def test_missing_extra_or_reordered_rows(self):
        for bad in (self.rows()[:6],self.rows()+self.rows()[:1],list(reversed(self.rows()))):
            with self.assertRaises(AssertionError):N['check_response_rows'](bad)
    def test_wrong_stage_or_step_types(self):
        for key,value in (('step',0.0),('stage',2.0),('step',0),('stage',6)):
            rows=self.rows();rows[7][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_detached_bypassed_wrong_mask_or_padding_refused(self):
        for key in ('raw_query_previous_padding_identities','previous_undetached','static_mask_features_preserved','independent_formula_checked'):
            for value in (False,1):
                rows=self.rows();rows[7][key]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_initial_nonzero_or_missing_second_response(self):
        for index,key,value in ((0,'weight_nonzero',True),(0,'residual_delta_max',.1),(7,'weight_nonzero',False),(7,'residual_delta_max',0.)):
            rows=self.rows();rows[index][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonboolean_response_refused(self):
        rows=self.rows();rows[7]['weight_nonzero']=1
        with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_nonfinite_or_negative_delta_refused(self):
        for value in (float('nan'),float('inf'),-.1):
            rows=self.rows();rows[7]['residual_delta_max']=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)
    def test_exact_CPU_tested_callback_core_retained(self):
        cpu=Path(__file__).with_name('probe_y78_cpu_attempt3.py').read_text(encoding='utf8');tree=ast.parse(cpu)
        names=('checked_prediction','capture_residual','checked_refined','before_indicator','before_coarse')
        old={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
        new={n.name:ast.dump(n,include_attributes=False) for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in names}
        self.assertEqual(old,new);self.assertEqual(len(new),5)
    def test_native_train_signature_unchanged(self):
        calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native' and n.func.attr=='train']
        self.assertEqual(len(calls),1);self.assertEqual([ast.unparse(n) for n in calls[0].args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])
if __name__=='__main__':unittest.main()
