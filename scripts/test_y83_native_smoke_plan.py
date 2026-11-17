"""Native evidence rejection controls; fixtures are not GPU execution."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).parent
TREE=ast.parse((ROOT/'probe_y83_native_smoke.py').read_text(encoding='utf-8'),feature_version=(3,8));N={}
nodes=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('check_norm_rows','check_path_calls')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<Y83-evidence-guards>','exec'),N)

class Guards(unittest.TestCase):
    def rows(self):
        return [dict(step=i//6,stage=i%6,original_self_identity=True,original_language_identity=True,sum_input_exact=True,
            both_inputs_undetached=True,private_module_identity=True,original_epsilon=True,float64_LayerNorm_reference=True) for i in range(12)]

    def state(self):
        return dict(qla=[None]*6,self=[None]*6,norm_values=[None]*6,ffn_consumers=list(range(6)),coarse_calls=6,
            queries=[None]*7,indicators=list(range(7)),query_norms=[True]*7,gates=[None]*7)

    def test_valid_schema_only(self):
        N['check_norm_rows'](self.rows());N['state']=self.state();N['check_path_calls']()

    def test_missing_or_extra_paths(self):
        for key in ('qla','self','norm_values','queries','gates'):
            for delta in (-1,1):
                s=self.state();s[key]=[None]*(len(s[key])+delta);N['state']=s
                with self.assertRaises(AssertionError):N['check_path_calls']()

    def test_wrong_or_reordered_FFN_consumers(self):
        for value in ([],list(reversed(range(6)))):
            s=self.state();s['ffn_consumers']=value;N['state']=s
            with self.assertRaises(AssertionError):N['check_path_calls']()

    def test_wrong_original_coarse_gate_count(self):
        s=self.state();s['coarse_calls']=0;N['state']=s
        with self.assertRaises(AssertionError):N['check_path_calls']()

    def test_shared_head_consumers_required(self):
        for key,value in [('indicators',[]),('query_norms',[False]*7)]:
            s=self.state();s[key]=value;N['state']=s
            with self.assertRaises(AssertionError):N['check_path_calls']()

    def test_missing_extra_or_reordered_rows(self):
        for rows in (self.rows()[:6],self.rows()+self.rows()[:1],list(reversed(self.rows()))):
            with self.assertRaises(AssertionError):N['check_norm_rows'](rows)

    def test_wrong_stage_step_or_type(self):
        for key,value in [('step',0),('step',1.),('stage',5),('stage',1.)]:
            rows=self.rows();rows[7][key]=value
            with self.assertRaises(AssertionError):N['check_norm_rows'](rows)

    def test_original_input_identity_undetached(self):
        for key in ('original_self_identity','original_language_identity','sum_input_exact','both_inputs_undetached'):
            for value in (False,1):
                rows=self.rows();rows[7][key]=value
                with self.assertRaises(AssertionError):N['check_norm_rows'](rows)

    def test_private_modules_original_epsilon_and_reference(self):
        for key in ('private_module_identity','original_epsilon','float64_LayerNorm_reference'):
            rows=self.rows();rows[7][key]=False
            with self.assertRaises(AssertionError):N['check_norm_rows'](rows)

    def test_exact_executed_CPU_callbacks_retained(self):
        core=ast.parse((ROOT/'y83_observer_core.py').read_text(encoding='utf-8'))
        expected={n.name:ast.dump(n,include_attributes=False) for n in core.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(len(expected),14)
        for tree in (TREE,ast.parse((ROOT/'probe_y83_cpu.py').read_text(encoding='utf-8'))):
            found={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in expected}
            self.assertEqual(found,expected)

    def test_original_native_train_entry(self):
        calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native' and n.func.attr=='train']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(x) for x in calls[0].args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])

if __name__=='__main__':unittest.main()
