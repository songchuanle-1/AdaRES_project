"""Native evidence rejection controls; synthetic fixtures are not GPU results."""
import ast
import copy
from pathlib import Path
import unittest

ROOT=Path(__file__).parent
TREE=ast.parse((ROOT/'probe_y82_native_smoke.py').read_text(encoding='utf-8'),feature_version=(3,8))
N={}
nodes=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('check_path_calls','check_response_rows')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<Y82-evidence-guards>','exec'),N)

class Guards(unittest.TestCase):
    def state(self):
        return dict(coarse=[None]*6,fine=[None]*6,fused=[None]*6,coarse_inputs=[None]*6,fine_inputs=[None]*6,
            queries=[None]*7,gates=[None]*7,qla_consumers=list(range(6)),indicators=list(range(7)),norms=[True]*7)

    def rows(self):
        return [dict(step=i//6,stage=i%6,weight_nonzero=i>=6,fusion_delta_max=.01 if i>=6 else 0.,
            original_coarse_identity=True,original_fine_identity=True,both_inputs_undetached=True,float64_product_projection_reference=True) for i in range(12)]

    def test_valid_schema_only(self):
        N['check_path_calls'](self.state());N['check_response_rows'](self.rows())

    def test_missing_extra_paths(self):
        for key in ('coarse','fine','fused','coarse_inputs','fine_inputs','queries','gates'):
            for n in (-1,1):
                state=self.state();state[key]=[None]*(len(state[key])+n)
                with self.assertRaises(AssertionError):N['check_path_calls'](state)

    def test_wrong_or_reordered_consumers(self):
        for key in ('qla_consumers','indicators'):
            state=self.state();state[key]=list(reversed(state[key]))
            with self.assertRaises(AssertionError):N['check_path_calls'](state)

    def test_missing_query_norm(self):
        state=self.state();state['norms']=[False]*7
        with self.assertRaises(AssertionError):N['check_path_calls'](state)

    def test_missing_extra_or_reordered_rows(self):
        for rows in (self.rows()[:6],self.rows()+self.rows()[:1],list(reversed(self.rows()))):
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_step_stage_types(self):
        for key,value in (('step',0.0),('stage',1.0),('step',0),('stage',5)):
            rows=self.rows();rows[7][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_wrong_branch_detach_or_formula(self):
        for key in ('original_coarse_identity','original_fine_identity','both_inputs_undetached','float64_product_projection_reference'):
            for value in (False,1):
                rows=self.rows();rows[7][key]=value
                with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_zero_then_nonzero_response(self):
        for i,key,value in ((0,'weight_nonzero',True),(0,'fusion_delta_max',.1),(7,'weight_nonzero',False),(7,'fusion_delta_max',0.),(7,'weight_nonzero',1)):
            rows=self.rows();rows[i][key]=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_nonfinite_or_negative(self):
        for value in (float('nan'),float('inf'),-.1):
            rows=self.rows();rows[7]['fusion_delta_max']=value
            with self.assertRaises(AssertionError):N['check_response_rows'](rows)

    def test_exact_executed_CPU_callbacks_retained(self):
        core=ast.parse((ROOT/'y82_observer_core.py').read_text(encoding='utf-8'))
        cpu=ast.parse((ROOT/'probe_y82_cpu.py').read_text(encoding='utf-8'))
        expected={n.name:ast.dump(n,include_attributes=False) for n in core.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(len(expected),12)
        for tree in (cpu,TREE):
            found={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in expected}
            self.assertEqual(found,expected)

    def test_original_native_train_entry(self):
        calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native' and n.func.attr=='train']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(x) for x in calls[0].args],['1','model','FirstTwoBatches()','optimizer','scheduler','cfg','logger','Writer()'])

if __name__=='__main__':unittest.main()
