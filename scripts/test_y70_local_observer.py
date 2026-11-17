"""Only the seven tested new Y70 files may extend the local preparation HEAD."""
import ast
from pathlib import Path
import unittest
S=Path(__file__).with_name('observe_server2_with_local_y70.py').read_text(encoding='utf-8')
T=ast.parse(S);N={}
names=('expected_additions','validate_y70_preparation')
nodes=[n for n in T.body if isinstance(n,ast.FunctionDef) and n.name in names]
nodes += [n for n in T.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODULE_FILES' for t in n.targets)]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<local-y70-extension-guards>','exec'),N)
class LocalPreparation(unittest.TestCase):
    def rows(self):return [['A',p] for p in sorted(N['expected_additions']('adares_y70_query_scale_fusion'))]
    def test_exact_new_files_only(self):N['validate_y70_preparation'](self.rows())
    def test_existing_source_edit_or_rename_refused(self):
        for status in ('M','D','R100'):
            rows=self.rows();rows[0][0]=status
            with self.assertRaises(AssertionError):N['validate_y70_preparation'](rows)
    def test_missing_extra_or_wrong_candidate_file_refused(self):
        for rows in (self.rows()[:-1],self.rows()+[['A','other.py']],self.rows()[:-1]+[['A','AdaRES/tools/train.py']]):
            with self.assertRaises(AssertionError):N['validate_y70_preparation'](rows)
if __name__=='__main__':unittest.main()
