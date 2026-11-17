"""Only the seven exact local Y71 additions may extend the preparation HEAD."""
import ast
from pathlib import Path
import unittest
S=Path(__file__).with_name('observe_y68_with_local_y71.py').read_text(encoding='utf-8');T=ast.parse(S);N={}
F=[n for n in T.body if isinstance(n,ast.FunctionDef) and n.name=='validate_local_y71'];assert len(F)==1
exec(compile(ast.Module(body=F,type_ignores=[]),'<local-Y71-guard>','exec'),N)
class LocalPreparation(unittest.TestCase):
    def rows(self):
        names=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py');package='adares_y71_visual_product_fusion'
        return [['A','AdaRES/ipdn/model_paper5/'+package+'/'+n] for n in names]+[['A','AdaRES/configs_song/model_paper5/'+package+'_res.yaml']]
    def test_exact_additions_only(self):N['validate_local_y71'](self.rows())
    def test_existing_source_edit_or_rename_refused(self):
        for status in ('M','D','R100'):
            rows=self.rows();rows[0][0]=status
            with self.assertRaises(AssertionError):N['validate_local_y71'](rows)
    def test_missing_extra_or_wrong_path_refused(self):
        for rows in (self.rows()[:-1],self.rows()+[['A','extra.py']],self.rows()[:-1]+[['A','AdaRES/tools/train.py']]):
            with self.assertRaises(AssertionError):N['validate_local_y71'](rows)
if __name__=='__main__':unittest.main()
