import unittest
from observe_y65_server2_additive_phase import validate_additive_marker
class Guards(unittest.TestCase):
    def setUp(self):
        self.old='882d61537fe7ffcb383c8f51f12fea36d1e53a6d';self.new='db35f2870d252e6888e3525b142785938f00ad4a'
        self.paths=['AdaRES/ipdn/model_paper5/adares_y67_stage_readout_norm/'+n for n in ('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')]+['AdaRES/configs_song/model_paper5/adares_y67_stage_readout_norm_res.yaml']
    def test_exact_additive_manifest_only(self):self.assertEqual(validate_additive_marker(self.old,self.new,self.paths),[self.old,self.new])
    def test_unknown_marker_refused(self):
        with self.assertRaises(AssertionError):validate_additive_marker(self.old,'a'*40,self.paths)
    def test_missing_or_extra_path_refused(self):
        for p in (self.paths[:-1],self.paths+['AdaRES/tools/train.py']):
            with self.assertRaises(AssertionError):validate_additive_marker(self.old,self.new,p)
    def test_changed_existing_source_or_original_refused(self):
        bad=self.paths[:];bad[-1]='AdaRES/ipdn/model_paper5/adares_y65_confidence_feedback/dec.py'
        with self.assertRaises(AssertionError):validate_additive_marker(self.old,self.new,bad)
        with self.assertRaises(AssertionError):validate_additive_marker('a'*40,self.new,self.paths)
if __name__=='__main__':unittest.main()
