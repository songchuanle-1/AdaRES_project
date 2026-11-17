"""Exact independent source and current-resource guards, not efficacy tests."""
import unittest
import yaml
from prepare_y88_mask_query_quadratic import ROOT, MODULE, files, PARENT
from prepare_y60_stage_heads import ref
from y88_deployment import Y84, Y85, Y87, Y88, contract, git, paths, validate_delta


class SourceGuards(unittest.TestCase):
    def test_reverse_only_three_edits_restores_original_W2(self):
        candidate = files()[2].read_text(encoding='utf-8')
        parent = (files()[2].parent.parent / PARENT / 'dec.py').read_text(encoding='utf-8')
        edits = [(MODULE + 'class DEC(nn.Module):', 'class DEC(nn.Module):'),
            ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.mask_query_quadratic = MaskQueryQuadratic(d_model)',
             '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)'),
            ('        pred_masks, attn_masks = self.get_mask(self.mask_query_quadratic(query), mask_feats, batch_mask)',
             '        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)')]
        for old, new in edits:
            self.assertEqual(candidate.count(old), 1)
            candidate = candidate.replace(old, new)
        self.assertEqual(candidate, parent)

    def test_other_native_model_files_and_YAML_identical(self):
        for path in files()[1:-1]:
            if path.name != 'dec.py':
                self.assertEqual(path.read_bytes(), (path.parent.parent / PARENT / path.name).read_bytes())
        config = ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')
        self.assertEqual(files()[-1].read_bytes(), config.read_bytes())

    def test_only_seven_committed_additions_no_failed_Y86(self):
        self.assertEqual(git('rev-parse', Y88 + '^').decode().strip(), Y87)
        rows = [row.split('\t') for row in git('diff', '--name-status', Y87, Y88).decode().splitlines()]
        validate_delta(rows, 'AdaRES-Y88')
        self.assertNotIn('AdaRES-Y86', __import__('y88_deployment').ADDITIONS.values())
        for path in paths('AdaRES-Y88'):
            self.assertEqual((ROOT / path).read_bytes(), git('show', Y88 + ':' + path))

    def test_original_sources_and_all_additive_edges(self):
        for cid, commit in (('AdaRES-Y84', Y84), ('AdaRES-Y85', Y85), ('AdaRES-Y87', Y87)):
            self.assertIn(Y88, contract(commit, Y88, cid))
        for commit, head in ((Y88, Y87), ('0' * 40, Y88)):
            with self.assertRaises(AssertionError):
                contract(commit, head, 'AdaRES-Y87')

    def test_current_pause_goal_and_whitelist_unchanged(self):
        servers = yaml.safe_load((ROOT / 'SERVERS.yaml').read_bytes())['servers']
        self.assertEqual(servers['server-1']['allowed_gpus'], [0])
        self.assertEqual(servers['server-2']['allowed_gpus'], [0])
        self.assertEqual(servers['server-6']['allowed_gpus'], [])
        self.assertEqual(ref('pipeline/goal.json')['sha256'],
                         '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a')
        self.assertEqual(ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256'],
                         '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45')


if __name__ == '__main__':
    unittest.main()
