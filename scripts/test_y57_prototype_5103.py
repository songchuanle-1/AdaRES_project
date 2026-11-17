"""Structural regressions for the exact independently realized Y57 package."""

import ast
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
CANDIDATE = ROOT / 'AdaRES/ipdn/model_paper5/adares_y57_peer_second_moment'


class PrototypeStructureTests(unittest.TestCase):
    def test_other_five_files_are_byte_equal(self):
        names = [p.name for p in CANDIDATE.glob('*.py')]
        self.assertEqual(len(names), 6)
        for name in names:
            if name != 'dec.py':
                self.assertEqual((CANDIDATE / name).read_bytes(), (PARENT / name).read_bytes())

    def test_all_other_decoder_nodes_identical(self):
        trees = [ast.parse((p / 'dec.py').read_text(encoding='utf-8')) for p in (PARENT, CANDIDATE)]
        nodes = [[ast.dump(n, include_attributes=False) for n in t.body if not (
            isinstance(n, ast.ClassDef) and n.name == 'QueryCompetitionIndicator')] for t in trees]
        self.assertEqual(*nodes)

    def test_no_cross_variant_imports(self):
        for path in CANDIDATE.glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn('adares_', node.module or '')

    def test_config_is_unchanged_and_unique(self):
        name = 'adares_y57_peer_second_moment_res.yaml'
        config = ROOT / 'AdaRES/configs_song/model_paper5' / name
        parent = config.with_name('adares_w2_pointmass_competition_res.yaml')
        self.assertEqual(yaml.safe_load(config.read_text(encoding='utf-8')),
                         yaml.safe_load(parent.read_text(encoding='utf-8')))
        self.assertEqual(len(list((ROOT / 'AdaRES').rglob(name))), 1)

    def test_full_channel_variance_path_without_channel_search(self):
        tree = ast.parse((CANDIDATE / 'dec.py').read_text(encoding='utf-8'))
        head = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'QueryCompetitionIndicator')
        source = ast.unparse(head)
        self.assertEqual(source.count('4 * d_model + 1'), 2)
        self.assertIn('normalized_query.square()', source)
        self.assertIn('second_moment - context.square()', source)
        self.assertIn('max_overlap, peer_variance', source)
        self.assertNotIn('topk', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
