"""Structural regressions for the independent exact-class Y56 package."""

import ast
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
CANDIDATE = ROOT / 'AdaRES/ipdn/model_paper5/adares_y56_unique_competitors'


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
        name = 'adares_y56_unique_competitors_res.yaml'
        config = ROOT / 'AdaRES/configs_song/model_paper5' / name
        parent = config.with_name('adares_w2_pointmass_competition_res.yaml')
        self.assertEqual(yaml.safe_load(config.read_text(encoding='utf-8')),
                         yaml.safe_load(parent.read_text(encoding='utf-8')))
        self.assertEqual(len(list((ROOT / 'AdaRES').rglob(name))), 1)

    def test_exact_class_path_and_unchanged_parameter_constructors(self):
        tree = ast.parse((CANDIDATE / 'dec.py').read_text(encoding='utf-8'))
        head = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'QueryCompetitionIndicator')
        source = ast.unparse(head)
        self.assertEqual(source.count('3 * d_model + 1'), 2)
        self.assertIn('torch.unique(', source)
        self.assertIn('query[index, duplicated].detach()', source)
        self.assertIn('detached_query.unsqueeze(2) == detached_query.unsqueeze(1)', source)
        self.assertIn('if not duplicate_flags[index]', source)
        self.assertIn('current.masked_fill(same_prototype, 0.0)', source)
        self.assertIn('counts[inverse]', source)
        self.assertNotIn('topk', source)
        parent = ast.parse((PARENT / 'dec.py').read_text(encoding='utf-8'))
        parent_head = next(n for n in parent.body if isinstance(n, ast.ClassDef)
                           and n.name == 'QueryCompetitionIndicator')
        constructors = [next(n for n in h.body if isinstance(n, ast.FunctionDef)
                             and n.name == '__init__') for h in (parent_head, head)]
        self.assertEqual(*(ast.dump(n, include_attributes=False) for n in constructors))


if __name__ == '__main__':
    unittest.main(verbosity=2)
