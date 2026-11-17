"""Focused regressions for the frozen diagnostic analysis, not efficacy tests."""

import ast
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "scripts/probe_y57_full_input_signal_6deb.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("predict", "bootstrap")]
namespace = {"np": np}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
predict, bootstrap = namespace["predict"], namespace["bootstrap"]


class DiagnosticAnalysisTests(unittest.TestCase):
    def test_constant_inputs_return_training_mean(self):
        actual = predict(np.zeros((8, 3)), np.arange(8), np.ones((4, 3)) * 1000)
        np.testing.assert_allclose(actual, np.repeat(3.5, 4), atol=1e-12)

    def test_duplicate_features_have_finite_equal_predictions(self):
        x = np.arange(12, dtype=float)
        actual = predict(np.column_stack((x, x)), x / 11,
                         np.column_stack((x, x)))
        self.assertTrue(np.isfinite(actual).all())
        self.assertLess(float(np.mean((actual - x / 11) ** 2)), 0.001)

    def test_prediction_is_independent_of_other_test_rows(self):
        x = np.arange(12, dtype=float)[:, None]
        alone = predict(x, np.arange(12) / 11, np.array([[4.0]]))
        mixed = predict(x, np.arange(12) / 11, np.array([[4.0], [1e10]]))
        np.testing.assert_allclose(alone[0], mixed[0], rtol=0, atol=1e-12)

    def test_scene_bootstrap_constant_and_mean(self):
        result = bootstrap(np.repeat(0.25, 8))
        self.assertEqual(result["mean"], 0.25)
        self.assertEqual(result["descriptive_scene_bootstrap95"], [0.25, 0.25])
        result = bootstrap(np.arange(8, dtype=float))
        self.assertEqual(result["mean"], 3.5)
        self.assertLess(result["descriptive_scene_bootstrap95"][0], 3.5)
        self.assertGreater(result["descriptive_scene_bootstrap95"][1], 3.5)

    def test_vector_negative_shifts_queries_not_channels(self):
        value = np.arange(12).reshape(4, 3)
        shifted = np.roll(value, 1, axis=0)
        np.testing.assert_array_equal(shifted[0], value[-1])
        np.testing.assert_array_equal(np.sort(shifted, axis=0), np.sort(value, axis=0))

    def test_native_followup_does_not_recursively_invoke_hooks(self):
        path = ROOT / "scripts/probe_y56_actual_transfer_6deb.py"
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        native = next(n for n in ast.walk(parsed)
                      if isinstance(n, ast.FunctionDef) and n.name == "native")
        calls = [n for n in ast.walk(native) if isinstance(n, ast.Call)]
        self.assertTrue(any(isinstance(n.func, ast.Attribute)
                            and n.func.attr == "forward" for n in calls))
        self.assertFalse(any(isinstance(n.func, ast.Name) and n.func.id == "module"
                             for n in calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
