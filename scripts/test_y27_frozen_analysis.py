"""Synthetic arithmetic checks, not scientific efficacy evidence."""

import copy
import unittest

from analyze_y27_frozen_terminal import comparison


def capture(value, alternating_repeat=False):
    rows = [dict(scan_id="scene%03d" % (i % 141), object_ids=[i], ann_id=str(i),
                 piou=value, target_points=1000, size="small") for i in range(9508)]
    second = copy.deepcopy(rows)
    if alternating_repeat:
        for i, row in enumerate(second):
            row["piou"] += 0.01 if i % 2 == 0 else -0.01
    return dict(passes=[dict(summary=dict(expressions=9508, scenes=141), rows=rows),
                        dict(summary=dict(expressions=9508, scenes=141), rows=second)])


class FrozenAnalysisTests(unittest.TestCase):
    def test_negative_ci_uses_frozen_harm_rule(self):
        result = comparison(capture(0.4), capture(0.5))["all"]
        self.assertTrue(result["frozen_material_harm_ci_upper_below_zero"])
        self.assertLess(result["scene_clustered_ci95"][1], 0)

    def test_zero_ci_is_not_harm_or_equivalence_proof(self):
        result = comparison(capture(0.5), capture(0.5))["all"]
        self.assertFalse(result["frozen_material_harm_ci_upper_below_zero"])
        self.assertEqual(result["scene_clustered_ci95"], [0, 0])

    def test_opposite_repeat_errors_do_not_cancel_absolute_noise(self):
        result = comparison(capture(0.5, True), capture(0.5))["all"]
        self.assertAlmostEqual(result["observed_group_mean_absolute_repeat_difference"], 0.01)
        self.assertAlmostEqual(result["historical_abs_mean_repeat_difference"], 0)

    def test_population_shape_fails_closed(self):
        changed = capture(0.5)
        changed["passes"][0]["summary"]["expressions"] -= 1
        with self.assertRaises(AssertionError):
            comparison(changed, capture(0.5))

    def test_identity_mismatch_fails_closed(self):
        changed = capture(0.5)
        changed["passes"][0]["rows"][0]["ann_id"] = "different"
        with self.assertRaises(ValueError):
            comparison(changed, capture(0.5))


if __name__ == "__main__":
    unittest.main()
