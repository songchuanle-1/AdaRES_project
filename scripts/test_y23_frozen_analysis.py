"""Controls for the preregistered repeat-noise definition."""

import unittest

from analyze_y23_frozen_terminal import comparison


def capture(first, second):
    def rows(values):
        return [dict(scan_id=str(i), object_ids=[1], ann_id="1", target_points=1,
                     size="small", piou=value) for i, value in enumerate(values)]
    return dict(passes=[dict(rows=rows(first)), dict(rows=rows(second))])


class FrozenNoiseControls(unittest.TestCase):
    def test_opposite_deviations_do_not_cancel(self):
        result = comparison(capture([.5, .5], [.7, .3]), capture([.5, .5], [.5, .5]))
        self.assertAlmostEqual(result["all"]["observed_group_mean_absolute_repeat_difference"], .2)
        self.assertAlmostEqual(result["all"]["historical_abs_mean_repeat_difference"], 0.)
        self.assertFalse(result["all"]["material_harm_within_observed_noise"])
        self.assertFalse(result["all"]["positive_difference_beyond_observed_noise"])
        self.assertEqual(result["medium"]["status"], "insufficient independent scenes")

    def test_zero_difference_control(self):
        result = comparison(capture([.4, .6], [.4, .6]), capture([.4, .6], [.4, .6]))["all"]
        self.assertEqual(result["observed_group_mean_absolute_repeat_difference"], 0.)
        self.assertEqual(result["scene_clustered_ci95"], [0., 0.])

    def test_harm_requires_crossing_noise_boundary(self):
        reference = capture([.8, .8], [.8, .8])
        noisy = comparison(capture([.6, .6], [.9, .3]), reference)["all"]
        quiet = comparison(capture([.6, .6], [.6, .6]), reference)["all"]
        self.assertFalse(noisy["material_harm_within_observed_noise"])
        self.assertTrue(quiet["material_harm_within_observed_noise"])

    def test_population_mismatch_fails(self):
        with self.assertRaises(ValueError):
            comparison(capture([.4, .6], [.4, .6]), capture([.4], [.4]))


if __name__ == "__main__":
    unittest.main()
