"""Проверки границ продаж, общей выборки и формул на искусственных данных."""
import unittest

import numpy as np
import pandas as pd

from src.scenarios import historical_prediction_rule, make_scenario_examples
from src.validation import aggregate_scores


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.opening = pd.Timestamp("2023-01-01")
        self.matches = pd.DataFrame({"match_id": ["M035", "M036", "M037"],
                                     "part": ["train"] * 3,
                                     "sales_open": [self.opening] * 3,
                                     "date": [self.opening + pd.Timedelta(days=d) for d in (28, 27, 29)]})
        self.sales = pd.DataFrame({"match_id": ["M035"] * 5, "zone": [1] * 5,
                                   "date": [self.opening + pd.Timedelta(days=d) for d in (0, 6, 7, 26, 27)],
                                   "tickets": [2, 3, 100, 4, 1000]})

    def test_cutoff_excludes_its_day_and_missing_days_mean_zero(self):
        examples = make_scenario_examples(self.matches, self.sales)
        selected = examples.loc[examples.match_id.eq("M035") & examples.zone.eq(1)].set_index("slice_id")
        self.assertEqual(selected.loc["d0", "sold"], 0)
        self.assertEqual(selected.loc["d7", "sold"], 5)
        self.assertEqual(selected.loc["d27", "sold"], 109)
        self.assertEqual(selected.loc["d27", "cutoff"], self.opening + pd.Timedelta(days=27))
        self.assertTrue(examples.loc[examples.zone.eq(2), "sold"].eq(0).all())

    def test_both_slices_use_same_long_matches_and_unit_total_weight(self):
        examples = make_scenario_examples(self.matches, self.sales)
        with_sales = examples.loc[examples.scenario.eq(1)]
        for _, group in with_sales.groupby("slice_id"):
            self.assertEqual(set(group.match_id), {"M035", "M037"})
        np.testing.assert_allclose(with_sales.groupby(["match_id", "zone"]).weight.sum(), 1)
        self.assertEqual(len(examples.loc[examples.scenario.eq(0)]), 24)

    def test_late_sales_cannot_change_observed_feature(self):
        before = make_scenario_examples(self.matches, self.sales)
        changed = self.sales.copy()
        changed.loc[changed.date.ge(self.opening + pd.Timedelta(days=27)), "tickets"] = 99999
        after = make_scenario_examples(self.matches, changed)
        pd.testing.assert_frame_equal(before, after)

    def test_rounding_and_floor_apply_without_historical_ceiling(self):
        batch = pd.DataFrame({"scenario": [0, 0, 1, 1, 1], "sold": [0, 0, 12, 0, 0]})
        predicted = historical_prediction_rule([2.5, -2.5, 3.1, 3.49, 10000.5], batch)
        np.testing.assert_array_equal(predicted, [3, 0, 12, 3, 10001])

    def test_R_mixes_mse_and_S_mixes_ready_R(self):
        windows = ["M035–M051", "M052–M068", "M069–M085"]
        summary = pd.DataFrame([
            {"window": window, "scenario": scenario, "zone": "all", "mse": value}
            for window, errors in zip(windows, ((100, 400), (100, 400), (900, 1600)))
            for scenario, value in enumerate(errors)
        ])
        scores_by_window, scores = aggregate_scores(summary)
        expected_R = np.sqrt([3500 / 17, 3500 / 17, 19500 / 17])
        np.testing.assert_allclose(scores_by_window.R, expected_R)
        self.assertAlmostEqual(scores["S"], .25 * expected_R[0] + .25 * expected_R[1] + .5 * expected_R[2])
        self.assertAlmostEqual(scores["S_0"], 20)
        self.assertAlmostEqual(scores["S_1"], 30)
        with self.assertRaises(ValueError):
            aggregate_scores(summary.iloc[:-1])


if __name__ == "__main__":
    unittest.main()
