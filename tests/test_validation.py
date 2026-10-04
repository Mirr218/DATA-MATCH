"""Проверки временных границ и весов на искусственных данных, без обучения."""
import unittest

import numpy as np
import pandas as pd

from src.validation import evaluate_method, rmse, rolling_windows


def artificial_tables():
    table = pd.DataFrame([
        {"match_id": f"M{i:03}", "zone": zone,
         "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i - 1),
         "tickets_total": 100, "cap": 200, "season_tickets": 20}
        for i in range(1, 86) for zone in range(1, 9)
    ])
    rows = []
    for start in (35, 52, 69):
        for number, slice_id, days, weight in (
            (start, "early", 2, 1 / 3), (start, "late", 1, 2 / 3),
            (start + 1, "single", 1, 1.0),
        ):
            date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=number - 1)
            rows.extend({"match_id": f"M{number:03}", "zone": zone,
                         "scenario": 1, "slice_id": slice_id,
                         "cutoff": date - pd.Timedelta(days=days), "weight": weight}
                        for zone in range(1, 9))
    return table, pd.DataFrame(rows)


def fixed_predictions(history, batch):
    """Заранее заданные числа для проверки арифметики, не модель."""
    return batch.slice_id.map({"early": 94.0, "late": 97.0, "single": 91.0}).to_numpy()


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.table, self.examples = artificial_tables()

    def test_rmse_and_invalid_inputs(self):
        self.assertAlmostEqual(rmse([0, 0], [3, 4]), np.sqrt(12.5))
        self.assertAlmostEqual(rmse([0, 0], [3, 4], [2, 1]), np.sqrt(34 / 3))
        for actual, predicted, weight in (([], [], None), ([0], [np.nan], None),
                                           ([0, 1], [0], None), ([0], [0], [-1])):
            with self.assertRaises(ValueError):
                rmse(actual, predicted, weight)

    def test_windows_keep_all_zones(self):
        windows = list(rolling_windows(self.table))
        self.assertEqual([len(history) for _, history, _ in windows], [272, 408, 544])
        self.assertEqual([len(checking) for _, _, checking in windows], [136] * 3)
        for _, history, checking in windows:
            self.assertFalse(set(history.match_id) & set(checking.match_id))

    def test_cutoff_and_hidden_answers(self):
        calls = []

        def inspect(history, batch):
            self.assertNotIn("tickets_total", batch)
            self.assertNotIn("cap", batch)
            self.assertNotIn("season_tickets", history)
            self.assertTrue(pd.to_datetime(history.date).lt(batch.cutoff.iloc[0].normalize()).all())
            window_start = batch.window.iloc[0].split("–")[0]
            self.assertTrue(history.match_id.lt(window_start).all())
            calls.append(len(history))
            return fixed_predictions(history, batch)

        evaluate_method(inspect, self.table, self.examples)
        self.assertTrue(calls)
        self.assertEqual(calls[0], 32 * 8)

    def test_repeated_slices_do_not_increase_match_weight(self):
        summary, details = evaluate_method(fixed_predictions, self.table, self.examples)
        overall = summary.loc[summary.zone.eq("all")]

        np.testing.assert_allclose(overall.mse, (18 + 81) / 2)
        self.assertTrue(overall.n_matches.eq(2).all())
        self.assertEqual(len(summary), 27)
        self.assertEqual(len(details), 72)

    def test_reject_invalid_weights_and_incomplete_match(self):
        bad_weight = self.examples.copy()
        bad_weight.loc[bad_weight.slice_id.eq("single"), "weight"] = 0.5
        incomplete = self.examples.drop(index=0)
        for examples in (bad_weight, incomplete):
            with self.assertRaises(ValueError):
                evaluate_method(fixed_predictions, self.table, examples)

    def test_reject_match_day_cutoff_and_missing_window(self):
        late = self.examples.copy()
        late["cutoff"] = pd.Timestamp("2030-01-01")
        missing_window = self.examples.loc[self.examples.match_id.lt("M069")]
        leaked = self.examples.assign(tickets_total=100)
        for examples in (late, missing_window, leaked):
            with self.assertRaises(ValueError):
                evaluate_method(fixed_predictions, self.table, examples)

    def test_prediction_rule_cannot_hide_wrong_prediction_order(self):
        def wrong_order(history, batch):
            return pd.Series(np.zeros(len(batch)), index=batch.index[::-1])

        with self.assertRaises(ValueError):
            evaluate_method(wrong_order, self.table, self.examples,
                            prediction_rule=lambda raw, batch: np.asarray(raw))

    def test_future_targets_cannot_change_predictions(self):
        _, before = evaluate_method(fixed_predictions, self.table, self.examples)
        changed = self.table.copy()
        changed.loc[changed.match_id.isin(self.examples.match_id), "tickets_total"] = 999
        _, after = evaluate_method(fixed_predictions, changed, self.examples)
        np.testing.assert_array_equal(before.prediction, after.prediction)
        self.assertFalse(before.squared_error.equals(after.squared_error))


if __name__ == "__main__":
    unittest.main()
