"""Временные границы ранних продаж на искусственных данных."""
import unittest

import numpy as np
import pandas as pd

from src.early_sales import (available_history, batch_timing, fallback_reason,
                             historical_analogs, make_early_sales_method,
                             predict_early_sales, prepare_sources, sales_at_cutoffs)
from src.scenarios import historical_prediction_rule


class EarlySalesTests(unittest.TestCase):
    def setUp(self):
        ids = ["M001", "M002", "M003", "M034", "M035", "M036"]
        self.matches = pd.DataFrame({
            "match_id": ids,
            "date": pd.to_datetime(["2023-01-10", "2023-01-20", "2023-01-15",
                                     "2023-01-31", "2023-02-05", "2023-02-01"]),
            "sales_open": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-01-01",
                                           "2023-01-01", "2023-01-20", "2023-01-20"]),
        })
        self.sales = pd.DataFrame({
            "match_id": ["M001"] * 3 + ["M002"] * 2 + ["M035"] * 3,
            "zone": [1] * 8,
            "date": pd.to_datetime(["2023-01-01", "2023-01-07", "2023-01-08",
                                     "2023-01-15", "2023-01-19", "2023-01-20",
                                     "2023-01-26", "2023-01-27"]),
            "tickets": [10, 20, 70, 40, 60, 3, 4, 1000],
        })
        self.calendar, self.daily = prepare_sources(self.matches, self.sales)
        self.history = self.matches[["match_id", "date"]].assign(zone=1, tickets_total=100)
        self.history.loc[self.history.match_id.eq("M003"), "tickets_total"] = 0
        self.batch = pd.DataFrame({"match_id": ["M035"], "zone": [1], "scenario": [1],
                                   "slice_id": ["d7"], "cutoff": [pd.Timestamp("2023-01-27")],
                                   "window": ["M035–M051"], "sold": [7],
                                   "match_date": [pd.Timestamp("2023-02-05")]}, index=[19])

    def test_current_sales_exclude_cutoff_day_and_preserve_order(self):
        query = batch_timing(self.batch, self.calendar)
        self.assertEqual(query.days_elapsed.iloc[0], 7)
        self.assertEqual(query.days_remaining.iloc[0], 9)
        self.assertEqual(query.sales_duration.iloc[0], 16)
        observed = sales_at_cutoffs(query, self.daily)
        pd.testing.assert_series_equal(observed, pd.Series([7], index=[19], name="sold"))
        pair = pd.concat([query, query.assign(zone=2), query.assign(cutoff=query.sales_open)])
        np.testing.assert_array_equal(sales_at_cutoffs(pair, self.daily), [7, 0, 0])

    def test_unfinished_and_entire_checking_window_are_excluded(self):
        query = batch_timing(self.batch, self.calendar)
        self.assertEqual(set(available_history(self.history, query).match_id),
                         {"M001", "M002", "M003"})
        later = query.assign(cutoff=pd.Timestamp("2023-02-03"))
        self.assertEqual(set(available_history(self.history, later).match_id),
                         {"M001", "M002", "M003", "M034"})
        same_day = query.assign(cutoff=pd.Timestamp("2023-01-20"))
        self.assertNotIn("M002", set(available_history(self.history, same_day).match_id))

    def test_A_uses_elapsed_days_excludes_short_and_zero_total(self):
        row = batch_timing(self.batch, self.calendar).iloc[0]
        analogs = historical_analogs(self.history, row, self.calendar, self.daily, "A")
        self.assertEqual(list(analogs.match_id), ["M001"])
        self.assertEqual(analogs.cutoff.iloc[0], pd.Timestamp("2023-01-08"))
        self.assertEqual(analogs.sold.iloc[0], 30)
        self.assertAlmostEqual(analogs.fraction.iloc[0], .3)

    def test_B_allows_opening_morning_but_not_before_opening(self):
        row = batch_timing(self.batch, self.calendar).iloc[0]
        analogs = historical_analogs(self.history, row, self.calendar, self.daily, "B")
        self.assertEqual(list(analogs.match_id), ["M001"])
        self.assertEqual(analogs.cutoff.iloc[0], pd.Timestamp("2023-01-01"))
        self.assertEqual(analogs.fraction.iloc[0], 0)

    def test_A_rejects_historical_cutoff_on_match_day(self):
        batch = self.batch.assign(cutoff=pd.Timestamp("2023-01-29"))
        row = batch_timing(batch, self.calendar).iloc[0]
        analogs = historical_analogs(self.history, row, self.calendar, self.daily, "A")
        self.assertTrue(analogs.empty)

    def test_future_sales_and_outside_history_cannot_change_analogs(self):
        row = batch_timing(self.batch, self.calendar).iloc[0]
        before = historical_analogs(self.history, row, self.calendar, self.daily, "A")
        changed = self.daily.copy()
        late = ((changed.match_id.eq("M001") & changed.date.ge("2023-01-08"))
                | changed.match_id.ne("M001"))
        changed.loc[late, "tickets"] = 999999
        after = historical_analogs(self.history, row, self.calendar, changed, "A")
        pd.testing.assert_frame_equal(before, after)
        again = historical_analogs(self.history, row, self.calendar, self.daily, "A")
        pd.testing.assert_frame_equal(before, again)
        restricted = self.history.loc[self.history.match_id.ne("M001")]
        self.assertTrue(historical_analogs(restricted, row, self.calendar, self.daily, "A").empty)

    def test_fixed_fallback_thresholds_and_equalities(self):
        self.assertEqual(fallback_reason(0, np.nan), "no_analogs")
        self.assertEqual(fallback_reason(4, .5), "too_few_analogs")
        self.assertEqual(fallback_reason(5, 0), "zero_fraction")
        self.assertEqual(fallback_reason(5, .149999), "low_fraction")
        self.assertEqual(fallback_reason(5, .15), "")
        self.assertEqual(fallback_reason(5, .7), "")

    def test_sources_and_functions_keep_no_learned_state(self):
        self.assertEqual(set(self.calendar), {"match_id", "date", "sales_open"})
        self.assertEqual(set(self.daily), {"match_id", "zone", "date", "tickets"})
        original = self.daily.copy(deep=True)
        self.sales.loc[:, "tickets"] = 99999
        row = batch_timing(self.batch, self.calendar).iloc[0]
        historical_analogs(self.history, row, self.calendar, self.daily, "A")
        historical_analogs(self.history, row, self.calendar, self.daily, "B")
        pd.testing.assert_frame_equal(self.daily, original)

    def five_analogs(self):
        ids = [f"M{i:03}" for i in range(1, 6)]
        totals, early = [100, 100, 200, 200, 400], [20, 40, 100, 120, 320]
        old = pd.DataFrame({"match_id": ids, "date": pd.Timestamp("2023-01-19"),
                            "sales_open": pd.Timestamp("2023-01-01")})
        calendar = pd.concat([old, self.calendar.loc[self.calendar.match_id.ge("M035")]])
        daily = pd.DataFrame([
            {"match_id": match, "zone": 1, "date": pd.Timestamp(day), "tickets": sold}
            for match, total, first in zip(ids, totals, early)
            for day, sold in (("2023-01-02", first), ("2023-01-08", total - first))
        ])
        daily = pd.concat([daily, self.daily.loc[self.daily.match_id.eq("M035")]])
        history = old[["match_id", "date"]].assign(zone=1, tickets_total=totals)
        return history, calendar.reset_index(drop=True), daily.reset_index(drop=True)

    def test_formula_uses_equal_match_fractions_and_preserves_index(self):
        history, calendar, daily = self.five_analogs()
        for alignment, expected, fraction in (("A", 14, .5), ("B", 7, 1)):
            predicted, report = predict_early_sales(history, self.batch, calendar, daily, alignment)
            self.assertEqual(list(predicted.index), [19])
            self.assertAlmostEqual(predicted.iloc[0], expected)
            self.assertAlmostEqual(report.mean_fraction.iloc[0], fraction)
            self.assertEqual(report.n_analogs.iloc[0], 5)
            self.assertFalse(report.fallback.iloc[0])

    def test_fallback_and_no_sales_keep_baseline_with_common_floor(self):
        history, calendar, daily = self.five_analogs()
        for alignment in ("A", "B"):
            predicted, report = predict_early_sales(history.iloc[:4], self.batch,
                                                    calendar, daily, alignment)
            self.assertAlmostEqual(predicted.iloc[0], 150)
            self.assertTrue(report.fallback.iloc[0])
            no_sales = self.batch.assign(scenario=0, slice_id="d0", sold=0,
                                         cutoff=pd.Timestamp("2023-01-20"))
            predicted, report = predict_early_sales(history, no_sales, calendar, daily, alignment)
            self.assertEqual(predicted.iloc[0], 200)
            self.assertTrue(report.uses_baseline.iloc[0])
            self.assertFalse(report.fallback.iloc[0])
        np.testing.assert_array_equal(historical_prediction_rule([1.5], self.batch.assign(seats=1000)), [7])

    def test_collector_and_previous_calls_cannot_change_predictions(self):
        history, calendar, daily = self.five_analogs()
        records = []
        method = make_early_sales_method(calendar, daily, "A", records)
        before = method(history, self.batch)
        method(history.iloc[:4], self.batch)
        records.clear()
        after = method(history, self.batch)
        pd.testing.assert_series_equal(before, after)
        self.assertEqual(len(records), 1)
        daily.loc[:, "tickets"] = 99999
        pd.testing.assert_series_equal(before, method(history, self.batch))


if __name__ == "__main__":
    unittest.main()
