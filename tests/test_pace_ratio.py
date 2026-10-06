"""Проверки фиксированного C на искусственных данных, без подбора параметров."""
import unittest

import numpy as np
import pandas as pd

from src.early_sales import prepare_sources
from src.pace_ratio import analog_stages, make_pace_ratio_method, predict_pace_ratio


class PaceRatioTests(unittest.TestCase):
    def setUp(self):
        ids = [f"M{i:03}" for i in range(1, 6)] + ["M035"]
        matches = pd.DataFrame({"match_id": ids,
            "date": pd.to_datetime(["2023-01-29"] * 5 + ["2023-03-01"]),
            "sales_open": pd.to_datetime(["2022-12-18"] + ["2023-01-01"] * 4 + ["2023-02-01"])})
        history = matches.iloc[:5][["match_id", "date"]].merge(pd.DataFrame({"zone": range(1, 9)}), how="cross")
        self.history = history.assign(tickets_total=100)
        batch = pd.DataFrame({"zone": range(1, 9)}, index=range(19, 27)).assign(
            match_id="M035", scenario=1, slice_id="d7", cutoff=pd.Timestamp("2023-02-08"),
            window="M035–M051", sold=20, match_date=pd.Timestamp("2023-03-01"))
        self.batch = batch
        records = []
        for row in matches.itertuples():
            day = 11 if row.match_id == "M001" else 7
            for zone in range(1, 9):
                for offset, tickets in ((0, 20 if row.match_id == "M035" else 10), (day, 999)):
                    records.append(dict(match_id=row.match_id, zone=zone,
                                        date=row.sales_open + pd.Timedelta(days=offset), tickets=tickets))
        self.calendar, self.daily = prepare_sources(matches, pd.DataFrame(records))

    def test_common_multiplier_half_trust_and_cutoff_exclusion(self):
        prediction, report, stages = predict_pace_ratio(self.history, self.batch, self.calendar, self.daily)
        np.testing.assert_array_equal(prediction, [150] * 8)
        self.assertTrue(prediction.index.equals(self.batch.index))
        self.assertTrue(report.n_analogs.eq(5).all())
        self.assertTrue(report.denominator.eq(80).all())
        self.assertTrue(report.m_raw.eq(2).all())
        self.assertTrue(stages.loc[stages.match_id.eq("M001"), "day"].eq(11).all())

    def test_less_than_five_returns_baseline_and_short_analog_is_excluded(self):
        calendar = self.calendar.copy()
        calendar.loc[calendar.match_id.eq("M005"), "sales_open"] = pd.Timestamp("2023-01-02")
        prediction, report, stages = predict_pace_ratio(self.history, self.batch, calendar, self.daily)
        np.testing.assert_array_equal(prediction, [100] * 8)
        self.assertTrue(report.fallback.all())
        self.assertTrue(report.n_analogs.eq(4).all())
        self.assertNotIn("M005", set(stages.match_id))

    def test_multiplier_limits_and_no_sales_scenario(self):
        for sold, expected, m in ((0, 75, .5), (200, 150, 2)):
            daily = self.daily.copy()
            daily.loc[daily.match_id.eq("M035") & daily.date.eq(pd.Timestamp("2023-02-01")), "tickets"] = sold
            prediction, report, _ = predict_pace_ratio(self.history, self.batch.assign(sold=sold), self.calendar, daily)
            np.testing.assert_array_equal(prediction, [expected] * 8)
            self.assertTrue(report.m.eq(m).all())
        batch = self.batch.assign(scenario=0, sold=0, cutoff=pd.Timestamp("2023-02-01"), slice_id="d0")
        prediction, report, stages = predict_pace_ratio(self.history, batch, self.calendar, self.daily)
        np.testing.assert_array_equal(prediction, [100] * 8)
        self.assertFalse(report.fallback.any())
        self.assertTrue(stages.empty)

    def test_day_limits_count_matches_and_zero_denominator_stops(self):
        row = pd.Series(dict(days_elapsed=0, sales_duration=28))
        stages = analog_stages(self.history, row, self.calendar, self.daily)
        self.assertTrue(stages.day.eq(1).all())
        self.assertEqual(stages.loc[stages.day_clipped, "match_id"].nunique(), 5)
        row = pd.Series(dict(days_elapsed=99, sales_duration=100))
        stages = analog_stages(self.history, row, self.calendar, self.daily)
        self.assertTrue(stages.day.eq(stages.length - 1).all())
        self.assertEqual(stages.loc[stages.day_clipped, "match_id"].nunique(), 5)
        daily = self.daily.assign(tickets=0)
        with self.assertRaisesRegex(ValueError, "нулевая сумма"):
            predict_pace_ratio(self.history, self.batch.assign(sold=0), self.calendar, daily)

    def test_future_records_and_calls_do_not_change_prediction(self):
        reference, _, _ = predict_pace_ratio(self.history, self.batch, self.calendar, self.daily)
        daily = self.daily.copy()
        daily.loc[daily.tickets.eq(999), "tickets"] = 99999
        extra = self.batch[["match_id", "zone", "match_date"]].rename(columns={"match_date": "date"})
        extra["tickets_total"] = 999999
        history = pd.concat([self.history, extra], ignore_index=True)
        changed, _, _ = predict_pace_ratio(history, self.batch, self.calendar, daily)
        pd.testing.assert_series_equal(reference, changed)
        records = []
        method = make_pace_ratio_method(self.calendar, self.daily, records)
        method(self.history, self.batch)
        records[0].loc[:, "prediction_raw"] = -99999
        method(self.history.assign(tickets_total=1000), self.batch)
        pd.testing.assert_series_equal(reference, method(self.history, self.batch))


if __name__ == "__main__":
    unittest.main()
