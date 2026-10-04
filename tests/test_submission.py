"""Проверки порядка шаблона, test-отсечки и противоречивых границ."""
import unittest

import pandas as pd

from src.make_baseline import build_submission, validate_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        test_ids = [f"M{i:03}" for i in range(86, 103)]
        self.template = pd.DataFrame([(m, z, None) for m in test_ids for z in range(1, 9)],
                                     columns=["match_id", "zone", "tickets_total"]).iloc[::-1].reset_index(drop=True)
        self.matches = pd.DataFrame({"match_id": ["M001", "M002"] + test_ids,
                                     "part": ["train", "train"] + ["test"] * 17})
        self.history = pd.DataFrame([
            {"match_id": m, "zone": z, "tickets_total": total + z,
             "date": pd.Timestamp("2025-12-01")}
            for m, total in (("M001", 100), ("M002", 101)) for z in range(1, 9)
        ])
        self.zones = self.template[["match_id", "zone"]].assign(seats=200, season_tickets=80)
        self.zones.loc[self.zones.zone.eq(2), ["seats", "season_tickets"]] = [120, 20]
        self.sales = pd.DataFrame({"match_id": ["M086", "M086"], "zone": [1, 1],
                                   "date": pd.to_datetime(["2025-12-14", "2025-12-15"]),
                                   "tickets": [110, 99999]})

    def test_template_order_rounding_cutoff_and_both_bounds(self):
        submission, bounds = build_submission(self.history, self.matches, self.zones, self.sales, self.template)
        self.assertTrue(submission[["match_id", "zone"]].equals(self.template[["match_id", "zone"]]))
        values = submission.set_index(["match_id", "zone"]).tickets_total
        self.assertEqual(values.loc[("M086", 1)], 110)
        self.assertEqual(values.loc[("M087", 1)], 102)
        self.assertEqual(values.loc[("M086", 2)], 100)
        self.assertEqual(bounds.sold.max(), 110)
        validate_submission(submission, self.template, bounds)

    def test_impossible_lower_and_upper_bounds_are_rejected(self):
        changed = self.sales.copy()
        changed.loc[0, "tickets"] = 121
        with self.assertRaises(ValueError):
            build_submission(self.history, self.matches, self.zones, changed, self.template)

    def test_wrong_order_fractional_prediction_and_duplicates_are_rejected(self):
        submission, bounds = build_submission(self.history, self.matches, self.zones, self.sales, self.template)
        wrong_order = submission.iloc[::-1].reset_index(drop=True)
        fractions = submission.copy()
        fractions["tickets_total"] = fractions.tickets_total.astype(float) + 0.1
        duplicates = submission.copy()
        duplicates.loc[1, ["match_id", "zone"]] = duplicates.loc[0, ["match_id", "zone"]].to_numpy()
        for invalid in (wrong_order, fractions, duplicates):
            with self.assertRaises(ValueError):
                validate_submission(invalid, self.template, bounds)

    def test_training_history_after_test_cutoff_is_rejected(self):
        changed = self.history.copy()
        changed.loc[0, "date"] = pd.Timestamp("2025-12-15")
        with self.assertRaises(ValueError):
            build_submission(changed, self.matches, self.zones, self.sales, self.template)


if __name__ == "__main__":
    unittest.main()
