"""Проверка средних по зонам и порядка ответов на искусственном примере."""
import unittest

import numpy as np
import pandas as pd

from src.baseline import zone_mean_baseline


class BaselineTests(unittest.TestCase):
    def test_means_use_only_history_and_keep_batch_order(self):
        history = pd.DataFrame({"match_id": ["M001", "M002", "M001"],
                                "zone": [1, 1, 2], "tickets_total": [100, 300, 50]})
        batch = pd.DataFrame({"zone": [2, 1, 1], "tickets_total": [999, 999, 999]},
                             index=[9, 3, 6])
        predicted = zone_mean_baseline(history, batch)
        np.testing.assert_array_equal(predicted.to_numpy(), [50, 200, 200])
        self.assertTrue(predicted.index.equals(batch.index))

    def test_unseen_zone_is_rejected(self):
        history = pd.DataFrame({"match_id": ["M001"], "zone": [1], "tickets_total": [100]})
        with self.assertRaises(ValueError):
            zone_mean_baseline(history, pd.DataFrame({"zone": [2]}))


if __name__ == "__main__":
    unittest.main()
