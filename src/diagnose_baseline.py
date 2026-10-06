"""Диагностика baseline на общей подвыборке; не меняет оценку из PROTOCOL."""
import numpy as np
import pandas as pd

from .baseline import zone_mean_baseline
from .baseline_data import ROOT, load_training_table
from .evaluate_baseline import evaluate_baseline
from .validation import KEYS, evaluate_method


def diagnose_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A: O+27/O+7 без sold; B: те же срезы с sold; C: утро O, те же матчи.

    Все варианты округляются .5 вверх. A/B имеют веса 2/3 и 1/3.
    Сравнение A/B изолирует вклад нижней границы. C против полного
    сценария 0 показывает выборку; C против A — доступность истории
    в разные моменты прогноза. Новую модель не обучаем и S не заменяем.
    """
    examples, official, _, _, _ = evaluate_baseline()
    matches = pd.read_csv(ROOT / "data/matches.csv", parse_dates=["date"])
    table = load_training_table()[KEYS + ["tickets_total"]].merge(
        matches[["match_id", "date"]], on="match_id", validate="many_to_one")
    sales_examples = examples.loc[examples.scenario.eq(1)].copy()
    same_matches = set(sales_examples.match_id)
    no_sales_examples = examples.loc[examples.scenario.eq(0) & examples.match_id.isin(same_matches)].copy()
    unbounded, _ = evaluate_method(zone_mean_baseline, table, sales_examples,
                                  prediction_rule=lambda raw, batch: np.floor(np.asarray(raw) + 0.5))
    no_sales_subset, _ = evaluate_method(zone_mean_baseline, table, no_sales_examples,
                                        prediction_rule=lambda raw, batch: np.floor(np.asarray(raw) + 0.5))
    def overall(frame, scenario):
        return frame.loc[frame.zone.eq("all") & frame.scenario.eq(scenario)].set_index("window")
    a, b, c, full = (overall(unbounded, 1), overall(official, 1),
                     overall(no_sales_subset, 0), overall(official, 0))
    comparison = pd.DataFrame({"matches": b.n_matches,
                               "A_unbounded": a.rmse, "B_sold_floor": b.rmse,
                               "C_no_sales_subset": c.rmse, "no_sales_all_17": full.rmse})
    comparison["gain_selection"] = comparison.no_sales_all_17 - comparison.C_no_sales_subset
    comparison["gain_history"] = comparison.C_no_sales_subset - comparison.A_unbounded
    comparison["gain_sold_floor"] = comparison.A_unbounded - comparison.B_sold_floor
    return comparison.reset_index(), official.loc[official.zone.ne("all")].copy()


if __name__ == "__main__":
    comparison, zones = diagnose_baseline()
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "baseline_diagnostics.csv", index=False)
    zones.to_csv(output / "baseline_diagnostics_zones.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(zones.pivot(index=["window", "scenario"], columns="zone", values="rmse")
          .to_string(float_format=lambda value: f"{value:.3f}"))
