"""Оба сценария baseline по PROTOCOL: python -X utf8 -m src.evaluate_baseline."""
import json

import pandas as pd

from .baseline import zone_mean_baseline
from .baseline_data import ROOT, load_training_table
from .scenarios import historical_prediction_rule, make_scenario_examples
from .validation import KEYS, aggregate_scores, evaluate_method


def evaluate_baseline():
    """Вернуть примеры, зональные ошибки, детали, R по окнам и S baseline."""
    matches = pd.read_csv(ROOT / "data/matches.csv", parse_dates=["date", "sales_open"])
    sales = pd.read_csv(ROOT / "data/sales_daily.csv", parse_dates=["date"])
    table = load_training_table()[KEYS + ["tickets_total"]]
    table = table.merge(matches[["match_id", "date"]], on="match_id", validate="many_to_one")
    examples = make_scenario_examples(matches, sales)
    seats = pd.read_csv(ROOT / "data/zones.csv", usecols=KEYS + ["seats"])
    examples = examples.merge(seats, on=KEYS, how="left", validate="many_to_one")
    summary, details = evaluate_method(zone_mean_baseline, table, examples,
                                      feature_columns=("sold", "seats"),
                                      prediction_rule=historical_prediction_rule)
    windows, scores = aggregate_scores(summary)
    return examples, summary, details, windows, scores


if __name__ == "__main__":
    examples, summary, details, windows, scores = evaluate_baseline()
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("examples", examples), ("summary", summary),
                        ("details", details), ("windows", windows)):
        frame.to_csv(output / f"baseline_scenarios_{name}.csv", index=False)
    (output / "baseline_scenarios_scores.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(windows.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("; ".join(f"{name} = {value:.3f}" for name, value in scores.items()))
    print(f"Примеров: {len(examples)}; округление .5 вверх, с историческим потолком seats.")
