"""Один фиксированный эксперимент C: python -X utf8 -m src.evaluate_pace_ratio."""
import hashlib
import json

import numpy as np
import pandas as pd

from .baseline_data import ROOT, load_training_table
from .early_sales import prepare_sources
from .evaluate_baseline import evaluate_baseline
from .pace_ratio import make_pace_ratio_method
from .scenarios import historical_prediction_rule
from .validation import KEYS, aggregate_scores, evaluate_method


def save(name, frame):
    frame.to_csv(ROOT / f"outputs/pace_ratio_C_{name}.csv", index=False)


def protected_hashes():
    paths = [ROOT / "predictions.csv", ROOT / "outputs/predictions_baseline.csv"]
    paths += sorted((ROOT / "outputs").glob("early_sales_A_*"))
    paths += sorted((ROOT / "outputs").glob("early_sales_B_*"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            if p.exists() else None for p in paths}


def criteria(windows, scores, baseline_windows, baseline_scores):
    improved = int(windows.R.lt(baseline_windows.R).sum())
    records = [
        ("S", scores["S"], baseline_scores["S"] * .98, scores["S"] <= baseline_scores["S"] * .98),
        ("improved_windows", improved, 2, improved >= 2),
        ("R_last", windows.R.iloc[-1], baseline_windows.R.iloc[-1] * 1.03,
         windows.R.iloc[-1] <= baseline_windows.R.iloc[-1] * 1.03),
        ("S_0", scores["S_0"], baseline_scores["S_0"] * 1.03,
         scores["S_0"] <= baseline_scores["S_0"] * 1.03),
        ("S_1", scores["S_1"], baseline_scores["S_1"] * 1.03,
         scores["S_1"] <= baseline_scores["S_1"] * 1.03)]
    return pd.DataFrame(records, columns=["criterion", "value", "threshold", "passed"])


def diagnostics(report, stages):
    """Диагностика после прогноза; не участвует в выборе или расчёте C."""
    sales = report.loc[report.scenario.eq(1)].copy()
    sales["ratio"] = sales.prediction / sales.tickets_total
    sales["duration_group"] = pd.cut(sales.sales_duration, [27, 35, 42, np.inf],
                                     labels=["28–35", "36–42", "43+"])
    fallback = sales.groupby("slice_id").agg(
        rows=("fallback", "size"), fallback_rows=("fallback", "sum"), fallback_rate=("fallback", "mean"))
    save("fallback", fallback.reset_index())
    duration = sales.groupby(["slice_id", "duration_group"], observed=True).agg(
        rows=("ratio", "size"), matches=("match_id", "nunique"),
        mean_ratio=("ratio", "mean"), median_ratio=("ratio", "median"))
    save("duration", duration.reset_index())
    rows = []
    for name, group in sales.groupby("slice_id"):
        total = float(group.squared_error.sum())
        largest = float(group.squared_error.nlargest(10).sum())
        rows.append(dict(group=name, rows=len(group), sse=total, top10_sse=largest,
                         top10_share=largest / total if total else 0))
        save(f"top10_{name}", group.nlargest(10, "squared_error"))
    weighted = sales.assign(weighted_error=sales.weight * sales.squared_error)
    total = float(weighted.weighted_error.sum())
    largest = float(weighted.weighted_error.nlargest(10).sum())
    rows.append(dict(group="pooled_sales_weighted", rows=len(sales), sse=total,
                     top10_sse=largest, top10_share=largest / total if total else 0))
    save("error_concentration", pd.DataFrame(rows))
    unique_stages = stages.drop_duplicates(["window", "predicted_match", "slice_id", "forecast_cutoff", "match_id"])
    clamps = unique_stages.groupby("slice_id").agg(
        comparisons=("day_clipped", "size"), day_clamps=("day_clipped", "sum"))
    save("day_clamps", clamps.reset_index())
    return fallback, duration, pd.DataFrame(rows), clamps


def main():
    before = protected_hashes()
    examples, _, baseline_details, baseline_windows, baseline_scores = evaluate_baseline()
    control = json.loads((ROOT / "outputs/pace_ratio_control_check.json").read_text())
    if baseline_scores != control["scores"] or len(examples) != 984:
        raise ValueError("Контроль изменился: сравнение C остановлено")
    matches = pd.read_csv(ROOT / "data/matches.csv", parse_dates=["date", "sales_open"])
    calendar, daily = prepare_sources(matches, pd.read_csv(ROOT / "data/sales_daily.csv"))
    table = load_training_table()[KEYS + ["tickets_total"]].merge(
        matches[["match_id", "date"]], on="match_id", validate="many_to_one")
    records, stage_records = [], []
    method = make_pace_ratio_method(calendar, daily, records, stage_records)
    summary, details = evaluate_method(method, table, examples, feature_columns=("sold", "seats"),
                                       prediction_rule=historical_prediction_rule)
    pd.testing.assert_frame_equal(details.loc[details.scenario.eq(0)],
                                  baseline_details.loc[baseline_details.scenario.eq(0)])
    keys = ["window", "match_id", "zone", "scenario", "slice_id", "cutoff"]
    report = pd.concat(records, ignore_index=True)
    columns = [c for c in report if c not in details or c in keys]
    report = details.merge(report[columns], on=keys, how="left", validate="one_to_one")
    report = report.merge(examples[KEYS + ["slice_id", "seats"]], on=KEYS + ["slice_id"], validate="one_to_one")
    if len(report) != 984 or report.prediction_raw.isna().any():
        raise ValueError("Не все примеры получили прогноз C")
    if not report.prediction.ge(report.sold).all() or not report.prediction.le(report.seats).all():
        raise ValueError("Нарушены общие границы")
    stages = pd.concat(stage_records, ignore_index=True)
    windows, scores = aggregate_scores(summary)
    checks = criteria(windows, scores, baseline_windows, baseline_scores)
    for name, frame in (("examples", examples), ("summary", summary), ("details", details),
                        ("windows", windows), ("diagnostics", report), ("analogs", stages), ("criteria", checks)):
        save(name, frame)
    diagnostics(report, stages)
    if protected_hashes() != before:
        raise RuntimeError("Изменён защищённый прогноз или прежний результат A/B")
    (ROOT / "outputs/pace_ratio_C_scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    (ROOT / "outputs/pace_ratio_C_protected_hashes.json").write_text(json.dumps(before, indent=2) + "\n")
    print(windows.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(json.dumps(scores))
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
