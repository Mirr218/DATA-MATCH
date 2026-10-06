"""Контроль (--control-only), признаки (--prepare-only), фиксированные A/B (--compare)."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from .baseline import zone_mean_baseline
from .baseline_data import ROOT
from .early_sales import (ALIGNMENTS, available_history, batch_timing,
                          fallback_reason, historical_analogs, make_early_sales_method,
                          prepare_sources, sales_at_cutoffs)
from .evaluate_baseline import evaluate_baseline
from .scenarios import historical_prediction_rule
from .validation import aggregate_scores, evaluate_method

EXPECTED_WINDOWS = np.array([
    [155.995, 151.394, 154.386],
    [105.410, 102.931, 104.542],
    [157.123, 132.333, 148.846],
])
EXPECTED_SCORES = {"S": 139.155, "S_0": 143.913, "S_1": 129.748}
REPORTED_TOLERANCE = 0.0005
PROTECTED_PATHS = ("docs/PROTOCOL.md", "predictions.csv",
                   "outputs/predictions_baseline.csv")


def protected_hashes():
    """Состояние файлов, которые эта сессия не должна изменять."""
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            if (ROOT / name).exists() else None for name in PROTECTED_PATHS}


def save_frame(name, frame):
    """Все новые отчёты остаются только внутри проекта."""
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"early_sales_{name}.csv", index=False)


def run_control():
    """Существующий baseline и точный пример HOWTO обязаны совпасть."""
    examples, summary, details, windows, scores = evaluate_baseline()
    np.testing.assert_allclose(windows[["rmse_no_sales", "rmse_sales", "R"]],
                               EXPECTED_WINDOWS, rtol=0, atol=REPORTED_TOLERANCE)
    for name, expected in EXPECTED_SCORES.items():
        if abs(scores[name] - expected) > REPORTED_TOLERANCE:
            raise ValueError(f"Контроль {name} не воспроизведён; сравнение остановлено")
    howto = (ROOT / "docs/HOWTO_EVALUATE.md").read_text(encoding="utf-8")
    code = howto.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(code, "docs/HOWTO_EVALUATE.md", "exec"), namespace)
    for name, expected in (("examples", examples), ("summary", summary),
                           ("details", details), ("windows", windows)):
        pd.testing.assert_frame_equal(namespace[name], expected)
    if namespace["scores"] != scores:
        raise ValueError("HOWTO и существующий baseline различаются")
    if len(namespace["table"]) != 680 or len(examples) != 984:
        raise ValueError("Изменилось число контрольных строк")
    for name, frame in (("examples", examples), ("control_summary", summary),
                        ("control_details", details), ("control_windows", windows)):
        save_frame(name, frame)
    output = ROOT / "outputs/early_sales_control_scores.json"
    output.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(windows.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("; ".join(f"{name} = {value:.3f}" for name, value in scores.items()))
    print("Контроль и точный пример HOWTO совпали; 680 итогов, 984 примера.")
    return namespace["table"], examples, summary, details, windows, scores


def check_available_features(table, examples, reference_details):
    """Проверить признаки на общих examples, возвращая только baseline."""
    matches = pd.read_csv(ROOT / "data/matches.csv")
    sales = pd.read_csv(ROOT / "data/sales_daily.csv")
    calendar, daily = prepare_sources(matches, sales)
    rows = []

    def inspect(history, batch):
        query = batch_timing(batch, calendar)
        np.testing.assert_array_equal(sales_at_cutoffs(query, daily), batch.sold)
        pd.testing.assert_frame_equal(available_history(history, query), history)
        for _, row in query.loc[query.scenario.eq(1)].iterrows():
            for alignment in ALIGNMENTS:
                analogs = historical_analogs(history, row, calendar, daily, alignment)
                if not (analogs.match_id.isin(history.match_id).all()
                        and analogs.date.lt(row.cutoff).all()
                        and analogs.cutoff.lt(row.cutoff).all()):
                    raise ValueError("Аналог вышел за разрешённую историю")
                fraction = analogs.fraction.mean()
                record = row.to_dict()
                record.update(alignment=alignment, n_analogs=len(analogs),
                              mean_fraction=fraction,
                              fallback_reason=fallback_reason(len(analogs), fraction))
                rows.append(record)
        return zone_mean_baseline(history, batch)

    _, details = evaluate_method(inspect, table, examples, feature_columns=("sold", "seats"),
                                 prediction_rule=historical_prediction_rule)
    pd.testing.assert_frame_equal(details, reference_details)
    checks = pd.DataFrame(rows)
    save_frame("feature_checks", checks)
    print(f"sold совпал во всех {len(examples)} примерах; проверено {len(checks)} срезов A/B.")
    print("Прогнозы остались контрольным baseline; A/B ещё не оценивались.")


def save_error_diagnostics(name, details):
    """Диагностика после прогноза; основной оценкой остаётся R/S всех examples."""
    pairs = details.assign(weighted_error=details.weight * details.squared_error)
    pair_errors = pairs.groupby(["window", "scenario", "match_id", "zone"],
                                 as_index=False).weighted_error.sum()
    match_errors = pair_errors.groupby(["window", "scenario", "match_id"],
                                       as_index=False).weighted_error.mean()
    match_errors["rmse"] = np.sqrt(match_errors.weighted_error)
    save_frame(f"{name}_matches", match_errors.rename(columns={"weighted_error": "mse"}))
    slices = details.groupby(["window", "scenario", "slice_id"], as_index=False).agg(
        mse=("squared_error", "mean"), n_matches=("match_id", "nunique"))
    slices["rmse"] = np.sqrt(slices.mse)
    save_frame(f"{name}_slices", slices)
    selected = pairs.loc[pairs.scenario.eq(1) & pairs.sales_duration.between(28, 42)]
    selected = selected.groupby(["window", "match_id", "zone"],
                                 as_index=False).weighted_error.sum()
    periods = selected.groupby("window", as_index=False).agg(
        mse=("weighted_error", "mean"), n_matches=("match_id", "nunique"),
        n_pairs=("zone", "size"))
    periods["rmse"] = np.sqrt(periods.mse)
    if periods.n_matches.tolist() != [10, 12, 9]:
        raise ValueError("Изменилась диагностическая выборка 28–42")
    save_frame(f"{name}_period_28_42", periods)


def save_fallback_rates(name, diagnostics):
    """Частота среди строк с продажами; сценарий 0 исключён из знаменателя."""
    sales = diagnostics.loc[diagnostics.scenario.eq(1)]
    rate = sales.groupby(["window", "slice_id"], as_index=False).agg(
        n_rows=("fallback", "size"), n_fallback=("fallback", "sum"),
        fallback_rate=("fallback", "mean"))
    overall = sales.groupby("slice_id", as_index=False).agg(
        n_rows=("fallback", "size"), n_fallback=("fallback", "sum"),
        fallback_rate=("fallback", "mean")).assign(window="all")
    save_frame(f"{name}_fallback", pd.concat([rate, overall[rate.columns]], ignore_index=True))
    return overall


def evaluate_candidate(alignment, table, examples, reference_details, calendar, daily):
    """A/B на неизменных examples; диагностические ответы присоединяются после прогноза."""
    records = []
    method = make_early_sales_method(calendar, daily, alignment, records)
    summary, details = evaluate_method(method, table, examples, feature_columns=("sold", "seats"),
                                       prediction_rule=historical_prediction_rule)
    pd.testing.assert_frame_equal(details.loc[details.scenario.eq(0)],
                                  reference_details.loc[reference_details.scenario.eq(0)])
    diagnostics = pd.concat(records, ignore_index=True)
    keys = ["window", "match_id", "zone", "scenario", "slice_id", "cutoff"]
    report = details.merge(diagnostics, on=keys, how="left", validate="one_to_one")
    if len(report) != len(examples) or report.prediction_raw.isna().any():
        raise ValueError("Кандидат не покрыл все общие примеры")
    if not report.prediction.ge(report.sold).all() or not (report.prediction % 1 == 0).all():
        raise ValueError("Не соблюдены общие границы или округление")
    windows, scores = aggregate_scores(summary)
    for name, frame in (("summary", summary), ("details", details),
                        ("windows", windows), ("diagnostics", report)):
        save_frame(f"{alignment}_{name}", frame)
    (ROOT / f"outputs/early_sales_{alignment}_scores.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rates = save_fallback_rates(alignment, diagnostics)
    save_error_diagnostics(alignment, report)
    print(f"Кандидат {alignment}:", flush=True)
    print(windows.to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
    print("; ".join(f"{name} = {value:.3f}" for name, value in scores.items()), flush=True)
    print(rates.to_string(index=False, float_format=lambda value: f"{value:.3%}"), flush=True)
    return windows, scores


def run_candidates(table, examples, reference_details):
    """Ровно A затем B, без изменения правил после результата A."""
    calendar, daily = prepare_sources(pd.read_csv(ROOT / "data/matches.csv"),
                                      pd.read_csv(ROOT / "data/sales_daily.csv"))
    dates = calendar.set_index("match_id")
    durations = (dates.date - dates.sales_open).dt.days
    baseline_details = reference_details.assign(sales_duration=reference_details.match_id.map(durations))
    save_error_diagnostics("control", baseline_details)
    return {alignment: evaluate_candidate(alignment, table, examples, reference_details,
                                          calendar, daily) for alignment in ALIGNMENTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--control-only", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    before = protected_hashes()
    table, examples, _, details, _, _ = run_control()
    if args.prepare_only:
        check_available_features(table, examples, details)
    elif args.compare:
        run_candidates(table, examples, details)
    if protected_hashes() != before:
        raise RuntimeError("Изменён защищённый файл")
    (ROOT / "outputs/early_sales_protected_hashes.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
