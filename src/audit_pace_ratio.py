"""Независимое восстановление C из CSV, без вызовов модели и validation."""
import hashlib
import json

import numpy as np
import pandas as pd

from .baseline_data import ROOT


def sources():
    matches = pd.read_csv(ROOT / "data/matches.csv", parse_dates=["date", "sales_open"])
    sales = pd.read_csv(ROOT / "data/sales_daily.csv", parse_dates=["date"])
    zones = pd.read_csv(ROOT / "data/zones.csv", usecols=["match_id", "zone", "seats"])
    pairs = matches.loc[matches.part.eq("train"), ["match_id", "date"]].merge(
        pd.DataFrame({"zone": range(1, 9)}), how="cross")
    totals = sales.groupby(["match_id", "zone"]).tickets.sum()
    pairs["tickets_total"] = [totals.get((m, z), 0) for m, z in zip(pairs.match_id, pairs.zone)]
    return matches.set_index("match_id"), sales, zones.set_index(["match_id", "zone"]), pairs


def check_examples(matches, report):
    expected = set()
    for first, last in ((35, 51), (52, 68), (69, 85)):
        for number in range(first, last + 1):
            match_id = f"M{number:03}"
            match = matches.loc[match_id]
            length = (match.date - match.sales_open).days
            slices = [(0, 0, 1.0)] + ([(1, 27, 2/3), (1, 7, 1/3)] if length >= 28 else [])
            for scenario, k, weight in slices:
                for zone in range(1, 9):
                    expected.add((match_id, zone, scenario, f"d{k}",
                                  match.sales_open + pd.Timedelta(days=k), weight))
    actual = set(report[["match_id", "zone", "scenario", "slice_id", "cutoff", "weight"]].itertuples(index=False, name=None))
    assert expected == actual and len(report) == len(expected) == 984


def analog_reference(available, match, cutoff, matches, daily_by_pair):
    k, length = (cutoff - match.sales_open).days, (match.date - match.sales_open).days
    amounts, stage_rows = [], []
    for analog_id in sorted(available.match_id.unique()):
        analog = matches.loc[analog_id]
        analog_length = (analog.date - analog.sales_open).days
        if analog_length < 28:
            continue
        assert analog.date < cutoff
        day_unclipped = (2 * k * analog_length + length) // (2 * length)
        day = min(max(day_unclipped, 1), analog_length - 1)
        analog_cutoff = analog.sales_open + pd.Timedelta(days=day)
        assert analog.sales_open < analog_cutoff < analog.date < cutoff
        row_amounts = []
        for zone in range(1, 9):
            daily = daily_by_pair.get((analog_id, zone))
            sold = 0 if daily is None else int(daily.loc[
                daily.date.ge(analog.sales_open) & daily.date.lt(analog_cutoff), "tickets"].sum())
            row_amounts.append(sold)
            stage_rows.append(dict(match_id=analog_id, zone=zone, day=day,
                                   day_unclipped=day_unclipped, day_clipped=day != day_unclipped,
                                   cutoff=analog_cutoff, sold=sold))
        amounts.append(row_amounts)
    return amounts, pd.DataFrame(stage_rows)


def reconstruct(matches, sales, zones, pairs, report, stages):
    daily_by_pair = dict(tuple(sales.groupby(["match_id", "zone"])))
    matched_stages, matched_rows = 0, 0
    for (window, match_id, scenario, slice_id, cutoff), group in report.groupby(
            ["window", "match_id", "scenario", "slice_id", "cutoff"]):
        group = group.sort_values("zone")
        match = matches.loc[match_id]
        available = pairs.loc[pairs.match_id.lt(window.split("–")[0]) & pairs.date.lt(cutoff)]
        baseline = available.groupby("zone").tickets_total.mean().reindex(range(1, 9)).to_numpy()
        own_sold = [int(sales.loc[sales.match_id.eq(match_id) & sales.zone.eq(z)
                    & sales.date.ge(match.sales_open) & sales.date.lt(cutoff), "tickets"].sum()) for z in range(1, 9)]
        np.testing.assert_array_equal(own_sold, group.sold)
        m, count, clipped = 1.0, 0, 0
        if scenario == 1:
            amounts, reference = analog_reference(available, match, cutoff, matches, daily_by_pair)
            count = len(amounts)
            actual = stages.loc[stages.predicted_match.eq(match_id) & stages.window.eq(window) & stages.slice_id.eq(slice_id)]
            columns = ["match_id", "zone", "day", "day_unclipped", "day_clipped", "cutoff", "sold"]
            pd.testing.assert_frame_equal(actual[columns].sort_values(["match_id", "zone"]).reset_index(drop=True),
                                          reference[columns].sort_values(["match_id", "zone"]).reset_index(drop=True))
            matched_stages += len(reference)
            clipped = reference.loc[reference.day_clipped, "match_id"].nunique()
            if count >= 5:
                denominator = float(np.asarray(amounts).mean(axis=0).sum())
                assert denominator > 0
                np.testing.assert_allclose(group.denominator, denominator, rtol=0, atol=1e-10)
                m_raw = sum(own_sold) / denominator
                np.testing.assert_allclose(group.m_raw, m_raw, rtol=0, atol=1e-12)
                m = min(max(m_raw, .5), 2)
        assert group.n_analogs.eq(count).all() and group.n_day_clipped.eq(clipped).all()
        assert group.fallback.eq(scenario == 1 and count < 5).all()
        raw = baseline * (1 + .5 * (m - 1))
        np.testing.assert_allclose(raw, group.prediction_raw, rtol=0, atol=1e-10)
        seats = [zones.loc[(match_id, z), "seats"] for z in range(1, 9)]
        predicted = np.minimum(np.maximum(np.floor(raw + .5), own_sold), seats)
        np.testing.assert_array_equal(predicted, group.prediction)
        actual_targets = pairs.loc[pairs.match_id.eq(match_id)].sort_values("zone").tickets_total
        np.testing.assert_array_equal(actual_targets, group.tickets_total)
        matched_rows += len(group)
    assert matched_rows == 984 and matched_stages == len(stages)
    return matched_rows, matched_stages


def independent_metrics(report):
    errors = report.assign(error=(report.prediction - report.tickets_total) ** 2)
    np.testing.assert_array_equal(errors.error, report.squared_error)
    pairs = errors.assign(weighted=errors.error * errors.weight).groupby(
        ["window", "scenario", "match_id", "zone"]).weighted.sum()
    means = pairs.groupby(["window", "scenario"]).mean().unstack("scenario")
    rmse0, rmse1 = np.sqrt(means[0]), np.sqrt(means[1])
    R = np.sqrt((11 / 17) * means[0] + (6 / 17) * means[1])
    weights = np.array([.25, .25, .5])
    scores = {"S": float(R.to_numpy() @ weights), "S_0": float(rmse0.to_numpy() @ weights),
              "S_1": float(rmse1.to_numpy() @ weights)}
    windows = pd.DataFrame(dict(window=means.index, rmse_no_sales=rmse0.to_numpy(),
                                rmse_sales=rmse1.to_numpy(), R=R.to_numpy()))
    saved = pd.read_csv(ROOT / "outputs/pace_ratio_C_windows.csv")
    np.testing.assert_allclose(windows.iloc[:, 1:], saved.iloc[:, 1:], rtol=0, atol=1e-10)
    saved_scores = json.loads((ROOT / "outputs/pace_ratio_C_scores.json").read_text())
    np.testing.assert_allclose(list(scores.values()), list(saved_scores.values()), rtol=0, atol=1e-10)
    baseline = pd.read_csv(ROOT / "outputs/pace_ratio_step2_baseline_windows.csv")
    baseline_scores = json.loads((ROOT / "outputs/pace_ratio_control_check.json").read_text())["scores"]
    flags = [scores["S"] <= .98 * baseline_scores["S"], int((R.to_numpy() < baseline.R).sum()) >= 2,
             R.iloc[-1] <= 1.03 * baseline.R.iloc[-1], scores["S_0"] <= 1.03 * baseline_scores["S_0"],
             scores["S_1"] <= 1.03 * baseline_scores["S_1"]]
    np.testing.assert_array_equal(flags, pd.read_csv(ROOT / "outputs/pace_ratio_C_criteria.csv").passed)
    return scores, [bool(flag) for flag in flags]


def main():
    report = pd.read_csv(ROOT / "outputs/pace_ratio_C_diagnostics.csv", parse_dates=["cutoff"])
    stages = pd.read_csv(ROOT / "outputs/pace_ratio_C_analogs.csv", parse_dates=["cutoff", "forecast_cutoff"])
    matches, sales, zones, pairs = sources()
    check_examples(matches, report)
    rows, analog_rows = reconstruct(matches, sales, zones, pairs, report, stages)
    scores, flags = independent_metrics(report)
    protected = json.loads((ROOT / "outputs/pace_ratio_C_protected_hashes.json").read_text())
    for name, expected in protected.items():
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        assert actual == expected, name
    names = ["data/matches.csv", "data/zones.csv", "data/sales_daily.csv", "docs/PROTOCOL.md",
             "src/pace_ratio.py", "src/evaluate_pace_ratio.py", "src/audit_pace_ratio.py",
             "src/scenarios.py", "src/validation.py", "src/early_sales.py", "src/baseline.py",
             "tests/test_pace_ratio.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    result = dict(verdict="OK", predictions_reconstructed=rows, analog_zone_rows_reconstructed=analog_rows,
                  scores=scores, criteria=flags, protected_files_unchanged=True, hashes=hashes)
    (ROOT / "outputs/pace_ratio_C_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Независимый аудит OK: {rows} прогнозов; {analog_rows} зональных срезов аналогов; R/S и критерии совпали.")


if __name__ == "__main__":
    main()
