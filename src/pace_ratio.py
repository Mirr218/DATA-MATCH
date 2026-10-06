"""Фиксированный C: общий темп восьми зон на той же доле окна продаж."""
import numpy as np
import pandas as pd

from .baseline import zone_mean_baseline
from .early_sales import available_history, batch_timing, sales_at_cutoffs
from .validation import KEYS

MIN_DURATION = 28
MIN_ANALOGS = 5
TRUST = 0.5
MULTIPLIER_MIN, MULTIPLIER_MAX = 0.5, 2.0


def analog_stages(history, row, calendar, daily):
    """Этап доступен на момент: расписание, history и продажи до среза аналога.

    Завершённость проверяется по утру прогноза, не по дню аналога.
    k/L точно; .5 вверх только для дня аналога, без round() Python.
    """
    dates = calendar.set_index("match_id")
    analogs = history[KEYS + ["date"]].copy()
    analogs["sales_open"] = analogs.match_id.map(dates.sales_open)
    if analogs.sales_open.isna().any() or not analogs.date.equals(analogs.match_id.map(dates.date)):
        raise ValueError("Календарь аналогов не совпадает с переданной history")
    analogs["length"] = (analogs.date - analogs.sales_open).dt.days
    analogs = analogs.loc[analogs.length.ge(MIN_DURATION)].copy()
    sizes = analogs.groupby("match_id").zone.agg(list)
    if not sizes.map(lambda zones: sorted(zones) == list(range(1, 9))).all():
        raise ValueError("Аналог должен иметь ровно восемь зон")
    k, length = int(row.days_elapsed), int(row.sales_duration)
    analogs["day_unclipped"] = (2 * k * analogs.length + length) // (2 * length)
    analogs["day"] = analogs.day_unclipped.clip(lower=1, upper=analogs.length - 1)
    analogs["day_clipped"] = analogs.day.ne(analogs.day_unclipped)
    analogs["cutoff"] = analogs.sales_open + pd.to_timedelta(analogs.day, unit="D")
    if not analogs.cutoff.lt(analogs.date).all():
        raise ValueError("Срез аналога должен быть строго до его матча")
    analogs["sold"] = sales_at_cutoffs(analogs, daily)
    return analogs


def predict_pace_ratio(history, batch, calendar, daily):
    """Прогноз доступен на момент: cutoff; все средние только из history.

    Свои продажи строго до утра cutoff, без продаж всегда baseline.
    Число аналогов и ограничения дня считаются по матчам, не по зонам.
    """
    query = batch_timing(batch, calendar)
    np.testing.assert_array_equal(sales_at_cutoffs(query, daily), batch.sold)
    allowed = available_history(history, query)
    baseline = zone_mean_baseline(allowed, batch)
    predicted, records, stages = baseline.copy(), [], []
    for match_id, group in query.groupby("match_id", sort=False):
        if sorted(group.zone) != list(range(1, 9)) or group.scenario.nunique() != 1:
            raise ValueError("Для общего темпа нужны восемь зон одного сценария")
        row = group.iloc[0]
        count, clipped, denominator, m_raw, m = 0, 0, np.nan, np.nan, 1.0
        fallback = False
        if row.scenario == 1:
            analogs = analog_stages(allowed, row, calendar, daily)
            count = analogs.match_id.nunique()
            clipped = analogs.loc[analogs.day_clipped, "match_id"].nunique()
            stages.append(analogs.assign(predicted_match=match_id, window=row.window,
                                         slice_id=row.slice_id, forecast_cutoff=row.cutoff))
            fallback = count < MIN_ANALOGS
            if not fallback:
                denominator = float(analogs.groupby("zone").sold.mean().sum())
                if not np.isfinite(denominator) or denominator <= 0:
                    raise ValueError("C не определён: нулевая сумма средних продаж аналогов")
                m_raw = float(group.sold.sum()) / denominator
                m = float(np.clip(m_raw, MULTIPLIER_MIN, MULTIPLIER_MAX))
                predicted.loc[group.index] = baseline.loc[group.index] * (1 + TRUST * (m - 1))
        report = group.copy()
        report["n_analogs"], report["n_day_clipped"] = count, clipped
        report["denominator"], report["numerator"] = denominator, group.sold.sum()
        report["m_raw"], report["m"] = m_raw, m
        report["fallback"], report["uses_baseline"] = fallback, fallback or row.scenario == 0
        report["baseline_raw"], report["prediction_raw"] = baseline.loc[group.index], predicted.loc[group.index]
        records.append(report)
    return predicted, pd.concat(records), pd.concat(stages, ignore_index=True) if stages else pd.DataFrame()


def make_pace_ratio_method(calendar, daily, diagnostics=None, analog_diagnostics=None):
    """Интерфейс evaluate_method; хранит источники, не обученное состояние.

    Диагностика только записывается; не используется для следующего прогноза.
    """
    source_calendar, source_daily = calendar.copy(deep=True), daily.copy(deep=True)

    def method(history, batch):
        predicted, report, stages = predict_pace_ratio(history, batch, source_calendar, source_daily)
        if diagnostics is not None:
            diagnostics.append(report)
        if analog_diagnostics is not None and not stages.empty:
            analog_diagnostics.append(stages)
        return predicted

    return method
