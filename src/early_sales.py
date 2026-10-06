"""Доступные срезы ранних продаж; обученного состояния между вызовами нет."""
import numpy as np
import pandas as pd

from .baseline import zone_mean_baseline
from .validation import HISTORY_COLUMNS, KEYS

MIN_ANALOGS = 5
MIN_MEAN_FRACTION = 0.15
ALIGNMENTS = ("A", "B")


def prepare_sources(matches, sales):
    """Скопировать только расписание и дневные продажи, без итогов/потолков.

    Календарь доступен на момент: заранее известное расписание.
    Признак продажи доступен на момент: только date < cutoff; обрезание
    выполняется в sales_at_cutoffs, а не при загрузке исходного файла.
    """
    calendar = matches[["match_id", "date", "sales_open"]].copy()
    daily = sales[KEYS + ["date", "tickets"]].copy()
    for frame, names in ((calendar, ("date", "sales_open")), (daily, ("date",))):
        for name in names:
            frame[name] = pd.to_datetime(frame[name], errors="raise").dt.normalize()
    if calendar.isna().any().any() or calendar.match_id.duplicated().any():
        raise ValueError("Нужен уникальный календарь без пропусков")
    if not calendar.sales_open.lt(calendar.date).all():
        raise ValueError("Открытие должно быть раньше матча")
    if daily.isna().any().any() or daily.duplicated(KEYS + ["date"]).any():
        raise ValueError("Пропуски или дубли дневных продаж")
    if (not np.isfinite(daily.tickets).all() or daily.tickets.lt(0).any()
            or (daily.tickets % 1 != 0).any()):
        raise ValueError("Нужны целые неотрицательные дневные продажи")
    if not daily.match_id.isin(calendar.match_id).all():
        raise ValueError("Продажи неизвестного календарю матча")
    return calendar, daily


def sales_at_cutoffs(pairs, daily):
    """Сумма по каждой строке, строго sales_open <= date < cutoff.

    sold доступен на момент: утро собственной cutoff каждой пары;
    отсутствующие дневные записи означают ноль, день cutoff исключён.
    Сохраняются порядок и индекс pairs, даже при повторных срезах пары.
    """
    query = pairs[KEYS + ["sales_open", "cutoff"]].copy()
    query["row_id"] = np.arange(len(query))
    allowed = daily.loc[daily.match_id.isin(query.match_id), KEYS + ["date", "tickets"]]
    observed = query.merge(allowed, on=KEYS, how="left", validate="many_to_many")
    observed = observed.loc[observed.date.ge(observed.sales_open)
                            & observed.date.lt(observed.cutoff)]
    totals = observed.groupby("row_id").tickets.sum()
    return pd.Series(totals.reindex(range(len(query)), fill_value=0).to_numpy(),
                     index=pairs.index, name="sold", dtype="int64")


def batch_timing(batch, calendar):
    """Календарный срок доступен на момент: расписание и утро cutoff."""
    query = batch[KEYS + ["scenario", "slice_id", "cutoff", "window", "sold"]].copy()
    dates = calendar.set_index("match_id")
    query["sales_open"] = query.match_id.map(dates.sales_open)
    query["match_date"] = query.match_id.map(dates.date)
    query["cutoff"] = pd.to_datetime(query.cutoff, errors="raise").dt.normalize()
    if query.isna().any().any() or not query.cutoff.lt(query.match_date).all():
        raise ValueError("Нужна известная отсечка строго до матча")
    if not query.cutoff.ge(query.sales_open).all():
        raise ValueError("Отсечка раньше открытия продаж")
    if not pd.to_datetime(batch.match_date).equals(query.match_date):
        raise ValueError("Дата матча batch не совпала с календарём")
    query["days_elapsed"] = (query.cutoff - query.sales_open).dt.days
    query["days_remaining"] = (query.match_date - query.cutoff).dt.days
    query["sales_duration"] = (query.match_date - query.sales_open).dt.days
    return query


def available_history(history, query):
    """Полный итог доступен на момент: завершённый до cutoff матч до окна.

    Общая validation уже обеспечивает эту границу; здесь дополнительно
    исключаем неподходящие строки. Никакие другие таблицы итогов не читаем.
    """
    if query.cutoff.nunique() != 1 or query.window.nunique() != 1:
        raise ValueError("Один вызов должен иметь одно окно и одно утро")
    if history.duplicated(KEYS).any():
        raise ValueError("Дубли в переданной истории")
    allowed = history[HISTORY_COLUMNS].copy()
    allowed["date"] = pd.to_datetime(allowed.date, errors="raise").dt.normalize()
    first_match = query.window.iloc[0].split("–")[0]
    cutoff = query.cutoff.iloc[0]
    allowed = allowed.loc[allowed.match_id.lt(first_match) & allowed.date.lt(cutoff)]
    if (allowed.isna().any().any() or not np.isfinite(allowed.tickets_total).all()
            or allowed.tickets_total.lt(0).any()):
        raise ValueError("Недопустимые доступные исторические итоги")
    return allowed


def historical_analogs(history, row, calendar, daily, alignment):
    """Признак доли доступен на момент: history, продажи до среза аналога.

    A: то же число дней от открытия; B: то же число дней до матча.
    Нулевой полный итог исключён; нулевая ранняя продажа остаётся в средней.
    Ответ проверяемого матча не читается, прошлые доли считаются заново.
    """
    if alignment not in ALIGNMENTS:
        raise ValueError("Разрешены только зафиксированные A и B")
    allowed = available_history(history, row.to_frame().T)
    analogs = allowed.loc[allowed.zone.eq(row.zone) & allowed.tickets_total.gt(0)].copy()
    dates = calendar.set_index("match_id")
    analogs["sales_open"] = analogs.match_id.map(dates.sales_open)
    if analogs.sales_open.isna().any() or not analogs.date.equals(analogs.match_id.map(dates.date)):
        raise ValueError("Исторический календарь не совпал с переданной history")
    if alignment == "A":
        analogs["cutoff"] = analogs.sales_open + pd.Timedelta(days=int(row.days_elapsed))
    else:
        analogs["cutoff"] = analogs.date - pd.Timedelta(days=int(row.days_remaining))
    analogs = analogs.loc[analogs.cutoff.ge(analogs.sales_open) & analogs.cutoff.lt(analogs.date)]
    analogs["sold"] = sales_at_cutoffs(analogs, daily)
    analogs["fraction"] = analogs.sold / analogs.tickets_total
    if not analogs.fraction.between(0, 1).all():
        raise ValueError("Историческая доля вне [0, 1]; проверить источники")
    return analogs


def fallback_reason(n_analogs, mean_fraction):
    """Фиксированное общее правило A/B; ровно 5 и 0.15 допускаются."""
    if n_analogs == 0:
        return "no_analogs"
    if not np.isfinite(mean_fraction) or not 0 <= mean_fraction <= 1:
        raise ValueError("Средняя доля должна быть конечной и в [0, 1]")
    reasons = []
    if n_analogs < MIN_ANALOGS:
        reasons.append("too_few_analogs")
    if mean_fraction == 0:
        reasons.append("zero_fraction")
    elif mean_fraction < MIN_MEAN_FRACTION:
        reasons.append("low_fraction")
    return ";".join(reasons)


def predict_early_sales(history, batch, calendar, daily, alignment):
    """Raw-прогноз доступен на момент: cutoff, разрешённая history и sold.

    Без продаж всегда baseline. С продажами: sold / средняя доля,
    либо baseline по фиксированному правилу. Округление/границы снаружи.
    Возвращает Series в порядке batch и независимые диагностические строки.
    """
    if alignment not in ALIGNMENTS:
        raise ValueError("Разрешены только зафиксированные A и B")
    query = batch_timing(batch, calendar)
    np.testing.assert_array_equal(sales_at_cutoffs(query, daily), batch.sold)
    allowed = available_history(history, query)
    baseline = zone_mean_baseline(allowed, batch)
    predicted = baseline.copy()
    records = []
    for index, row in query.iterrows():
        count, fraction, reason = 0, np.nan, "no_sales_scenario"
        if row.scenario == 1:
            analogs = historical_analogs(allowed, row, calendar, daily, alignment)
            count, fraction = len(analogs), analogs.fraction.mean()
            reason = fallback_reason(count, fraction)
            if not reason:
                predicted.loc[index] = row.sold / fraction
        record = row.to_dict()
        record.update(alignment=alignment, n_analogs=count, mean_fraction=fraction,
                      fallback_reason=reason, fallback=bool(reason) and row.scenario == 1,
                      uses_baseline=bool(reason), baseline_raw=float(baseline.loc[index]),
                      prediction_raw=float(predicted.loc[index]))
        records.append(record)
    return predicted, pd.DataFrame(records, index=query.index)


def make_early_sales_method(calendar, daily, alignment, diagnostics=None):
    """Подключение method(history, batch); средние заново при каждом вызове.

    Замыкание хранит только копии исходных источников. Диагностика лишь
    записывается, никогда не читается методом и не влияет на прогноз.
    """
    source_calendar, source_daily = calendar.copy(deep=True), daily.copy(deep=True)

    def method(history, batch):
        predicted, report = predict_early_sales(history, batch, source_calendar,
                                               source_daily, alignment)
        if diagnostics is not None:
            diagnostics.append(report)
        return predicted

    return method
