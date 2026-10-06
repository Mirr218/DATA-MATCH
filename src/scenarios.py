"""Фиксированные исторические примеры по PROTOCOL, согласованному 04.10.2026."""
import numpy as np
import pandas as pd

from .validation import KEYS, WINDOWS

MIN_SALES_DURATION = 28
SALES_SLICES = ((27, 2 / 3), (7, 1 / 3))


def make_scenario_examples(matches: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Сценарий 0: утро O, все матчи окна; 1: O+27/O+7, только L ≥ 28.

    У обоих срезов сценария 1 одна выборка. sold включает только даты
    O ≤ date < cutoff; дневные записи на cutoff и позже не используются.
    Выход совместим с evaluate_method, feature_columns=('sold',).
    cutoff доступен на момент: заранее известное расписание открытия продаж.
    sold доступен на момент: утро cutoff, продажи по предыдущий день включительно.
    """
    calendar = matches.loc[matches.part.eq("train"), ["match_id", "date", "sales_open"]].copy()
    for column in ("date", "sales_open"):
        calendar[column] = pd.to_datetime(calendar[column], errors="raise")
    daily = sales[KEYS + ["date", "tickets"]].copy()
    daily["date"] = pd.to_datetime(daily.date, errors="raise")
    if calendar.match_id.duplicated().any() or calendar.isna().any().any():
        raise ValueError("Нужны уникальные матчи с известными датами")
    if daily.isna().any().any() or daily.duplicated(KEYS + ["date"]).any():
        raise ValueError("Пропуски или дубли дневных продаж")
    if (not np.isfinite(daily.tickets).all() or daily.tickets.lt(0).any()
            or (daily.tickets % 1 != 0).any()):
        raise ValueError("Дневные продажи должны быть конечными, целыми и неотрицательными")
    frames = []
    for first, last in WINDOWS:
        checking = calendar.loc[calendar.match_id.between(f"M{first:03}", f"M{last:03}")]
        eligible = checking.loc[(checking.date - checking.sales_open).dt.days.ge(MIN_SALES_DURATION)]
        for scenario, days, weight, group in (
            (0, 0, 1.0, checking),
            *((1, days, weight, eligible) for days, weight in SALES_SLICES),
        ):
            frame = group[["match_id", "sales_open"]].merge(
                pd.DataFrame({"zone": range(1, 9)}), how="cross")
            frame["scenario"], frame["slice_id"], frame["weight"] = scenario, f"d{days}", weight

            frame["cutoff"] = frame.sales_open + pd.Timedelta(days=days)
            observed = frame.merge(daily, on=KEYS, how="left", validate="one_to_many")

            observed = observed.loc[observed.date.ge(observed.sales_open) & observed.date.lt(observed.cutoff)]
            sold = observed.groupby(KEYS, as_index=False).tickets.sum().rename(columns={"tickets": "sold"})
            frame = frame.merge(sold, on=KEYS, how="left", validate="one_to_one")
            frame["sold"] = frame.sold.fillna(0).astype("int64")
            frames.append(frame.drop(columns="sales_open"))
    return pd.concat(frames, ignore_index=True)


def historical_prediction_rule(raw, batch: pd.DataFrame) -> np.ndarray:
    """Единое округление, минимум sold и максимум seats для всех методов.

    sold доступен на момент: cutoff; значение обрезано в общем примере.
    seats доступен на момент: cutoff по согласованному допущению PROTOCOL.
    """
    predicted = np.asarray(raw, dtype=float)
    sold = batch.sold.to_numpy(dtype=float)
    if predicted.shape != sold.shape or not np.isfinite(predicted).all():
        raise ValueError("Нужен один конечный прогноз на строку")
    if not np.isfinite(sold).all() or (sold < 0).any() or (sold % 1 != 0).any():
        raise ValueError("Наблюдаемые продажи должны быть целыми и неотрицательными")
    if not batch.scenario.isin([0, 1]).all() or ((batch.scenario == 0) & (batch.sold != 0)).any():
        raise ValueError("В сценарии без продаж sold должен быть равен нулю")

    seats = batch.seats.to_numpy(dtype=float)
    if (seats.shape != sold.shape or not np.isfinite(seats).all()
            or (seats < 0).any() or (seats % 1 != 0).any()):
        raise ValueError("Число мест должно быть конечным, целым и неотрицательным")
    if (sold > seats).any():
        raise ValueError("Противоречие: уже продано больше seats")
    return np.minimum(np.maximum(np.floor(predicted + 0.5), sold), seats)
