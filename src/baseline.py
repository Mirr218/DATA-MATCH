"""Среднее по зоне; запуск исходного ориентира: python -X utf8 -m src.baseline.

Ориентир использует всю историю до окна, без округления и cap.
Он не имитирует ранние сценарные отсечки и не является оценкой R/S.
"""

import numpy as np
import pandas as pd

from .baseline_data import ROOT, load_training_table
from .validation import HISTORY_COLUMNS, KEYS, rmse, rolling_windows


def zone_mean_baseline(history: pd.DataFrame, batch: pd.DataFrame) -> pd.Series:
    """Прогноз среднего tickets_total по зоне, только из переданной истории.

    Совместим с evaluate_method: history уже отфильтрована по окну и cutoff.
    Правильные ответы batch не используются. Ограничения и округление
    применяются отдельно общим prediction_rule после согласования.
    доступен на момент: cutoff; только завершённая переданная история до окна.
    """
    if history.empty or history.duplicated(KEYS).any():
        raise ValueError("Нужна непустая история без повторов матч–зона")
    if not np.isfinite(history.tickets_total).all() or history.tickets_total.lt(0).any():
        raise ValueError("Исторические итоги должны быть конечными и неотрицательными")


    means = history.groupby("zone").tickets_total.mean()
    predicted = batch.zone.map(means)
    if predicted.isna().any():
        raise ValueError("В истории нет одной из прогнозируемых зон")
    return predicted.rename("prediction")


def evaluate_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Воспроизвести схему из DATA_FACTS, отдельно от сценарной проверки.

    Средние фиксированы на окно. Проверяем, что история завершилась до
    дня первого матча окна; это НЕ ранее согласованные сценарные отсечки.
    Возвращает RMSE общего окна/восьми зон и подробности 408 прогнозов.
    """
    table = load_training_table()[KEYS + ["tickets_total"]]
    calendar = pd.read_csv(ROOT / "data/matches.csv", usecols=["match_id", "date"])
    calendar["date"] = pd.to_datetime(calendar.date, errors="raise")
    table = table.merge(calendar, on="match_id", validate="many_to_one")
    rows, details = [], []
    for window, history, checking in rolling_windows(table):
        reference_moment = checking.date.min().normalize()
        if not history.date.lt(reference_moment).all():
            raise ValueError("В исходной истории есть незавершённые к началу окна матчи")

        predicted = zone_mean_baseline(history[HISTORY_COLUMNS], checking[KEYS])
        detail = checking[KEYS + ["tickets_total"]].copy()
        detail["window"] = window
        detail["prediction"] = predicted
        detail["squared_error"] = (detail.tickets_total - detail.prediction) ** 2
        details.append(detail)
        row = {
            "window": window,
            "train_matches": history.match_id.nunique(),
            "check_matches": checking.match_id.nunique(),
            "rmse": rmse(detail.tickets_total, detail.prediction),
        }
        for zone, group in detail.groupby("zone"):
            row[f"z{zone}"] = rmse(group.tickets_total, group.prediction)
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


if __name__ == "__main__":
    metrics, details = evaluate_reference()
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "baseline_reference_metrics.csv", index=False)
    details.to_csv(output / "baseline_reference_details.csv", index=False)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Исходный ориентир: без округления, cap и сценарных отсечек; это не R/S.")
