"""Общая временная проверка; правила сценарных срезов задаёт PROTOCOL.md.

Этот модуль не выбирает отсечки, исторический cap или округление.
Все три окна служат для выбора метода; независимого финального теста нет.
"""

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

WINDOWS = ((35, 51), (52, 68), (69, 85))
KEYS = ["match_id", "zone"]
HISTORY_COLUMNS = KEYS + ["date", "tickets_total"]
EXAMPLE_COLUMNS = KEYS + ["scenario", "slice_id", "cutoff", "weight"]
FORBIDDEN_FEATURES = {"tickets_total", "cap", "season_tickets"}


def rmse(y_true, y_pred, sample_weight=None) -> float:
    """Корень среднего квадрата ошибки; веса необязательны, без округления."""
    actual, predicted = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not actual.size:
        raise ValueError("Нужны два непустых одномерных массива одинаковой длины")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Цели и прогнозы должны быть конечными числами")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights is not None:
        if (
            weights.shape != actual.shape
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "Веса должны быть конечными, неотрицательными и с суммой > 0"
            )
    return float(np.sqrt(np.average((actual - predicted) ** 2, weights=weights)))


def _check_pairs(frame: pd.DataFrame, groups: list[str]) -> None:
    """Одна запись каждой из восьми зон внутри группы."""
    if frame.empty or frame.duplicated(groups + ["zone"]).any():
        raise ValueError("Пустая таблица или повторяющиеся зоны внутри группы")
    complete = frame.groupby(groups, dropna=False).zone.apply(
        lambda values: set(values) == set(range(1, 9)) and len(values) == 8
    )
    if not complete.all():
        raise ValueError("В каждой группе должны быть все восемь зон")


def rolling_windows(
    table: pd.DataFrame,
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Выдать (имя окна, история до окна, проверочные 17 матчей).

    table: match_id, zone, date (дата матча), tickets_total; только train.
    Все восемь зон остаются вместе. История расширяется: 34, 51, 68 матчей.
    Это разбиение по датам матчей, не доказательство доступности истории:
    evaluate_method дополнительно исключает незавершённые к cutoff матчи.
    """
    _check_pairs(table, ["match_id"])
    dates = pd.to_datetime(table["date"], errors="raise")
    if dates.isna().any() or table.groupby("match_id")["date"].nunique().ne(1).any():
        raise ValueError("У каждого матча должна быть одна непустая дата")
    calendar = (
        table.assign(date=dates).drop_duplicates("match_id").sort_values("match_id")
    )
    expected = [f"M{i:03}" for i in range(1, 86)]
    if (
        calendar.match_id.tolist() != expected
        or not calendar.date.is_monotonic_increasing
    ):
        raise ValueError("Нужны M001–M085 в порядке дат, без test")
    for first, last in WINDOWS:
        start, end = f"M{first:03}", f"M{last:03}"
        history = table.loc[table.match_id.lt(start)].copy()
        checking = table.loc[table.match_id.between(start, end)].copy()
        yield f"{start}–{end}", history, checking


def _prepare_examples(examples: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Проверить явные отсечки и равные суммарные веса матчей."""
    if not set(EXAMPLE_COLUMNS).issubset(examples.columns):
        raise ValueError(f"В примерах нужны столбцы {EXAMPLE_COLUMNS}")
    if FORBIDDEN_FEATURES & set(examples.columns):
        raise ValueError(
            "Примеры не должны содержать цели или окончательные исторические границы"
        )
    result = examples.copy().reset_index(drop=True)
    if result[EXAMPLE_COLUMNS].isna().any().any():
        raise ValueError("Ключи, отсечки и веса не должны содержать пропуски")
    if not result.scenario.isin([0, 1]).all():
        raise ValueError("Сценарий: 0 без продаж или 1 с продажами")
    result["cutoff"] = pd.to_datetime(result.cutoff, errors="raise")
    if result.cutoff.dt.tz is not None:
        raise ValueError("Отсечки задаются местным временем без часового пояса")
    result["weight"] = pd.to_numeric(result.weight, errors="raise")
    if not np.isfinite(result.weight).all() or result.weight.le(0).any():
        raise ValueError("Вес каждого среза должен быть конечным и положительным")
    slice_keys = ["scenario", "match_id", "slice_id"]
    _check_pairs(result, slice_keys)
    grouped = result.groupby(slice_keys)
    if grouped.cutoff.nunique().ne(1).any() or grouped.weight.nunique().ne(1).any():
        raise ValueError("Восемь зон одного среза имеют одну отсечку и один вес")
    sums = result.groupby(["scenario"] + KEYS).weight.sum()
    if not np.allclose(sums, 1.0, rtol=0, atol=1e-12):
        raise ValueError("Веса срезов каждой пары матч–зона должны суммироваться в 1")
    calendar = table[KEYS + ["date"]].rename(columns={"date": "match_date"})
    result = result.merge(calendar, on=KEYS, how="left", validate="many_to_one")
    result["match_date"] = pd.to_datetime(result.match_date, errors="raise")
    if result.match_date.isna().any():
        raise ValueError("Проверочные пары отсутствуют в train")
    if result.cutoff.ge(result.match_date.dt.normalize()).any():
        raise ValueError("Момент прогноза должен быть раньше дня проверочного матча")
    result["window"] = ""
    for first, last in WINDOWS:
        mask = result.match_id.between(f"M{first:03}", f"M{last:03}")
        result.loc[mask, "window"] = f"M{first:03}–M{last:03}"
    if result.window.eq("").any():
        raise ValueError("Примеры должны принадлежать одному из трёх окон")
    if result.groupby("scenario").window.nunique().ne(len(WINDOWS)).any():
        raise ValueError("Каждый сценарий должен быть представлен во всех трёх окнах")
    return result


def _summarize(details: pd.DataFrame) -> pd.DataFrame:
    """Сначала усреднить квадраты ошибок срезов, затем пары матч–зона."""
    pair_keys = ["window", "scenario"] + KEYS
    pairs = details.assign(weighted_error=details.squared_error * details.weight)
    pairs = pairs.groupby(pair_keys, as_index=False).weighted_error.sum()
    rows = []
    for (window, scenario), group in pairs.groupby(["window", "scenario"]):
        for zone in ["all"] + list(range(1, 9)):
            selected = group if zone == "all" else group.loc[group.zone.eq(zone)]
            mse = float(selected.weighted_error.mean())
            rows.append(
                {
                    "window": window,
                    "scenario": scenario,
                    "zone": zone,
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "n_matches": selected.match_id.nunique(),
                    "n_pairs": len(selected),
                }
            )
    return pd.DataFrame(rows)


def evaluate_method(
    method: Callable,
    table: pd.DataFrame,
    examples: pd.DataFrame,
    feature_columns: tuple[str, ...] = (),
    prediction_rule: Callable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Общий интерфейс метода: summary, details = evaluate_method(...).

    table: 680 известных train-итогов с match_id, zone, date, tickets_total.
    examples: фиксированные ПРОТОКОЛОМ примеры с match_id, zone, scenario
    (0/1), slice_id, cutoff (момент прогноза), weight. Восемь зон среза
    имеют одну отсечку и один вес; сумма весов срезов пары в сценарии = 1.
    Каждый представленный сценарий покрывает все три окна. Состав
    примеров/веса одни и те же для baseline и всех кандидатов.
    Готовые правила отсечек и смешивания 27/7 этот модуль не назначает.

    method(history, batch) -> одномерный массив прогнозов в порядке batch.
    Вызов отдельный для каждого окна и cutoff. history содержит только
    HISTORY_COLUMNS: матчи ДО окна, завершённые до дня cutoff. batch
    содержит EXAMPLE_COLUMNS, match_date, window и явные feature_columns.
    Проверочный tickets_total и окончательные cap/season_tickets методу
    не передаются. Метод не должен сохранять обученное состояние между
    вызовами; построение его признаков обязано соблюдать PROTOCOL.md,
    в том числе собственные исторические отсечки обучающих примеров.
    История доступна на момент: утро cutoff, только итоги матчей прошлых дней.

    prediction_rule(raw, batch) -> массив: единые ограничения/округление
    для всех методов, если согласованы; None оставляет прогнозы как есть.
    Ошибка считается только ПОСЛЕ этого общего правила. Доступность
    дополнительных признаков/границ проверяется отдельным leakage-audit.
    Согласованные 04.10.2026 examples и prediction_rule строит scenarios.py.
    Произвольные отсечки/веса не являются оценкой по принятому протоколу.

    summary: window, scenario, zone ('all' или 1..8), mse, rmse,
    n_matches, n_pairs. details: ключи/отсечки/веса, цель, прогноз и квадрат
    ошибки для каждого среза. Сначала суммируем weight * squared_error
    внутри пары матч–зона, затем усредняем пары; каждый матч весит равно.
    Для агрегатов R/S использовать mse строки zone='all' по PROTOCOL.md.
    """
    payload_columns = EXAMPLE_COLUMNS + ["match_date", "window"]
    invalid = set(feature_columns) & (FORBIDDEN_FEATURES | set(payload_columns))
    if invalid or len(set(feature_columns)) != len(feature_columns):
        raise ValueError(f"Запрещённые, служебные или повторные признаки: {invalid}")
    if not set(feature_columns).issubset(examples.columns):
        raise ValueError("Заявленные признаки отсутствуют в examples")
    windows = list(rolling_windows(table))
    prepared = _prepare_examples(examples, table)
    targets = table[KEYS + ["tickets_total"]]
    if not np.isfinite(targets.tickets_total).all() or targets.tickets_total.lt(0).any():
        raise ValueError("Train-итоги должны быть конечными и неотрицательными")
    rows = []
    for window, history, _ in windows:
        checking = prepared.loc[prepared.window.eq(window)]
        for cutoff, batch in checking.groupby("cutoff", sort=True):

            available = history.loc[pd.to_datetime(history.date).lt(cutoff.normalize())]
            if available.empty:
                raise ValueError(f"Нет завершённой истории для {window}, {cutoff}")
            payload = batch[payload_columns + list(feature_columns)].copy()
            raw = method(available[HISTORY_COLUMNS].copy(), payload.copy())
            if isinstance(raw, pd.Series) and not raw.index.equals(payload.index):
                raise ValueError("Индекс исходных прогнозов должен совпадать с batch")
            predicted = (
                raw if prediction_rule is None else prediction_rule(raw, payload.copy())
            )
            if isinstance(predicted, pd.Series) and not predicted.index.equals(
                payload.index
            ):
                raise ValueError("Индекс Series с прогнозами должен совпадать с batch")
            actual = batch.merge(targets, on=KEYS, validate="many_to_one")
            rmse(actual.tickets_total.to_numpy(), predicted)
            detail = batch[EXAMPLE_COLUMNS + ["window"]].copy()
            detail["tickets_total"] = actual.tickets_total.to_numpy()
            detail["prediction"] = np.asarray(predicted, dtype=float)
            detail["squared_error"] = (detail.tickets_total - detail.prediction) ** 2
            rows.append(detail)
    details = pd.concat(rows, ignore_index=True)
    return _summarize(details), details


def aggregate_scores(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Вычислить R[w], S и S[0]/S[1] строго по PROTOCOL.md.

    Использует только mse строк zone='all': ровно два сценария каждого
    из трёх окон. R смешивает MSE с весами 11/17 и 6/17; S смешивает
    готовые R с весами 0.25/0.25/0.50. Зональные строки не участвуют.
    """
    window_names = [f"M{first:03}–M{last:03}" for first, last in WINDOWS]
    overall = summary.loc[summary.zone.eq("all"), ["window", "scenario", "mse"]]
    expected = {(window, scenario) for window in window_names for scenario in (0, 1)}
    if overall.duplicated(["window", "scenario"]).any() or set(
        map(tuple, overall[["window", "scenario"]].to_numpy())
    ) != expected:
        raise ValueError("Для R/S нужны оба сценария всех трёх окон, без дублей")
    if not np.isfinite(overall.mse).all() or overall.mse.lt(0).any():
        raise ValueError("MSE должна быть конечной и неотрицательной")
    mse = overall.pivot(index="window", columns="scenario", values="mse").loc[window_names]
    windows = pd.DataFrame({"window": window_names,
                            "rmse_no_sales": np.sqrt(mse[0].to_numpy()),
                            "rmse_sales": np.sqrt(mse[1].to_numpy()),
                            "R": np.sqrt((11 / 17) * mse[0].to_numpy() + (6 / 17) * mse[1].to_numpy())})
    window_weights = np.array([0.25, 0.25, 0.50])
    scores = {"S": float(window_weights @ windows.R.to_numpy()),
              "S_0": float(window_weights @ windows.rmse_no_sales.to_numpy()),
              "S_1": float(window_weights @ windows.rmse_sales.to_numpy())}
    return windows, scores
