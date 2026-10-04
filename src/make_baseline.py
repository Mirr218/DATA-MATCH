"""Сборка test: python -X utf8 -m src.make_baseline; только outputs/predictions_baseline.csv."""
import numpy as np
import pandas as pd

from .baseline import zone_mean_baseline
from .baseline_data import ROOT, load_training_table
from .scenarios import historical_prediction_rule
from .validation import KEYS

TEST_CUTOFF = pd.Timestamp("2025-12-15")
TEST_ROWS = 136
TRAIN_MATCHES = 85
OUTPUT = ROOT / "outputs/predictions_baseline.csv"


def validate_submission(submission, template, bounds) -> None:
    """Проверить формат, порядок и границы до записи и после чтения CSV."""
    expected_columns = KEYS + ["tickets_total"]
    if template.columns.tolist() != expected_columns or submission.columns.tolist() != expected_columns:
        raise ValueError("Столбцы должны точно совпадать с шаблоном организаторов")
    if len(submission) != TEST_ROWS or len(template) != TEST_ROWS:
        raise ValueError("Нужно ровно 136 строк данных")
    if submission.isna().any().any() or submission.duplicated(KEYS).any():
        raise ValueError("Пропуски или дубли в прогнозе")
    if not submission[KEYS].equals(template[KEYS]):
        raise ValueError("Пары матч–зона и порядок должны точно совпадать с шаблоном")
    values = submission.tickets_total
    if not pd.api.types.is_integer_dtype(values) or not np.isfinite(values).all():
        raise ValueError("Прогнозы должны быть конечными целыми числами")
    checked = submission.merge(bounds[KEYS + ["sold", "cap"]], on=KEYS,
                               how="left", validate="one_to_one")
    if checked.isna().any().any():
        raise ValueError("Для каждой пары нужны границы")
    if not np.isfinite(checked[["sold", "cap"]]).all().all():
        raise ValueError("Границы должны быть конечными")
    if checked.sold.lt(0).any() or checked.cap.lt(checked.sold).any():
        raise ValueError("Противоречивые границы: продано больше потолка или отрицательная продажа")
    if values.lt(0).any() or checked.tickets_total.lt(checked.sold).any() or checked.tickets_total.gt(checked.cap).any():
        raise ValueError("Прогноз вне допустимых границ")


def build_submission(history, matches, zones, sales, template):
    """Создать ответ в порядке шаблона, применить округление и test-границы.

    доступен на момент: утро 15.12.2025; история только завершённых матчей,
    sold только по 14.12 включительно. cap доступен на момент: предоставленные
    организаторами test-параметры zones, как предписывает PROTOCOL.
    Исторические абонементы и КХЛ не входят в прогноз среднего по зоне.
    """
    test_ids = set(matches.loc[matches.part.eq("test"), "match_id"])
    train_ids = set(matches.loc[matches.part.eq("train"), "match_id"])
    if set(template.match_id) != test_ids or template[KEYS].isna().any().any():
        raise ValueError("Шаблон должен содержать только все test-матчи")
    expected_pairs = {(match_id, zone) for match_id in test_ids for zone in range(1, 9)}
    if set(map(tuple, template[KEYS].to_numpy())) != expected_pairs:
        raise ValueError("В шаблоне нужны все восемь зон каждого test-матча")
    if not pd.to_datetime(history.date).lt(TEST_CUTOFF).all() or not set(history.match_id).issubset(train_ids):
        raise ValueError("История должна содержать только завершённые к test-отсечке train-матчи")
    daily = sales.loc[sales.match_id.isin(test_ids)].copy()
    daily["date"] = pd.to_datetime(daily.date, errors="raise")
    daily = daily.loc[daily.date.lt(TEST_CUTOFF)]
    if daily[KEYS + ["date", "tickets"]].isna().any().any() or daily.duplicated(KEYS + ["date"]).any():
        raise ValueError("Пропуски или дубли известных test-продаж")
    if not np.isfinite(daily.tickets).all() or daily.tickets.lt(0).any() or (daily.tickets % 1 != 0).any():
        raise ValueError("Известные продажи должны быть целыми неотрицательными числами")
    sold = daily.groupby(KEYS, as_index=False).tickets.sum().rename(columns={"tickets": "sold"})
    test_zones = zones.loc[zones.match_id.isin(test_ids), KEYS + ["seats", "season_tickets"]]
    bounds = template[KEYS].merge(test_zones, on=KEYS, how="left", validate="one_to_one")
    bounds = bounds.merge(sold, on=KEYS, how="left", validate="one_to_one")
    bounds["sold"] = bounds.sold.fillna(0).astype("int64")
    bounds["cap"] = bounds.seats - bounds.season_tickets
    if bounds.isna().any().any() or not np.isfinite(bounds[["seats", "season_tickets", "cap"]]).all().all():
        raise ValueError("Пропуски или неконечные test-параметры зон")
    if (bounds.cap % 1 != 0).any() or bounds.cap.lt(bounds.sold).any():
        raise ValueError("Потолок должен быть целым и не меньше уже проданного")
    bounds["scenario"] = bounds.match_id.isin(set(daily.match_id)).astype("int64")
    raw = zone_mean_baseline(history, bounds)
    bounded = historical_prediction_rule(raw, bounds)
    submission = template[KEYS].copy()
    submission["tickets_total"] = np.minimum(bounded, bounds.cap.to_numpy()).astype("int64")
    validate_submission(submission, template, bounds)
    return submission, bounds


def main() -> None:
    """Проверить перед сохранением, записать только OUTPUT, проверить чтение CSV."""
    matches = pd.read_csv(ROOT / "data/matches.csv", parse_dates=["date", "sales_open"])
    zones = pd.read_csv(ROOT / "data/zones.csv")
    sales = pd.read_csv(ROOT / "data/sales_daily.csv", parse_dates=["date"])
    template = pd.read_csv(ROOT / "data/submission_template.csv")
    history = load_training_table()[KEYS + ["tickets_total"]].merge(
        matches[["match_id", "date"]], on="match_id", validate="many_to_one")
    if history.match_id.nunique() != TRAIN_MATCHES:
        raise ValueError("Для итогового baseline нужны все 85 train-матчей")
    submission, bounds = build_submission(history, matches, zones, sales, template)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT, index=False)
    validate_submission(pd.read_csv(OUTPUT), template, bounds)
    print(submission.head(5).to_string(index=False))
    print(f"Строк: {len(submission)}; история: {history.match_id.nunique()} матчей.")
    print("Формат, порядок, целочисленность, минимум sold и потолок: OK.")
    print(f"Сохранён только: {OUTPUT}")


if __name__ == "__main__":
    main()
