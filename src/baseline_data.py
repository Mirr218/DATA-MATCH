"""Таблица известных итогов train; параметры зон здесь описательные, не признаки."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["match_id", "zone"]
ZONES = set(range(1, 9))


def load_training_table(root: Path = ROOT) -> pd.DataFrame:
    """Суммировать tickets только для train, присоединить параметры зон.

    cap — описательный окончательный потолок. Его историческая доступность
    не установлена: нельзя автоматически использовать его в валидации.
    Полные test-итоги неизвестны, поэтому test в результат не входит.
    По README отсутствие записей продаж означает ноль.
    """

    matches = pd.read_csv(root / "data/matches.csv")
    zones = pd.read_csv(root / "data/zones.csv")
    sales = pd.read_csv(root / "data/sales_daily.csv")

    train = matches.loc[matches.part.eq("train"), ["match_id"]]
    totals = (
        sales.loc[sales.match_id.isin(train.match_id)]
        .groupby(KEYS, as_index=False)
        .tickets.sum()
        .rename(columns={"tickets": "tickets_total"})
    )
    table = train.merge(zones, on="match_id", validate="one_to_many")
    table = table.merge(totals, on=KEYS, how="left", validate="one_to_one")


    table["tickets_total"] = table.tickets_total.fillna(0).astype("int64")

    table["cap"] = table.seats - table.season_tickets

    assert not table.duplicated(KEYS).any(), "Дубли матч–зона"
    assert not table.isna().any().any(), "Пропуски"
    assert set(map(tuple, table[KEYS].to_numpy())) == {
        (match_id, zone) for match_id in train.match_id for zone in ZONES
    }, "У каждого матча должны быть все восемь зон"
    assert table.cap.ge(0).all(), "Отрицательный потолок"
    assert table.tickets_total.between(0, table.cap).all(), "Итог вне границ"

    columns = KEYS + ["tickets_total", "seats", "season_tickets", "cap"]

    return table.sort_values(KEYS)[columns].reset_index(drop=True)


if __name__ == "__main__":
    table = load_training_table()
    output = ROOT / "outputs/match_zone_train.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.head(5).to_string(index=False))
    print(f"Строк: {len(table)}; матчей: {table.match_id.nunique()}")
