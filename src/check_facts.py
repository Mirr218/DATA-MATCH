"""Независимая проверка исходных утверждений DATA_FACTS; запуск: python src/check_facts.py.

Ожидания ниже переписаны из исходной версии документа, результаты считаются из CSV.
Скрипт сохраняет отчёт, но не меняет данные, DATA_FACTS или predictions.csv.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CUTOFF = pd.Timestamp("2025-12-14")
ROWS = []


def record(fact, expected, actual, matches=None):
    status = "НЕ ПРОВЕРЯЕТСЯ ПО CSV" if matches is None else ("ДА" if matches else "НЕТ")
    if str(expected).startswith("уточнение"):
        status = "УТОЧНЕНИЕ"
    ROWS.append((fact, str(expected), str(actual), status))


def rounded(values, digits=0):
    return [round(float(value), digits) for value in values]


def vector(fact, expected, values, digits=0):
    actual = rounded(values, digits)
    record(fact, expected, actual, actual == expected)


def load_data():
    names = (
        "khl_games",
        "matches",
        "sales_daily",
        "season_tickets_daily",
        "submission_template",
        "zones",
    )
    tables = {name: pd.read_csv(DATA / f"{name}.csv") for name in names}
    m, s, z, k = (
        tables[name] for name in ("matches", "sales_daily", "zones", "khl_games")
    )
    for table, columns in ((m, ["date", "sales_open"]), (s, ["date"]), (k, ["date"])):
        for column in columns:
            table[column] = pd.to_datetime(table[column], errors="raise")
    for table, keys in (
        (m, ["match_id"]),
        (z, ["match_id", "zone"]),
        (s, ["match_id", "zone", "date"]),
    ):
        assert not table.duplicated(keys).any(), f"Повторяющиеся ключи: {keys}"
        assert not table.isna().any().any(), f"Пропуски в таблице с ключами {keys}"
    assert set(s.match_id) <= set(m.match_id) and set(z.match_id) == set(m.match_id)
    assert set(zip(z.match_id, z.zone)) == {
        (mid, zone) for mid in m.match_id for zone in range(1, 9)
    }
    assert set(zip(s.match_id, s.zone)) <= set(zip(z.match_id, z.zone))
    assert s.tickets.ge(0).all() and (s.tickets % 1 == 0).all()
    totals = s.groupby(["match_id", "zone"]).tickets.sum().rename("total")
    pairs = z.merge(m, on="match_id", validate="many_to_one").merge(
        totals, on=["match_id", "zone"], how="left", validate="one_to_one"
    )
    pairs["total"] = pairs.total.fillna(0)
    pairs["cap"] = pairs.seats - pairs.season_tickets
    assert pairs.cap.gt(0).all() and pairs.total.le(pairs.cap).all()
    return tables, m, s, z, k, pairs


def check_structure(m, s, pairs):
    train, test = m[m.part.eq("train")], m[m.part.eq("test")]
    expected_ids = (
        [f"M{i:03}" for i in range(1, 86)],
        [f"M{i:03}" for i in range(86, 103)],
    )
    actual = (len(m), m.groupby("season").size().tolist(), len(train), len(test))
    record(
        "Матчи, сезоны, train/test, идентификаторы",
        "102; 3×34; M001–085 / M086–102; test 2025/26",
        actual,
        actual == (102, [34, 34, 34], 85, 17)
        and (sorted(train.match_id), sorted(test.match_id)) == expected_ids
        and set(test.season) == {"2025/26"},
    )
    record(
        "Определение цели и отсутствующей строки",
        "Сумма tickets; нет строки = 0",
        "Правило README; суммы вычислены; смысл отсутствия строки не доказуем по CSV",
    )
    tz = pairs[pairs.part.eq("test")].season_tickets
    vector(
        "Абонементы в тестовых зонах: минимум, максимум, пропуски",
        [126, 654, 0],
        [tz.min(), tz.max(), tz.isna().sum()],
    )
    record(
        "Определение cap",
        "seats − season_tickets",
        "Правило README; доступные суммы продаж не превышают cap",
    )
    joined = s.merge(
        m[["match_id", "date", "sales_open"]],
        on="match_id",
        suffixes=("_sale", "_match"),
        validate="many_to_one",
    )
    bad = (joined.days_before != (joined.date_match - joined.date_sale).dt.days).sum()
    record("days_before совпадает с разностью дат", "0 ошибок", int(bad), bad == 0)
    record(
        "Продажи не позже 14.12.2025",
        "≤ 2025-12-14",
        str(s.date.max().date()),
        s.date.max() <= CUTOFF,
    )
    assert (
        joined.date_sale.ge(joined.sales_open).all()
        and joined.date_sale.le(joined.date_match).all()
    )
    for first, last, opening, days, ahead in (
        (86, 89, "2025-11-18", 27, [10, 12, 14, 16]),
        (90, 91, "2025-12-08", 7, [30, 32]),
    ):
        ids = [f"M{i:03}" for i in range(first, last + 1)]
        group = test.set_index("match_id").loc[ids]
        dates = (
            s[s.match_id.isin(ids)]
            .groupby("match_id")
            .date.agg(["min", "max", "nunique"])
        )
        actual = f"открытие {group.sales_open.dt.strftime('%Y-%m-%d').tolist()}; дней с продажами {dates['nunique'].tolist()}; до матчей {(group.date-CUTOFF).dt.days.tolist()}"
        ok = (
            group.sales_open.eq(pd.Timestamp(opening)).all()
            and dates["min"].eq(pd.Timestamp(opening)).all()
            and dates["max"].eq(CUTOFF).all()
            and dates["nunique"].eq(days).all()
            and (group.date - CUTOFF).dt.days.tolist() == ahead
        )
        record(
            f"M{first:03}–M{last:03}: начало, длительность, до матча",
            f"{opening}; {days} дней включительно; {ahead}",
            actual,
            bool(ok),
        )
        if first == 86:
            vector(
                "M086–089: разность даты матча и открытия",
                [36, 42],
                [
                    (group.date - group.sales_open).dt.days.min(),
                    (group.date - group.sales_open).dt.days.max(),
                ],
            )
    absent = sorted(set(test.match_id) - set(s.match_id))
    record(
        "11 тестовых матчей без продаж",
        "M092–M102",
        absent,
        absent == [f"M{i:03}" for i in range(92, 103)],
    )
    for label, group, expected in (
        ("train", train, [10, 63, 32]),
        ("test", test, [28, 42, 36]),
    ):
        duration = (group.date - group.sales_open).dt.days
        vector(
            f"Окно продаж {label}: min/max/median",
            expected,
            [duration.min(), duration.max(), duration.median()],
        )
    record(
        "11 матчей ещё не открылись",
        "sales_open > 14.12.2025",
        int(test.sales_open.gt(CUTOFF).sum()),
        set(test.loc[test.sales_open.gt(CUTOFF), "match_id"]) == set(absent),
    )


def check_khl(m, k):
    last = k.date.max()
    record(
        "Последняя дата КХЛ",
        "2025-11-19",
        str(last.date()),
        last == pd.Timestamp("2025-11-19"),
    )
    count = int(k.season.eq("2025/26").sum())
    record("Игры КХЛ сезона 2025/26", 302, count, count == 302)
    home = m[m.date.le(last)].sort_values("date").iloc[-1]
    record(
        "Последний домашний матч до среза КХЛ",
        "M083 / 2025-11-13",
        f"{home.match_id} / {home.date.date()}",
        home.match_id == "M083" and home.date == pd.Timestamp("2025-11-13"),
    )
    later = sorted(m.loc[m.date.gt(last), "match_id"])
    record(
        "Матчи после среза КХЛ",
        "M084–M102",
        later,
        later == [f"M{i:03}" for i in range(84, 103)],
    )
    farthest = m.loc[m.part.eq("test"), "date"].max()
    record(
        "Самый дальний test / давность КХЛ",
        "2026-03-13 / около 4 месяцев",
        f"{farthest.date()} / {(farthest-last).days} дней",
        farthest == pd.Timestamp("2026-03-13") and 105 <= (farthest - last).days <= 135,
    )
    clubs = set(k.home) | set(k.away)
    missing = sorted(set(m.opponent_khl) - clubs)
    record(
        "Название клуба и покрытие соперников",
        "HC Avangard Omsk; все соперники есть",
        f"клуб есть: {'HC Avangard Omsk' in clubs}; отсутствуют: {missing}",
        "HC Avangard Omsk" in clubs and not missing,
    )


def check_demand(train):
    grouped = train.groupby("zone").total
    vector(
        "Средний итог по зонам 1–8",
        [465, 308, 633, 782, 767, 487, 781, 450],
        grouped.mean(),
    )
    vector(
        "Стандартное отклонение по зонам (ddof=1)",
        [36, 40, 62, 168, 133, 239, 205, 75],
        grouped.std(ddof=1),
    )
    for season, expected in (
        ("2023/24", [609, 337, 1035]),
        ("2024/25", [285, 147, 368]),
        ("2025/26", [649, 236, 894]),
    ):
        group = train.loc[train.zone.eq(6) & train.season.eq(season), "total"]
        vector(
            f"Зона 6, {season}: среднее/min/max",
            expected,
            [group.mean(), group.min(), group.max()],
        )
    for zone, expected in ((4, [55, 17]), (5, [49, 27])):
        values = train.loc[train.zone.eq(zone), "total"]
        vector(
            f"Зона {zone}: количества [750,900] и [300,750)",
            expected,
            [values.between(750, 900).sum(), (values.ge(300) & values.lt(750)).sum()],
        )
        rest = values[~values.between(750, 900)]
        record(
            f"Зона {zone}: остальные ниже 750",
            "все остальные ниже",
            f"выше 900: {int(values.gt(900).sum())}; ниже 300: {int(values.lt(300).sum())}",
            rest.lt(750).all(),
        )
    zone7 = train.loc[train.zone.eq(7), "total"]
    vector("Зона 7: фактический min/max", [300, 1350], [zone7.min(), zone7.max()])
    record(
        "Зона 7: нет явной полки",
        "без явной полки",
        "Нет определения полки; диапазон сам по себе не доказывает форму распределения",
    )
    maxima = (train.total / train.cap).groupby(train.zone).max()
    vector(
        "Максимум total/cap по зонам 1–8",
        [0.84, 0.90, 0.72, 0.88, 0.87, 0.77, 0.93, 0.97],
        maxima,
        2,
    )


def check_occupancy(train):
    match = train.groupby("match_id").agg(
        total=("total", "sum"),
        cap=("cap", "sum"),
        weekday=("weekday", "first"),
        opponent=("opponent", "first"),
        time=("time", "first"),
    )
    match["occupancy"] = match.total / match.cap
    week = match.groupby("weekday").occupancy.mean()
    opp = match.groupby("opponent").occupancy.agg(["mean", "size"])
    vector(
        "Заполняемость: min/max средних по дням, std по матчам",
        [0.62, 0.70, 0.07],
        [week.min(), week.max(), match.occupancy.std(ddof=1)],
        2,
    )
    vector(
        "Заполняемость по соперникам: min/max",
        [0.54, 0.70],
        [opp["mean"].min(), opp["mean"].max()],
        2,
    )
    vector(
        "Соперников всего / с ≥3 матчами; время 17:00 или 19:30",
        [22, 18, 83],
        [len(opp), opp["size"].ge(3).sum(), match.time.isin(["17:00", "19:30"]).sum()],
    )
    record(
        "Разница сопоставима с шумом; по зонам не проверялось",
        "интерпретация и история анализа",
        "Нужны критерий сравнения и отдельный анализ; историю чужих проверок CSV не подтверждает",
    )


def check_pace(train, s):
    totals = train.set_index(["match_id", "zone"]).total
    assert totals.gt(0).all(), "Доли не определены при нулевом итоге"
    sales = s[s.match_id.isin(train.match_id.unique())]
    groups = ([1, 2, 3], [4], [5, 6], [7, 8])
    for threshold, expected in ((10, [65, 62, 55, 51]), (16, [50, 44, 36, 32])):
        early = (
            sales[sales.days_before.ge(threshold)]
            .groupby(["match_id", "zone"])
            .tickets.sum()
            .reindex(totals.index, fill_value=0)
        )
        fractions = early / totals
        values = [
            100 * fractions[fractions.index.get_level_values("zone").isin(group)].mean()
            for group in groups
        ]
        vector(
            f"Накопление к {threshold} дням: группы 1–3 / 4 / 5–6 / 7–8, %",
            expected,
            values,
        )
        record(
            f"Накопление к {threshold} дням: точные средние по зонам, %",
            "уточнение группового описания",
            rounded(100 * fractions.groupby("zone").mean(), 2),
        )
    late = (
        sales[sales.days_before.between(0, 3)]
        .groupby(["match_id", "zone"])
        .tickets.sum()
        .reindex(totals.index, fill_value=0)
        / totals
    )
    per_zone = 100 * late.groupby("zone").mean()
    vector(
        "Последние 4 дня: min/max средних долей зон, %",
        [15, 26],
        [per_zone.min(), per_zone.max()],
    )
    for label, mask, expected in (
        (">700", totals.gt(700), [15, 21]),
        ("<450", totals.lt(450), [26, 50]),
    ):
        selected = late[mask & (totals.index.get_level_values("zone") == 6)]
        vector(
            f"Зона 6, итог {label}: доля последних 4 дней, % / матчей",
            expected,
            [100 * selected.mean(), len(selected)],
        )
    peak = (
        sales.groupby(["match_id", "zone"]).tickets.max().reindex(totals.index) / totals
    )
    vector(
        "Максимальный день: средняя доля, доля пар >28%, максимум, %",
        [16, 10, 62],
        [100 * peak.mean(), 100 * peak.gt(0.28).mean(), 100 * peak.max()],
    )
    record(
        "Причина скачков зоны 6",
        "не выяснена",
        "Причинность не устанавливается описательными суммами",
    )
    for hypothesis in (
        "Ранний темп различает горячие и холодные матчи",
        "Скачки зоны 6 связаны с ценой/правилами",
        "Режимы зон 4–5 связаны с соперником/датой",
    ):
        record(
            "Гипотеза: " + hypothesis,
            "НЕ проверена",
            "Остаётся гипотезой; независимая проверка потребует отдельного исследования",
        )


def check_baseline(train):
    """Только воспроизведение опубликованной арифметики, без обучения модели."""
    windows = (
        (35, 51, 156, [48, 65, 74, 121, 152, 342, 157, 60]),
        (52, 68, 103, [46, 29, 57, 117, 110, 204, 86, 62]),
        (69, 85, 157, [34, 57, 77, 141, 139, 288, 242, 79]),
    )
    for first, last, expected_rmse, expected_zones in windows:
        validation = train[train.match_id.between(f"M{first:03}", f"M{last:03}")].copy()
        learning = train[train.date.lt(validation.date.min())]
        assert len(validation) == 17 * 8 and len(learning) == (first - 1) * 8
        assert learning.date.max() < validation.date.min()
        assert set(learning.match_id).isdisjoint(validation.match_id)
        averages = learning.groupby("zone").total.mean()
        validation["squared_error"] = (
            validation.total - validation.zone.map(averages)
        ) ** 2
        rmse = np.sqrt(validation.squared_error.mean())
        zone_rmse = np.sqrt(validation.groupby("zone").squared_error.mean())
        vector(
            f"Baseline M{first:03}–M{last:03}: RMSE и зоны 1–8",
            [expected_rmse] + expected_zones,
            [rmse] + zone_rmse.tolist(),
        )
        record(
            f"Baseline M{first:03}–M{last:03}: точные RMSE",
            "уточнение округления",
            rounded([rmse] + zone_rmse.tolist(), 3),
        )
    shares = (
        validation.groupby("zone").squared_error.sum() / validation.squared_error.sum()
    )
    vector(
        "Последнее окно: вклад зон 6+7 / 4+5 в SSE, %",
        [72, 20],
        [100 * shares.loc[[6, 7]].sum(), 100 * shares.loc[[4, 5]].sum()],
    )


def save_report(tables):
    lines = [
        "# Независимая проверка исходного DATA_FACTS.md",
        "",
        "Запуск: `python src/check_facts.py`. Ожидания относятся к версии ДО исправления документа.",
        "Порядок зон: 1–8; std выборочное (ddof=1); округление до указанной точности.",
        "Доли усредняются по всем парам матч–зона, а не взвешиваются числом билетов. До открытия продаж доля равна нулю.",
        "К d дням: days_before >= d; последние четыре дня: 0 <= days_before <= 3.",
        "Baseline: среднее только до начала окна, без округления и ограничения cap; все зоны матча вместе.",
        "«УТОЧНЕНИЕ» — дополнительный расчёт; «НЕ ПРОВЕРЯЕТСЯ ПО CSV» — правило, интерпретация или гипотеза.",
        "",
        "| Факт | Исходное утверждение | Что получилось | Совпало |",
        "|---|---|---|---|",
    ]
    for row in ROWS:
        lines.append(
            "| "
            + " | ".join(value.replace("|", "/").replace("\n", " ") for value in row)
            + " |"
        )
    lines += ["", "## Исходные файлы", "", "| Файл | Строк | SHA-256 |", "|---|---:|---|"]
    for name, table in tables.items():
        digest = hashlib.sha256((DATA / f"{name}.csv").read_bytes()).hexdigest()
        lines.append(f"| {name}.csv | {len(table)} | {digest} |")
    path = ROOT / "docs" / "DATA_FACTS_AUDIT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Проверок/уточнений: {len(ROWS)}; совпало: {sum(r[3] == 'ДА' for r in ROWS)}; расхождений: {sum(r[3] == 'НЕТ' for r in ROWS)}"
    )
    print(f"Отчёт: {path}")
    for fact, expected, actual, status in ROWS:
        if status == "НЕТ":
            print(f"НЕТ: {fact}: ожидалось {expected}; получилось {actual}")


def main():
    ROWS.clear()
    tables, m, s, z, k, pairs = load_data()
    record(
        "В файлах нет цен",
        "нет столбцов цен",
        {name: list(table.columns) for name, table in tables.items()},
        all(
            not any(token in column.lower() for token in ("price", "цена", "стоимость"))
            for table in tables.values()
            for column in table.columns
        ),
    )
    train = pairs[pairs.part.eq("train")].copy()
    check_structure(m, s, pairs)
    check_khl(m, k)
    check_demand(train)
    check_occupancy(train)
    check_pace(train, s)
    check_baseline(train)
    save_report(tables)


if __name__ == "__main__":
    main()
