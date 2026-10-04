# Подключение метода к общей проверке

Работайте из корня проекта; правила — в `docs/PROTOCOL.md`, их не меняйте.
Метод храните в своём согласованном файле/ветке; общая validation.py принадлежит лидеру.
Интерфейс: `method(history, batch)` возвращает одномерный массив в порядке batch либо Series с тем же индексом.
history: match_id, zone, date, tickets_total; только матчи до окна, завершённые к cutoff.
batch: ключи, scenario, slice_id, cutoff, weight, match_date, window и явные feature_columns; правильных ответов нет.
Вызов отдельный для окна и cutoff: заново обучайте метод только на history, не сохраняйте модель между вызовами.
Дополнительные признаки передавайте через feature_columns; train-признаки строятся по своим историческим отсечкам.
summary содержит MSE/RMSE по окну, сценарию и зоне; details — ошибки каждого среза; aggregate_scores даёт R/S.

Пример ровно на 10 строк: игрушечный метод повторяет среднее baseline.
```python
import pandas as pd
from src.baseline_data import ROOT, load_training_table
from src.scenarios import make_scenario_examples, historical_prediction_rule
from src.validation import evaluate_method, aggregate_scores
matches = pd.read_csv(ROOT / "data/matches.csv", parse_dates=["date", "sales_open"])
table = load_training_table()[["match_id", "zone", "tickets_total"]].merge(matches[["match_id", "date"]], on="match_id", validate="many_to_one")
examples = make_scenario_examples(matches, pd.read_csv(ROOT / "data/sales_daily.csv", parse_dates=["date"]))
def toy(history, batch): return batch.zone.map(history.groupby("zone").tickets_total.mean())
summary, details = evaluate_method(toy, table, examples, feature_columns=("sold",), prediction_rule=historical_prediction_rule)
windows, scores = aggregate_scores(summary)
```

table: 680 train-пар; examples: 984 среза из локальных CSV. Контроль: `python -X utf8 -m src.evaluate_baseline`; пример должен дать те же числа:

| Окно | RMSE без продаж | RMSE с продажами | R |
|---|---:|---:|---:|
| M035–M051 | 155.995 | 151.394 | 154.386 |
| M052–M068 | 105.410 | 102.931 | 104.542 |
| M069–M085 | 157.123 | 132.333 | 148.846 |

S = 139.155; S[0] = 143.913; S[1] = 129.748. Числа отчёта округлены, сравнение идёт по полным значениям.
Исходный ориентир `python -X utf8 -m src.baseline`: 155.999 / 103.139 / 157.144; другая схема, не S.
Нельзя менять примеры, веса, отсечки, окна, формулы и пороги; нельзя перемешивать матчи или делить их зоны.
Нельзя читать поздние продажи/проверочные цели, учить средние или преобразования на всей таблице, применять исторический cap.
Всегда передавайте historical_prediction_rule: .5 вверх, минимум sold; для новых признаков описывайте доступность в docstring.
Фиксируйте random_state; внешние данные запрещены, КХЛ/абонементы требуют согласованных правил и отдельного аудита.
Все три окна участвуют в выборе; результат и leakage-audit передайте лидеру для EXPERIMENTS, сравнение — с тем же baseline.
