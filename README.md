# DATA MATCH: прогноз спроса на билеты ХК «Авангард»

Задача: итог обычных продаж для 17 тестовых матчей M086–M102 по 8 зонам, 136 целых неотрицательных прогнозов. Выбран baseline — среднее по зоне из доступной истории. A, B и C отклонены; основания и контрольные числа приведены в [STATE](docs/STATE.md).

Порядок чтения: README → [STATE](docs/STATE.md) → [PROTOCOL](docs/PROTOCOL.md) → [HOWTO_EVALUATE](docs/HOWTO_EVALUATE.md) → при необходимости [EXPERIMENTS](docs/EXPERIMENTS.md). Полный [индекс документов](docs/README.md); [индекс аудитов](docs/audits/README.md).

## Запуск и воспроизведение

Рабочий каталог — корень проекта. Данные организаторов располагаются локально в data; схема — [data/README.md](data/README.md), условия — [data/LICENSE.md](data/LICENSE.md). Данные и outputs не входят в Git. Документационная проверка не запускает модели или сборку прогнозов.

- `python -X utf8 -m src.evaluate_baseline` — обе сценарные ошибки, R/S и файлы в outputs.
- `python -X utf8 -m src.diagnose_baseline` — вклад выборки, истории и нижней границы.
- `python -X utf8 -m src.make_baseline` — только outputs/predictions_baseline.csv; `python -X utf8 -m unittest discover -s tests -v` — проверки.
- `python -X utf8 -m src.evaluate_early_sales --compare` — контроль и фиксированные A/B; `--control-only` — только контроль, `--prepare-only` — временные срезы. Эти команды не перезаписывают prediction-файлы.
Сохранённые результаты A/B — до seats; повторный --compare с нынешним общим правилом даст отдельный пересчёт, после seats не выполнялся.
- `python -X utf8 -m src.evaluate_pace_ratio` — C и отдельные outputs/pace_ratio_C_*; `python -X utf8 -m src.audit_pace_ratio` — независимое восстановление. Файлы сдачи и прежние отчёты A/B не перезаписывают.
В среде запуска Python из PATH не содержит pandas; рабочий интерпретатор — `C:/Users/cocos/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe` (Python 3.12.14, numpy 2.3.5, pandas 3.0.1). В PowerShell применяется `& "полный путь/python.exe"` вместо `python`.
Команда make_baseline указана как справочная: её повторный запуск перезаписывает резерв и требует отдельного разрешения на запись.

## Состояние реализации

Сохранён резерв outputs/predictions_baseline.csv; корневой predictions.csv отсутствует. Ноутбук и презентация ещё не готовы; инструкция README описывает существующие команды, а комплект сдачи остаётся незавершённым. Целевая дата и ближайшие действия — STATE. Baseline, validation и их тесты уже присутствуют в истории ветки; текущие изменения развивают общее правило seats и добавляют фиксированные A/B/C.

Правила честности: только временные окна, все зоны матча вместе, история до окна и завершённая к cutoff, продажи строго до утра отсечки. Исторические ограничения и допущение о seats — PROTOCOL. Параметры и критерии после просмотра результатов не подбираются. Внешние данные и праздники не используются; доступность исторических абонементов остаётся открытой.

## Локальные рабочие копии

Worktree: `C:/Users/cocos/.codex/worktrees/5ed2/Data Match Техинческий трек`, ветка `baseline`.
Основная копия по прежней записи: `C:/Users/cocos/OneDrive/Dokumenti/ChatGPT/Data Match Техинческий трек`, `main`; состояние других рабочих копий не проверяется этим обзором.
Статус: изменения рабочей копии локальные.

AGENTS.md и .agents содержат локальные инструкции агента и исключены .gitignore; правила для команды находятся в PROTOCOL. Исторические NEXT_SESSION.md и PROMPT.md сохраняются как локальные черновики; текущий план — STATE. Отчёты аудитов описывают версии кода на даты проверок; старые хеши и расчётные артефакты сохраняются без переоформления.
