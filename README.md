# ВКР: Agent + API + SQL/1C

Проект работает в режиме `NL -> LLM API -> Agent -> SQL/1C`.

Текущий рабочий фокус: реальная 1C интеграция (без заглушек).

## Основные режимы

1. Чат агента в терминале.
2. HTTP API для интеграции.
3. Бенчмарк на Spider-1C c метриками времени/качества.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
```

## Ключевая настройка

В `.env` обязательно:

```env
ONEC_MOCK=false
```

и реальные параметры 1C endpoint:

```env
ONEC_BASE_URL=http://<1c-host>:<port>
ONEC_QUERY_PATH=/query
ONEC_USERNAME=<login>
ONEC_PASSWORD=<password>
```

Подробная пошаговая инструкция: [METHODIC_1C.md](/home/alpc/nl21c/METHODIC_1C.md)

## Проверка подключения 1C

```bash
python -m vkr_stage1.onec_check
```

## Чат в терминале

```bash
python -m vkr_stage1.chat_cli --tool auto --dataset-profile erp-medium
```

## API сервер

```bash
uvicorn vkr_stage1.api.server:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

## Бенчмарк на датасете

```bash
python -m vkr_stage1.eval.benchmark_spider1c \
  --parquet data/Spider-1C/data/train-00000-of-00001.parquet \
  --limit 200 \
  --progress-every 10 \
  --out data/bench_spider1c.json
```

## Структура проекта

Главные программы (точки входа):

- `python -m vkr_stage1.chat_cli` — интерактивный чат-агент в терминале.
- `uvicorn vkr_stage1.api.server:app` — HTTP API для интеграции (`/query`).
- `python -m vkr_stage1.main --query "..."` — одиночный запуск запроса из CLI.
- `python -m vkr_stage1.onec_check` — проверка подключения к реальному endpoint 1C.
- `python -m vkr_stage1.eval.benchmark_spider1c` — бенчмарк на Spider-1C.

```text
n21c_code/
├─ src/vkr_stage1/
│  ├─ main.py                  # CLI-раннер одного запроса (NL -> agent -> SQL/1C)
│  ├─ chat_cli.py              # интерактивный терминальный чат
│  ├─ onec_check.py            # проверка живого 1C endpoint
│  ├─ api/server.py            # FastAPI-сервер с endpoint /query
│  ├─ agents/
│  │  ├─ router.py             # выбор ветки: SQL или 1C
│  │  └─ service.py            # оркестрация маршрута и выполнение запроса
│  ├─ pipeline/nl2sql.py       # пайплайн Text-to-SQL
│  ├─ llm/openai_client.py     # клиент LLM: генерация SQL и 1C-запросов
│  ├─ connectors/
│  │  ├─ sql_connector.py      # SQLite: схема, генерация demo/ERP данных, SELECT
│  │  └─ onec_connector.py     # HTTP-коннектор к 1C
│  ├─ eval/
│  │  ├─ benchmark_spider1c.py # массовый бенчмарк, latency и exact match
│  │  └─ spider1c_eval.py      # sample-eval на датасете
│  └─ core/
│     ├─ config.py             # загрузка настроек из .env
│     └─ logger.py             # базовая настройка логирования
├─ tests/                      # unit-тесты роутера, сервиса, API и NL2SQL
├─ data/                       # SQLite БД, результаты бенчмарка, Spider-1C
├─ METHODIC_1C.md              # пошаговое подключение к реальной 1C
├─ pyproject.toml              # зависимости и конфиг проекта
└─ README.md                   # быстрый старт и основные команды
```
