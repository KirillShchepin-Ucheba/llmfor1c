# Методичка: подключение реальной 1C (без заглушки - на будущее)

## 1. Цель
Подключить проект к реальному HTTP endpoint 1C и запускать:
- чат-агент в терминале,
- API,
- бенчмарк по датасету.

## 2. Что должно быть на стороне 1C
Нужен HTTP сервис, который принимает POST JSON:

```json
{"query": "ВЫБРАТЬ ..."}
```

и возвращает JSON-ответ, например:

```json
{"rows": [{"Поле": "Значение"}]}
```

Endpoint в примерах: `http://<host>:<port>/query`.

## 3. Настройка .env
Создай `.env` из примера и заполни:

```env
LLM_API_BASE=https://api.vsellm.ru/v1
LLM_API_KEY=...
LLM_MODEL=anthropic/claude-haiku-4.5

ONEC_BASE_URL=http://<1c-host>:<port>
ONEC_QUERY_PATH=/query
ONEC_USERNAME=<если есть>
ONEC_PASSWORD=<если есть>
ONEC_MOCK=false
```

Важно: `ONEC_MOCK=false`.

## 4. Проверка 1C подключения

```bash
source .venv/bin/activate
python -m vkr_stage1.onec_check
```

Если endpoint отвечает, увидишь JSON-ответ 1C.

## 5. Запуск чата агента (терминал)

```bash
python -m vkr_stage1.chat_cli --tool auto --dataset-profile erp-medium
```

Проверочные запросы:
- SQL-ветка: `топ 5 клиентов по сумме заказов`
- 1C-ветка: `покажи остатки по складу по номенклатуре`

## 6. Запуск API

```bash
uvicorn vkr_stage1.api.server:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
```

Запрос:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "покажи остатки по складу по номенклатуре",
    "tool": "auto",
    "dataset_profile": "erp-medium",
    "reset_db": false
  }'
```

## 7. Бенчмарк на многих запросах

```bash
python -m vkr_stage1.eval.benchmark_spider1c \
  --parquet data/Spider-1C/data/train-00000-of-00001.parquet \
  --limit 200 \
  --progress-every 10 \
  --out data/bench_spider1c.json
```

Метрики в отчете:
- `success_rate`
- `exact_match_rate`
- `latency_ms.avg/p50/max`
- ошибки по каждому сэмплу

## 8. Типичные ошибки
- `ONEC_MOCK=true`: реальная проверка отключена, поставь `false`.
- `HTTP 401/403`: неверные `ONEC_USERNAME/PASSWORD`.
- `HTTP 404`: проверь `ONEC_QUERY_PATH`.
- `Connection refused/timeout`: 1C endpoint недоступен по сети.
