from __future__ import annotations

import pytest

from vkr_stage1.connectors.sql_connector import SQLConnector
from vkr_stage1.pipeline.nl2sql import run_nl2sql


class FakeLLM:
    def __init__(self, sql: str) -> None:
        self.sql = sql

    def generate_sql(self, question: str, schema: str) -> str:
        return self.sql


class HistoryAwareFakeLLM:
    def __init__(self) -> None:
        self.last_history: list[dict[str, str]] | None = None

    def generate_sql(self, question: str, schema: str, history: list[dict[str, str]] | None = None) -> str:
        self.last_history = history
        return "SELECT name FROM clients ORDER BY id LIMIT 1"


def test_nl2sql_pipeline_returns_rows() -> None:
    sql_connector = SQLConnector("./data/test_stage1.db")
    sql_connector.bootstrap_demo_data()

    llm = FakeLLM("SELECT name, city FROM clients ORDER BY id")
    result = run_nl2sql("Покажи клиентов", sql_connector, llm)

    assert "generated_sql" in result
    assert len(result["rows"]) >= 1


def test_execute_select_rejects_non_select() -> None:
    sql_connector = SQLConnector("./data/test_stage1.db")
    sql_connector.bootstrap_demo_data()

    with pytest.raises(ValueError):
        sql_connector.execute_select("DELETE FROM clients")


def test_execute_select_allows_with_query() -> None:
    sql_connector = SQLConnector("./data/test_stage1.db")
    sql_connector.bootstrap_demo_data()

    rows = sql_connector.execute_select(
        "WITH cte AS (SELECT name FROM clients) SELECT name FROM cte LIMIT 1;"
    )
    assert len(rows) == 1


def test_nl2sql_pipeline_passes_history() -> None:
    sql_connector = SQLConnector("./data/test_stage1.db")
    sql_connector.bootstrap_demo_data()
    llm = HistoryAwareFakeLLM()
    history = [
        {"role": "user", "content": "show clients"},
        {"role": "assistant", "content": "tool=sql"},
    ]

    result = run_nl2sql("top 1", sql_connector, llm, dialog_history=history)

    assert llm.last_history == history
    assert len(result["rows"]) == 1
