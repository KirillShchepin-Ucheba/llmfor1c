from __future__ import annotations

from vkr_stage1.agents.service import run_agent_query
from vkr_stage1.connectors.sql_connector import SQLConnector


class FakeLLM:
    def generate_sql(self, question: str, schema: str) -> str:
        return "SELECT name FROM clients ORDER BY id LIMIT 2"

    def generate_1c_query(self, question: str, schema: str) -> str:
        return "ВЫБРАТЬ Номенклатура, Остаток ИЗ РегистрНакопления.ОстаткиТоваров"


class FakeOneCConnector:
    def execute_query(self, query_text: str) -> dict:
        return {
            "rows": [
                {"Номенклатура": "Ноутбук", "Остаток": 12},
                {"Номенклатура": "Мышь", "Остаток": 57},
            ],
            "query": query_text,
        }


class HistoryAwareLLM:
    def __init__(self) -> None:
        self.last_history: list[dict[str, str]] | None = None

    def generate_sql(self, question: str, schema: str, history: list[dict[str, str]] | None = None) -> str:
        self.last_history = history
        return "SELECT name FROM clients ORDER BY id LIMIT 2"

    def generate_1c_query(self, question: str, schema: str, history: list[dict[str, str]] | None = None) -> str:
        self.last_history = history
        if history and any("teachers" in item.get("content", "").lower() for item in history):
            return "ВЫБРАТЬ Учителя.Ссылка ИЗ Справочник.Учителя КАК Учителя"
        return "ВЫБРАТЬ Предметы.Ссылка ИЗ Справочник.Предметы КАК Предметы"


def test_agent_query_sql_branch() -> None:
    sql_connector = SQLConnector("./data/test_agent.db")
    sql_connector.bootstrap_demo_data()

    result = run_agent_query(
        question="Покажи клиентов",
        tool="sql",
        sql_connector=sql_connector,
        onec_connector=FakeOneCConnector(),
        llm_client=FakeLLM(),
    )

    assert result["tool"] == "sql"
    assert len(result["rows"]) == 2


def test_agent_query_onec_branch() -> None:
    sql_connector = SQLConnector("./data/test_agent.db")
    sql_connector.bootstrap_demo_data()

    result = run_agent_query(
        question="Покажи остатки на складе",
        tool="auto",
        sql_connector=sql_connector,
        onec_connector=FakeOneCConnector(),
        llm_client=FakeLLM(),
    )

    assert result["tool"] == "1c"
    assert len(result["onec_result"]["rows"]) > 0


def test_agent_query_onec_receives_history() -> None:
    sql_connector = SQLConnector("./data/test_agent.db")
    sql_connector.bootstrap_demo_data()
    llm = HistoryAwareLLM()
    history = [
        {"role": "user", "content": "show teachers"},
        {"role": "assistant", "content": "tool=1c\nquery=..."},
    ]

    result = run_agent_query(
        question="in one column",
        tool="1c",
        sql_connector=sql_connector,
        onec_connector=FakeOneCConnector(),
        llm_client=llm,
        dialog_history=history,
    )

    assert llm.last_history == history
    assert "Учителя" in result["generated_1c_query"]
