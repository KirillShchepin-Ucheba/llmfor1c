from __future__ import annotations

from typing import Any

from vkr_stage1.connectors.sql_connector import SQLConnector


class SQLGeneratorProtocol:
    def generate_sql(
        self, question: str, schema: str, history: list[dict[str, str]] | None = None
    ) -> str:  # pragma: no cover
        raise NotImplementedError


def run_nl2sql(
    question: str,
    sql_connector: SQLConnector,
    llm_client: SQLGeneratorProtocol,
    dialog_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    schema = sql_connector.get_schema_text()
    if dialog_history:
        generated_sql = llm_client.generate_sql(question=question, schema=schema, history=dialog_history)
    else:
        generated_sql = llm_client.generate_sql(question=question, schema=schema)
    rows = sql_connector.execute_select(generated_sql)

    return {
        "question": question,
        "schema": schema,
        "generated_sql": generated_sql,
        "rows": rows,
    }
