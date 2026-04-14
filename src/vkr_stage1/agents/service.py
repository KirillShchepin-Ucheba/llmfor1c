from __future__ import annotations

from typing import Any

from vkr_stage1.agents.router import choose_tool
from vkr_stage1.connectors.onec_connector import OneCConnector
from vkr_stage1.connectors.sql_connector import SQLConnector
from vkr_stage1.pipeline.nl2sql import run_nl2sql

DEFAULT_ONEC_SCHEMA = (
    "Справочник.Номенклатура(Ссылка, Наименование);\n"
    "РегистрНакопления.ОстаткиТоваров(Номенклатура, Склад, Остаток)"
)


class OneCGeneratorProtocol:
    def generate_1c_query(
        self, question: str, schema: str, history: list[dict[str, str]] | None = None
    ) -> str:  # pragma: no cover
        raise NotImplementedError


class SQLGeneratorProtocol:
    def generate_sql(
        self, question: str, schema: str, history: list[dict[str, str]] | None = None
    ) -> str:  # pragma: no cover
        raise NotImplementedError


def run_agent_query(
    question: str,
    tool: str,
    sql_connector: SQLConnector,
    onec_connector: OneCConnector,
    llm_client: SQLGeneratorProtocol | OneCGeneratorProtocol,
    dialog_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    route = choose_tool(question=question, forced_tool=tool)

    if route.tool == "sql":
        result = run_nl2sql(question, sql_connector, llm_client, dialog_history=dialog_history)  # type: ignore[arg-type]
        result["tool"] = "sql"
        result["route_reason"] = route.reason
        return result

    onec_schema_source = "fallback:static"
    try:
        onec_schema = onec_connector.fetch_schema_text()
        onec_schema_source = "dynamic:endpoint"
    except Exception:
        onec_schema = DEFAULT_ONEC_SCHEMA

    if dialog_history:
        onec_query = llm_client.generate_1c_query(  # type: ignore[attr-defined]
            question=question,
            schema=onec_schema,
            history=dialog_history,
        )
    else:
        onec_query = llm_client.generate_1c_query(question=question, schema=onec_schema)  # type: ignore[attr-defined]
    retry_used = False
    try:
        onec_result = onec_connector.execute_query(onec_query)
    except Exception as exc:
        message = str(exc)
        # Retry once with strict instruction when model used a missing field.
        if "Поле не найдено" in message or "field not found" in message.lower():
            retry_used = True
            fixed_question = (
                f"{question}\n\n"
                "ВАЖНО: используй только поля из схемы. "
                "Если нужного поля нет, не используй его и убери такой фильтр."
            )
            if dialog_history:
                onec_query = llm_client.generate_1c_query(  # type: ignore[attr-defined]
                    question=fixed_question,
                    schema=onec_schema,
                    history=dialog_history,
                )
            else:
                onec_query = llm_client.generate_1c_query(question=fixed_question, schema=onec_schema)  # type: ignore[attr-defined]
            onec_result = onec_connector.execute_query(onec_query)
        else:
            raise
    return {
        "question": question,
        "tool": "1c",
        "route_reason": route.reason,
        "onec_schema_source": onec_schema_source,
        "onec_retry_used": retry_used,
        "generated_1c_query": onec_query,
        "onec_result": onec_result,
    }
