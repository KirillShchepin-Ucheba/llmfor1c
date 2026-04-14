from __future__ import annotations

import argparse
import json
from typing import Any

from vkr_stage1.agents.service import run_agent_query
from vkr_stage1.connectors.onec_connector import OneCConnector
from vkr_stage1.connectors.sql_connector import SQLConnector
from vkr_stage1.core.config import get_settings
from vkr_stage1.core.logger import setup_logger
from vkr_stage1.llm.openai_client import OpenAILLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: NL -> LLM API -> SQL -> DB")
    parser.add_argument("--query", required=True, help="NL-запрос пользователя")
    parser.add_argument("--tool", choices=["auto", "sql", "1c"], default="auto", help="Маршрутизация агента")
    parser.add_argument("--format", choices=["human", "json"], default="human")
    parser.add_argument("--show-schema", action="store_true", help="Показать schema в выводе")
    parser.add_argument(
        "--dataset-profile",
        choices=["demo", "erp-small", "erp-medium", "erp-large"],
        default="erp-medium",
        help="Профиль данных для теста NL->SQL",
    )
    parser.add_argument("--reset-db", action="store_true", help="Пересоздать БД под выбранный профиль")
    return parser.parse_args()


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(0 rows)")
        return

    cols = list(rows[0].keys())
    widths = {c: len(str(c)) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    header = "| " + " | ".join(str(c).ljust(widths[c]) for c in cols) + " |"
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols) + " |")
    print(sep)
    print(f"{len(rows)} rows")


def _print_human(result: dict[str, Any], show_schema: bool) -> None:
    print(f"Question: {result['question']}")
    if show_schema and "schema" in result:
        print(f"Schema: {result['schema']}")
    print(f"Tool: {result.get('tool', 'sql')} ({result.get('route_reason', 'n/a')})")
    if "generated_sql" in result:
        print(f"SQL: {result['generated_sql']}")
        _print_table(result["rows"])
        return

    print(f"1C Query: {result['generated_1c_query']}")
    _print_table(result["onec_result"].get("rows", []))


def main() -> None:
    logger = setup_logger()
    args = parse_args()
    settings = get_settings()
    if not args.query.strip(". ").strip():
        raise ValueError("Передай реальный текст запроса вместо '...'.")

    db_path = settings.db_path
    if args.dataset_profile != "demo" and settings.db_path == "./data/stage1_demo.db":
        db_path = f"./data/{args.dataset_profile}.db"

    sql_connector = SQLConnector(db_path)
    if args.dataset_profile == "demo":
        sql_connector.bootstrap_demo_data()
    else:
        size = args.dataset_profile.split("-", 1)[1]
        sql_connector.bootstrap_erp_data(size=size, reset=args.reset_db)

    llm_client = OpenAILLMClient(
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    onec_connector = OneCConnector(
        base_url=settings.onec_base_url,
        query_path=settings.onec_query_path,
        schema_path=settings.onec_schema_path,
        username=settings.onec_username,
        password=settings.onec_password,
        mock_mode=settings.onec_mock,
    )

    result = run_agent_query(
        question=args.query,
        tool=args.tool,
        sql_connector=sql_connector,
        onec_connector=onec_connector,
        llm_client=llm_client,
    )
    logger.info("Agent query executed")
    counts = sql_connector.table_counts()
    logger.info("DB profile: %s | db=%s | tables=%d", args.dataset_profile, db_path, len(counts))
    if args.format == "json":
        if not args.show_schema:
            result = {k: v for k, v in result.items() if k != "schema"}
        result["dataset_profile"] = args.dataset_profile
        result["table_counts"] = counts
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"Dataset profile: {args.dataset_profile}")
    print(f"DB path: {db_path}")
    _print_human(result, show_schema=args.show_schema)


if __name__ == "__main__":
    main()
