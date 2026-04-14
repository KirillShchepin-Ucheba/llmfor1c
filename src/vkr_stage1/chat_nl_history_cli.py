from __future__ import annotations

import argparse
import json
from typing import Any

from vkr_stage1.agents.service import run_agent_query
from vkr_stage1.connectors.onec_connector import OneCConnector
from vkr_stage1.connectors.sql_connector import SQLConnector
from vkr_stage1.core.config import get_settings
from vkr_stage1.llm.openai_client import OpenAILLMClient


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
    print(sep)
    print("| " + " | ".join(str(c).ljust(widths[c]) for c in cols) + " |")
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols) + " |")
    print(sep)
    print(f"{len(rows)} rows")


def _result_to_history_message(result: dict[str, Any]) -> str:
    if result.get("tool") == "sql":
        rows = result.get("rows", [])
        cols = ", ".join(rows[0].keys()) if rows else "no columns"
        return (
            "tool=sql\n"
            f"query={result.get('generated_sql', '')}\n"
            f"columns={cols}\n"
            f"rows={len(rows)}"
        )

    rows = result.get("onec_result", {}).get("rows", [])
    cols = ", ".join(rows[0].keys()) if rows else "no columns"
    return (
        "tool=1c\n"
        f"query={result.get('generated_1c_query', '')}\n"
        f"columns={cols}\n"
        f"rows={len(rows)}"
    )


def _push_history(history: list[dict[str, str]], role: str, content: str, max_turns: int) -> None:
    history.append({"role": role, "content": content})
    if max_turns <= 0:
        history.clear()
        return

    max_messages = max_turns * 2
    if len(history) > max_messages:
        del history[:-max_messages]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NL-only chat with explicit message history support")
    parser.add_argument("--tool", choices=["auto", "sql", "1c"], default="1c")
    parser.add_argument(
        "--dataset-profile",
        choices=["demo", "erp-small", "erp-medium", "erp-large"],
        default="erp-medium",
    )
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--show-generated-query", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    db_path = settings.db_path
    if args.dataset_profile != "demo" and settings.db_path == "./data/stage1_demo.db":
        db_path = f"./data/{args.dataset_profile}.db"

    sql_connector = SQLConnector(db_path)
    if args.dataset_profile == "demo":
        sql_connector.bootstrap_demo_data()
    else:
        size = args.dataset_profile.split("-", 1)[1]
        sql_connector.bootstrap_erp_data(size=size, reset=args.reset_db)

    onec_connector = OneCConnector(
        base_url=settings.onec_base_url,
        query_path=settings.onec_query_path,
        schema_path=settings.onec_schema_path,
        username=settings.onec_username,
        password=settings.onec_password,
        mock_mode=settings.onec_mock,
    )
    llm_client = OpenAILLMClient(
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    history: list[dict[str, str]] = []
    print("NL chat with history started. Commands: /help, /history, /clear, /exit")
    print(f"profile={args.dataset_profile}, tool={args.tool}, history_turns={args.history_turns}")

    while True:
        user_q = input("nl> ").strip()
        if not user_q:
            continue
        if user_q in {"/exit", "exit", "quit"}:
            break
        if user_q == "/help":
            print("Type a natural-language request. Follow-up requests can be short: 'in one column'.")
            continue
        if user_q == "/history":
            print(json.dumps(history, ensure_ascii=False, indent=2))
            continue
        if user_q == "/clear":
            history.clear()
            print("history cleared")
            continue

        try:
            result = run_agent_query(
                question=user_q,
                tool=args.tool,
                sql_connector=sql_connector,
                onec_connector=onec_connector,
                llm_client=llm_client,
                dialog_history=history,
            )
        except Exception as exc:
            print(f"agent.error: {exc}")
            continue

        if args.show_generated_query:
            if result.get("tool") == "sql":
                print(f"sql: {result.get('generated_sql')}")
            else:
                print(f"1c: {result.get('generated_1c_query')}")

        rows = result.get("rows") if result.get("tool") == "sql" else result.get("onec_result", {}).get("rows", [])
        _print_table(rows or [])
        _push_history(history, "user", user_q, args.history_turns)
        _push_history(history, "assistant", _result_to_history_message(result), args.history_turns)


if __name__ == "__main__":
    main()
