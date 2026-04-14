from __future__ import annotations

from vkr_stage1.agents.router import choose_tool


def test_choose_tool_onec_keyword() -> None:
    route = choose_tool("Покажи остатки по складу")
    assert route.tool == "1c"


def test_choose_tool_default_sql() -> None:
    route = choose_tool("Топ клиентов по выручке")
    assert route.tool == "sql"


def test_choose_tool_forced_sql() -> None:
    route = choose_tool("Покажи остатки по складу", forced_tool="sql")
    assert route.tool == "sql"
