from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolRoute:
    tool: str
    reason: str


ONEC_KEYWORDS = (
    "1с",
    "остат",
    "регистр",
    "документ",
    "справочник",
    "проведен",
    "номенклатур",
    "контрагент",
    "склад",
)


def choose_tool(question: str, forced_tool: str = "auto") -> ToolRoute:
    force = forced_tool.strip().lower()
    if force in {"sql", "1c"}:
        return ToolRoute(tool=force, reason=f"forced:{force}")

    text = question.lower()
    if any(k in text for k in ONEC_KEYWORDS):
        return ToolRoute(tool="1c", reason="keyword:1c-domain")
    return ToolRoute(tool="sql", reason="default:analytics-sql")
