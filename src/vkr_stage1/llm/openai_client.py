from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal, Sequence, TypedDict

import openai

PROJECT_CONTEXT = (
    "Project context: You are an assistant for a graduation project on Text-to-1C and Text-to-SQL. "
    "Prioritize business-like analytics queries and deterministic output."
)


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass
class OpenAILLMClient:
    api_base: str
    api_key: str
    model: str
    embedding_model: str = "text-embedding-3-small"

    def _chat(self, system_prompt: str, user_prompt: str, history: Sequence[ChatMessage] | None = None) -> str:
        if not self.api_key:
            raise ValueError("LLM_API_KEY is empty")

        client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for msg in history or []:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})

        resp = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=messages,
        )
        return (resp.choices[0].message.content or "").strip()

    def generate_sql(self, question: str, schema: str, history: Sequence[ChatMessage] | None = None) -> str:
        raw = self._chat(
            system_prompt=(
                f"{PROJECT_CONTEXT} "
                "You convert natural language to SQL. "
                "Use dialog history to resolve follow-up requests with omitted subject. "
                "Return only one SQL query for SQLite without markdown."
            ),
            user_prompt=(
                "Schema:\n"
                f"{schema}\n\n"
                "Question:\n"
                f"{question}\n\n"
                "SQL:"
            ),
            history=history,
        )
        return _extract_select_sql(raw)

    def generate_1c_query(self, question: str, schema: str, history: Sequence[ChatMessage] | None = None) -> str:
        raw = self._chat(
            system_prompt=(
                f"{PROJECT_CONTEXT} "
                "You convert natural language to 1C query language. "
                "Use dialog history to resolve follow-up requests with omitted subject. "
                "Return only one 1C query starting with 'ВЫБРАТЬ' without markdown. "
                "Use only entities and fields explicitly present in the schema. "
                "Do not invent fields. "
                "If an exact field for the requested intent is missing, produce the closest executable approximation "
                "using available schema entities/fields (best-effort), while preserving user intent as much as possible. "
                "Return query text only, without explanations."
            ),
            user_prompt=(
                "1C Schema:\n"
                f"{schema}\n\n"
                "Question:\n"
                f"{question}\n\n"
                "1C Query:"
            ),
            history=history,
        )
        return _extract_select_1c(raw)

    def get_embedding(self, text: str) -> list[float]:
        if not self.api_key:
            raise ValueError("LLM_API_KEY is empty")

        client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        resp = client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return list(resp.data[0].embedding)

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding vectors must have the same length")

        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            raise ValueError("Embedding vectors must be non-zero")
        return dot / (left_norm * right_norm)


def _extract_select_sql(raw: str) -> str:
    text = raw.replace("```sql", "").replace("```", "").strip()
    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if not match:
        preview = text[:200].replace("\n", " ")
        raise ValueError(f"LLM did not return SQL SELECT query. Model output: {preview}")

    sql = text[match.start() :].strip()
    semi = sql.find(";")
    if semi != -1:
        sql = sql[: semi + 1]
    return sql


def _extract_select_1c(raw: str) -> str:
    text = raw.replace("```1c", "").replace("```", "").strip()
    match = re.search(r"\b(ВЫБРАТЬ|SELECT)\b", text, flags=re.IGNORECASE)
    if not match:
        preview = text[:200].replace("\n", " ")
        raise ValueError(f"LLM did not return 1C query starting with ВЫБРАТЬ. Model output: {preview}")

    query = text[match.start() :].strip()
    semi = query.find(";")
    if semi != -1:
        query = query[: semi + 1]
    return query
