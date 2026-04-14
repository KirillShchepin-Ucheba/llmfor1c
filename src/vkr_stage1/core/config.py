from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("DB_PATH", "./data/stage1_demo.db")
    llm_api_base: str = os.getenv("LLM_API_BASE", "https://api.vsellm.ru/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "anthropic/claude-haiku-4.5")

    onec_base_url: str = os.getenv("ONEC_BASE_URL", "http://localhost:8080")
    onec_query_path: str = os.getenv("ONEC_QUERY_PATH", "/query")
    onec_schema_path: str = os.getenv("ONEC_SCHEMA_PATH", "/schema")
    onec_username: str = os.getenv("ONEC_USERNAME", "")
    onec_password: str = os.getenv("ONEC_PASSWORD", "")
    onec_mock: bool = os.getenv("ONEC_MOCK", "false").lower() == "true"


def get_settings() -> Settings:
    return Settings()
