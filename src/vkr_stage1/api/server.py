from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

from vkr_stage1.agents.service import run_agent_query
from vkr_stage1.connectors.onec_connector import OneCConnector
from vkr_stage1.connectors.sql_connector import SQLConnector
from vkr_stage1.core.config import get_settings
from vkr_stage1.llm.openai_client import OpenAILLMClient


class QueryRequest(BaseModel):
    question: str
    tool: str = "auto"  # auto | sql | 1c
    dataset_profile: str = "erp-medium"  # demo | erp-small | erp-medium | erp-large
    reset_db: bool = False
    dialog_history: list[dict[str, str]] = Field(default_factory=list)


app = FastAPI(title="VKR Agent API", version="0.1.0")


@app.get("/")
def root() -> dict:
    return {"service": "vkr-agent-api", "status": "ok", "docs": "/docs", "query_endpoint": "/query"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    try:
        settings = get_settings()
        if not req.question.strip(". ").strip():
            raise ValueError("question must not be empty")

        db_path = settings.db_path
        if req.dataset_profile != "demo" and settings.db_path == "./data/stage1_demo.db":
            db_path = f"./data/{req.dataset_profile}.db"

        sql_connector = SQLConnector(db_path)
        if req.dataset_profile == "demo":
            sql_connector.bootstrap_demo_data()
        else:
            size = req.dataset_profile.split("-", 1)[1]
            sql_connector.bootstrap_erp_data(size=size, reset=req.reset_db)

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

        result = run_agent_query(
            question=req.question,
            tool=req.tool,
            sql_connector=sql_connector,
            onec_connector=onec_connector,
            llm_client=llm_client,
            dialog_history=req.dialog_history,
        )
        result["dataset_profile"] = req.dataset_profile
        result["table_counts"] = sql_connector.table_counts()
        return result
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc))
