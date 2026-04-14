from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class OneCConnector:
    base_url: str
    query_path: str
    schema_path: str = "/schema"
    username: str = ""
    password: str = ""
    mock_mode: bool = False

    def _build_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.query_path.lstrip('/')}"

    def _build_schema_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.schema_path.lstrip('/')}"

    def fetch_schema_text(self) -> str:
        if self.mock_mode:
            raise RuntimeError("Cannot fetch 1C schema in mock mode")

        auth = (self.username, self.password) if self.username else None
        try:
            response = httpx.get(
                self._build_schema_url(),
                auth=auth,
                timeout=30.0,
                trust_env=False,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body_preview = (exc.response.text or "")[:400]
            raise RuntimeError(
                f"1C schema endpoint error at {self._build_schema_url()}: {exc}. Response body: {body_preview}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"1C schema endpoint error at {self._build_schema_url()}: {exc}") from exc

        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            data = response.json()
            if isinstance(data, dict):
                schema = data.get("schema")
                if isinstance(schema, str) and schema.strip():
                    return schema.strip()
            raise RuntimeError("1C schema endpoint returned JSON without non-empty 'schema' field")

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("1C schema endpoint returned empty schema text")
        return text

    def execute_query(self, query_text: str) -> dict[str, Any]:
        if self.mock_mode:
            raise RuntimeError(
                "ONEC_MOCK=true is disabled for real runs. Set ONEC_MOCK=false and configure real 1C endpoint."
            )

        auth = (self.username, self.password) if self.username else None
        payload = {"query": query_text}

        try:
            response = httpx.post(
                self._build_url(),
                json=payload,
                auth=auth,
                timeout=60.0,
                trust_env=False,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            body_preview = (exc.response.text or "")[:400]
            raise RuntimeError(
                f"1C endpoint error at {self._build_url()}: {exc}. Response body: {body_preview}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"1C endpoint error at {self._build_url()}: {exc}") from exc

        if isinstance(data, dict):
            return data
        return {"rows": data}
