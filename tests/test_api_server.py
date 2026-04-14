from __future__ import annotations

from vkr_stage1.api.server import health, root


def test_root_endpoint() -> None:
    payload = root()
    assert payload["service"] == "vkr-agent-api"
    assert payload["status"] == "ok"


def test_health_endpoint() -> None:
    payload = health()
    assert payload["status"] == "ok"
