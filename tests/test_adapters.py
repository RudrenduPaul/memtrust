"""Adapter tests. Every HTTP-based adapter is exercised via pytest-httpx
(no real network calls); MemPalaceAdapter is exercised via a fake
in-memory Palace injected through its constructor, matching the
_PalaceProtocol shape defined in mempalace_adapter.py.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from memtrust.adapters.base import (
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
)
from memtrust.adapters.mem0_adapter import Mem0Adapter, Mem0SelfHostedAdapter
from memtrust.adapters.mempalace_adapter import MemPalaceAdapter
from memtrust.adapters.openviking_adapter import OpenVikingAdapter
from memtrust.adapters.zep_graphiti_adapter import ZepGraphitiAdapter

# ---------------------------------------------------------------------------
# BackendNotConfiguredError -- every adapter, no env var set
# ---------------------------------------------------------------------------


def test_mem0_raises_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    with pytest.raises(BackendNotConfiguredError) as excinfo:
        Mem0Adapter()
    assert excinfo.value.missing_env_var == "MEM0_API_KEY"
    assert excinfo.value.backend_name == "mem0"


def test_zep_raises_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZEP_API_KEY", raising=False)
    with pytest.raises(BackendNotConfiguredError) as excinfo:
        ZepGraphitiAdapter()
    assert excinfo.value.missing_env_var == "ZEP_API_KEY"


def test_openviking_raises_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENVIKING_API_KEY", raising=False)
    with pytest.raises(BackendNotConfiguredError) as excinfo:
        OpenVikingAdapter()
    assert excinfo.value.missing_env_var == "OPENVIKING_API_KEY"


def test_mempalace_raises_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMPALACE_STORAGE_PATH", raising=False)
    with pytest.raises(BackendNotConfiguredError) as excinfo:
        MemPalaceAdapter()
    assert excinfo.value.missing_env_var == "MEMPALACE_STORAGE_PATH"


def test_backend_not_configured_error_message_mentions_methodology() -> None:
    err = BackendNotConfiguredError("mem0", "MEM0_API_KEY")
    assert "docs/methodology.md" in str(err)
    assert "MEM0_API_KEY" in str(err)


# ---------------------------------------------------------------------------
# Mem0Adapter
# ---------------------------------------------------------------------------


def test_mem0_store_query_update(monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    adapter = Mem0Adapter()

    httpx_mock.add_response(
        method="POST",
        url="https://api.mem0.ai/v1/memories/",
        json={"id": "mem-1"},
    )
    store_result = adapter.store("session-1", "I like tea.")
    assert store_result.memory_id == "mem-1"
    assert store_result.latency_ms >= 0

    httpx_mock.add_response(
        method="POST",
        url="https://api.mem0.ai/v1/memories/search/",
        json={"results": [{"id": "mem-1", "memory": "I like tea.", "score": 0.9}]},
    )
    query_result = adapter.query("session-1", "what do I like?")
    assert len(query_result.records) == 1
    assert query_result.records[0].content == "I like tea."
    assert query_result.conflict_signal == ConflictSignal.NOT_APPLICABLE

    httpx_mock.add_response(
        method="PUT",
        url="https://api.mem0.ai/v1/memories/mem-1/",
        json={"id": "mem-1"},
    )
    update_result = adapter.update("session-1", "mem-1", "I like coffee now.")
    assert update_result.acknowledged is True
    adapter.close()


def test_mem0_store_raises_backend_api_error_on_http_failure(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    adapter = Mem0Adapter()
    httpx_mock.add_response(status_code=500)
    with pytest.raises(BackendAPIError):
        adapter.store("session-1", "content")
    adapter.close()


def test_mem0_hosted_adapter_unaffected_by_selfhosted_addition(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """Regression guard: adding Mem0SelfHostedAdapter must not change one
    byte of Mem0Adapter's request shape. Base URL stays api.mem0.ai, the
    route stays under /v1/, and query() still only accepts the base
    (session_id, query, top_k) signature -- passing run_id would be a
    TypeError, proving hosted and self-hosted did not merge into a single
    branchy method.
    """
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    adapter = Mem0Adapter()
    assert adapter._http.base_url == "https://api.mem0.ai"

    httpx_mock.add_response(
        method="POST",
        url="https://api.mem0.ai/v1/memories/search/",
        json={"results": [{"id": "mem-1", "memory": "I like tea.", "score": 0.9}]},
    )
    query_result = adapter.query("session-1", "what do I like?")
    assert len(query_result.records) == 1

    with pytest.raises(TypeError):
        adapter.query("session-1", "what do I like?", run_id="")  # type: ignore[call-arg]
    adapter.close()


# ---------------------------------------------------------------------------
# Mem0SelfHostedAdapter
# ---------------------------------------------------------------------------


def test_mem0_selfhosted_raises_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEM0_SELFHOSTED_BASE_URL", raising=False)
    with pytest.raises(BackendNotConfiguredError) as excinfo:
        Mem0SelfHostedAdapter()
    assert excinfo.value.missing_env_var == "MEM0_SELFHOSTED_BASE_URL"
    assert excinfo.value.backend_name == "mem0_selfhosted"


def test_mem0_selfhosted_store_query_update_use_unprefixed_routes(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """The self-hosted OSS server mounts its FastAPI router at unprefixed
    paths (POST /memories, POST /search, PUT /memories/{id}), confirmed
    against the real server/main.py source -- not /v1/memories/... like
    the hosted Platform API. pytest-httpx matches the exact URL, so this
    fails loudly if the adapter ever regresses to the hosted route shape.
    """
    monkeypatch.setenv("MEM0_SELFHOSTED_BASE_URL", "http://localhost:8888")
    adapter = Mem0SelfHostedAdapter()
    assert adapter._http.base_url == "http://localhost:8888"

    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8888/memories",
        json={"id": "mem-1"},
    )
    store_result = adapter.store("session-1", "I like tea.")
    assert store_result.memory_id == "mem-1"

    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8888/search",
        json={"results": [{"id": "mem-1", "memory": "I like tea.", "score": 0.9}]},
    )
    query_result = adapter.query("session-1", "what do I like?")
    assert len(query_result.records) == 1
    assert query_result.records[0].content == "I like tea."
    assert query_result.conflict_signal == ConflictSignal.NOT_APPLICABLE

    httpx_mock.add_response(
        method="PUT",
        url="http://localhost:8888/memories/mem-1",
        json={"id": "mem-1"},
    )
    update_result = adapter.update("session-1", "mem-1", "I like coffee now.")
    assert update_result.acknowledged is True
    adapter.close()


def test_mem0_selfhosted_query_with_empty_string_run_id_and_agent_id(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """Deliberately-empty-string run_id/agent_id must not crash, and must
    reach the server as literal empty strings inside `filters` (an
    `is not None` check, not a truthy check) -- this is the exact filter
    shape mem0ai/mem0#5973 (entity-id filter scoping) describes, and the
    self-hosted server's own deprecated-field merge path is documented
    (see module docstring) to silently drop falsy values here instead.
    memtrust must not reproduce that same drop itself, or the eval could
    never observe the vendor's behavior.
    """
    monkeypatch.setenv("MEM0_SELFHOSTED_BASE_URL", "http://localhost:8888")
    adapter = Mem0SelfHostedAdapter()

    captured_request: dict[str, Any] = {}

    def _capture(request: Any) -> Any:
        import json as _json

        captured_request.update(_json.loads(request.content))
        return httpx.Response(status_code=200, json={"results": []})

    httpx_mock.add_callback(_capture, method="POST", url="http://localhost:8888/search")

    result = adapter.query("session-1", "what do I like?", run_id="", agent_id="")

    assert result.records == []
    assert captured_request["filters"] == {"user_id": "session-1", "run_id": "", "agent_id": ""}
    adapter.close()


def test_mem0_selfhosted_query_omits_run_id_and_agent_id_when_not_passed(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEM0_SELFHOSTED_BASE_URL", "http://localhost:8888")
    adapter = Mem0SelfHostedAdapter()

    captured_request: dict[str, Any] = {}

    def _capture(request: Any) -> Any:
        import json as _json

        captured_request.update(_json.loads(request.content))
        return httpx.Response(status_code=200, json={"results": []})

    httpx_mock.add_callback(_capture, method="POST", url="http://localhost:8888/search")

    adapter.query("session-1", "what do I like?")

    assert captured_request["filters"] == {"user_id": "session-1"}
    adapter.close()


def test_mem0_selfhosted_query_passes_threshold_when_given(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEM0_SELFHOSTED_BASE_URL", "http://localhost:8888")
    adapter = Mem0SelfHostedAdapter()

    captured_request: dict[str, Any] = {}

    def _capture(request: Any) -> Any:
        import json as _json

        captured_request.update(_json.loads(request.content))
        return httpx.Response(status_code=200, json={"results": []})

    httpx_mock.add_callback(_capture, method="POST", url="http://localhost:8888/search")

    adapter.query("session-1", "what do I like?", threshold=0.4)

    assert captured_request["threshold"] == 0.4
    adapter.close()


def test_mem0_selfhosted_store_raises_backend_api_error_on_http_failure(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEM0_SELFHOSTED_BASE_URL", "http://localhost:8888")
    adapter = Mem0SelfHostedAdapter()
    httpx_mock.add_response(status_code=500)
    with pytest.raises(BackendAPIError):
        adapter.store("session-1", "content")
    adapter.close()


def test_mem0_selfhosted_uses_explicit_base_url_and_api_key_over_env(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.delenv("MEM0_SELFHOSTED_BASE_URL", raising=False)
    monkeypatch.delenv("MEM0_SELFHOSTED_API_KEY", raising=False)
    adapter = Mem0SelfHostedAdapter(base_url="http://example-host:9000", api_key="secret-key")
    assert adapter._http.base_url == "http://example-host:9000"
    assert adapter._http.headers["x-api-key"] == "secret-key"
    adapter.close()


# ---------------------------------------------------------------------------
# ZepGraphitiAdapter
# ---------------------------------------------------------------------------


def test_zep_query_flags_invalidated_edge(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("ZEP_API_KEY", "test-key")
    adapter = ZepGraphitiAdapter()
    httpx_mock.add_response(
        method="POST",
        url="https://api.getzep.com/graph/search",
        json={
            "edges": [
                {"uuid": "e1", "fact": "meeting at 2pm", "invalid_at": "2026-06-01T00:00:00Z"},
                {"uuid": "e2", "fact": "meeting at 3pm", "invalid_at": None},
            ]
        },
    )
    result = adapter.query("session-1", "what time is the meeting?")
    assert result.conflict_signal == ConflictSignal.FLAGGED
    assert len(result.records) == 2
    adapter.close()


def test_zep_query_served_stale_when_no_invalidation(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("ZEP_API_KEY", "test-key")
    adapter = ZepGraphitiAdapter()
    httpx_mock.add_response(
        method="POST",
        url="https://api.getzep.com/graph/search",
        json={"edges": [{"uuid": "e1", "fact": "meeting at 2pm", "invalid_at": None}]},
    )
    result = adapter.query("session-1", "what time is the meeting?")
    assert result.conflict_signal == ConflictSignal.SERVED_STALE
    adapter.close()


def test_zep_update_aliases_to_store(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("ZEP_API_KEY", "test-key")
    adapter = ZepGraphitiAdapter()
    httpx_mock.add_response(
        method="POST", url="https://api.getzep.com/graph/episodes", json={"uuid": "e2"}
    )
    result = adapter.update("session-1", "e1", "meeting at 3pm")
    assert result.memory_id == "e2"
    adapter.close()


def test_zep_store_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("ZEP_API_KEY", "test-key")
    adapter = ZepGraphitiAdapter()
    httpx_mock.add_response(status_code=503)
    with pytest.raises(BackendAPIError):
        adapter.store("session-1", "content")
    adapter.close()


# ---------------------------------------------------------------------------
# OpenVikingAdapter
# ---------------------------------------------------------------------------


def test_openviking_store_and_query(monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()

    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/write",
        json={"path": "viking://memory/session-1/abc"},
    )
    store_result = adapter.store("session-1", "I prefer dark mode.")
    assert "viking://" in store_result.memory_id

    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/search",
        json={
            "results": [{"path": "viking://memory/session-1/abc", "content": "I prefer dark mode."}]
        },
    )
    query_result = adapter.query("session-1", "what mode do I prefer?")
    assert len(query_result.records) == 1
    assert query_result.conflict_signal == ConflictSignal.NOT_APPLICABLE
    adapter.close()


def test_openviking_uses_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    monkeypatch.setenv("OPENVIKING_BASE_URL", "https://self-hosted.example.com")
    adapter = OpenVikingAdapter()
    assert str(adapter._http.base_url) == "https://self-hosted.example.com"
    adapter.close()


def test_openviking_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(status_code=401)
    with pytest.raises(BackendAPIError):
        adapter.store("session-1", "content")
    adapter.close()


# ---------------------------------------------------------------------------
# MemPalaceAdapter (fake in-memory Palace, no chromadb dependency required)
# ---------------------------------------------------------------------------


class FakePalace:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._next_id = 0

    def remember(self, room: str, content: str, metadata: dict[str, str]) -> str:
        self._next_id += 1
        memory_id = f"palace-{self._next_id}"
        self._store[memory_id] = {"room": room, "content": content, "metadata": metadata}
        return memory_id

    def recall(self, room: str, query: str, top_k: int) -> list[dict[str, Any]]:
        return [
            {"id": mid, "content": v["content"], "metadata": v["metadata"]}
            for mid, v in self._store.items()
            if v["room"] == room
        ][:top_k]

    def invalidate(self, room: str, memory_id: str, content: str) -> dict[str, Any]:
        new_id = self.remember(room, content, {"invalidated": "false"})
        if memory_id in self._store:
            self._store[memory_id]["metadata"]["invalidated"] = "true"
        return {"id": new_id}


def test_mempalace_store_query_update_with_fake_palace() -> None:
    palace = FakePalace()
    adapter = MemPalaceAdapter(palace=palace)

    store_result = adapter.store("room-1", "My dog is named Baxter.")
    assert store_result.memory_id == "palace-1"

    query_result = adapter.query("room-1", "what is my dog's name?")
    assert len(query_result.records) == 1
    assert query_result.conflict_signal == ConflictSignal.NOT_APPLICABLE

    adapter.update("room-1", store_result.memory_id, "My dog is actually named Max.")
    query_result_2 = adapter.query("room-1", "what is my dog's name?")
    invalidated = [r for r in query_result_2.records if r.metadata.get("invalidated") == "true"]
    assert len(invalidated) == 1
    assert query_result_2.conflict_signal == ConflictSignal.FLAGGED


def test_mempalace_wraps_vendor_exceptions_in_backend_api_error() -> None:
    class BrokenPalace:
        def remember(self, room: str, content: str, metadata: dict[str, str]) -> str:
            raise RuntimeError("vendor exploded")

        def recall(self, room: str, query: str, top_k: int) -> list[dict[str, Any]]:
            raise RuntimeError("vendor exploded")

        def invalidate(self, room: str, memory_id: str, content: str) -> dict[str, Any]:
            raise RuntimeError("vendor exploded")

    adapter = MemPalaceAdapter(palace=BrokenPalace())
    with pytest.raises(BackendAPIError):
        adapter.store("room-1", "content")
    with pytest.raises(BackendAPIError):
        adapter.query("room-1", "query")
    with pytest.raises(BackendAPIError):
        adapter.update("room-1", "id", "content")


def test_mempalace_get_palace_raises_clear_error_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real `mempalace` package is not a memtrust dependency (kept
    # optional per docs/methodology.md), so in this test environment it
    # is genuinely not installed -- this exercises the real ImportError
    # path, not a simulated one.
    monkeypatch.setenv("MEMPALACE_STORAGE_PATH", "/tmp/fake-palace")
    adapter = MemPalaceAdapter()
    with pytest.raises(BackendAPIError, match="not installed"):
        adapter.store("room-1", "content")
