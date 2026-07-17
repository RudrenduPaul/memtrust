"""Adapter tests. Every HTTP-based adapter is exercised via pytest-httpx
(no real network calls); MemPalaceAdapter is exercised two ways: a fast,
offline fake (`FakeMCPTools`, injected via the `mcp_tools=` constructor
kwarg, matching the confirmed-real `_MCPToolsProtocol` shape defined in
mempalace_adapter.py) for most tests, plus a set of `test_real_mempalace_*`
integration tests near the end of the MemPalaceAdapter section that run
against the actual installed `mempalace` package (gated by
`pytest.importorskip("mempalace")`, requires the optional
`mempalace-direct` extra) -- see mempalace_adapter.py's module docstring
for why there is no fictional-API-era `_PalaceProtocol`/`Palace` fake left
to speak of.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from memtrust.adapters.base import (
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
    CrashSignal,
    DeletePrefixResult,
    DeleteResult,
    ExtractionSignal,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    RankingSignal,
    RetrievalWarning,
    StoreResult,
    UpdateResult,
)
from memtrust.adapters.mem0_adapter import Mem0Adapter, Mem0SelfHostedAdapter
from memtrust.adapters.mempalace_adapter import MemPalaceAdapter
from memtrust.adapters.openviking_adapter import OpenVikingAdapter
from memtrust.adapters.zep_graphiti_adapter import ZepGraphitiAdapter
from memtrust.adapters.zep_graphiti_selfhosted_adapter import (
    ZepGraphitiSelfHostedAdapter,
    _classify_crash,
    _parse_falkordb_url,
)

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


def test_graphiti_selfhosted_raises_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPHITI_NEO4J_URI", raising=False)
    monkeypatch.delenv("GRAPHITI_FALKORDB_URL", raising=False)
    with pytest.raises(BackendNotConfiguredError) as excinfo:
        ZepGraphitiSelfHostedAdapter()
    assert excinfo.value.missing_env_var == "GRAPHITI_NEO4J_URI"
    assert excinfo.value.backend_name == "graphiti_selfhosted"


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
    assert store_result.extraction_signal == ExtractionSignal.FACTS_EXTRACTED

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


def test_mem0_store_reports_empty_extraction_signal_when_no_id_in_response(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """mem0ai/mem0#5178: a store() call can complete with a 200 and a
    normal-shaped body that nonetheless carries no usable memory id --
    Mem0's own extraction pipeline decided there was nothing worth
    persisting. Before this fix, `_extract_memory_id()` silently returned
    "" and the resulting StoreResult looked identical to a genuine
    successful store. This proves the adapter now flags that gap.
    """
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    adapter = Mem0Adapter()
    httpx_mock.add_response(
        method="POST",
        url="https://api.mem0.ai/v1/memories/",
        json={"results": []},
    )
    store_result = adapter.store("session-1", "just saying hi, nothing to remember")
    assert store_result.memory_id == ""
    assert store_result.extraction_signal == ExtractionSignal.EMPTY_EXTRACTION
    adapter.close()


def test_mem0_delete_success(monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    adapter = Mem0Adapter()
    httpx_mock.add_response(
        method="DELETE",
        url="https://api.mem0.ai/v1/memories/mem-1/",
        json={"message": "Memory deleted successfully!"},
    )
    result = adapter.delete("mem-1")
    assert result.success is True
    assert result.memory_id == "mem-1"
    assert result.latency_ms >= 0
    adapter.close()


def test_mem0_delete_raises_backend_api_error_on_http_failure(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    adapter = Mem0Adapter()
    httpx_mock.add_response(method="DELETE", status_code=404)
    with pytest.raises(BackendAPIError):
        adapter.delete("does-not-exist")
    adapter.close()


def test_mem0_delete_many_aggregates_all_results_via_real_http_calls(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """delete_many() must return one result per input id, in order, even
    when some deletes fail -- this is the exact aggregation shape the
    mem0ai/mem0#5936 / #5970 truncation bug got wrong (client code kept
    only the last response instead of all N)."""
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    adapter = Mem0Adapter()
    httpx_mock.add_response(
        method="DELETE", url="https://api.mem0.ai/v1/memories/mem-1/", json={"ok": True}
    )
    httpx_mock.add_response(
        method="DELETE", url="https://api.mem0.ai/v1/memories/mem-2/", status_code=500
    )
    httpx_mock.add_response(
        method="DELETE", url="https://api.mem0.ai/v1/memories/mem-3/", json={"ok": True}
    )

    results = adapter.delete_many(["mem-1", "mem-2", "mem-3"])

    assert len(results) == 3
    assert [r.memory_id for r in results] == ["mem-1", "mem-2", "mem-3"]
    assert [r.success for r in results] == [True, False, True]


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
    assert store_result.extraction_signal == ExtractionSignal.FACTS_EXTRACTED

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


def test_mem0_selfhosted_store_reports_empty_extraction_signal_when_no_id_in_response(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """Same mem0ai/mem0#5178 gap as Mem0Adapter above, reproduced against
    the self-hosted server's unprefixed POST /memories route.
    """
    monkeypatch.setenv("MEM0_SELFHOSTED_BASE_URL", "http://localhost:8888")
    adapter = Mem0SelfHostedAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8888/memories",
        json={"results": []},
    )
    store_result = adapter.store("session-1", "just saying hi, nothing to remember")
    assert store_result.memory_id == ""
    assert store_result.extraction_signal == ExtractionSignal.EMPTY_EXTRACTION
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


def test_mem0_selfhosted_delete_uses_unprefixed_route(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    # Same delete_many() aggregation contract as the hosted adapter, but
    # against the self-hosted server's unprefixed DELETE /memories/{id}
    # route rather than /v1/memories/{id}/.
    monkeypatch.setenv("MEM0_SELFHOSTED_BASE_URL", "http://localhost:8888")
    adapter = Mem0SelfHostedAdapter()
    httpx_mock.add_response(
        method="DELETE",
        url="http://localhost:8888/memories/mem-1",
        json={"message": "deleted"},
    )
    result = adapter.delete("mem-1")
    assert result.success is True
    assert result.memory_id == "mem-1"
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


def test_zep_delete_success(monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("ZEP_API_KEY", "test-key")
    adapter = ZepGraphitiAdapter()
    httpx_mock.add_response(
        method="DELETE", url="https://api.getzep.com/graph/episodes/e1", json={}
    )
    result = adapter.delete("e1")
    assert result.success is True
    assert result.memory_id == "e1"
    adapter.close()


def test_zep_delete_raises_backend_api_error_on_http_failure(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("ZEP_API_KEY", "test-key")
    adapter = ZepGraphitiAdapter()
    httpx_mock.add_response(method="DELETE", status_code=500)
    with pytest.raises(BackendAPIError):
        adapter.delete("e1")
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
# OpenVikingAdapter -- error detail includes real response body
# (volcengine/OpenViking#1227)
#
# `str(exc)` on an httpx.HTTPStatusError is only the status line (e.g.
# "Client error '400 Bad Request' for url ..."), never the real response
# body -- a server-side Pydantic validation error like `"id"
# extra_forbidden` was silently swallowed down to that useless status
# line before this fix. The raised BackendAPIError's detail must contain
# the actual response body, not just the generic status line.
# ---------------------------------------------------------------------------


def test_openviking_store_error_detail_includes_response_body(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/write",
        status_code=400,
        json={
            "detail": [
                {
                    "type": "extra_forbidden",
                    "loc": ["body", "id"],
                    "msg": "Extra inputs are not permitted",
                }
            ]
        },
    )
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.store("session-1", "content")
    assert "extra_forbidden" in exc_info.value.detail
    assert "Extra inputs are not permitted" in exc_info.value.detail
    # The generic status-line message alone (pre-fix behavior) must not be
    # all that's captured -- the real body content has to be present too.
    assert (
        exc_info.value.detail
        != "Client error '400 Bad Request' for url 'http://localhost:1933/v1/fs/write'"
    )
    adapter.close()


def test_openviking_query_error_detail_includes_response_body(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/search",
        status_code=422,
        json={"detail": "path_prefix: field required"},
    )
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.query("session-1", "what mode do I prefer?")
    assert "path_prefix: field required" in exc_info.value.detail
    adapter.close()


def test_openviking_error_detail_falls_back_to_str_exc_when_no_response_body(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    # A 401 with an empty body must not crash trying to append an empty
    # body -- it falls back to the plain str(exc) status-line message.
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(status_code=401)
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.store("session-1", "content")
    assert "response body" not in exc_info.value.detail
    adapter.close()


def test_openviking_delete_success(monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/delete",
        json={"deleted": True},
    )
    result = adapter.delete("viking://memory/session-1/abc")
    assert result.success is True
    assert result.memory_id == "viking://memory/session-1/abc"
    adapter.close()


def test_openviking_delete_raises_backend_api_error_on_http_failure(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(method="POST", status_code=500)
    with pytest.raises(BackendAPIError):
        adapter.delete("viking://memory/session-1/abc")
    adapter.close()


# ---------------------------------------------------------------------------
# OpenVikingAdapter.get_stats() -- volcengine/OpenViking#1255
#
# #1255 (real bug, reported by SeeYangZhi): GET /api/v1/stats/memories
# returns an all-zero count even when memories genuinely exist (confirmed
# via filesystem listing and /v1/search/find in the issue's own repro).
# These tests confirm this adapter parses the real, issue-quoted response
# shape correctly -- they do not, and cannot, prove the live endpoint
# still exhibits the undercounting bug; see evals/stats_accuracy.py for
# the eval that classifies undercounting against an independently
# verified count.
# ---------------------------------------------------------------------------


def test_openviking_get_stats_parses_total_memories(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:1933/api/v1/stats/memories",
        json={
            "total_memories": 10,
            "by_category": {"profile": 1, "preferences": 3, "entities": 6},
        },
    )
    result = adapter.get_stats()
    assert result.total_memories == 10
    assert result.raw["by_category"]["entities"] == 6
    adapter.close()


def test_openviking_get_stats_reproduces_1255_zero_count_shape(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """The exact response shape quoted verbatim in #1255's bug report --
    this adapter must parse it faithfully (0, not None or an error) so
    evals/stats_accuracy.py's undercount comparison has a real number to
    compare against."""
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:1933/api/v1/stats/memories",
        json={
            "total_memories": 0,
            "by_category": {
                "profile": 0,
                "preferences": 0,
                "entities": 0,
                "events": 0,
                "cases": 0,
                "patterns": 0,
                "tools": 0,
                "skills": 0,
            },
            "hotness_distribution": {"cold": 0, "warm": 0, "hot": 0},
        },
    )
    result = adapter.get_stats()
    assert result.total_memories == 0
    adapter.close()


def test_openviking_get_stats_raises_backend_api_error_on_http_failure(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(method="GET", status_code=500)
    with pytest.raises(BackendAPIError):
        adapter.get_stats()
    adapter.close()


def test_openviking_get_stats_missing_field_returns_none(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:1933/api/v1/stats/memories",
        json={"unexpected_shape": True},
    )
    result = adapter.get_stats()
    assert result.total_memories is None
    adapter.close()


def test_openviking_supports_stats_is_true() -> None:
    assert OpenVikingAdapter.supports_stats is True


# ---------------------------------------------------------------------------
# OpenVikingAdapter.store() resource_path -- volcengine/OpenViking#1703 gap
#
# #1703 (real bug, reported by SonicBotMan): OpenViking's own
# index_resource() skipped every subdirectory during reindex, so nested-
# directory content was never vectorized and searches over it silently
# returned nothing. That bug is only reachable by this harness at all if
# memtrust's own store() actually constructs a real nested directory tree
# against OpenViking in the first place -- these tests prove store() now
# does that when a caller supplies `resource_path` metadata, and that it
# still falls back to the pre-existing flat content-hash path when no
# caller does (no regression for evals/contradiction.py, evals/
# compression.py, longmemeval.py, locomo.py -- none of which pass
# resource_path).
# ---------------------------------------------------------------------------


def test_openviking_store_with_resource_path_writes_real_nested_path(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/write",
        match_json={
            "path": "viking://memory/session-1/entities/people/jordan-lee.md",
            "content": "Jordan Lee prefers async standups.",
            "metadata": {"resource_path": "entities/people/jordan-lee.md", "origin": "user"},
        },
        json={"path": "viking://memory/session-1/entities/people/jordan-lee.md"},
    )

    result = adapter.store(
        "session-1",
        "Jordan Lee prefers async standups.",
        metadata={"resource_path": "entities/people/jordan-lee.md", "origin": "user"},
    )

    assert result.memory_id == "viking://memory/session-1/entities/people/jordan-lee.md"
    adapter.close()


def test_openviking_store_without_resource_path_falls_back_to_flat_hash(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """No regression: a caller that never sets `resource_path` in metadata
    (every eval except resource_sync_safety.py) must keep writing to the
    same flat memory/{session_id}/{sha256(content)[:16]} path as before."""
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()

    def capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        assert body["path"].startswith("viking://memory/session-1/")
        suffix = body["path"].removeprefix("viking://memory/session-1/")
        assert "/" not in suffix  # flat, single-level -- no nested directory
        assert len(suffix) == 16  # sha256(content)[:16]
        return httpx.Response(status_code=200, json={"path": body["path"]})

    httpx_mock.add_callback(
        capture, method="POST", url="http://localhost:1933/v1/fs/write", is_reusable=True
    )

    adapter.store("session-1", "I prefer dark mode.")
    adapter.store("session-1", "I prefer dark mode.", metadata={"origin": "user"})
    adapter.close()


def test_openviking_store_resource_path_with_leading_slash_is_normalized(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/write",
        match_json={
            "path": "viking://memory/session-1/preferences/user-482/notifications.md",
            "content": "Notifications muted after 9pm.",
            "metadata": {"resource_path": "/preferences/user-482/notifications.md"},
        },
        json={"path": "viking://memory/session-1/preferences/user-482/notifications.md"},
    )

    result = adapter.store(
        "session-1",
        "Notifications muted after 9pm.",
        metadata={"resource_path": "/preferences/user-482/notifications.md"},
    )

    assert result.memory_id == "viking://memory/session-1/preferences/user-482/notifications.md"
    adapter.close()


# ---------------------------------------------------------------------------
# OpenVikingAdapter.list_resource_paths() -- real recursive tree walk
# ---------------------------------------------------------------------------


def test_openviking_list_resource_paths_walks_nested_directories(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """A single flat response is not assumed to already contain every
    nested file: a directory entry in the first response must be
    descended into with a follow-up call, and the final list must contain
    the real leaf paths from every level, not just the top level."""
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()

    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/list",
        match_json={"path_prefix": "viking://memory/session-1/entities"},
        json={
            "entries": [
                {"path": "viking://memory/session-1/entities/skills.md", "type": "file"},
                {"path": "viking://memory/session-1/entities/people", "type": "directory"},
            ]
        },
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/list",
        match_json={"path_prefix": "viking://memory/session-1/entities/people"},
        json={
            "entries": [
                {
                    "path": "viking://memory/session-1/entities/people/jordan-lee.md",
                    "type": "file",
                },
                {
                    "path": "viking://memory/session-1/entities/people/alex-kim.md",
                    "type": "file",
                },
            ]
        },
    )

    paths = adapter.list_resource_paths("memory/session-1/entities")

    assert sorted(paths) == sorted(
        [
            "viking://memory/session-1/entities/skills.md",
            "viking://memory/session-1/entities/people/jordan-lee.md",
            "viking://memory/session-1/entities/people/alex-kim.md",
        ]
    )
    adapter.close()


def test_openviking_list_resource_paths_treats_trailing_slash_as_directory(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """Bare path-string entries (not dicts) that end in "/" must also be
    recursed into -- the directory marker is not assumed to always arrive
    as a {"type": "directory"} dict."""
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()

    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/list",
        match_json={"path_prefix": "viking://memory/session-1/preferences"},
        json={"paths": ["viking://memory/session-1/preferences/user-482/"]},
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/list",
        match_json={"path_prefix": "viking://memory/session-1/preferences/user-482"},
        json={"paths": ["viking://memory/session-1/preferences/user-482/notifications.md"]},
    )

    paths = adapter.list_resource_paths("memory/session-1/preferences")

    assert paths == ["viking://memory/session-1/preferences/user-482/notifications.md"]
    adapter.close()


def test_openviking_list_resource_paths_respects_max_depth(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """max_depth=0 must not recurse at all -- a directory entry at the
    top level is dropped rather than descended into."""
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()

    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/list",
        match_json={"path_prefix": "viking://memory/session-1/entities"},
        json={
            "entries": [
                {"path": "viking://memory/session-1/entities/skills.md", "type": "file"},
                {"path": "viking://memory/session-1/entities/people", "type": "directory"},
            ]
        },
    )

    paths = adapter.list_resource_paths("memory/session-1/entities", max_depth=0)

    assert paths == ["viking://memory/session-1/entities/skills.md"]
    adapter.close()


# ---------------------------------------------------------------------------
# OpenVikingAdapter -- CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE
# (volcengine/OpenViking#2966, lRoccoon)
#
# A legacy uint16-length-truncated record's `fields` JSON crashes
# LocalIndex's internal delta-list conversion with a bare
# json.decoder.JSONDecodeError on the delete()/upsert (store()/update())
# write paths. This adapter's own resp.json() calls would otherwise let
# that raw exception escape unclassified -- these tests prove it is caught
# and classified instead.
# ---------------------------------------------------------------------------


def test_openviking_store_raises_crash_signal_on_malformed_json_response(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/write",
        status_code=200,
        content=b'{"path": "viking://memory/session-1/abc", "corrupted": tru',
    )
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.store("session-1", "content")
    assert exc_info.value.crash_signal == CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE
    adapter.close()


def test_openviking_update_raises_crash_signal_on_malformed_json_response(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/write",
        status_code=200,
        content=b"not json at all",
    )
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.update("session-1", "viking://memory/session-1/abc", "new content")
    assert exc_info.value.crash_signal == CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE
    adapter.close()


def test_openviking_delete_raises_crash_signal_on_malformed_json_response(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/delete",
        status_code=200,
        content=b'{"unterminated": "strin',
    )
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.delete("viking://memory/session-1/abc")
    assert exc_info.value.crash_signal == CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE
    adapter.close()


def test_delete_result_corruption_signal_defaults_to_not_applicable() -> None:
    """DeleteResult now mirrors StoreResult/UpdateResult's corruption_signal
    field -- confirms the default without needing a live adapter."""
    from memtrust.adapters.base import CorruptionSignal

    result = DeleteResult(success=True, memory_id="m1", latency_ms=1.0)
    assert result.corruption_signal == CorruptionSignal.NOT_APPLICABLE


class LegacyCorruptRecordFakeAdapter(MemoryBackendAdapter):
    """In-memory fake modeling volcengine/OpenViking#2966's exact shape:
    delete() raises BackendAPIError with
    CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE for ids the fake treats
    as legacy-corrupt, and succeeds normally for every other id. Same
    precedent as evals/crash_recovery.py's CrashRecoveryFakeAdapter (see
    test_evals.py) -- a purpose-built fake proving the taxonomy classifies
    correctly, independent of any live OpenViking instance."""

    name = "fake-legacy-corrupt"
    env_var = "FAKE_API_KEY"

    def __init__(self, corrupt_ids: set[str]) -> None:
        self._corrupt_ids = corrupt_ids
        self._store: dict[str, str] = {}

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        memory_id = f"{session_id}-{len(self._store)}"
        self._store[memory_id] = content
        return StoreResult(memory_id=memory_id, latency_ms=0.1)

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        return QueryResult(
            records=[], conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        if memory_id in self._corrupt_ids:
            raise BackendAPIError(
                self.name,
                "legacy-corrupt record",
                crash_signal=CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE,
            )
        self._store[memory_id] = content
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        if memory_id in self._corrupt_ids:
            raise BackendAPIError(
                self.name,
                "legacy-corrupt record: JSONDecodeError during delete_data()",
                crash_signal=CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE,
            )
        self._store.pop(memory_id, None)
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=0.1)


def test_legacy_corrupt_record_delete_raises_classified_crash_signal() -> None:
    adapter = LegacyCorruptRecordFakeAdapter(corrupt_ids={"corrupt-1"})
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.delete("corrupt-1")
    assert exc_info.value.crash_signal == CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE


def test_legacy_corrupt_record_delete_many_records_failure_per_id_without_crashing() -> None:
    """delete_many()'s existing per-id aggregation (base.py) must not let
    one legacy-corrupt record's exception truncate or crash the batch --
    it should record that id as a failure and keep processing the rest."""
    adapter = LegacyCorruptRecordFakeAdapter(corrupt_ids={"corrupt-1"})
    store_1 = adapter.store("s1", "clean content one")
    store_2 = adapter.store("s1", "clean content two")

    results = adapter.delete_many([store_1.memory_id, "corrupt-1", store_2.memory_id])

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is False
    assert "legacy-corrupt" in str(results[1].raw.get("error", ""))
    assert results[2].success is True


# ---------------------------------------------------------------------------
# OpenVikingAdapter.store() -- ExtractionSignal (volcengine/OpenViking#2751,
# gleydson115-code)
# ---------------------------------------------------------------------------


def test_openviking_store_sets_facts_extracted_when_total_memories_absent(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/write",
        json={"path": "viking://memory/session-1/abc"},
    )
    result = adapter.store("session-1", "I prefer dark mode.")
    assert result.extraction_signal == ExtractionSignal.FACTS_EXTRACTED
    adapter.close()


def test_openviking_store_sets_empty_extraction_when_total_memories_is_zero(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """volcengine/OpenViking#2751: the OpenAI VLM backend's hardcoded
    max_tokens=32768 exceeds gpt-4o-mini's real 16384 cap; the resulting
    API 400 gets swallowed inside compressor_v2, and the write commit
    still returns 200/accepted with total_memories staying 0, silent."""
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/write",
        json={"path": "viking://memory/session-1/abc", "total_memories": 0},
    )
    result = adapter.store("session-1", "some chit-chat with no facts")
    assert result.extraction_signal == ExtractionSignal.EMPTY_EXTRACTION
    adapter.close()


def test_openviking_store_sets_facts_extracted_when_total_memories_nonzero(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/write",
        json={"path": "viking://memory/session-1/abc", "total_memories": 3},
    )
    result = adapter.store("session-1", "I prefer dark mode.")
    assert result.extraction_signal == ExtractionSignal.FACTS_EXTRACTED
    adapter.close()


# ---------------------------------------------------------------------------
# OpenVikingAdapter.query() -- RankingSignal.RERANK_FALLBACK
# (volcengine/OpenViking#1737 wychosenone, #2739/#2880 hhspiny)
# ---------------------------------------------------------------------------


def test_openviking_query_flags_rerank_fallback_on_empty_document(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/search",
        json={
            "results": [
                {"path": "viking://memory/session-1/a", "content": ""},
                {"path": "viking://memory/session-1/b", "content": "some real content"},
            ]
        },
    )
    result = adapter.query("session-1", "anything")
    assert result.ranking_signal == RankingSignal.RERANK_FALLBACK
    adapter.close()


def test_openviking_query_flags_rerank_fallback_on_oversized_batch(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """hhspiny's #2880 shape: L2 abstracts collectively exceed the
    reranker's real token budget (~4096 tokens, MAX_RERANK_TOKENS)."""
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    oversized_content = "x" * 20000  # ~5000 estimated tokens, over budget alone
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/search",
        json={
            "results": [
                {"path": "viking://memory/session-1/a", "content": oversized_content},
                {"path": "viking://memory/session-1/b", "content": "short"},
            ]
        },
    )
    result = adapter.query("session-1", "anything")
    assert result.ranking_signal == RankingSignal.RERANK_FALLBACK
    adapter.close()


def test_openviking_query_does_not_flag_rerank_fallback_under_budget(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/search",
        json={
            "results": [
                {"path": "viking://memory/session-1/a", "content": "a normal short memory"},
                {"path": "viking://memory/session-1/b", "content": "another normal memory"},
            ]
        },
    )
    result = adapter.query("session-1", "anything")
    assert result.ranking_signal == RankingSignal.NOT_APPLICABLE
    adapter.close()


def test_openviking_query_single_record_never_flags_rerank_fallback(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """A single-candidate response has nothing rerank would meaningfully
    reorder -- must not be flagged even if that one record is empty."""
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/search",
        json={"results": [{"path": "viking://memory/session-1/a", "content": ""}]},
    )
    result = adapter.query("session-1", "anything")
    assert result.ranking_signal == RankingSignal.NOT_APPLICABLE
    adapter.close()


# ---------------------------------------------------------------------------
# OpenVikingAdapter.delete_prefix() (volcengine/OpenViking#3064, AcTiveXXX)
# ---------------------------------------------------------------------------


def test_openviking_delete_prefix_deletes_discovered_children_and_root(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/list",
        match_json={"path_prefix": "viking://orphan-test"},
        json={
            "entries": [
                {"path": "viking://orphan-test/child1.md", "type": "file"},
                {"path": "viking://orphan-test/child2.md", "type": "file"},
            ]
        },
    )
    for _ in range(3):  # 2 discovered children + the prefix root itself
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:1933/v1/fs/delete",
            json={"deleted": True},
        )
    result = adapter.delete_prefix("orphan-test", recursive=True)

    assert set(result.deleted_paths) == {
        "viking://orphan-test/child1.md",
        "viking://orphan-test/child2.md",
        "viking://orphan-test",
    }
    assert result.failed_paths == []
    adapter.close()


def test_openviking_delete_prefix_only_deletes_root_when_listing_finds_nothing(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """Reproduces the client-visible shape of volcengine/OpenViking#3064:
    when the underlying listing call reports zero children (the exact
    consequence of the server's own bare `except: pass` on a directory
    that no longer exists in AGFS), this adapter can only discover and
    delete the root URI -- it has no way to independently discover the
    orphaned children through this endpoint alone. This is exactly why
    evals/orphan_cleanup.py's classification re-queries for seeded content
    afterward instead of trusting an empty listing as proof of a clean
    delete -- see VectorIntegritySignal.ORPHANED_VECTOR_ENTRY."""
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/list",
        json={"entries": []},
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/delete",
        json={"deleted": True},
    )
    result = adapter.delete_prefix("orphan-test", recursive=True)

    assert result.deleted_paths == ["viking://orphan-test"]
    adapter.close()


def test_openviking_delete_prefix_records_failed_child_deletes(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    adapter = OpenVikingAdapter()
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:1933/v1/fs/list",
        json={"entries": [{"path": "viking://orphan-test/child1.md", "type": "file"}]},
    )
    httpx_mock.add_response(
        method="POST", url="http://localhost:1933/v1/fs/delete", status_code=500
    )
    httpx_mock.add_response(
        method="POST", url="http://localhost:1933/v1/fs/delete", json={"deleted": True}
    )
    result = adapter.delete_prefix("orphan-test", recursive=True)

    assert "viking://orphan-test/child1.md" in result.failed_paths
    adapter.close()


class OrphanedChildFakeAdapter(MemoryBackendAdapter):
    """In-memory fake modeling volcengine/OpenViking#3064's exact shape at
    the harness level: delete_prefix() only ever removes the root URI from
    the backing store (mirroring the bug's "only the root URI gets
    deleted" outcome), list_resource_paths() correctly reports the
    directory as empty afterward (the AGFS-listing-level view says
    "gone"), but query() still surfaces the orphaned children (the vector
    index disagrees). Same precedent as test_evals.py's
    CrashRecoveryFakeAdapter/MigrationRollbackFakeAdapter -- a
    purpose-built fake proving the classifier tells CLEAN and
    ORPHANED_VECTOR_ENTRY apart correctly."""

    name = "fake-orphaned-child"
    env_var = "FAKE_API_KEY"
    supports_resource_sync = True
    supports_prefix_delete = True

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        path = (metadata or {}).get("resource_path", f"{session_id}-{len(self._store)}")
        self._store[path] = content
        return StoreResult(memory_id=path, latency_ms=0.1)

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        records = [
            MemoryRecord(memory_id=path, content=content)
            for path, content in self._store.items()
            if query.lower() in content.lower()
        ]
        return QueryResult(
            records=records, conflict_signal=ConflictSignal.NOT_APPLICABLE, latency_ms=0.1
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        self._store[memory_id] = content
        return UpdateResult(memory_id=memory_id, acknowledged=True, latency_ms=0.1)

    def delete(self, memory_id: str) -> DeleteResult:
        existed = self._store.pop(memory_id, None) is not None
        return DeleteResult(success=existed, memory_id=memory_id, latency_ms=0.1)

    def list_resource_paths(self, prefix: str) -> list[str]:
        # The buggy AGFS-listing view: unconditionally reports nothing --
        # this models #3064's bare `except: pass` swallowing a listing
        # failure against a parent directory already gone from AGFS, the
        # same failure that made _collect_uris() return an empty list
        # regardless of how many child vector-index entries actually
        # still exist.
        return []

    def delete_prefix(self, prefix: str, recursive: bool = True) -> DeletePrefixResult:
        # Models the #3064 bug directly: only the root URI is ever
        # removed from the backing store, regardless of how many children
        # exist beneath it.
        existed = self._store.pop(prefix, None) is not None
        deleted = [prefix] if existed else []
        return DeletePrefixResult(
            prefix=prefix, deleted_paths=deleted, failed_paths=[], latency_ms=0.1
        )


def test_orphaned_child_fake_adapter_leaves_query_matches_after_delete_prefix() -> None:
    adapter = OrphanedChildFakeAdapter()
    adapter.store("s1", "root marker", metadata={"resource_path": "parent"})
    adapter.store("s1", "child alpha content", metadata={"resource_path": "parent/child-alpha.md"})
    adapter.store("s1", "child beta content", metadata={"resource_path": "parent/child-beta.md"})

    adapter.delete_prefix("parent", recursive=True)

    # The buggy listing view reports the prefix as fully gone...
    assert adapter.list_resource_paths("parent") == []
    # ...but the children are still there and still searchable.
    query_result = adapter.query("s1", "child alpha")
    assert len(query_result.records) == 1
    assert query_result.records[0].content == "child alpha content"


# ---------------------------------------------------------------------------
# MemPalaceAdapter (fake mempalace.mcp_server module, no chromadb
# dependency required)
#
# FakeMCPTools stands in for the real `mempalace.mcp_server` module.
# Unlike the removed FakePalace (which stood in for a fictional
# `mempalace.Palace` class that never existed in the real package -- see
# mempalace_adapter.py's module docstring), every method here mirrors a
# function confirmed real and live-verified against the installed
# `mempalace` package (version 3.5.0). See
# test_real_mempalace_* below for the actual live-package integration
# tests these fake-backed tests are a fast, offline complement to.
# ---------------------------------------------------------------------------


class FakeMCPTools:
    def __init__(self) -> None:
        self._drawers: dict[str, dict[str, Any]] = {}
        self._next_id = 0
        self.status_response: dict[str, Any] = {
            "total_drawers": 0,
            "wings": {},
            "rooms": {},
            "backend": "chroma",
        }
        self.list_wings_response: dict[str, Any] = {"wings": {}}
        self.list_rooms_response: dict[str, Any] = {"wing": "all", "rooms": {}}
        self.list_rooms_calls: list[str | None] = []
        self.add_drawer_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.kg_facts: list[dict[str, Any]] = []
        self.reconnect_call_count = 0

    def tool_reconnect(self) -> dict[str, Any]:
        self.reconnect_call_count += 1
        return {"success": True, "message": "Reconnected to palace", "drawers": len(self._drawers)}

    def tool_status(self) -> dict[str, Any]:
        return self.status_response

    def tool_list_wings(self) -> dict[str, Any]:
        return self.list_wings_response

    def tool_list_rooms(self, wing: str | None = None) -> dict[str, Any]:
        self.list_rooms_calls.append(wing)
        return self.list_rooms_response

    def tool_add_drawer(
        self,
        wing: str,
        room: str,
        content: str,
        source_file: str | None = None,
        added_by: str = "mcp",
    ) -> dict[str, Any]:
        self.add_drawer_calls.append(
            {
                "wing": wing,
                "room": room,
                "content": content,
                "source_file": source_file,
                "added_by": added_by,
            }
        )
        for drawer_id, d in self._drawers.items():
            if d["wing"] == wing and d["room"] == room and d["content"] == content:
                return {"success": True, "reason": "already_exists", "drawer_id": drawer_id}
        self._next_id += 1
        drawer_id = f"drawer-{self._next_id}"
        self._drawers[drawer_id] = {
            "wing": wing,
            "room": room,
            "content": content,
            "source_file": source_file,
            "created_at": f"2026-01-{self._next_id:02d}T00:00:00",
        }
        return {"success": True, "drawer_id": drawer_id, "wing": wing, "room": room, "chunks": 1}

    def tool_search(
        self,
        query: str,
        limit: int = 5,
        wing: str | None = None,
        room: str | None = None,
        source_file: str | None = None,
        max_distance: float = 1.5,
        min_similarity: float | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        self.search_calls.append({"query": query, "limit": limit, "wing": wing, "room": room})
        matches = [
            (drawer_id, d)
            for drawer_id, d in self._drawers.items()
            if (wing is None or d["wing"] == wing) and (room is None or d["room"] == room)
        ]
        results = []
        for i, (_drawer_id, d) in enumerate(matches[:limit]):
            similarity = round(0.9 - (i * 0.1), 3)
            results.append(
                {
                    "text": d["content"],
                    "wing": d["wing"],
                    "room": d["room"],
                    "source_file": d.get("source_file") or "",
                    "source_path": d.get("source_file") or "",
                    "created_at": d["created_at"],
                    "similarity": similarity,
                    "distance": round(1 - similarity, 4),
                    "effective_distance": round(1 - similarity, 4),
                    "closet_boost": 0.0,
                    "matched_via": "drawer",
                    "bm25_score": 1.0,
                }
            )
        return {
            "query": query,
            "filters": {"wing": wing, "room": room, "source_file": source_file},
            "total_before_filter": len(matches),
            "results": results,
        }

    def tool_update_drawer(
        self,
        drawer_id: str,
        content: str | None = None,
        wing: str | None = None,
        room: str | None = None,
    ) -> dict[str, Any]:
        if drawer_id not in self._drawers:
            return {"success": False, "error": f"Drawer not found: {drawer_id}"}
        if content is not None:
            self._drawers[drawer_id]["content"] = content
        if wing is not None:
            self._drawers[drawer_id]["wing"] = wing
        if room is not None:
            self._drawers[drawer_id]["room"] = room
        return {
            "success": True,
            "drawer_id": drawer_id,
            "wing": self._drawers[drawer_id]["wing"],
            "room": self._drawers[drawer_id]["room"],
        }

    def tool_delete_drawer(self, drawer_id: str) -> dict[str, Any]:
        if drawer_id not in self._drawers:
            return {"success": False, "error": f"Drawer not found: {drawer_id}"}
        del self._drawers[drawer_id]
        return {
            "success": True,
            "drawer_id": drawer_id,
            "deleted_ids": [drawer_id],
            "chunks_deleted": 1,
        }

    def tool_kg_add(
        self,
        subject: str,
        predicate: str,
        object: str,  # noqa: A002 - mirrors the real tool_kg_add() parameter name
        valid_from: str | None = None,
        valid_to: str | None = None,
        source_closet: str | None = None,
        source_file: str | None = None,
        source_drawer_id: str | None = None,
    ) -> dict[str, Any]:
        for f in self.kg_facts:
            if f["subject"] == subject and f["predicate"] == predicate and f["object"] == object:
                return {
                    "success": True,
                    "triple_id": f["triple_id"],
                    "fact": f"{subject} → {predicate} → {object}",
                }
        triple_id = f"t_{subject}_{predicate}_{object}_{len(self.kg_facts)}"
        self.kg_facts.append(
            {
                "triple_id": triple_id,
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "confidence": 1.0,
                "source_closet": source_closet,
                "current": valid_to is None,
            }
        )
        return {
            "success": True,
            "triple_id": triple_id,
            "fact": f"{subject} → {predicate} → {object}",
        }

    def tool_kg_invalidate(
        self,
        subject: str,
        predicate: str,
        object: str,  # noqa: A002 - mirrors the real tool_kg_invalidate() parameter name
        ended: str | None = None,
    ) -> dict[str, Any]:
        resolved = ended or "2026-01-01"
        for f in self.kg_facts:
            if f["subject"] == subject and f["predicate"] == predicate and f["object"] == object:
                f["valid_to"] = resolved
                f["current"] = False
        return {
            "success": True,
            "fact": f"{subject} → {predicate} → {object}",
            "ended": resolved,
        }

    def tool_kg_query(
        self, entity: str, as_of: str | None = None, direction: str = "both"
    ) -> dict[str, Any]:
        facts = []
        for f in self.kg_facts:
            if direction in ("outgoing", "both") and f["subject"] == entity:
                facts.append({**f, "direction": "outgoing"})
            elif direction in ("incoming", "both") and f["object"] == entity:
                facts.append({**f, "direction": "incoming"})
        return {"entity": entity, "as_of": as_of, "facts": facts, "count": len(facts)}


def test_mempalace_store_query_update_delete_with_fake_mcp_tools() -> None:
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)

    store_result = adapter.store("room-1", "My dog is named Baxter.")
    assert store_result.memory_id == "drawer-1"

    query_result = adapter.query("room-1", "what is my dog's name?")
    assert len(query_result.records) == 1
    assert query_result.records[0].content == "My dog is named Baxter."
    # See mempalace_adapter.py's module docstring "NO PER-RECORD ID"
    # section: the real tool_search response never carries an id.
    assert query_result.records[0].memory_id == ""
    assert query_result.conflict_signal == ConflictSignal.NOT_APPLICABLE

    update_result = adapter.update(
        "room-1", store_result.memory_id, "My dog is actually named Max."
    )
    assert update_result.acknowledged is True
    query_result_2 = adapter.query("room-1", "what is my dog's name?")
    assert query_result_2.records[0].content == "My dog is actually named Max."

    delete_result = adapter.delete(store_result.memory_id)
    assert delete_result.success is True
    query_result_3 = adapter.query("room-1", "what is my dog's name?")
    assert len(query_result_3.records) == 0


def test_mempalace_store_reads_room_source_file_added_by_from_metadata() -> None:
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)

    adapter.store(
        "room-1",
        "content",
        metadata={"room": "custom-room", "source_file": "chat.log", "added_by": "custom-agent"},
    )

    assert tools.add_drawer_calls == [
        {
            "wing": "room-1",
            "room": "custom-room",
            "content": "content",
            "source_file": "chat.log",
            "added_by": "custom-agent",
        }
    ]


def test_mempalace_store_defaults_room_and_added_by_when_metadata_omitted() -> None:
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)

    adapter.store("room-1", "content")

    assert tools.add_drawer_calls == [
        {
            "wing": "room-1",
            "room": MemPalaceAdapter.DEFAULT_ROOM,
            "content": "content",
            "source_file": None,
            "added_by": "memtrust",
        }
    ]


def test_mempalace_query_scopes_search_by_session_id_as_wing() -> None:
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)

    adapter.query("room-1", "some query", top_k=3)

    assert tools.search_calls == [
        {"query": "some query", "limit": 3, "wing": "room-1", "room": None}
    ]


def test_mempalace_supported_modes_is_confirmed_empty() -> None:
    # See mempalace_adapter.py's module docstring "Mode variants" section
    # -- neither tool_add_drawer nor tool_search accepts a `mode` keyword
    # in the real, installed package; the old ("raw", "AAAK") guess is
    # now confirmed absent, not just unconfirmed.
    assert MemPalaceAdapter.supported_modes == ()


def test_mempalace_mode_param_accepted_and_ignored_as_a_noop() -> None:
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)

    # Must not raise, and must not change the underlying vendor call shape
    # -- the real tool_add_drawer/tool_search have no mode parameter to
    # forward this to.
    adapter.store("room-1", "content", mode="AAAK")
    adapter.query("room-1", "query", mode="AAAK")

    assert tools.add_drawer_calls[0] == {
        "wing": "room-1",
        "room": MemPalaceAdapter.DEFAULT_ROOM,
        "content": "content",
        "source_file": None,
        "added_by": "memtrust",
    }
    assert tools.search_calls[0] == {"query": "query", "limit": 5, "wing": "room-1", "room": None}


def test_mempalace_store_raises_backend_api_error_on_vendor_reported_failure() -> None:
    class RejectingTools(FakeMCPTools):
        def tool_add_drawer(
            self,
            wing: str,
            room: str,
            content: str,
            source_file: str | None = None,
            added_by: str = "mcp",
        ) -> dict[str, Any]:
            return {"success": False, "error": "room contains invalid characters"}

    adapter = MemPalaceAdapter(mcp_tools=RejectingTools())
    with pytest.raises(BackendAPIError, match="invalid characters"):
        adapter.store("room-1", "content")


def test_mempalace_wraps_vendor_exceptions_in_backend_api_error() -> None:
    class BrokenTools:
        def tool_add_drawer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("vendor exploded")

        def tool_search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("vendor exploded")

        def tool_update_drawer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("vendor exploded")

        def tool_delete_drawer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("vendor exploded")

        def tool_kg_add(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("vendor exploded")

        def tool_kg_invalidate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("vendor exploded")

        def tool_kg_query(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("vendor exploded")

    adapter = MemPalaceAdapter(mcp_tools=BrokenTools())
    with pytest.raises(BackendAPIError):
        adapter.store("room-1", "content")
    with pytest.raises(BackendAPIError):
        adapter.query("room-1", "query")
    with pytest.raises(BackendAPIError):
        adapter.update("room-1", "id", "content")
    with pytest.raises(BackendAPIError):
        adapter.delete("id")
    with pytest.raises(BackendAPIError):
        adapter.kg_add("s", "p", "o")
    with pytest.raises(BackendAPIError):
        adapter.kg_invalidate("s", "p", "o")
    with pytest.raises(BackendAPIError):
        adapter.kg_query("s")


def test_mempalace_get_mcp_tools_raises_clear_error_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real `mempalace` package is not a memtrust dependency (kept
    # optional -- see pyproject.toml's `mempalace-direct` extra), so in
    # this test environment it is genuinely not installed -- this
    # exercises the real ImportError path, not a simulated one. Every
    # vendor call in this adapter now shares the single `_get_mcp_tools()`
    # lazy-import point, so one such test covers store()/query()/
    # update()/delete()/metadata_overview()/kg_*() alike.
    monkeypatch.setenv("MEMPALACE_STORAGE_PATH", "/tmp/fake-palace")
    adapter = MemPalaceAdapter()
    with pytest.raises(BackendAPIError, match="not installed"):
        adapter.store("room-1", "content")


# ---------------------------------------------------------------------------
# Read-after-write verification (StoreResult.verified / verify_store)
# ---------------------------------------------------------------------------


def test_store_result_defaults_to_verified_none() -> None:
    adapter = MemPalaceAdapter(mcp_tools=FakeMCPTools())
    result = adapter.store("room-1", "My dog is named Baxter.")
    assert result.verified is None


def test_mempalace_verify_true_confirms_readable_write() -> None:
    """(a) verify=True with a fake that returns the just-stored content
    (via its content-substring fallback, since the real tool_search
    response carries no id -- see the module docstring) confirms
    verified=True."""
    adapter = MemPalaceAdapter(mcp_tools=FakeMCPTools())
    result = adapter.store("room-1", "My dog is named Baxter.", verify=True)
    assert result.verified is True


def test_mempalace_verify_true_detects_silently_dropped_write() -> None:
    """(b) verify=True with a fake tool_search that comes back empty sets
    verified=False rather than raising -- this is the "store() didn't
    raise, but the write was silently dropped/corrupted" failure mode
    (MemPalace issues #1929, #1977) this feature exists to catch."""

    class SilentlyDroppingTools(FakeMCPTools):
        def tool_add_drawer(
            self,
            wing: str,
            room: str,
            content: str,
            source_file: str | None = None,
            added_by: str = "mcp",
        ) -> dict[str, Any]:
            return {
                "success": True,
                "drawer_id": "drawer-ghost-1",
                "wing": wing,
                "room": room,
                "chunks": 1,
            }

        def tool_search(
            self,
            query: str,
            limit: int = 5,
            wing: str | None = None,
            room: str | None = None,
            source_file: str | None = None,
            max_distance: float = 1.5,
            min_similarity: float | None = None,
            context: str | None = None,
        ) -> dict[str, Any]:
            return {
                "query": query,
                "filters": {"wing": wing, "room": room, "source_file": source_file},
                "total_before_filter": 0,
                "results": [],
            }

    adapter = MemPalaceAdapter(mcp_tools=SilentlyDroppingTools())
    result = adapter.store("room-1", "My dog is named Baxter.", verify=True)
    assert result.verified is False
    # Crucially, this must not raise -- a failed verification is a
    # reported fact about the write, not an exception.


def test_mempalace_verify_true_detects_wrong_content_on_readback() -> None:
    """Same failure mode as above, but tool_search returns *something* --
    just not the content that was actually stored (corruption, not a
    total drop). Still verified=False, still no exception."""

    class CorruptingTools(FakeMCPTools):
        def tool_add_drawer(
            self,
            wing: str,
            room: str,
            content: str,
            source_file: str | None = None,
            added_by: str = "mcp",
        ) -> dict[str, Any]:
            return {
                "success": True,
                "drawer_id": "drawer-corrupt-1",
                "wing": wing,
                "room": room,
                "chunks": 1,
            }

        def tool_search(
            self,
            query: str,
            limit: int = 5,
            wing: str | None = None,
            room: str | None = None,
            source_file: str | None = None,
            max_distance: float = 1.5,
            min_similarity: float | None = None,
            context: str | None = None,
        ) -> dict[str, Any]:
            return {
                "query": query,
                "filters": {"wing": wing, "room": room, "source_file": source_file},
                "total_before_filter": 1,
                "results": [
                    {
                        "text": "\x00\x00\x00",
                        "wing": wing,
                        "room": room,
                        "source_file": "",
                        "source_path": "",
                        "created_at": "2026-01-01T00:00:00",
                        "similarity": 0.9,
                        "distance": 0.1,
                        "effective_distance": 0.1,
                        "closet_boost": 0.0,
                        "matched_via": "drawer",
                        "bm25_score": 1.0,
                    }
                ],
            }

    adapter = MemPalaceAdapter(mcp_tools=CorruptingTools())
    result = adapter.store("room-1", "My dog is named Baxter.", verify=True)
    assert result.verified is False


def test_mempalace_verify_false_by_default_does_not_call_search() -> None:
    """(c) verify=False (default) behavior: no query() call happens,
    verified stays None."""

    class SearchTrackingTools(FakeMCPTools):
        def __init__(self) -> None:
            super().__init__()
            self.search_call_count = 0

        def tool_search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.search_call_count += 1
            return super().tool_search(*args, **kwargs)

    tools = SearchTrackingTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)

    result = adapter.store("room-1", "My dog is named Baxter.")
    assert result.verified is None
    assert tools.search_call_count == 0

    # Explicit verify=False must behave identically to the omitted default.
    result_explicit = adapter.store("room-1", "My cat is named Whiskers.", verify=False)
    assert result_explicit.verified is None
    assert tools.search_call_count == 0


def test_verify_store_raises_backend_api_error_when_query_itself_fails() -> None:
    """A genuine vendor/network failure during the verification query()
    call must still propagate as BackendAPIError, not be swallowed into
    verified=False -- only an absent/wrong record on a successful query
    means "the write was silently dropped," not a query that itself
    errored."""

    class QueryFailsTools(FakeMCPTools):
        def tool_search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("vendor exploded during verification query")

    adapter = MemPalaceAdapter(mcp_tools=QueryFailsTools())
    with pytest.raises(BackendAPIError):
        adapter.store("room-1", "content", verify=True)


# ---------------------------------------------------------------------------
# MemPalaceAdapter.update() / delete() -- now genuinely implemented against
# the real, confirmed tool_update_drawer/tool_delete_drawer, closing the
# gap the old, fictional-API adapter left (delete() previously always
# raised "not implemented" -- see the module docstring).
# ---------------------------------------------------------------------------


def test_mempalace_update_reports_not_acknowledged_without_raising_when_drawer_missing() -> None:
    adapter = MemPalaceAdapter(mcp_tools=FakeMCPTools())
    result = adapter.update("room-1", "nonexistent-id", "new content")
    assert result.acknowledged is False
    assert result.memory_id == "nonexistent-id"


def test_mempalace_delete_now_implemented_against_real_tool_delete_drawer() -> None:
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)
    store_result = adapter.store("room-1", "content")

    delete_result = adapter.delete(store_result.memory_id)

    assert delete_result.success is True
    assert delete_result.memory_id == store_result.memory_id


def test_mempalace_delete_reports_unsuccessful_without_raising_when_drawer_missing() -> None:
    adapter = MemPalaceAdapter(mcp_tools=FakeMCPTools())
    result = adapter.delete("nonexistent-id")
    assert result.success is False
    assert result.memory_id == "nonexistent-id"


# ---------------------------------------------------------------------------
# MemPalaceAdapter.query() -- RankingSignal detection
#
# Re-pointed at `similarity` -- the real, confirmed field tool_search's own
# ranking is actually driven by -- rather than the fictional-API-era
# importance/emotional_weight/weight/authored_at keys, which belong to a
# different method (`Layer1.generate()`'s "wake-up" sort,
# mempalace/mempalace#1733) this adapter never calls. See
# mempalace_adapter.py's module docstring "RANKING SIGNAL" section for the
# full reasoning.
# ---------------------------------------------------------------------------


def test_mempalace_query_flags_missing_ordering_key_when_similarity_constant() -> None:
    class ConstantSimilarityTools(FakeMCPTools):
        def tool_search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raw = super().tool_search(*args, **kwargs)
            for item in raw["results"]:
                item["similarity"] = 0.5
            return raw

    tools = ConstantSimilarityTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)
    for content in ["Had coffee with Alex.", "Signed the lease.", "Grandmother's health scare."]:
        adapter.store("room-1", content)

    query_result = adapter.query("room-1", "wake me up with important memories", top_k=10)
    assert query_result.ranking_signal == RankingSignal.MISSING_ORDERING_KEY


def test_mempalace_query_reports_signal_driven_when_similarity_genuinely_varies() -> None:
    # Negative control: this must NOT be flagged -- FakeMCPTools.tool_search
    # already assigns a genuinely varying similarity per result (matching
    # the real tool_search's own effective_distance-driven ranking).
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)
    adapter.store("room-1", "Renewed car registration.")
    adapter.store("room-1", "Grandmother's health scare.")
    adapter.store("room-1", "Signed the lease.")

    query_result = adapter.query("room-1", "wake me up with important memories", top_k=10)
    assert query_result.ranking_signal == RankingSignal.SIGNAL_DRIVEN


def test_mempalace_query_ranking_signal_not_applicable_with_fewer_than_two_records() -> None:
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)
    adapter.store("room-1", "Had coffee with Alex.")

    query_result = adapter.query("room-1", "wake me up with important memories")
    assert query_result.ranking_signal == RankingSignal.NOT_APPLICABLE


def test_mempalace_query_falls_back_to_authored_at_when_similarity_absent() -> None:
    """`authored_at` is a real, confirmed, flat field on `tool_search`
    results (MemPalace/mempalace#1890, contributor JosefAschauer) --
    `_RANKING_METADATA_KEYS` checks it as a fallback signal when
    `similarity` itself is entirely absent from a response. This proves
    the fallback against `MemPalaceAdapter` itself, not a synthetic fake
    adapter standing in for it."""

    class NoSimilarityVariedAuthoredAtTools(FakeMCPTools):
        def tool_search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raw = super().tool_search(*args, **kwargs)
            for i, item in enumerate(raw["results"]):
                del item["similarity"]
                item["authored_at"] = f"2026-01-{i + 1:02d}T00:00:00"
            return raw

    tools = NoSimilarityVariedAuthoredAtTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)
    for content in ["Had coffee with Alex.", "Signed the lease.", "Grandmother's health scare."]:
        adapter.store("room-1", content)

    query_result = adapter.query("room-1", "wake me up with important memories", top_k=10)
    assert query_result.ranking_signal == RankingSignal.SIGNAL_DRIVEN


def test_mempalace_query_flags_missing_ordering_key_when_authored_at_constant() -> None:
    class NoSimilarityConstantAuthoredAtTools(FakeMCPTools):
        def tool_search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raw = super().tool_search(*args, **kwargs)
            for item in raw["results"]:
                del item["similarity"]
                item["authored_at"] = "2026-01-01T00:00:00"
            return raw

    tools = NoSimilarityConstantAuthoredAtTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)
    for content in ["Had coffee with Alex.", "Signed the lease.", "Grandmother's health scare."]:
        adapter.store("room-1", content)

    query_result = adapter.query("room-1", "wake me up with important memories", top_k=10)
    assert query_result.ranking_signal == RankingSignal.MISSING_ORDERING_KEY


# ---------------------------------------------------------------------------
# MemPalaceAdapter.query() -- no per-record id, metadata carries every
# other real response field (see the module docstring's "NO PER-RECORD ID"
# section).
# ---------------------------------------------------------------------------


def test_mempalace_query_records_always_have_empty_memory_id() -> None:
    """The real tool_search response never carries an id -- this adapter
    reports memory_id="" explicitly rather than guessing one (see the
    module docstring's "NO PER-RECORD ID" section)."""
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)
    adapter.store("room-1", "content")

    query_result = adapter.query("room-1", "content")

    assert query_result.records[0].memory_id == ""


def test_mempalace_query_metadata_surfaces_real_response_fields() -> None:
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)
    adapter.store("room-1", "Filed the annual tax return.", metadata={"source_file": "notes.md"})

    query_result = adapter.query("room-1", "tax return")

    metadata = query_result.records[0].metadata
    assert metadata["wing"] == "room-1"
    assert metadata["source_file"] == "notes.md"
    assert metadata["matched_via"] == "drawer"
    assert "similarity" in metadata
    assert "text" not in metadata  # text becomes .content, not a metadata key


# ---------------------------------------------------------------------------
# MemPalaceAdapter.query() -- conflict_signal is now honestly
# NOT_APPLICABLE for every drawer-backed query (see the module docstring's
# "CONFLICT SIGNAL" section) -- the real tool_search response has no
# invalidation marker at all; that concept only exists on the KG side (see
# the kg_* tests below).
# ---------------------------------------------------------------------------


def test_mempalace_query_conflict_signal_always_not_applicable() -> None:
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)
    store_result = adapter.store("room-1", "My dog is named Baxter.")
    adapter.update("room-1", store_result.memory_id, "My dog is actually named Max.")

    query_result = adapter.query("room-1", "what is my dog's name?")

    assert query_result.conflict_signal == ConflictSignal.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# MemPalaceAdapter.query() -- degraded-retrieval detection
#
# `warnings`/`available_in_scope` parsing is kept for forward-compat, but
# never observed populated against the real, installed 3.5.0 package (see
# the module docstring's "DEGRADED-RETRIEVAL WARNINGS" section) -- these
# tests exercise the parsing logic against a synthetic response shape, not
# a shape any real, confirmed code path in the installed package produces.
# ---------------------------------------------------------------------------


class DegradedRetrievalTools(FakeMCPTools):
    """Fake mempalace.mcp_server whose tool_search() returns a
    caller-supplied raw response directly -- lets tests exercise the
    warnings/available_in_scope parsing path query() still supports."""

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__()
        self._response = response

    def tool_search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._response


def test_mempalace_query_sets_degraded_retrieval_when_backend_warns() -> None:
    tools = DegradedRetrievalTools(
        {
            "results": [
                {"text": "kiyo xhci fix notes", "wing": "room-1", "room": "memtrust"},
            ],
            "warnings": ["hnsw drift detected"],
            "available_in_scope": 50,
        }
    )
    adapter = MemPalaceAdapter(mcp_tools=tools)

    query_result = adapter.query("room-1", "kiyo xhci")

    assert len(query_result.records) == 1
    assert query_result.degraded_retrieval == RetrievalWarning(
        warnings=["hnsw drift detected"], available_in_scope=50
    )


def test_mempalace_query_degraded_retrieval_unset_on_clean_response() -> None:
    """A clean response (no warnings) must leave degraded_retrieval unset
    (None) -- an empty `warnings` list is not itself a degradation
    signal."""
    tools = DegradedRetrievalTools(
        {
            "results": [
                {"text": "unrelated content", "wing": "room-1", "room": "memtrust"},
            ],
            "warnings": [],
            "available_in_scope": 1,
        }
    )
    adapter = MemPalaceAdapter(mcp_tools=tools)

    query_result = adapter.query("room-1", "query")

    assert query_result.degraded_retrieval is None


def test_mempalace_query_degraded_retrieval_unset_for_ordinary_response() -> None:
    """The ordinary, real response shape (FakeMCPTools' default
    tool_search, matching the confirmed real shape) never sets
    degraded_retrieval -- there is no `warnings` key on that shape."""
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)
    adapter.store("room-1", "My dog is named Baxter.")

    query_result = adapter.query("room-1", "what is my dog's name?")

    assert query_result.degraded_retrieval is None


def test_mempalace_query_available_in_scope_none_when_backend_omits_it() -> None:
    tools = DegradedRetrievalTools(
        {
            "results": [],
            "warnings": ["vector search unavailable: filter planner error"],
        }
    )
    adapter = MemPalaceAdapter(mcp_tools=tools)

    query_result = adapter.query("room-1", "query")

    assert query_result.degraded_retrieval == RetrievalWarning(
        warnings=["vector search unavailable: filter planner error"],
        available_in_scope=None,
    )


def test_mempalace_query_available_in_scope_ignores_non_int_value() -> None:
    """A malformed/mocked available_in_scope (not a real int) must not be
    trusted as a number -- treated the same as absent, never coerced."""
    tools = DegradedRetrievalTools(
        {
            "results": [],
            "warnings": ["vector search unavailable: boom"],
            "available_in_scope": "not-a-number",
        }
    )
    adapter = MemPalaceAdapter(mcp_tools=tools)

    query_result = adapter.query("room-1", "query")

    assert query_result.degraded_retrieval is not None
    assert query_result.degraded_retrieval.available_in_scope is None


def test_mempalace_query_raises_backend_api_error_when_dict_missing_results_key() -> None:
    """A dict-shaped response with no `results` key at all is the real,
    confirmed shape tool_search returns on a sanitizer rejection --
    {"error": ...}. That must fail loudly as BackendAPIError, not a
    confusing KeyError or a silent empty-records response."""
    tools = DegradedRetrievalTools({"error": "wing contains invalid characters"})
    adapter = MemPalaceAdapter(mcp_tools=tools)

    with pytest.raises(BackendAPIError, match="results"):
        adapter.query("room-1", "query")


# ---------------------------------------------------------------------------
# MemPalaceAdapter.kg_add() / kg_invalidate() / kg_query() -- the new,
# additive knowledge-graph capability (see the module docstring's
# "CONTRADICTION/STALENESS DETECTION MOVED TO THE KG API" section). Not
# part of the shared MemoryBackendAdapter contract.
# ---------------------------------------------------------------------------


def test_mempalace_kg_add_returns_triple_id_and_fact() -> None:
    adapter = MemPalaceAdapter(mcp_tools=FakeMCPTools())

    result = adapter.kg_add("user", "favorite_color", "blue")

    assert result.success is True
    assert result.triple_id is not None
    assert result.fact == "user → favorite_color → blue"


def test_mempalace_kg_add_idempotent_on_repeat() -> None:
    adapter = MemPalaceAdapter(mcp_tools=FakeMCPTools())

    first = adapter.kg_add("user", "favorite_color", "blue")
    second = adapter.kg_add("user", "favorite_color", "blue")

    assert first.triple_id == second.triple_id


def test_mempalace_kg_invalidate_marks_current_false_and_stamps_ended() -> None:
    adapter = MemPalaceAdapter(mcp_tools=FakeMCPTools())
    adapter.kg_add("user", "favorite_color", "blue")

    before = adapter.kg_query("user")
    assert before.facts[0].current is True

    invalidate_result = adapter.kg_invalidate("user", "favorite_color", "blue", ended="2026-06-01")
    assert invalidate_result.success is True
    assert invalidate_result.ended == "2026-06-01"

    after = adapter.kg_query("user")
    assert after.facts[0].current is False
    assert after.facts[0].valid_to == "2026-06-01"


def test_mempalace_kg_invalidate_succeeds_even_if_fact_never_added() -> None:
    """Documents a real, live-verified quirk: tool_kg_invalidate does not
    check prior existence -- it succeeds unconditionally. This adapter
    does not second-guess that (see kg_invalidate()'s docstring)."""
    adapter = MemPalaceAdapter(mcp_tools=FakeMCPTools())

    result = adapter.kg_invalidate("nobody", "nonexistent_pred", "nothing")

    assert result.success is True


def test_mempalace_kg_query_reports_outgoing_facts_for_entity() -> None:
    adapter = MemPalaceAdapter(mcp_tools=FakeMCPTools())
    adapter.kg_add("user", "favorite_color", "blue")
    adapter.kg_add("user", "favorite_language", "python")

    result = adapter.kg_query("user")

    assert result.entity == "user"
    assert result.count == 2
    assert {f.object for f in result.facts} == {"blue", "python"}
    assert all(f.direction == "outgoing" for f in result.facts)


def test_mempalace_kg_query_raises_backend_api_error_when_facts_key_missing() -> None:
    class NoFactsKeyTools(FakeMCPTools):
        def tool_kg_query(
            self, entity: str, as_of: str | None = None, direction: str = "both"
        ) -> dict[str, Any]:
            return {"error": "entity name contains invalid characters"}

    adapter = MemPalaceAdapter(mcp_tools=NoFactsKeyTools())
    with pytest.raises(BackendAPIError, match="facts"):
        adapter.kg_query("bad entity!!")


# ---------------------------------------------------------------------------
# MemPalaceAdapter.metadata_overview() / list_metadata_categories() /
# list_metadata_subcategories() -- the confirmed-real
# mempalace.mcp_server.tool_status/tool_list_wings/tool_list_rooms wrapper
# (see mempalace_adapter.py's "MCP metadata-tool coverage" module docstring
# section and MemPalace/mempalace#1871, contributor alionar). Unchanged by
# this rewrite except for sharing `_get_mcp_tools()` with every other
# method.
# ---------------------------------------------------------------------------


class _RaisingMCPTools:
    """Models the real mempalace.mcp_server functions raising outright
    (as opposed to returning an `{"error": ...}` dict) -- e.g. an
    unexpected exception inside the vendor's own code."""

    def tool_status(self) -> dict[str, Any]:
        raise RuntimeError("sqlite database is locked")

    def tool_list_wings(self) -> dict[str, Any]:
        raise RuntimeError("sqlite database is locked")

    def tool_list_rooms(self, wing: str | None = None) -> dict[str, Any]:
        raise RuntimeError("sqlite database is locked")


def test_mempalace_supports_metadata_overview_is_true() -> None:
    assert MemPalaceAdapter.supports_metadata_overview is True


def test_mempalace_metadata_overview_reports_wing_room_counts() -> None:
    tools = FakeMCPTools()
    tools.status_response = {
        "total_drawers": 500,
        "wings": {"wing_0": 250, "wing_1": 250},
        "rooms": {"room_0": 100, "room_1": 400},
        "backend": "chroma",
    }
    adapter = MemPalaceAdapter(mcp_tools=tools)

    overview = adapter.metadata_overview()

    assert overview.total_records == 500
    assert overview.categories == {"wing_0": 250, "wing_1": 250}
    assert overview.subcategories == {"room_0": 100, "room_1": 400}
    assert overview.partial is False
    assert overview.error is None
    assert overview.latency_ms >= 0


def test_mempalace_list_metadata_categories_wraps_tool_list_wings() -> None:
    tools = FakeMCPTools()
    tools.list_wings_response = {"wings": {"wing_a": 3, "wing_b": 7}}
    adapter = MemPalaceAdapter(mcp_tools=tools)

    result = adapter.list_metadata_categories()

    assert result.counts == {"wing_a": 3, "wing_b": 7}
    assert result.scope is None
    assert result.error is None


def test_mempalace_list_metadata_subcategories_scopes_by_category() -> None:
    tools = FakeMCPTools()
    tools.list_rooms_response = {"wing": "wing_a", "rooms": {"room_x": 2, "room_y": 1}}
    adapter = MemPalaceAdapter(mcp_tools=tools)

    result = adapter.list_metadata_subcategories(category="wing_a")

    assert tools.list_rooms_calls == ["wing_a"]
    assert result.counts == {"room_x": 2, "room_y": 1}
    assert result.scope == "wing_a"


def test_mempalace_list_metadata_subcategories_unscoped_passes_none_wing() -> None:
    tools = FakeMCPTools()
    adapter = MemPalaceAdapter(mcp_tools=tools)

    adapter.list_metadata_subcategories()

    assert tools.list_rooms_calls == [None]


def test_mempalace_metadata_overview_flags_partial_on_backend_error() -> None:
    """Mirrors the rest of this adapter's "never trust the call didn't
    raise" rule: a `{"error": ...}` response (the real package's own
    partial-failure shape, confirmed via tool_status()'s sqlite-integrity
    and metadata-fetch-exception fallback paths) must surface as
    partial=True / a non-None error, not silently look like a clean 0."""
    tools = FakeMCPTools()
    tools.status_response = {
        "total_drawers": 100,
        "wings": {"wing_0": 100},
        "rooms": {},
        "error": "tool_status metadata fetch failed",
        "partial": True,
    }
    adapter = MemPalaceAdapter(mcp_tools=tools)

    overview = adapter.metadata_overview()

    assert overview.partial is True
    assert overview.error == "tool_status metadata fetch failed"


def test_mempalace_metadata_calls_wrap_vendor_exceptions_in_backend_api_error() -> None:
    adapter = MemPalaceAdapter(mcp_tools=_RaisingMCPTools())

    with pytest.raises(BackendAPIError, match="sqlite database is locked"):
        adapter.metadata_overview()
    with pytest.raises(BackendAPIError, match="sqlite database is locked"):
        adapter.list_metadata_categories()
    with pytest.raises(BackendAPIError, match="sqlite database is locked"):
        adapter.list_metadata_subcategories()


def test_mempalace_sync_mcp_palace_path_bridges_storage_path_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mempalace.mcp_server reads MEMPALACE_PALACE_PATH (confirmed real),
    a different env var name than this adapter's own
    MEMPALACE_STORAGE_PATH -- metadata_overview() must bridge the two
    before every call rather than silently assuming they line up."""
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.setenv("MEMPALACE_STORAGE_PATH", "/tmp/a-real-palace-path")
    adapter = MemPalaceAdapter(mcp_tools=FakeMCPTools())

    adapter.metadata_overview()

    assert os.environ.get("MEMPALACE_PALACE_PATH") == "/tmp/a-real-palace-path"


def test_mempalace_sync_calls_reconnect_only_when_path_actually_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """See mempalace_adapter.py's module docstring "CROSS-PATH RELIABILITY
    GAP" section: a live-reproduced real vendor bug means switching
    MEMPALACE_PALACE_PATH to a different value within one process can
    leave mempalace's own client/vector-disabled cache stale unless
    tool_reconnect() is called once right after the switch. This adapter
    now does that automatically, but only when the path actually changes
    -- never on every call, which would cost a real vendor call per
    store()/query()/update()/delete() for the common single-path-per-run
    case."""
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)

    monkeypatch.setenv("MEMPALACE_STORAGE_PATH", "/tmp/palace-x")
    tools_x = FakeMCPTools()
    adapter_x = MemPalaceAdapter(mcp_tools=tools_x)
    adapter_x.store("room-1", "content")
    adapter_x.store("room-1", "more content")
    # First-ever sync in this process: MEMPALACE_PALACE_PATH had no prior
    # value, so this is not treated as a "change" -- no reconnect call.
    assert tools_x.reconnect_call_count == 0

    monkeypatch.setenv("MEMPALACE_STORAGE_PATH", "/tmp/palace-y")
    tools_y = FakeMCPTools()
    adapter_y = MemPalaceAdapter(mcp_tools=tools_y)
    adapter_y.store("room-1", "content")
    # A real path change happened (palace-x -> palace-y) -- exactly one
    # reconnect call, on the first sync that observes it.
    assert tools_y.reconnect_call_count == 1
    adapter_y.store("room-1", "more content")
    adapter_y.query("room-1", "content")
    # Subsequent calls against the SAME (now-current) path do not
    # re-trigger reconnect.
    assert tools_y.reconnect_call_count == 1


# ---------------------------------------------------------------------------
# Real, live integration tests against the actual installed `mempalace`
# package -- requires the optional `mempalace-direct` dependency group
# (`pip install -e ".[dev,mempalace-direct]"`). `real_mempalace_adapter`
# below calls `pytest.importorskip("mempalace")` itself so only these
# tests skip (not the whole module) when that group isn't installed --
# matching the convention test_mempalace_metadata_scale.py's
# test_real_chroma_seeder_reports_correct_counts_at_small_scale already
# established. These are what actually prove the shapes documented in
# mempalace_adapter.py's module docstring, not a restatement of them.
#
# All drawer tests below share ONE real palace directory and ONE real
# MemPalaceAdapter instance (module-scoped fixture), using a distinct
# `session_id`/wing per test for isolation instead of a fresh palace
# directory per test. This is deliberate, not a shortcut: see the module
# docstring's "CROSS-PATH RELIABILITY GAP" section -- switching
# MEMPALACE_PALACE_PATH between distinct real palace directories within
# one process is a live-reproduced, real vendor reliability gap, and
# reusing one path throughout matches memtrust's own normal single-run
# usage pattern (one adapter, one fixed storage path, for a whole eval
# run) exactly. The KG tests use a per-run-unique subject entity instead
# -- see the module docstring's "KNOWLEDGE-GRAPH STORAGE IGNORES
# MEMPALACE_PALACE_PATH" section: the real KG store is one fixed file in
# the calling user's home directory, shared globally, with no per-path
# isolation to rely on at all.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_mempalace_adapter() -> Any:
    pytest.importorskip(
        "mempalace",
        reason=(
            "requires the optional `mempalace-direct` extra: "
            "pip install -e '.[dev,mempalace-direct]'. See "
            "mempalace_adapter.py's module docstring."
        ),
    )
    import tempfile

    from memtrust.adapters.mempalace_adapter import MemPalaceAdapter as _RealMemPalaceAdapter

    storage_path = os.path.join(tempfile.mkdtemp(prefix="memtrust-mempalace-real-"), "palace")
    os.makedirs(storage_path, exist_ok=True)
    previous_storage = os.environ.get("MEMPALACE_STORAGE_PATH")
    os.environ["MEMPALACE_STORAGE_PATH"] = storage_path
    # Deliberately do NOT touch MEMPALACE_PALACE_PATH here -- leave it
    # exactly as any earlier test/process activity left it, so the
    # adapter's own _sync_mcp_palace_path() gets an undisturbed view of
    # whatever the real mempalace.mcp_server module last saw and can
    # correctly detect a real path change (see the module docstring's
    # "CROSS-PATH RELIABILITY GAP" section -- clearing this env var here
    # would defeat that change-detection on the very first real call).
    adapter = _RealMemPalaceAdapter()
    yield adapter
    if previous_storage is None:
        os.environ.pop("MEMPALACE_STORAGE_PATH", None)
    else:
        os.environ["MEMPALACE_STORAGE_PATH"] = previous_storage


def test_real_mempalace_store_query_update_delete_round_trip(real_mempalace_adapter: Any) -> None:
    adapter = real_mempalace_adapter

    store_result = adapter.store("round-trip-session", "My dog is named Baxter.")
    assert store_result.memory_id  # real drawer_id, non-empty

    query_result = adapter.query("round-trip-session", "what is my dog's name?")
    assert any("Baxter" in r.content for r in query_result.records)
    # Confirmed real gap -- see the module docstring's "NO PER-RECORD ID"
    # section.
    assert all(r.memory_id == "" for r in query_result.records)

    update_result = adapter.update(
        "round-trip-session", store_result.memory_id, "My dog is actually named Max."
    )
    assert update_result.acknowledged is True

    query_after_update = adapter.query("round-trip-session", "what is my dog's name?")
    assert any("Max" in r.content for r in query_after_update.records)
    assert not any("Baxter" in r.content for r in query_after_update.records)

    delete_result = adapter.delete(store_result.memory_id)
    assert delete_result.success is True

    query_after_delete = adapter.query("round-trip-session", "what is my dog's name?")
    assert not any("Max" in r.content for r in query_after_delete.records)


def test_real_mempalace_query_isolates_by_session_id_as_wing(real_mempalace_adapter: Any) -> None:
    adapter = real_mempalace_adapter

    adapter.store("isolation-session-a", "session A's secret is the number 42.")
    adapter.store("isolation-session-b", "session B's secret is the color teal.")

    result_a = adapter.query("isolation-session-a", "secret")
    result_b = adapter.query("isolation-session-b", "secret")

    assert any("42" in r.content for r in result_a.records)
    assert not any("42" in r.content for r in result_b.records)
    assert any("teal" in r.content for r in result_b.records)
    assert not any("teal" in r.content for r in result_a.records)


def test_real_mempalace_store_idempotent_on_identical_content(real_mempalace_adapter: Any) -> None:
    adapter = real_mempalace_adapter

    first = adapter.store("idempotency-session", "identical content for idempotency check")
    second = adapter.store("idempotency-session", "identical content for idempotency check")

    assert first.memory_id == second.memory_id


def test_real_mempalace_update_delete_report_unsuccessful_for_nonexistent_id(
    real_mempalace_adapter: Any,
) -> None:
    adapter = real_mempalace_adapter

    update_result = adapter.update("notfound-session", "drawer-does-not-exist", "new content")
    assert update_result.acknowledged is False

    delete_result = adapter.delete("drawer-does-not-exist")
    assert delete_result.success is False


def test_real_mempalace_verify_store_confirms_readable_write(real_mempalace_adapter: Any) -> None:
    adapter = real_mempalace_adapter

    result = adapter.store("verify-session", "verifiable content marker QRSTUV", verify=True)

    assert result.verified is True


def test_real_mempalace_chunked_content_update_delete_round_trip(
    real_mempalace_adapter: Any,
) -> None:
    """Confirms the module docstring's "CHUNKED CONTENT" section live:
    content over mempalace's real default chunk_size (800 chars) gets
    split into multiple physical chunk drawers on store(), but
    update()/delete() against the logical drawer_id store() returns still
    work correctly -- despite tool_add_drawer's own docstring in the
    installed package claiming update/delete "report 'not found' on the
    chunked path," which does not hold up against this version's actual
    _logical_drawer_record()/_logical_chunk_group() resolution logic."""
    adapter = real_mempalace_adapter

    long_content = "This is a long memory sentence about the project. " * 40  # well over 800 chars
    assert len(long_content) > 800
    store_result = adapter.store("chunked-session", long_content)

    update_result = adapter.update("chunked-session", store_result.memory_id, "replacement content")
    assert update_result.acknowledged is True

    query_after_update = adapter.query("chunked-session", "replacement content")
    assert any("replacement content" in r.content for r in query_after_update.records)

    another_long_content = "A separate long memory sentence for delete. " * 40
    delete_store_result = adapter.store("chunked-session", another_long_content)
    delete_result = adapter.delete(delete_store_result.memory_id)
    assert delete_result.success is True

    query_after_delete = adapter.query("chunked-session", "separate long memory sentence")
    assert not any("separate long memory sentence" in r.content for r in query_after_delete.records)


def test_real_mempalace_kg_add_invalidate_query_round_trip(real_mempalace_adapter: Any) -> None:
    # See the module docstring's "KNOWLEDGE-GRAPH STORAGE IGNORES
    # MEMPALACE_PALACE_PATH" section: the real KG store is one fixed,
    # environment-global file with no per-test/per-path isolation at all
    # -- a randomized subject makes this test's assertions valid
    # regardless of what any other test run (in this process or a prior
    # one) already wrote there.
    adapter = real_mempalace_adapter
    subject = f"memtrust-kg-test-user-{uuid.uuid4().hex}"

    add_result = adapter.kg_add(subject, "favorite_color", "blue", source_closet="kg-session")
    assert add_result.success is True
    assert add_result.triple_id is not None

    before = adapter.kg_query(subject)
    assert before.count == 1
    assert before.facts[0].current is True
    assert before.facts[0].valid_to is None

    invalidate_result = adapter.kg_invalidate(subject, "favorite_color", "blue")
    assert invalidate_result.success is True
    assert invalidate_result.ended is not None

    after = adapter.kg_query(subject)
    assert after.facts[0].current is False
    assert after.facts[0].valid_to == invalidate_result.ended


# ---------------------------------------------------------------------------
# MemoryBackendAdapter.delete_many() -- base-class aggregation contract
#
# This is the primitive an eval needs to reproduce the real, merged
# mem0ai/mem0#5936 / #5970 bug class: a multi-entity delete whose
# client-side aggregation silently truncated to only the last response
# instead of all N. These tests exercise the base class's own
# delete_many() in isolation (via a minimal fake adapter, not tied to any
# one vendor's HTTP shape) to prove it does not repeat that bug: every
# id gets exactly one result, in the original order, whether it
# succeeded or failed.
# ---------------------------------------------------------------------------


class _FakeDeleteAdapter(MemoryBackendAdapter):
    """Minimal concrete adapter used only to exercise the base class's
    default delete_many() loop -- store/query/update are irrelevant here
    and deliberately left unimplemented."""

    name = "fake"
    env_var = "FAKE_KEY"

    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self._fail_ids = fail_ids or set()
        self.delete_calls: list[str] = []

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        raise NotImplementedError

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        raise NotImplementedError

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        raise NotImplementedError

    def delete(self, memory_id: str) -> DeleteResult:
        self.delete_calls.append(memory_id)
        if memory_id in self._fail_ids:
            raise BackendAPIError(self.name, f"vendor rejected {memory_id}")
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=1.0)


def test_delete_many_calls_delete_once_per_id_in_order() -> None:
    adapter = _FakeDeleteAdapter()
    ids = [f"mem-{i}" for i in range(5)]

    results = adapter.delete_many(ids)

    assert adapter.delete_calls == ids
    assert [r.memory_id for r in results] == ids
    assert all(r.success for r in results)
    assert len(results) == len(ids)


def test_delete_many_aggregates_mixed_success_and_failure_without_truncation() -> None:
    # The N-entity-delete-truncation reproduction: 5 ids, 2 of which fail
    # server-side. A buggy client (the shape of mem0ai/mem0#5936/#5970)
    # would keep only the last response; delete_many() must return all 5,
    # each mapped to the correct outcome, in the original order.
    adapter = _FakeDeleteAdapter(fail_ids={"mem-1", "mem-3"})
    ids = ["mem-0", "mem-1", "mem-2", "mem-3", "mem-4"]

    results = adapter.delete_many(ids)

    assert len(results) == 5
    assert [r.memory_id for r in results] == ids
    assert [r.success for r in results] == [True, False, True, False, True]
    # A failed delete() call must not be silently dropped -- it is
    # recorded as a failed DeleteResult at its original position, not
    # omitted from the list.
    assert sum(1 for r in results if not r.success) == 2


def test_delete_many_empty_list_returns_empty_list() -> None:
    adapter = _FakeDeleteAdapter()
    assert adapter.delete_many([]) == []


# ---------------------------------------------------------------------------
# ZepGraphitiSelfHostedAdapter -- self-hosted graphiti-core
#
# The real `graphiti-core` package is not installed in this environment
# (confirmed: `ModuleNotFoundError: No module named 'graphiti_core'`), and no
# Neo4j/FalkorDB instance is reachable here either. Every test below exercises
# this adapter's own logic against a fake client injected through the
# `graphiti_client=` constructor kwarg -- the same convention
# MemPalaceAdapter's `palace=` param uses -- conforming to the
# `_GraphitiProtocol` shape defined in zep_graphiti_selfhosted_adapter.py.
# These tests prove the adapter's internal logic is correct given a response
# shape; they do not prove that shape matches a live graphiti-core instance.
# See that module's docstring and docs/methodology.md for the full caveat.
# ---------------------------------------------------------------------------


class _FakeEntityEdge:
    """Stands in for a real `graphiti_core.edges.EntityEdge` -- exposes
    `.model_dump()` the same way the real Pydantic model does, so
    `zep_graphiti_selfhosted_adapter._to_plain_dict()` handles it exactly
    as it would the real class."""

    def __init__(
        self,
        uuid: str,
        fact: str,
        source_node_uuid: str | None = "node-source",
        target_node_uuid: str | None = "node-target",
        invalid_at: str | None = None,
        valid_at: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.uuid = uuid
        self.fact = fact
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid
        self.invalid_at = invalid_at
        self.valid_at = valid_at
        self.attributes = attributes or {}

    def model_dump(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "invalid_at": self.invalid_at,
            "valid_at": self.valid_at,
            "attributes": self.attributes,
        }


class _FakeEpisodicNode:
    def __init__(self, uuid: str) -> None:
        self.uuid = uuid


class _FakeAddEpisodeResults:
    """Stands in for a real `graphiti_core.graphiti.AddEpisodeResults`."""

    def __init__(self, episode_uuid: str) -> None:
        self.episode = _FakeEpisodicNode(episode_uuid)

    def model_dump(self) -> dict[str, Any]:
        return {"episode": {"uuid": self.episode.uuid}}


class FakeGraphitiClient:
    """Fake conforming to `_GraphitiProtocol` -- records every call it
    receives so tests can assert on exactly what this adapter sent."""

    def __init__(self) -> None:
        self.add_episode_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.remove_episode_calls: list[str] = []
        self.closed = False
        self._next_search_result: list[_FakeEntityEdge] = []
        self._episode_counter = 0

    def set_search_result(self, edges: list[_FakeEntityEdge]) -> None:
        self._next_search_result = edges

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: Any,
        group_id: str | None = None,
        update_communities: bool = False,
    ) -> _FakeAddEpisodeResults:
        self._episode_counter += 1
        self.add_episode_calls.append(
            {
                "name": name,
                "episode_body": episode_body,
                "source_description": source_description,
                "reference_time": reference_time,
                "group_id": group_id,
                "update_communities": update_communities,
            }
        )
        return _FakeAddEpisodeResults(f"episode-{self._episode_counter}")

    async def search(
        self, query: str, group_ids: list[str] | None = None, num_results: int = 10
    ) -> list[_FakeEntityEdge]:
        self.search_calls.append(
            {"query": query, "group_ids": group_ids, "num_results": num_results}
        )
        return self._next_search_result

    async def remove_episode(self, episode_uuid: str) -> None:
        self.remove_episode_calls.append(episode_uuid)

    async def close(self) -> None:
        self.closed = True


class _BrokenGraphitiClient:
    """Every call raises, same convention as MemPalaceAdapter's
    `BrokenPalace` test double above -- exercises the adapter's
    vendor-exception-wrapping path."""

    async def add_episode(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("vendor exploded")

    async def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError("vendor exploded")

    async def remove_episode(self, episode_uuid: str) -> None:
        raise RuntimeError("vendor exploded")

    async def close(self) -> None:
        pass


class _UnpackErrorGraphitiClient:
    """`add_episode()` raises the exact `ValueError` shape/message
    getzep/graphiti#836's filed traceback reports -- a tuple/list unpacking
    count mismatch inside `add_episode(update_communities=True)`'s
    community-update branch. Only `add_episode` needs to raise this; the
    other Protocol methods are unused by the tests that inject this
    client."""

    async def add_episode(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("too many values to unpack (expected 2)")

    async def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise NotImplementedError

    async def remove_episode(self, episode_uuid: str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class _TypeComparisonErrorGraphitiClient:
    """`add_episode()` raises the exact `TypeError` shape/message
    getzep/graphiti#920's filed traceback reports -- a tz-naive vs.
    tz-aware datetime comparison inside `resolve_edge_contradictions()`,
    reached from `add_episode()`."""

    async def add_episode(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("can't compare offset-naive and offset-aware datetimes")

    async def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise NotImplementedError

    async def remove_episode(self, episode_uuid: str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class _EmptyQuerySanitizationErrorGraphitiClient:
    """`search()` raises the exact RediSearch `Syntax error` shape
    getzep/graphiti#1222's own filed reproduction reports (`gh issue view
    1222 --repo getzep/graphiti`): an empty (or all-stopword) query string
    makes `build_fulltext_query()` in `falkordb_driver.py` append empty
    parentheses to the group filter -- `(@group_id:"my_graph") ()` -- which
    FalkorDB's RediSearch engine rejects with this exact message. The real
    exception FalkorDB raises here is a `redis`-client `ResponseError`;
    this test double raises a plain `Exception` carrying the real message
    text instead of importing `redis` (an optional dependency scoped to
    the `mem0-direct` extras group only, per pyproject.toml -- this file's
    own tests must keep working with no vendor package installed at all),
    matching `_classify_crash()`'s message-only, non-`isinstance` match
    for this shape."""

    async def add_episode(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise Exception("RediSearch: Syntax error at offset 22 near my_graph")

    async def remove_episode(self, episode_uuid: str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class _PipeSlashSanitizationErrorGraphitiClient:
    """`search()` raises the exact RediSearch `Syntax error` shape
    getzep/graphiti#1183's own filed reproduction reports (`gh pr view
    1183 --repo getzep/graphiti`): episode text containing an unescaped
    pipe (e.g. `"install.sh | bash"`) survives `sanitize()` (before that
    PR's fix), tokenizes into a stray `|`, and gets rejoined as an empty
    token between RediSearch OR-pipe delimiters -- `"sh | | | bash"` --
    which FalkorDB's RediSearch engine rejects with this exact message.
    Same "plain Exception, not a real redis import" reasoning as
    `_EmptyQuerySanitizationErrorGraphitiClient` above."""

    async def add_episode(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise Exception("RediSearch: Syntax error at offset 178 near sh")

    async def remove_episode(self, episode_uuid: str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class _NulByteQueryParameterErrorGraphitiClient:
    """`add_episode()` raises the exact `redis.exceptions.ResponseError`
    shape/message getzep/graphiti#1525's own filed reproduction reports
    (`gh issue view 1525 --repo getzep/graphiti`): a NUL byte embedded in
    episode content survives FalkorDB's client-side query-parameter
    serialization and makes FalkorDB's parser reject the entire bulk
    episode-save query. Same "plain Exception, not a real redis import"
    reasoning as the RediSearch-syntax-error fakes above -- the real
    exception is a `redis`-client `ResponseError`, this test double
    carries the real message text instead."""

    async def add_episode(self, *args: Any, **kwargs: Any) -> Any:
        raise Exception("Failed to parse query parameter 'episodes' value")

    async def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise NotImplementedError

    async def remove_episode(self, episode_uuid: str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class _EmbeddingBatchCountMismatchGraphitiClient:
    """`add_episode()` raises the exact `ValueError` shape/message
    getzep/graphiti#1467's own filed reproduction reports (`gh issue view
    1467 --repo getzep/graphiti`): `GeminiEmbedder.create_batch()`
    silently returning fewer vectors than inputs for the
    gemini-embedding-2* model family eventually trips a
    `zip(..., strict=True)` count-mismatch several frames away in
    graphiti-core's own dedup pipeline, reached from `add_episode()`."""

    async def add_episode(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("zip() argument 2 is shorter than argument 1")

    async def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise NotImplementedError

    async def remove_episode(self, episode_uuid: str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass


def test_graphiti_selfhosted_configured_via_falkordb_url_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAPHITI_NEO4J_URI", raising=False)
    monkeypatch.setenv("GRAPHITI_FALKORDB_URL", "redis://localhost:6379")
    # Must not raise BackendNotConfiguredError -- GRAPHITI_FALKORDB_URL
    # alone satisfies configuration, the same "one env var, or SKIPPED"
    # contract GRAPHITI_NEO4J_URI satisfies on its own.
    ZepGraphitiSelfHostedAdapter()


# ---------------------------------------------------------------------------
# GeminiEmbedder support (getzep/graphiti#1467, elimydlarz) -- embedder
# selection is a genuinely new capability this adapter had no code path for
# at all before this build. `_build_embedder()` is unit-tested directly
# (it needs no live graphiti_core/Neo4j/FalkorDB connection); the actual
# `Graphiti(embedder=...)` construction call it feeds is confirmed by
# reading the real graphiti_core source only, same convention this file's
# module docstring already applies to neo4j_user/falkordb host/port
# threading -- graphiti-core is not installed in this test environment
# (see FakeGraphitiClient's own section header above), so _get_client()'s
# real Graphiti(...) construction is never exercised end-to-end here.
# ---------------------------------------------------------------------------


def test_graphiti_selfhosted_gemini_provider_without_api_key_raises_not_configured() -> None:
    with pytest.raises(BackendNotConfiguredError):
        ZepGraphitiSelfHostedAdapter(neo4j_uri="bolt://localhost:7687", embedder_provider="gemini")


def test_graphiti_selfhosted_gemini_provider_with_injected_client_does_not_require_api_key() -> (
    None
):
    # Test-injection path (graphiti_client=) bypasses real construction
    # entirely, same convention every other constructor guard in this
    # adapter follows -- a caller supplying a fake/real client directly
    # should never be blocked by a credential check that only matters for
    # real construction.
    ZepGraphitiSelfHostedAdapter(graphiti_client=FakeGraphitiClient(), embedder_provider="gemini")


def test_graphiti_selfhosted_build_embedder_returns_none_by_default() -> None:
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=FakeGraphitiClient())
    assert adapter._build_embedder() is None


def test_graphiti_selfhosted_build_embedder_unsupported_provider_raises() -> None:
    adapter = ZepGraphitiSelfHostedAdapter(
        graphiti_client=FakeGraphitiClient(), embedder_provider="cohere"
    )
    with pytest.raises(BackendAPIError, match="unsupported embedder_provider"):
        adapter._build_embedder()


def test_graphiti_selfhosted_build_embedder_gemini_raises_clear_error_when_package_missing() -> (
    None
):
    # graphiti-core is not installed in this test environment (confirmed
    # elsewhere in this file) -- this exercises the real ImportError path
    # for the Gemini embedder wiring, not a simulated one, same convention
    # test_graphiti_selfhosted_get_client_raises_clear_error_when_package_missing
    # already establishes for _get_client() itself.
    adapter = ZepGraphitiSelfHostedAdapter(
        graphiti_client=FakeGraphitiClient(), embedder_provider="gemini", gemini_api_key="key-123"
    )
    with pytest.raises(BackendAPIError):
        adapter._build_embedder()


def test_graphiti_selfhosted_store_query_update_with_fake_client() -> None:
    client = FakeGraphitiClient()
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)

    store_result = adapter.store("session-1", "Priya is the lead on payments.")
    assert store_result.memory_id == "episode-1"
    call = client.add_episode_calls[0]
    assert call["group_id"] == "session-1"
    assert call["episode_body"] == "Priya is the lead on payments."
    assert call["update_communities"] is False

    client.set_search_result(
        [
            _FakeEntityEdge(
                uuid="edge-1",
                fact="Sam is the lead on payments.",
                attributes={"confidence": 0.9, "source": "correction"},
            )
        ]
    )
    query_result = adapter.query("session-1", "Who leads payments?")
    assert len(query_result.records) == 1
    record = query_result.records[0]
    assert record.memory_id == "edge-1"
    assert record.content == "Sam is the lead on payments."
    assert query_result.conflict_signal == ConflictSignal.SERVED_STALE

    adapter.update("session-1", store_result.memory_id, "Sam is now the lead.")
    assert len(client.add_episode_calls) == 2
    assert client.add_episode_calls[1]["episode_body"] == "Sam is now the lead."


def test_graphiti_selfhosted_query_populates_attributes_from_mock_response() -> None:
    """query() populates MemoryRecord.attributes from the backend's
    structured per-edge attributes dict -- the field this build added to
    base.py specifically so graphiti_core's EntityEdge.attributes survives
    the adapter boundary instead of being dropped or flattened into
    `raw` only."""
    client = FakeGraphitiClient()
    client.set_search_result(
        [
            _FakeEntityEdge(
                uuid="edge-9",
                fact="The migration ticket is priority P0.",
                attributes={"ticket_id": "OPS-4471", "team": "platform-infra"},
            )
        ]
    )
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    result = adapter.query("session-x", "priority?")
    assert result.records[0].attributes == {"ticket_id": "OPS-4471", "team": "platform-infra"}


def test_graphiti_selfhosted_query_attributes_default_empty_when_absent() -> None:
    client = FakeGraphitiClient()
    client.set_search_result([_FakeEntityEdge(uuid="edge-2", fact="a fact with no attributes")])
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    result = adapter.query("session-1", "q")
    assert result.records[0].attributes == {}


def test_graphiti_selfhosted_query_flags_invalidated_edge() -> None:
    client = FakeGraphitiClient()
    client.set_search_result(
        [_FakeEntityEdge(uuid="e1", fact="old fact", invalid_at="2026-06-01T00:00:00Z")]
    )
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    result = adapter.query("session-1", "q")
    assert result.conflict_signal == ConflictSignal.FLAGGED
    assert result.records[0].metadata == {"invalid_at": "2026-06-01T00:00:00Z"}


def test_graphiti_selfhosted_query_served_stale_when_no_invalidation() -> None:
    client = FakeGraphitiClient()
    client.set_search_result([_FakeEntityEdge(uuid="e1", fact="a fact")])
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    result = adapter.query("session-1", "q")
    assert result.conflict_signal == ConflictSignal.SERVED_STALE


def test_graphiti_selfhosted_query_raw_carries_edge_endpoint_uuids() -> None:
    """The exact structural shape evals/contradiction.py's
    EDGE_INTEGRITY_VIOLATION check reads: MemoryRecord.raw must carry the
    edge's source_node_uuid/target_node_uuid so that check can inspect
    them without this adapter needing its own bespoke field for it."""
    client = FakeGraphitiClient()
    client.set_search_result(
        [_FakeEntityEdge(uuid="e1", fact="a fact", source_node_uuid="n1", target_node_uuid="n2")]
    )
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    result = adapter.query("session-1", "q")
    assert result.records[0].raw["source_node_uuid"] == "n1"
    assert result.records[0].raw["target_node_uuid"] == "n2"


def test_graphiti_selfhosted_store_threads_update_communities_true() -> None:
    """The update_communities=True toggle this build added, demonstrated
    reaching add_episode() -- see this adapter module's docstring for the
    honest limitation this does NOT prove: that it actually triggers
    getzep/graphiti#836's ValueError, which needs a live instance and real
    entity extraction to observe."""
    client = FakeGraphitiClient()
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client, update_communities=True)
    adapter.store("session-1", "content")
    assert client.add_episode_calls[0]["update_communities"] is True


def test_graphiti_selfhosted_update_communities_defaults_false() -> None:
    client = FakeGraphitiClient()
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    adapter.store("session-1", "content")
    assert client.add_episode_calls[0]["update_communities"] is False


def test_graphiti_selfhosted_update_communities_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHITI_UPDATE_COMMUNITIES", "true")
    client = FakeGraphitiClient()
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    adapter.store("session-1", "content")
    assert client.add_episode_calls[0]["update_communities"] is True


def test_graphiti_selfhosted_explicit_update_communities_kwarg_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPHITI_UPDATE_COMMUNITIES", "true")
    client = FakeGraphitiClient()
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client, update_communities=False)
    adapter.store("session-1", "content")
    assert client.add_episode_calls[0]["update_communities"] is False


def test_graphiti_selfhosted_delete_calls_remove_episode() -> None:
    client = FakeGraphitiClient()
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    result = adapter.delete("episode-7")
    assert result.success is True
    assert result.memory_id == "episode-7"
    assert client.remove_episode_calls == ["episode-7"]


def test_graphiti_selfhosted_wraps_vendor_exceptions_in_backend_api_error() -> None:
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=_BrokenGraphitiClient())
    with pytest.raises(BackendAPIError):
        adapter.store("session-1", "content")
    with pytest.raises(BackendAPIError):
        adapter.query("session-1", "q")
    with pytest.raises(BackendAPIError):
        adapter.delete("id-1")


# ---------------------------------------------------------------------------
# CrashSignal classification -- closes the gap where getzep/graphiti#836
# and getzep/graphiti#920 both surfaced as an identical opaque
# BackendAPIError with no way to distinguish "this specific known
# graphiti-core bug crashed" from "some other failure happened." See
# CrashSignal in base.py and _classify_crash() in
# zep_graphiti_selfhosted_adapter.py.
# ---------------------------------------------------------------------------


def test_classify_crash_unpack_value_error_matches_836_shape() -> None:
    assert (
        _classify_crash(ValueError("too many values to unpack (expected 2)"))
        == CrashSignal.UNPACK_ERROR
    )
    # #836's traceback can also read "not enough values to unpack" depending
    # on which side of 2 the extracted node count lands on -- both must
    # classify the same way, since both come from the same semaphore_gather
    # unpack-of-a-list-of-2-tuples root cause.
    assert (
        _classify_crash(ValueError("not enough values to unpack (expected 2, got 1)"))
        == CrashSignal.UNPACK_ERROR
    )


def test_classify_crash_datetime_type_error_matches_920_shape() -> None:
    assert (
        _classify_crash(TypeError("can't compare offset-naive and offset-aware datetimes"))
        == CrashSignal.TYPE_COMPARISON_ERROR
    )


def test_classify_crash_generic_runtime_error_is_unknown() -> None:
    assert _classify_crash(RuntimeError("vendor exploded")) == CrashSignal.UNKNOWN


def test_classify_crash_unrelated_value_error_is_unknown() -> None:
    # A ValueError that is NOT the #836 unpack shape must not be
    # miscategorized as UNPACK_ERROR just because it's a ValueError.
    assert _classify_crash(ValueError("invalid literal for int()")) == CrashSignal.UNKNOWN


def test_classify_crash_unrelated_type_error_is_unknown() -> None:
    # Same guard, for TypeError -- must not be miscategorized as
    # TYPE_COMPARISON_ERROR just because it's a TypeError.
    assert _classify_crash(TypeError("unsupported operand type(s)")) == CrashSignal.UNKNOWN


def test_classify_crash_empty_query_redisearch_syntax_error_matches_1222_shape() -> None:
    # getzep/graphiti#1222's exact filed reproduction message: an empty
    # sanitized query produces "(@group_id:...) ()", which RediSearch
    # rejects with this message shape.
    assert (
        _classify_crash(Exception("RediSearch: Syntax error at offset 22 near my_graph"))
        == CrashSignal.QUERY_SANITIZATION_ERROR
    )


def test_classify_crash_pipe_slash_redisearch_syntax_error_matches_1183_shape() -> None:
    # getzep/graphiti#1183's exact filed reproduction message: an
    # unescaped pipe in episode text produces an empty token between
    # RediSearch OR-pipe delimiters, rejected with this message shape.
    assert (
        _classify_crash(Exception("RediSearch: Syntax error at offset 178 near sh"))
        == CrashSignal.QUERY_SANITIZATION_ERROR
    )


def test_classify_crash_redisearch_message_requires_both_substrings() -> None:
    # A generic Python SyntaxError (or any message containing only "syntax
    # error" without "redisearch") must NOT be miscategorized as
    # QUERY_SANITIZATION_ERROR -- both substrings are required so this
    # never fires on an unrelated syntax error from a different source.
    assert _classify_crash(Exception("Syntax error near token")) == CrashSignal.UNKNOWN
    assert _classify_crash(Exception("RediSearch: connection refused")) == CrashSignal.UNKNOWN


def test_classify_crash_embedding_batch_count_mismatch_matches_1467_shape() -> None:
    # getzep/graphiti#1467's exact filed reproduction message: a strict-zip
    # count mismatch several frames away from GeminiEmbedder.create_batch()
    # silently returning too few vectors.
    assert (
        _classify_crash(ValueError("zip() argument 2 is shorter than argument 1"))
        == CrashSignal.EMBEDDING_BATCH_COUNT_MISMATCH
    )
    # The reverse-argument-order phrasing (whichever list came up short)
    # must classify the same way.
    assert (
        _classify_crash(ValueError("zip() argument 1 is shorter than argument 2"))
        == CrashSignal.EMBEDDING_BATCH_COUNT_MISMATCH
    )


def test_classify_crash_zip_message_requires_both_substrings() -> None:
    # A ValueError that only mentions "zip()" or only "shorter than" (not
    # both) must NOT be miscategorized as EMBEDDING_BATCH_COUNT_MISMATCH.
    assert _classify_crash(ValueError("zip() takes no keyword arguments")) == CrashSignal.UNKNOWN
    assert _classify_crash(ValueError("list is shorter than expected")) == CrashSignal.UNKNOWN


def test_graphiti_selfhosted_store_classifies_1467_embedding_batch_count_mismatch() -> None:
    """A fake client raising getzep/graphiti#1467's exact ValueError shape
    is classified as CrashSignal.EMBEDDING_BATCH_COUNT_MISMATCH on the
    raised BackendAPIError, not the generic default."""
    adapter = ZepGraphitiSelfHostedAdapter(
        graphiti_client=_EmbeddingBatchCountMismatchGraphitiClient()
    )
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.store("session-1", "content extracting 2+ entities")
    assert exc_info.value.crash_signal == CrashSignal.EMBEDDING_BATCH_COUNT_MISMATCH


def test_classify_crash_nul_byte_query_parameter_error_matches_1525_shape() -> None:
    # getzep/graphiti#1525's exact filed reproduction message: a NUL byte
    # embedded in episode content makes FalkorDB reject the entire bulk
    # episode-save query with this message shape.
    assert (
        _classify_crash(Exception("Failed to parse query parameter 'episodes' value"))
        == CrashSignal.QUERY_PARAMETER_PARSE_ERROR
    )
    # Case-insensitive, and matches regardless of which parameter name is
    # embedded in the message.
    assert (
        _classify_crash(Exception("failed to parse query parameter 'content' value"))
        == CrashSignal.QUERY_PARAMETER_PARSE_ERROR
    )


def test_classify_crash_query_parameter_message_requires_both_substrings() -> None:
    # A message that only contains one of the two required substrings must
    # NOT be miscategorized as QUERY_PARAMETER_PARSE_ERROR.
    assert _classify_crash(Exception("Failed to parse query parameter")) == CrashSignal.UNKNOWN
    assert _classify_crash(Exception("invalid value for parameter")) == CrashSignal.UNKNOWN


def test_graphiti_selfhosted_store_classifies_1525_nul_byte_query_parameter_error() -> None:
    """A fake client raising getzep/graphiti#1525's exact ResponseError
    shape is classified as CrashSignal.QUERY_PARAMETER_PARSE_ERROR on the
    raised BackendAPIError, not the generic default -- store() classifies
    this via its existing generic `except Exception` fallback (the real
    exception is neither ValueError nor TypeError, so it never hits the
    (ValueError, TypeError) pre-filter #836/#920 use)."""
    adapter = ZepGraphitiSelfHostedAdapter(
        graphiti_client=_NulByteQueryParameterErrorGraphitiClient()
    )
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.store("session-1", "content with a \x00 nul byte")
    assert exc_info.value.crash_signal == CrashSignal.QUERY_PARAMETER_PARSE_ERROR


def test_graphiti_selfhosted_store_classifies_836_unpack_error() -> None:
    """A fake client raising getzep/graphiti#836's exact ValueError shape
    is classified as CrashSignal.UNPACK_ERROR on the raised
    BackendAPIError, not the generic default."""
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=_UnpackErrorGraphitiClient())
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.store("session-1", "content")
    assert exc_info.value.crash_signal == CrashSignal.UNPACK_ERROR


def test_graphiti_selfhosted_store_classifies_920_type_comparison_error() -> None:
    """A fake client raising getzep/graphiti#920's exact TypeError shape is
    classified as CrashSignal.TYPE_COMPARISON_ERROR on the raised
    BackendAPIError, not the generic default."""
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=_TypeComparisonErrorGraphitiClient())
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.store("session-1", "content")
    assert exc_info.value.crash_signal == CrashSignal.TYPE_COMPARISON_ERROR


def test_graphiti_selfhosted_store_classifies_generic_failure_as_unknown() -> None:
    """A fake client raising an unrelated RuntimeError still raises
    BackendAPIError (existing behavior, unchanged) but now carries an
    explicit crash_signal=CrashSignal.UNKNOWN rather than leaving the
    caller with no way to tell this apart from a classified crash."""
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=_BrokenGraphitiClient())
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.store("session-1", "content")
    assert exc_info.value.crash_signal == CrashSignal.UNKNOWN


def test_graphiti_selfhosted_query_classifies_1222_empty_query_syntax_error() -> None:
    """query() -- previously never classified crashes at all, unlike
    store() -- now recognizes getzep/graphiti#1222's exact RediSearch
    `Syntax error` shape (empty sanitized query -> "(@group_id:...) ()")
    and attaches CrashSignal.QUERY_SANITIZATION_ERROR to the raised
    BackendAPIError instead of leaving crash_signal unset/None."""
    adapter = ZepGraphitiSelfHostedAdapter(
        graphiti_client=_EmptyQuerySanitizationErrorGraphitiClient()
    )
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.query("session-1", "")
    assert exc_info.value.crash_signal == CrashSignal.QUERY_SANITIZATION_ERROR


def test_graphiti_selfhosted_query_classifies_1183_pipe_slash_syntax_error() -> None:
    """Same query()-classification wiring as the #1222 test above, for
    getzep/graphiti#1183's distinct trigger: episode/query text containing
    unescaped pipe/slash characters producing the identical RediSearch
    `Syntax error` message shape via a different upstream root cause."""
    adapter = ZepGraphitiSelfHostedAdapter(
        graphiti_client=_PipeSlashSanitizationErrorGraphitiClient()
    )
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.query("session-1", "install.sh | bash")
    assert exc_info.value.crash_signal == CrashSignal.QUERY_SANITIZATION_ERROR


def test_graphiti_selfhosted_query_classifies_generic_failure_as_unknown() -> None:
    """A fake client's search() raising an unrelated RuntimeError still
    raises BackendAPIError (existing behavior, unchanged) but now carries
    an explicit crash_signal=CrashSignal.UNKNOWN -- proving query()'s new
    classification distinguishes an unrelated failure from both #1222/
    #1183's QUERY_SANITIZATION_ERROR shape above and from the unrelated
    #836/#920 shapes store() classifies, not just a blanket "anything
    non-empty counts as classified."""
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=_BrokenGraphitiClient())
    with pytest.raises(BackendAPIError) as exc_info:
        adapter.query("session-1", "q")
    assert exc_info.value.crash_signal == CrashSignal.UNKNOWN


def test_graphiti_selfhosted_close_calls_client_close() -> None:
    client = FakeGraphitiClient()
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    adapter.close()
    assert client.closed is True


def test_graphiti_selfhosted_close_is_a_noop_when_client_never_constructed() -> None:
    adapter = ZepGraphitiSelfHostedAdapter(neo4j_uri="bolt://localhost:7687")
    adapter.close()  # must not raise -- _get_client() was never called


def test_graphiti_selfhosted_get_client_raises_clear_error_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # graphiti-core is not a memtrust dependency and is genuinely not
    # installed in this test environment -- this exercises the real
    # ImportError path, not a simulated one, same convention as
    # MemPalaceAdapter's equivalent test above.
    monkeypatch.setenv("GRAPHITI_NEO4J_URI", "bolt://localhost:7687")
    adapter = ZepGraphitiSelfHostedAdapter()
    with pytest.raises(BackendAPIError, match="not installed"):
        adapter.store("session-1", "content")


def test_parse_falkordb_url_bare_host_port() -> None:
    assert _parse_falkordb_url("localhost:6379") == ("localhost", 6379, None, None)


def test_parse_falkordb_url_full_redis_uri() -> None:
    assert _parse_falkordb_url("redis://admin:secret@falkor-host:6380") == (
        "falkor-host",
        6380,
        "admin",
        "secret",
    )


def test_parse_falkordb_url_defaults_port_when_omitted() -> None:
    host, port, _user, _password = _parse_falkordb_url("falkor-host")
    assert host == "falkor-host"
    assert port == 6379
