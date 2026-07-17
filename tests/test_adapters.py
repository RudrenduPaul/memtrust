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
    CrashSignal,
    DeleteResult,
    ExtractionSignal,
    MemoryBackendAdapter,
    QueryResult,
    RankingSignal,
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
# MemPalaceAdapter (fake in-memory Palace, no chromadb dependency required)
# ---------------------------------------------------------------------------


class FakePalace:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._next_id = 0
        self.remember_modes: list[str | None] = []
        self.recall_modes: list[str | None] = []

    def remember(
        self, room: str, content: str, metadata: dict[str, str], mode: str | None = None
    ) -> str:
        self.remember_modes.append(mode)
        self._next_id += 1
        memory_id = f"palace-{self._next_id}"
        self._store[memory_id] = {"room": room, "content": content, "metadata": metadata}
        return memory_id

    def recall(
        self, room: str, query: str, top_k: int, mode: str | None = None
    ) -> list[dict[str, Any]]:
        self.recall_modes.append(mode)
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


def test_mempalace_threads_mode_through_to_palace_calls() -> None:
    palace = FakePalace()
    adapter = MemPalaceAdapter(palace=palace)

    adapter.store("room-1", "content", mode="AAAK")
    adapter.query("room-1", "query", mode="AAAK")
    assert palace.remember_modes == ["AAAK"]
    assert palace.recall_modes == ["AAAK"]

    # Not passing `mode` at all (the default) must not change the
    # underlying call shape -- `None` is forwarded, exactly as before this
    # parameter existed.
    adapter.store("room-1", "content")
    adapter.query("room-1", "query")
    assert palace.remember_modes == ["AAAK", None]
    assert palace.recall_modes == ["AAAK", None]


def test_mempalace_supported_modes_reports_raw_and_aaak() -> None:
    assert MemPalaceAdapter.supported_modes == ("raw", "AAAK")


def test_mempalace_wraps_vendor_exceptions_in_backend_api_error() -> None:
    class BrokenPalace:
        def remember(
            self, room: str, content: str, metadata: dict[str, str], mode: str | None = None
        ) -> str:
            raise RuntimeError("vendor exploded")

        def recall(
            self, room: str, query: str, top_k: int, mode: str | None = None
        ) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Read-after-write verification (StoreResult.verified / verify_store)
# ---------------------------------------------------------------------------


def test_store_result_defaults_to_verified_none() -> None:
    palace = FakePalace()
    adapter = MemPalaceAdapter(palace=palace)
    result = adapter.store("room-1", "My dog is named Baxter.")
    assert result.verified is None


def test_mempalace_verify_true_confirms_readable_write() -> None:
    """(a) verify=True with a mock adapter that returns the just-stored
    content confirms verified=True."""
    palace = FakePalace()
    adapter = MemPalaceAdapter(palace=palace)
    result = adapter.store("room-1", "My dog is named Baxter.", verify=True)
    assert result.verified is True


def test_mempalace_verify_true_detects_silently_dropped_write() -> None:
    """(b) verify=True with a mock returning empty/wrong content sets
    verified=False rather than raising -- this is the exact "store()
    didn't raise, but the write was silently dropped/corrupted" failure
    mode (MemPalace issues #1929, #1977) this feature exists to catch.
    """

    class SilentlyDroppingPalace:
        """remember() returns a normal memory_id and never raises, but
        the write never actually lands -- recall() always comes back
        empty, simulating checkpoint corruption or a stale/self-
        deadlocked lock silently no-oping the write server-side.
        """

        def remember(
            self, room: str, content: str, metadata: dict[str, str], mode: str | None = None
        ) -> str:
            return "palace-ghost-1"

        def recall(
            self, room: str, query: str, top_k: int, mode: str | None = None
        ) -> list[dict[str, Any]]:
            return []

        def invalidate(self, room: str, memory_id: str, content: str) -> dict[str, Any]:
            raise NotImplementedError

    adapter = MemPalaceAdapter(palace=SilentlyDroppingPalace())
    result = adapter.store("room-1", "My dog is named Baxter.", verify=True)
    assert result.verified is False
    # Crucially, this must not raise -- a failed verification is a
    # reported fact about the write, not an exception.


def test_mempalace_verify_true_detects_wrong_content_on_readback() -> None:
    """Same failure mode as above, but recall() returns *something* --
    just not the content that was actually stored (corruption, not a
    total drop). Still verified=False, still no exception."""

    class CorruptingPalace:
        def remember(
            self, room: str, content: str, metadata: dict[str, str], mode: str | None = None
        ) -> str:
            return "palace-corrupt-1"

        def recall(
            self, room: str, query: str, top_k: int, mode: str | None = None
        ) -> list[dict[str, Any]]:
            return [{"id": "palace-corrupt-1", "content": "\x00\x00\x00", "metadata": {}}]

        def invalidate(self, room: str, memory_id: str, content: str) -> dict[str, Any]:
            raise NotImplementedError

    adapter = MemPalaceAdapter(palace=CorruptingPalace())
    result = adapter.store("room-1", "My dog is named Baxter.", verify=True)
    assert result.verified is False


def test_mempalace_verify_false_by_default_does_not_call_recall() -> None:
    """(c) verify=False (default) behavior is unchanged from before this
    fix -- no query() call happens, verified stays None."""

    class RecallTrackingPalace(FakePalace):
        def __init__(self) -> None:
            super().__init__()
            self.recall_call_count = 0

        def recall(
            self, room: str, query: str, top_k: int, mode: str | None = None
        ) -> list[dict[str, Any]]:
            self.recall_call_count += 1
            return super().recall(room, query, top_k, mode)

    palace = RecallTrackingPalace()
    adapter = MemPalaceAdapter(palace=palace)

    result = adapter.store("room-1", "My dog is named Baxter.")
    assert result.verified is None
    assert palace.recall_call_count == 0

    # Explicit verify=False must behave identically to the omitted default.
    result_explicit = adapter.store("room-1", "My cat is named Whiskers.", verify=False)
    assert result_explicit.verified is None
    assert palace.recall_call_count == 0


def test_verify_store_raises_backend_api_error_when_query_itself_fails() -> None:
    """A genuine vendor/network failure during the verification query()
    call must still propagate as BackendAPIError, not be swallowed into
    verified=False -- only an absent/wrong record on a successful query
    means "the write was silently dropped," not a query that itself
    errored."""

    class QueryFailsPalace:
        def remember(
            self, room: str, content: str, metadata: dict[str, str], mode: str | None = None
        ) -> str:
            return "palace-1"

        def recall(
            self, room: str, query: str, top_k: int, mode: str | None = None
        ) -> list[dict[str, Any]]:
            raise RuntimeError("vendor exploded during verification query")

        def invalidate(self, room: str, memory_id: str, content: str) -> dict[str, Any]:
            raise NotImplementedError

    adapter = MemPalaceAdapter(palace=QueryFailsPalace())
    with pytest.raises(BackendAPIError):
        adapter.store("room-1", "content", verify=True)


def test_mempalace_delete_raises_clear_not_implemented_backend_api_error() -> None:
    # MemPalace has no confirmed delete/forget primitive (see module
    # docstring) -- delete() must still exist and fail with a typed,
    # documented BackendAPIError rather than an AttributeError or a
    # silent no-op.
    adapter = MemPalaceAdapter(palace=FakePalace())
    with pytest.raises(BackendAPIError, match="not implemented"):
        adapter.delete("palace-1")


# ---------------------------------------------------------------------------
# MemPalaceAdapter.query() -- RankingSignal detection
#
# Reproduces the exact mempalace/mempalace#1733 shape (GitHub user
# Kartalops): `Layer1.generate()` sorts drawers by `importance`/
# `emotional_weight`/`weight`, but no ingest path ever writes those keys
# (confirmed 0/45,969 drawers on a real palace), so the field silently
# defaults to a constant and the sort degenerates to insertion order.
# ---------------------------------------------------------------------------


def test_mempalace_query_flags_missing_ordering_key_when_importance_is_constant() -> None:
    palace = FakePalace()
    adapter = MemPalaceAdapter(palace=palace)
    for content in ["Had coffee with Alex.", "Signed the lease.", "Grandmother's health scare."]:
        adapter.store("room-1", content, metadata={"importance": "0.5"})

    query_result = adapter.query("room-1", "wake me up with important memories")
    assert query_result.ranking_signal == RankingSignal.MISSING_ORDERING_KEY


def test_mempalace_query_flags_missing_ordering_key_when_importance_never_written() -> None:
    # The exact #1733 shape: no ingest path ever wrote the field at all,
    # not even a default -- indistinguishable from a constant default from
    # this adapter's black-box view, and flagged the same way.
    palace = FakePalace()
    adapter = MemPalaceAdapter(palace=palace)
    for content in ["Had coffee with Alex.", "Signed the lease.", "Grandmother's health scare."]:
        adapter.store("room-1", content)

    query_result = adapter.query("room-1", "wake me up with important memories")
    assert query_result.ranking_signal == RankingSignal.MISSING_ORDERING_KEY


def test_mempalace_query_reports_signal_driven_when_importance_genuinely_varies() -> None:
    # Negative control: this must NOT be flagged -- a real per-record
    # signal exists here.
    palace = FakePalace()
    adapter = MemPalaceAdapter(palace=palace)
    adapter.store("room-1", "Renewed car registration.", metadata={"importance": "0.2"})
    adapter.store("room-1", "Grandmother's health scare.", metadata={"importance": "0.9"})
    adapter.store("room-1", "Signed the lease.", metadata={"importance": "0.6"})

    query_result = adapter.query("room-1", "wake me up with important memories")
    assert query_result.ranking_signal == RankingSignal.SIGNAL_DRIVEN


def test_mempalace_query_ranking_signal_not_applicable_with_fewer_than_two_records() -> None:
    palace = FakePalace()
    adapter = MemPalaceAdapter(palace=palace)
    adapter.store("room-1", "Had coffee with Alex.", metadata={"importance": "0.5"})

    query_result = adapter.query("room-1", "wake me up with important memories")
    assert query_result.ranking_signal == RankingSignal.NOT_APPLICABLE


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


def test_graphiti_selfhosted_configured_via_falkordb_url_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAPHITI_NEO4J_URI", raising=False)
    monkeypatch.setenv("GRAPHITI_FALKORDB_URL", "redis://localhost:6379")
    # Must not raise BackendNotConfiguredError -- GRAPHITI_FALKORDB_URL
    # alone satisfies configuration, the same "one env var, or SKIPPED"
    # contract GRAPHITI_NEO4J_URI satisfies on its own.
    ZepGraphitiSelfHostedAdapter()


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
