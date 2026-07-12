"""Adapter tests. Every HTTP-based adapter is exercised via pytest-httpx
(no real network calls); MemPalaceAdapter is exercised via a fake
in-memory Palace injected through its constructor, matching the
_PalaceProtocol shape defined in mempalace_adapter.py.
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from memtrust.adapters.base import (
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
)
from memtrust.adapters.mem0_adapter import Mem0Adapter
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

        def remember(self, room: str, content: str, metadata: dict[str, str]) -> str:
            return "palace-ghost-1"

        def recall(self, room: str, query: str, top_k: int) -> list[dict[str, Any]]:
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
        def remember(self, room: str, content: str, metadata: dict[str, str]) -> str:
            return "palace-corrupt-1"

        def recall(self, room: str, query: str, top_k: int) -> list[dict[str, Any]]:
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

        def recall(self, room: str, query: str, top_k: int) -> list[dict[str, Any]]:
            self.recall_call_count += 1
            return super().recall(room, query, top_k)

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
        def remember(self, room: str, content: str, metadata: dict[str, str]) -> str:
            return "palace-1"

        def recall(self, room: str, query: str, top_k: int) -> list[dict[str, Any]]:
            raise RuntimeError("vendor exploded during verification query")

        def invalidate(self, room: str, memory_id: str, content: str) -> dict[str, Any]:
            raise NotImplementedError

    adapter = MemPalaceAdapter(palace=QueryFailsPalace())
    with pytest.raises(BackendAPIError):
        adapter.store("room-1", "content", verify=True)
