"""Tests for evals/episode_temporal_leak.py and
ZepGraphitiSelfHostedAdapter.retrieve_episodes() -- the new driver-level
primitive this eval needs (`Graphiti.search()` only ever returns edges,
never episodes, so this eval cannot be built on the generic
`MemoryBackendAdapter.query()` interface every other eval in
tests/test_evals.py exercises). Kept in a dedicated file rather than
folded into test_adapters.py/test_evals.py because this capability is
deliberately adapter-specific, not part of the shared interface -- see
`episode_temporal_leak.py`'s module docstring.

graphiti-core is not installed in this test environment (confirmed
elsewhere in this repo's test suite) -- every test here exercises this
adapter's own logic against a fake driver/episode-ops double injected via
the `graphiti_client=` constructor kwarg, the same convention
test_adapters.py's FakeGraphitiClient already establishes. These tests
prove the classification logic is correct given a response shape; they do
not prove that shape matches a live FalkorDB instance. See
zep_graphiti_selfhosted_adapter.py's module docstring for the full
caveat.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from memtrust.adapters.base import BackendAPIError
from memtrust.adapters.zep_graphiti_selfhosted_adapter import ZepGraphitiSelfHostedAdapter
from memtrust.evals.episode_temporal_leak import (
    EpisodeTemporalSignal,
    classify_episode_temporal_leak,
    run_episode_temporal_leak_eval,
)


class _FakeEpisodicNode:
    """Stands in for a real `graphiti_core.nodes.EpisodicNode` -- exposes
    `.model_dump()` the same way the real Pydantic model does, matching
    test_adapters.py's `_FakeEntityEdge` convention for the edge side."""

    def __init__(self, uuid: str, name: str, valid_at: datetime) -> None:
        self.uuid = uuid
        self.name = name
        self.valid_at = valid_at

    def model_dump(self) -> dict[str, Any]:
        return {"uuid": self.uuid, "name": self.name, "valid_at": self.valid_at}


class _FakeEpisodeNodeOps:
    """Stands in for a real `graphiti_core.driver.operations.episode_node_ops
    .EpisodeNodeOperations` instance -- records every call so tests can
    assert on exactly what `retrieve_episodes()` sent."""

    def __init__(self, episodes: list[_FakeEpisodicNode]) -> None:
        self._episodes = episodes
        self.calls: list[dict[str, Any]] = []

    async def retrieve_episodes(
        self,
        executor: Any,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: str | None = None,
        saga: str | None = None,
    ) -> list[_FakeEpisodicNode]:
        self.calls.append(
            {
                "executor": executor,
                "reference_time": reference_time,
                "last_n": last_n,
                "group_ids": group_ids,
                "source": source,
                "saga": saga,
            }
        )
        return self._episodes


class _FakeDriverWithEpisodeOps:
    def __init__(self, episode_node_ops: _FakeEpisodeNodeOps | None) -> None:
        self.episode_node_ops = episode_node_ops


class _FakeGraphitiClientWithDriver:
    """Minimal `_GraphitiProtocol`-conforming fake exposing only the
    `.driver` attribute `retrieve_episodes()` actually reads -- the other
    Protocol methods are unused by the tests that inject this client."""

    def __init__(self, driver: _FakeDriverWithEpisodeOps | None) -> None:
        self.driver = driver

    async def add_episode(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise NotImplementedError

    async def remove_episode(self, episode_uuid: str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# classify_episode_temporal_leak() -- pure classification logic
# ---------------------------------------------------------------------------


def test_classify_no_leak_when_every_episode_before_reference_time() -> None:
    reference_time = datetime(2024, 2, 1, tzinfo=UTC)
    episodes = [
        {"uuid": "e1", "name": "past episode", "valid_at": datetime(2024, 1, 1, tzinfo=UTC)}
    ]
    signal, leaked = classify_episode_temporal_leak(episodes, reference_time)
    assert signal == EpisodeTemporalSignal.NO_LEAK
    assert leaked == []


def test_classify_temporal_leak_when_future_episode_returned() -> None:
    """The exact getzep/graphiti#1625 shape: a future-dated episode
    (valid_at=2024-03-01) returned for a reference_time=2024-02-01
    query."""
    reference_time = datetime(2024, 2, 1, tzinfo=UTC)
    episodes = [
        {"uuid": "e1", "name": "past episode", "valid_at": datetime(2024, 1, 1, tzinfo=UTC)},
        {"uuid": "e2", "name": "future episode", "valid_at": datetime(2024, 3, 1, tzinfo=UTC)},
    ]
    signal, leaked = classify_episode_temporal_leak(episodes, reference_time)
    assert signal == EpisodeTemporalSignal.TEMPORAL_LEAK
    assert leaked == ["future episode"]


def test_classify_handles_iso_string_valid_at() -> None:
    """A real, installed EpisodicNode's model_dump() serializes `valid_at`
    to an ISO-8601 string (pydantic's default datetime serialization) --
    classify_episode_temporal_leak() must parse that, not just accept a
    real datetime object."""
    reference_time = datetime(2024, 2, 1, tzinfo=UTC)
    episodes = [{"uuid": "e2", "name": "future episode", "valid_at": "2024-03-01T00:00:00+00:00"}]
    signal, leaked = classify_episode_temporal_leak(episodes, reference_time)
    assert signal == EpisodeTemporalSignal.TEMPORAL_LEAK
    assert leaked == ["future episode"]


def test_classify_skips_episodes_with_missing_or_unparseable_valid_at() -> None:
    reference_time = datetime(2024, 2, 1, tzinfo=UTC)
    episodes: list[dict[str, Any]] = [
        {"uuid": "e1", "name": "no valid_at field"},
        {"uuid": "e2", "name": "garbage valid_at", "valid_at": "not-a-date"},
    ]
    signal, leaked = classify_episode_temporal_leak(episodes, reference_time)
    assert signal == EpisodeTemporalSignal.NO_LEAK
    assert leaked == []


def test_classify_falls_back_to_uuid_then_unknown_for_leaked_episode_label() -> None:
    reference_time = datetime(2024, 2, 1, tzinfo=UTC)
    episodes = [{"uuid": "e2", "valid_at": datetime(2024, 3, 1, tzinfo=UTC)}]
    _, leaked = classify_episode_temporal_leak(episodes, reference_time)
    assert leaked == ["e2"]

    episodes_no_id: list[dict[str, Any]] = [{"valid_at": datetime(2024, 3, 1, tzinfo=UTC)}]
    _, leaked_no_id = classify_episode_temporal_leak(episodes_no_id, reference_time)
    assert leaked_no_id == ["unknown"]


# ---------------------------------------------------------------------------
# ZepGraphitiSelfHostedAdapter.retrieve_episodes() -- the driver-level
# primitive
# ---------------------------------------------------------------------------


def test_retrieve_episodes_calls_real_signature_shape() -> None:
    """Confirms this adapter calls episode_node_ops.retrieve_episodes()
    with the real, source-confirmed argument shape: executor=driver
    (self-referential, per the issue's own repro code), plus
    reference_time/last_n/group_ids threaded through unchanged."""
    reference_time = datetime(2024, 2, 1, tzinfo=UTC)
    ops = _FakeEpisodeNodeOps(
        [_FakeEpisodicNode("e1", "past episode", datetime(2024, 1, 1, tzinfo=UTC))]
    )
    driver = _FakeDriverWithEpisodeOps(ops)
    client = _FakeGraphitiClientWithDriver(driver)
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)

    episodes = adapter.retrieve_episodes(reference_time, group_ids=["g1"], last_n=5)

    assert len(episodes) == 1
    assert episodes[0]["name"] == "past episode"
    call = ops.calls[0]
    assert call["executor"] is driver
    assert call["reference_time"] == reference_time
    assert call["last_n"] == 5
    assert call["group_ids"] == ["g1"]


def test_retrieve_episodes_raises_when_no_episode_node_ops_surface() -> None:
    client = _FakeGraphitiClientWithDriver(_FakeDriverWithEpisodeOps(None))
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    with pytest.raises(BackendAPIError, match="episode_node_ops"):
        adapter.retrieve_episodes(datetime(2024, 2, 1, tzinfo=UTC))


def test_retrieve_episodes_raises_when_driver_attribute_missing() -> None:
    # A client double with no .driver attribute at all (e.g. FakeGraphitiClient
    # from test_adapters.py) -- getattr(..., "driver", None) degrades to
    # None cleanly, same error as the "no episode_node_ops" case above.
    class _NoDriverClient:
        async def add_episode(self, *a: Any, **k: Any) -> Any:
            raise NotImplementedError

        async def search(self, *a: Any, **k: Any) -> list[Any]:
            raise NotImplementedError

        async def remove_episode(self, episode_uuid: str) -> None:
            raise NotImplementedError

        async def close(self) -> None:
            pass

    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=_NoDriverClient())
    with pytest.raises(BackendAPIError, match="episode_node_ops"):
        adapter.retrieve_episodes(datetime(2024, 2, 1, tzinfo=UTC))


def test_retrieve_episodes_wraps_vendor_exception_as_backend_api_error() -> None:
    class _BrokenEpisodeNodeOps:
        async def retrieve_episodes(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("vendor exploded")

    client = _FakeGraphitiClientWithDriver(
        _FakeDriverWithEpisodeOps(_BrokenEpisodeNodeOps())  # type: ignore[arg-type]
    )
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)
    with pytest.raises(BackendAPIError):
        adapter.retrieve_episodes(datetime(2024, 2, 1, tzinfo=UTC))


# ---------------------------------------------------------------------------
# run_episode_temporal_leak_eval() -- full pipeline
# ---------------------------------------------------------------------------


def test_run_episode_temporal_leak_eval_reproduces_1625_shape_end_to_end() -> None:
    """pcy06's own reproduction, end to end: a past episode and a
    future-dated episode both returned by retrieve_episodes() for a
    reference_time between them -- classified TEMPORAL_LEAK."""
    reference_time = datetime(2024, 2, 1, tzinfo=UTC)
    ops = _FakeEpisodeNodeOps(
        [
            _FakeEpisodicNode("e1", "past episode", datetime(2024, 1, 1, tzinfo=UTC)),
            _FakeEpisodicNode("e2", "future episode", datetime(2024, 3, 1, tzinfo=UTC)),
        ]
    )
    client = _FakeGraphitiClientWithDriver(_FakeDriverWithEpisodeOps(ops))
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)

    result = run_episode_temporal_leak_eval(adapter, reference_time, group_ids=["repro"])

    assert result.signal == EpisodeTemporalSignal.TEMPORAL_LEAK
    assert result.leaked_episode_names == ["future episode"]
    assert result.total_episodes_returned == 2
    assert result.error is None


def test_run_episode_temporal_leak_eval_no_leak_when_falkordb_behaves() -> None:
    reference_time = datetime(2024, 2, 1, tzinfo=UTC)
    ops = _FakeEpisodeNodeOps(
        [_FakeEpisodicNode("e1", "past episode", datetime(2024, 1, 1, tzinfo=UTC))]
    )
    client = _FakeGraphitiClientWithDriver(_FakeDriverWithEpisodeOps(ops))
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)

    result = run_episode_temporal_leak_eval(adapter, reference_time)
    assert result.signal == EpisodeTemporalSignal.NO_LEAK
    assert result.leaked_episode_names == []


def test_run_episode_temporal_leak_eval_not_applicable_on_backend_error() -> None:
    client = _FakeGraphitiClientWithDriver(_FakeDriverWithEpisodeOps(None))
    adapter = ZepGraphitiSelfHostedAdapter(graphiti_client=client)

    result = run_episode_temporal_leak_eval(adapter, datetime(2024, 2, 1, tzinfo=UTC))
    assert result.signal == EpisodeTemporalSignal.NOT_APPLICABLE
    assert result.error is not None
