"""Adapter for self-hosted `graphiti-core` (https://github.com/getzep/graphiti),
the open-source temporal-knowledge-graph engine that also powers Zep Cloud.

Confidence: MEDIUM on wire-level shape, LOW on live end-to-end behavior.

`zep_graphiti_adapter.py`'s `ZepGraphitiAdapter` targets Zep Cloud's hosted
REST API. This adapter is the second, separately-configured adapter
docs/methodology.md already calls for under "Why Zep targets the hosted
Cloud API, not self-hosted Graphiti": *"If self-hosted Graphiti support is
wanted later, it should be a second adapter (e.g.
`zep_graphiti_selfhosted_adapter.py`) with its own configuration story, not
a silent branch inside this one."* That is exactly what this file is.

## Why this adapter exists

Four real, independently-verified graphiti-core bugs live entirely in the
self-hosted library layer that `ZepGraphitiAdapter` (hosted REST) cannot
reach at all, because Zep Cloud's REST surface never exposes graphiti-core's
internals to a caller:

  * **getzep/graphiti#1302** (open as of this build): `lucene_sanitize()`
    in `graphiti_core/helpers.py` builds a `str.translate()` escape map
    that includes every uppercase `O`, `R`, `N`, `T`, `A`, `D` character,
    intending to escape the Lucene boolean operators `AND`/`OR`/`NOT` --
    but the map operates per-character, not per-token, so it backslash-
    escapes those letters *anywhere* they appear in a query string (e.g.
    "ORION" becomes "\\O\\RION"), silently degrading BM25 full-text
    ranking for any query containing those letters. No exception, no
    error -- just worse-ranked (or unranked) results. Confirmed by fetching
    `graphiti_core/helpers.py` from `getzep/graphiti`'s `main` branch
    directly on 2026-07-16; the six single-letter translate entries
    (`'O': r'\\O', 'R': r'\\R', 'N': r'\\N', 'T': r'\\T', 'A': r'\\A',
    'D': r'\\D'`) are present verbatim, confirming the bug is still live
    upstream as of this adapter's build.
  * **getzep/graphiti#836** (open as of this build): `add_episode(...,
    update_communities=True)` raises `ValueError: too many values to
    unpack` (or "not enough values to unpack") whenever the episode's
    extracted node count is not exactly 2. Confirmed by fetching
    `graphiti_core/graphiti.py` and
    `graphiti_core/utils/maintenance/community_operations.py` from
    `main` on 2026-07-16: `add_episode()`'s community-update branch does
        `communities, community_edges = await semaphore_gather(
            *[update_community(...) for node in nodes], ...
        )`
    but `semaphore_gather` returns a plain `list` with one 2-tuple
    `(list[CommunityNode], list[CommunityEdge])` per node in `nodes` --
    so this line only succeeds when `len(nodes) == 2`. Any episode that
    extracts 0, 1, or 3+ entities (the overwhelmingly common case) raises
    `ValueError` from this unpack, not from anything resembling
    documented, expected behavior.
  * **getzep/graphiti#1013** (merged/fixed upstream): the Neo4j bulk
    edge-save Cypher query used to build its `SET` clause from an
    enumerated field list that omitted `EntityEdge.attributes` and
    `reference_time`, so a bulk-saved edge could silently come back
    missing properties a single-edge `save()` call would have written.
    Confirmed fixed on `main` as of 2026-07-16:
    `graphiti_core/models/edges/edge_db_queries.py`'s
    `get_entity_edge_save_bulk_query()` now emits `SET e = edge` for the
    Neo4j provider (the default case; FalkorDB's own branch of the same
    function uses `SET r = edge`, same shape, different Cypher variable
    name) -- an unconditional property copy, not an enumerated list --
    which is the documented fix shape for #1013.
  * **getzep/graphiti#1001** (closed via #1013): FalkorDB's older
    `add_triplet()`-era edge-creation path created edges via `MATCH`
    (which silently no-ops if either endpoint node is absent) and never
    set `source_node_uuid`/`target_node_uuid` as edge properties at all.
    Confirmed closed: `add_triplet()` no longer exists anywhere in
    `graphiti_core/driver/falkordb_driver.py` on `main` as of 2026-07-16
    -- the FalkorDB driver was rewritten around a
    `graphiti_core/driver/falkordb/operations/` module structure that
    post-dates this bug report.

None of the above four confirmations came from running graphiti-core
against a live Neo4j or FalkorDB instance in this environment -- they came
from fetching the real source files from `getzep/graphiti`'s `main` branch
on GitHub (`raw.githubusercontent.com/getzep/graphiti/main/...`) during
this build, 2026-07-16, and reading them directly. That is stronger
evidence than documentation, but it is still a snapshot of an unpinned,
moving branch, and it is not a substitute for exercising this adapter
against a real deployment. See "What this adapter does NOT prove" below.

## Design: why a separate class/file, not a flag on ZepGraphitiAdapter

Same reasoning `Mem0SelfHostedAdapter` documents in `mem0_adapter.py`:
hosted Zep Cloud and self-hosted graphiti-core are materially different
deployment shapes (REST vs. a direct Python client against a graph
driver), with different configuration stories (one API key vs. a database
connection) and different confidence levels. Collapsing them into one
class with an if/else would hide that from callers and from
docs/methodology.md's confidence table.

## Configuration: two backends, one env-var contract

Self-hosted graphiti-core has no single API key -- it needs a running
graph database (Neo4j by default, or FalkorDB) plus its own LLM/embedder
credentials for entity extraction. This adapter is gated on
`GRAPHITI_NEO4J_URI` (its primary, class-level `env_var`, consistent with
the "one env var, or SKIPPED" contract every other adapter follows) OR
`GRAPHITI_FALKORDB_URL` as an alternate. `BackendNotConfiguredError` is
only raised when *neither* is set. Supporting env vars:

  * `GRAPHITI_NEO4J_URI` -- e.g. `bolt://localhost:7687`
  * `GRAPHITI_NEO4J_USER` -- defaults to `"neo4j"` if unset
  * `GRAPHITI_NEO4J_PASSWORD`
  * `GRAPHITI_FALKORDB_URL` -- e.g. `redis://localhost:6379`; when set,
    this takes precedence over the Neo4j variables and a `FalkorDriver`
    is constructed instead of the default `Neo4jDriver`.
  * `GRAPHITI_UPDATE_COMMUNITIES` -- `"1"`/`"true"`/`"yes"` (case
    insensitive) threads `update_communities=True` through every
    `add_episode()` call this adapter makes, which is the toggle that
    would reach getzep/graphiti#836's code path. Defaults to `False`.
    Can also be set per-instance via the `update_communities` constructor
    kwarg, which takes precedence over the env var when explicitly passed
    (not `None`).

graphiti-core's real public API (`Graphiti.add_episode()`,
`Graphiti.search()`, `Graphiti.remove_episode()`) is entirely `async def`.
`MemoryBackendAdapter`'s `store()`/`query()`/`update()`/`delete()` are
synchronous, matching every other adapter in this repo -- this adapter
bridges the two with `asyncio.run(...)` inside each sync method, the same
pattern any synchronous caller of an async-only library has to use.

## What this adapter does NOT prove

This adapter was built and unit-tested (see tests/test_adapters.py)
entirely against a `graphiti_core`-shaped Protocol double -- the real
`graphiti-core` package is not installed in this build environment
(`ModuleNotFoundError: No module named 'graphiti_core'`), and no Neo4j or
FalkorDB instance was started or reached during this build. That means:

  * The exact constructor/method signatures this file calls
    (`Graphiti(uri=, user=, password=)`, `Graphiti(graph_driver=)`,
    `FalkorDriver(host=, port=, username=, password=)`,
    `add_episode(name=, episode_body=, source_description=,
    reference_time=, group_id=, update_communities=)`,
    `search(query, group_ids=, num_results=)`,
    `remove_episode(episode_uuid)`) were confirmed by reading the real
    source on GitHub, not by importing and calling the real package.
  * The `update_communities` toggle demonstrably threads through to
    `add_episode()` (see `store()` below and its unit test), but this
    build has no way to demonstrate it actually *triggers*
    getzep/graphiti#836's `ValueError` without a live Neo4j instance and
    real LLM credentials to run entity extraction against -- the bug
    lives inside `semaphore_gather`'s interaction with however many
    entities the LLM extraction step returns, which nothing in this
    adapter's mocked test path exercises.
  * `lucene_sanitize()` (getzep/graphiti#1302) is exercised nowhere in
    this adapter's own code -- it is an internal helper graphiti-core's
    search pipeline calls on the caller's query string before it ever
    reaches this adapter. This adapter cannot patch, bypass, or directly
    unit-test that function; it can only route eval queries containing
    the trigger characters (see tests/fixtures/contradiction_cases.json)
    at a live self-hosted instance, so a contributor who runs this
    adapter against a real deployment can compare BM25 ranking with and
    without those letters and observe the degradation directly.
  * `EDGE_INTEGRITY_VIOLATION` (getzep/graphiti#1013/#1001) is checked at
    the harness level, in `evals/contradiction.py`'s `classify_case()`,
    against whatever `source_node_uuid`/`target_node_uuid` values this
    adapter's `query()` puts in `MemoryRecord.raw`. Since both bugs are
    confirmed fixed/closed on the version of graphiti-core this adapter
    was built against, a live run today should not actually trigger this
    signal -- it exists so the check is *available* if this adapter is
    ever pinned to, or someone's self-hosted deployment happens to run,
    an older graphiti-core version where the bug is still present.

Anyone relying on this adapter to reproduce any of the four issues above
against a real deployment should verify against a live Neo4j/FalkorDB
instance first, exactly the same caveat docs/methodology.md already
states for `Mem0SelfHostedAdapter`, `MemPalaceAdapter`, and
`OpenVikingAdapter`.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlsplit

from memtrust.adapters.base import (
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
    DeleteResult,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    StoreResult,
    UpdateResult,
)

DEFAULT_NEO4J_USER = "neo4j"


class _GraphitiProtocol(Protocol):
    """Shape this adapter expects from a real `graphiti_core.Graphiti`
    instance (Neo4j- or FalkorDB-backed). Defined as a Protocol, not
    imported from the package, so tests can inject a fake/mock
    implementation without the real `graphiti-core` package (and its
    Neo4j/FalkorDB driver dependencies) installed -- the same convention
    `mempalace_adapter.py`'s `_PalaceProtocol` already establishes in this
    repo.
    """

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        group_id: str | None = None,
        update_communities: bool = False,
    ) -> Any: ...

    async def search(
        self, query: str, group_ids: list[str] | None = None, num_results: int = 10
    ) -> list[Any]: ...

    async def remove_episode(self, episode_uuid: str) -> None: ...

    async def close(self) -> None: ...


def _to_plain_dict(obj: object) -> dict[str, object]:
    """Best-effort conversion of a graphiti_core Pydantic model (or a test
    double standing in for one) to a plain dict, without importing
    pydantic or graphiti_core here -- this adapter only duck-types against
    whatever `model_dump()`/dict shape it's handed, matching how
    `_GraphitiProtocol` above never imports the real package either."""
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        result = model_dump()
        return result if isinstance(result, dict) else {}
    if isinstance(obj, dict):
        return obj
    try:
        return dict(vars(obj))
    except TypeError:
        return {}


def _parse_falkordb_url(url: str) -> tuple[str, int, str | None, str | None]:
    """Parse `GRAPHITI_FALKORDB_URL` into the (host, port, username,
    password) tuple `graphiti_core.driver.falkordb_driver.FalkorDriver`'s
    constructor accepts. Accepts a bare `host:port` as well as a full
    `redis://user:pass@host:port` URL -- FalkorDB is Redis-protocol
    compatible and its own docs use `redis://` URLs, but a caller who just
    writes `localhost:6379` should not be forced to add a scheme.
    """
    candidate = url if "://" in url else f"redis://{url}"
    parsed = urlsplit(candidate)
    return (parsed.hostname or "localhost", parsed.port or 6379, parsed.username, parsed.password)


class ZepGraphitiSelfHostedAdapter(MemoryBackendAdapter):
    """Adapter for self-hosted `graphiti-core` (Neo4j by default, optional
    FalkorDB). See this module's docstring for the confidence level, the
    four bug classes this adapter exists to make reachable, and exactly
    what was -- and was not -- confirmed against real source vs. a live
    instance.
    """

    name = "graphiti_selfhosted"
    env_var = "GRAPHITI_NEO4J_URI"
    supports_update = True

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        falkordb_url: str | None = None,
        update_communities: bool | None = None,
        graphiti_client: _GraphitiProtocol | None = None,
    ) -> None:
        resolved_neo4j_uri = neo4j_uri or os.environ.get(self.env_var)
        resolved_falkordb_url = falkordb_url or os.environ.get("GRAPHITI_FALKORDB_URL")
        if graphiti_client is None and not resolved_neo4j_uri and not resolved_falkordb_url:
            # Either GRAPHITI_NEO4J_URI or GRAPHITI_FALKORDB_URL configures
            # this adapter -- env_var names the primary one for the error
            # message the same way every other adapter's does, but the
            # constructor itself accepts either. See module docstring.
            raise BackendNotConfiguredError(self.name, self.env_var)
        if update_communities is None:
            update_communities = os.environ.get(
                "GRAPHITI_UPDATE_COMMUNITIES", ""
            ).strip().lower() in ("1", "true", "yes")
        self._update_communities = update_communities
        self._client = graphiti_client
        self._neo4j_uri = resolved_neo4j_uri
        self._neo4j_user = neo4j_user or os.environ.get("GRAPHITI_NEO4J_USER", DEFAULT_NEO4J_USER)
        self._neo4j_password = neo4j_password or os.environ.get("GRAPHITI_NEO4J_PASSWORD")
        self._falkordb_url = resolved_falkordb_url

    def _get_client(self) -> _GraphitiProtocol:
        if self._client is not None:
            return self._client
        try:
            from graphiti_core import Graphiti  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendAPIError(
                self.name,
                "the `graphiti-core` package is not installed. Install it "
                "with `pip install graphiti-core` (add the `[falkordb]` "
                "extra for FalkorDB support). See docs/methodology.md for "
                "this adapter's confidence level and what was confirmed "
                "against the real graphiti_core source vs. a live instance.",
            ) from exc

        if self._falkordb_url:
            try:
                from graphiti_core.driver.falkordb_driver import (  # type: ignore[import-not-found]
                    FalkorDriver,
                )
            except ImportError as exc:
                raise BackendAPIError(
                    self.name,
                    "graphiti-core is installed without FalkorDB support. "
                    "Install `pip install graphiti-core[falkordb]`, or unset "
                    "GRAPHITI_FALKORDB_URL and configure GRAPHITI_NEO4J_URI "
                    "instead.",
                ) from exc
            host, port, username, password = _parse_falkordb_url(self._falkordb_url)
            driver = FalkorDriver(host=host, port=port, username=username, password=password)
            self._client = Graphiti(graph_driver=driver)
        else:
            self._client = Graphiti(
                uri=self._neo4j_uri, user=self._neo4j_user, password=self._neo4j_password
            )
        return self._client

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
        *,
        verify: bool = False,
    ) -> StoreResult:
        """Store an episode via `Graphiti.add_episode()`.

        `mode` is a no-op, same convention as every other adapter (see
        `MemoryBackendAdapter.supported_modes`). `metadata` is also
        accepted-and-ignored: graphiti-core's real `add_episode()`
        signature (confirmed against `graphiti_core/graphiti.py` on
        `main`, 2026-07-16) has no generic key/value metadata parameter --
        unlike Mem0/MemPalace, there is nothing to thread it into without
        fabricating a parameter the real API doesn't accept. This is
        stated here explicitly rather than silently dropping the value
        with no comment.
        """
        del mode
        del metadata
        timer = self._timed()
        client = self._get_client()
        try:
            result = asyncio.run(
                client.add_episode(
                    name=f"memtrust-{session_id}",
                    episode_body=content,
                    source_description="memtrust-eval",
                    reference_time=datetime.now(UTC),
                    group_id=session_id,
                    update_communities=self._update_communities,
                )
            )
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc

        result_dict = _to_plain_dict(result)
        episode = getattr(result, "episode", None) if not isinstance(result, dict) else None
        if episode is not None:
            episode_uuid = str(getattr(episode, "uuid", "") or "")
        else:
            episode_field = result_dict.get("episode")
            episode_uuid = (
                str(episode_field.get("uuid", "")) if isinstance(episode_field, dict) else ""
            )

        store_result = StoreResult(
            memory_id=episode_uuid, latency_ms=timer.elapsed_ms(), raw=result_dict
        )
        if verify:
            store_result.verified = self.verify_store(store_result, session_id, content)
        return store_result

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        """Search via `Graphiti.search()`, which returns `list[EntityEdge]`
        directly (confirmed against `graphiti_core/graphiti.py` on `main`,
        2026-07-16) -- unlike the hosted `ZepGraphitiAdapter`, there is no
        REST envelope to unwrap.

        Note on getzep/graphiti#1302 (`lucene_sanitize`): the `query`
        string passed here reaches graphiti-core's own internal sanitizer
        before it ever hits the database -- this adapter has no surface to
        intercept or bypass that. Fixture cases containing uppercase
        O/R/N/T/A/D (tests/fixtures/contradiction_cases.json) are what let
        an eval run against a *live* self-hosted instance demonstrate the
        BM25 ranking degradation; against the mocked test double used in
        this repo's own test suite, the sanitizer never runs at all -- see
        module docstring's "What this adapter does NOT prove."
        """
        del mode
        timer = self._timed()
        client = self._get_client()
        try:
            edges = asyncio.run(client.search(query, group_ids=[session_id], num_results=top_k))
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc

        records: list[MemoryRecord] = []
        any_invalidated = False
        for edge in edges:
            edge_dict = _to_plain_dict(edge)
            invalid_at = edge_dict.get("invalid_at")
            if invalid_at:
                any_invalidated = True
            created_at = edge_dict.get("valid_at") or edge_dict.get("created_at")
            attributes = edge_dict.get("attributes")
            records.append(
                MemoryRecord(
                    memory_id=str(edge_dict.get("uuid", "")),
                    content=str(edge_dict.get("fact", "")),
                    score=None,
                    created_at=str(created_at) if created_at else None,
                    metadata={"invalid_at": str(invalid_at)} if invalid_at else {},
                    attributes=dict(attributes) if isinstance(attributes, dict) else {},
                    raw=edge_dict,
                )
            )

        # graphiti-core's bi-temporal invalid_at is the same documented
        # signal ZepGraphitiAdapter (hosted) reads -- see that module's
        # docstring for the citation. No invalidated edge in the result
        # set does not prove no contradiction occurred, only that this
        # call alone cannot tell FLAGGED from an invisible resolution --
        # same disambiguation note as the hosted adapter.
        conflict_signal = ConflictSignal.FLAGGED if any_invalidated else ConflictSignal.SERVED_STALE
        return QueryResult(
            records=records,
            conflict_signal=conflict_signal,
            latency_ms=timer.elapsed_ms(),
            raw={"edges": [r.raw for r in records]},
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        # graphiti-core has no separate "update" verb, same as hosted Zep
        # Cloud -- a contradicting fact is submitted through add_episode()
        # again and the extraction pipeline resolves the contradiction
        # itself. See ZepGraphitiAdapter.update() for the identical
        # reasoning against the hosted API.
        timer = self._timed()
        result = self.store(session_id, content)
        return UpdateResult(
            memory_id=result.memory_id,
            acknowledged=True,
            latency_ms=timer.elapsed_ms(),
            raw=result.raw,
        )

    def delete(self, memory_id: str) -> DeleteResult:
        """Delete via `Graphiti.remove_episode(episode_uuid)`, confirmed
        against `graphiti_core/graphiti.py` on `main` (2026-07-16): it
        looks up the episode, deletes entity edges/nodes exclusively
        mentioned by it, and deletes the episode itself.
        """
        timer = self._timed()
        client = self._get_client()
        try:
            asyncio.run(client.remove_episode(memory_id))
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=timer.elapsed_ms())

    def close(self) -> None:
        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            asyncio.run(close())
