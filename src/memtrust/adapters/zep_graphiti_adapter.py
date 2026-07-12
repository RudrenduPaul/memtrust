"""Adapter for Zep / Graphiti (https://getzep.com, https://github.com/getzep/graphiti).

Confidence: MEDIUM-HIGH on behavior, MEDIUM on exact wire format.

Graphiti is the open-source temporal-knowledge-graph engine that powers
Zep. Its documented core operations are `add_episode()` (ingest a unit of
conversation, which triggers entity/relationship extraction, dedup, and
contradiction detection against the existing graph) and `search()`
(hybrid semantic + BM25 + graph-traversal retrieval). Confirmed via
Graphiti's own docs and DeepWiki: when a new episode contradicts an
existing fact, Graphiti stamps the old graph edge `invalid_at` rather
than deleting it -- old and new facts both remain inspectable, bi-
temporally. That is a real, documented product behavior, not a memtrust
assumption, and it maps directly onto ConflictSignal.FLAGGED when the
query response surfaces an edge with a non-null `invalid_at`.

This adapter targets Zep Cloud's hosted API (`ZEP_API_KEY`), which wraps
Graphiti, rather than a self-hosted `graphiti-core` + Neo4j/FalkorDB
deployment. Self-hosted Graphiti has no single API key to gate on -- it
requires a running graph database plus LLM credentials, which is
infrastructure memtrust cannot assume exists in a fresh clone or CI run.
The hosted-API choice keeps this adapter consistent with the rest of the
harness's "one env var, or SKIPPED" contract. See docs/methodology.md.
Exact REST paths below are a best-effort reconstruction; the fact-
invalidation behavior they read from is the documented part.
"""

from __future__ import annotations

import os

import httpx

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

DEFAULT_BASE_URL = "https://api.getzep.com"


class ZepGraphitiAdapter(MemoryBackendAdapter):
    name = "zep"
    env_var = "ZEP_API_KEY"
    supports_update = True

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        api_key = os.environ.get(self.env_var)
        if not api_key:
            raise BackendNotConfiguredError(self.name, self.env_var)
        self._http = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Api-Key {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        timer = self._timed()
        payload: dict[str, object] = {
            "group_id": session_id,
            "data": content,
            "type": "text",
            "source_description": "memtrust-eval",
        }
        if metadata:
            payload["metadata"] = metadata
        try:
            resp = self._http.post("/graph/episodes", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        episode_id = str(data.get("uuid", data.get("episode_id", "")))
        return StoreResult(memory_id=episode_id, latency_ms=timer.elapsed_ms(), raw=data)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        timer = self._timed()
        payload = {"group_id": session_id, "query": query, "limit": top_k}
        try:
            resp = self._http.post("/graph/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc

        edges = data.get("edges", data.get("facts", []))
        records: list[MemoryRecord] = []
        any_invalidated = False
        for edge in edges:
            invalid_at = edge.get("invalid_at")
            if invalid_at:
                any_invalidated = True
            records.append(
                MemoryRecord(
                    memory_id=str(edge.get("uuid", "")),
                    content=str(edge.get("fact", edge.get("name", ""))),
                    score=edge.get("score"),
                    created_at=edge.get("valid_at") or edge.get("created_at"),
                    metadata={"invalid_at": str(invalid_at)} if invalid_at else {},
                    raw=edge,
                )
            )

        # A returned edge stamped invalid_at, alongside a live edge for the
        # same fact slot, is Graphiti's documented mechanism for surfacing
        # a superseded fact -- that is a FLAGGED signal by definition (the
        # contradiction is visible in the response, not hidden). If no
        # edge in the result set carries invalid_at, the harness cannot
        # tell from this call alone whether that means "no contradiction
        # occurred" or "the backend resolved it invisibly" -- the eval
        # layer (evals/contradiction.py) disambiguates using the known
        # eval fixture, not this adapter.
        conflict_signal = ConflictSignal.FLAGGED if any_invalidated else ConflictSignal.SERVED_STALE
        return QueryResult(
            records=records,
            conflict_signal=conflict_signal,
            latency_ms=timer.elapsed_ms(),
            raw=data,
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        # Graphiti has no separate "update" verb in its documented API --
        # a contradicting fact is submitted the same way a new fact is,
        # through add_episode/store(), and the graph's own extraction
        # pipeline resolves the contradiction (invalidating the old edge)
        # during ingestion. This method exists to satisfy the shared
        # interface and to make that behavior explicit and testable,
        # rather than silently aliasing to store() without comment.
        timer = self._timed()
        result = self.store(session_id, content)
        return UpdateResult(
            memory_id=result.memory_id,
            acknowledged=True,
            latency_ms=timer.elapsed_ms(),
            raw=result.raw,
        )

    def delete(self, memory_id: str) -> DeleteResult:
        # Best-effort reconstruction, same confidence level as store()'s
        # /graph/episodes path above: Zep's hosted API is not confirmed
        # here to expose a documented "delete episode" verb distinct from
        # its bi-temporal invalidate-on-contradiction behavior (see the
        # module docstring). This targets the REST path symmetrical with
        # store()'s POST /graph/episodes -- DELETE /graph/episodes/{uuid}
        # -- and should be corrected by whoever verifies it against a
        # live Zep instance if the real surface differs.
        timer = self._timed()
        try:
            resp = self._http.delete(f"/graph/episodes/{memory_id}")
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        return DeleteResult(
            success=True, memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=data
        )

    def close(self) -> None:
        self._http.close()
