"""Adapters for Mem0 (https://mem0.ai): hosted Platform API and self-hosted
OSS server.

`Mem0Adapter` (hosted, this module's original adapter) and
`Mem0SelfHostedAdapter` (below) are deliberately two separate classes, not
one adapter with a deployment flag, following the precedent this codebase
already set for Zep/Graphiti in docs/methodology.md ("If self-hosted
Graphiti support is wanted later, it should be a second adapter ... with
its own configuration story, not a silent branch inside this one"). The
two Mem0 deployments have different base URLs, different route prefixes,
different auth defaults, and different confidence levels -- collapsing
them into one class with an if/else would hide that from callers and from
the confidence table in docs/methodology.md.

## Mem0Adapter (hosted Platform API)

Confidence: HIGH. Mem0 publishes a documented Python SDK (`mem0ai` on PyPI)
built around `MemoryClient(api_key=...)` with `.add()`, `.search()`, and
`.update()` methods, and reads `MEM0_API_KEY` from the environment when no
key is passed explicitly (docs.mem0.ai/platform/quickstart, June 2026 SDK
v2.0.8 release notes). This adapter talks to the same REST surface the
official SDK wraps, using httpx directly rather than depending on the
`mem0ai` package, to keep memtrust's own dependency tree small and to keep
every HTTP call independently mockable in tests without a vendor SDK
installed. Endpoint paths below are best-effort reconstructions of the
REST API the documented SDK calls sit on top of -- see
docs/methodology.md for what is confirmed vs. inferred.

Mem0's own memory pipeline decides, per stored fact, whether to ADD a new
memory, UPDATE an existing one, or leave it alone -- this is documented
Mem0 behavior, not a memtrust assumption, and it is exactly the kind of
vendor-side conflict resolution the contradiction-detection eval is built
to observe rather than short-circuit.

## Mem0SelfHostedAdapter (self-hosted OSS `server/` FastAPI wrapper)

Confidence: MEDIUM-HIGH on route shape and request/response models, LOW on
end-to-end live behavior. Unlike the hosted adapter above (reconstructed
from SDK docs) or MemPalace/OpenViking below (reconstructed from product
docs that could not be fully fetched), this adapter's route shape was
confirmed by fetching the *actual source code* of `server/main.py` and
`server/auth.py` from the `mem0ai/mem0` GitHub repository's `main` branch
directly (`raw.githubusercontent.com/mem0ai/mem0/main/server/...`) during
this build, 2026-07-11. That is stronger evidence than a documentation
page, but it is still: (a) a snapshot of an unpinned `main` branch that
can drift out from under any specific self-hosted deployment's actual
server version, and (b) never exercised against a live running instance
in this environment -- no self-hosted mem0 server was started and hit
with a real HTTP request during this build. Confirmed from that source
read:

  * Routes are unprefixed -- `POST /memories`, `GET /memories`,
    `GET /memories/{id}`, `PUT /memories/{id}`, `DELETE /memories/{id}`,
    `DELETE /memories`, `POST /search`, `GET /memories/{id}/history`,
    `POST /reset` -- there is no `/v1/...` prefix anywhere in this
    router, unlike the hosted Platform API.
  * `POST /memories` request body (`MemoryCreate`): `messages`, `user_id`,
    `agent_id`, `run_id`, `metadata`, `expiration_date`, `infer`,
    `memory_type`, `prompt`. The handler 400s unless at least one of
    `user_id`/`agent_id`/`run_id` is set.
  * `POST /search` request body (`SearchRequest`): `query`, `filters`
    (a dict), `top_k`, `threshold`, `explain`, `show_expired`, plus
    deprecated top-level `user_id`/`run_id`/`agent_id` fields that the
    handler merges into `filters` -- but only `if entity_val:`, a
    *truthy* check, not `is not None`. That means the self-hosted
    server's own deprecated-field merge path silently drops a
    deliberately-empty-string `run_id`/`agent_id` instead of scoping the
    filter to "must be empty" -- this is the concrete, source-confirmed
    shape of the entity-id filter-scoping issue (mem0ai/mem0#5973) that
    this adapter exists to make reachable. This adapter always sends
    `run_id`/`agent_id` inside the `filters` dict directly (never through
    the deprecated top-level fields) using an `is not None` check, so a
    caller's deliberate empty string reaches the server's filter-matching
    logic intact rather than being dropped by memtrust itself -- whether
    the self-hosted `Memory.search()` implementation then handles that
    empty-string filter correctly is exactly what the eval is meant to
    observe, not something this adapter should paper over.
  * `PUT /memories/{id}` body (`MemoryUpdate`): `text`, `metadata`,
    `expiration_date`.
  * No auth by default: `verify_auth` requires a bearer JWT or
    `X-API-Key` header unless `AUTH_DISABLED` is set server-side, and
    the project's own Docker self-hosting guide (mem0.ai/blog/
    self-host-mem0-docker) ships `AUTH_DISABLED` on by default for local
    use, mapping the container to `localhost:8888`. That is why this
    adapter's required configuration is a base URL, not an API key (the
    same reasoning docs/methodology.md already gives for
    `MEMPALACE_STORAGE_PATH`), with an optional API key for deployments
    that front the server with their own auth.

Not confirmed by this build, and explicitly out of scope for this change:
the exact JSON shape `Memory.search()`/`Memory.add()` return (the FastAPI
handlers pass that dict through unmodified, so this adapter reuses the
hosted adapter's `{"results": [...]}` parsing, which is the OSS `mem0ai`
SDK's own documented response shape, but that specific field name was not
re-verified against `server/main.py` beyond what's shown above); the
`DELETE /memories` multi-filter route that mem0ai/mem0#5936/#5970
describe as truncating results (there is no `delete()` method on
`MemoryBackendAdapter` for this adapter to implement it against, and
adding one is a larger interface change than this backlog item scopes to
-- see docs/methodology.md); and the embedding-dimension-mismatch failure
mem0ai/mem0#4297 describes, which lives entirely in self-hosted vector
store configuration this adapter has no surface to trigger or observe --
merely routing eval traffic at a self-hosted instance is what makes that
bug class newly *reachable* by a memtrust user, not something this
adapter's code exercises directly.
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

DEFAULT_BASE_URL = "https://api.mem0.ai"
DEFAULT_SELFHOSTED_BASE_URL = "http://localhost:8888"


class Mem0Adapter(MemoryBackendAdapter):
    name = "mem0"
    env_var = "MEM0_API_KEY"
    supports_update = True

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        api_key = os.environ.get(self.env_var)
        if not api_key:
            raise BackendNotConfiguredError(self.name, self.env_var)
        self._http = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        timer = self._timed()
        payload: dict[str, object] = {
            "messages": [{"role": "user", "content": content}],
            "user_id": session_id,
        }
        if metadata:
            payload["metadata"] = metadata
        try:
            resp = self._http.post("/v1/memories/", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        memory_id = _extract_memory_id(data)
        return StoreResult(memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=data)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        timer = self._timed()
        payload = {"query": query, "filters": {"user_id": session_id}, "top_k": top_k}
        try:
            resp = self._http.post("/v1/memories/search/", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc

        raw_results = data.get("results", data if isinstance(data, list) else [])
        records = [
            MemoryRecord(
                memory_id=str(item.get("id", "")),
                content=str(item.get("memory", item.get("text", ""))),
                score=item.get("score"),
                created_at=item.get("created_at"),
                metadata=item.get("metadata") or {},
                raw=item,
            )
            for item in raw_results
        ]
        # Mem0's search response does not carry an explicit "this result
        # superseded a prior one" marker in the documented API surface --
        # conflict handling happens inside Mem0's own add/update pipeline,
        # invisibly to the caller. Until Mem0 documents a per-result
        # conflict marker, memtrust conservatively records SILENT_OVERWRITE
        # when exactly one record is returned for a query that the
        # contradiction eval knows should have two candidate facts, and
        # FLAGGED only if the response ever includes more than one
        # conflicting record for the same fact slot. See
        # evals/contradiction.py for how this signal is actually derived
        # per test case; this adapter only reports the raw record set.
        conflict_signal = ConflictSignal.NOT_APPLICABLE
        return QueryResult(
            records=records,
            conflict_signal=conflict_signal,
            latency_ms=timer.elapsed_ms(),
            raw=data,
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        timer = self._timed()
        try:
            resp = self._http.put(f"/v1/memories/{memory_id}/", json={"text": content})
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        return UpdateResult(
            memory_id=memory_id, acknowledged=True, latency_ms=timer.elapsed_ms(), raw=data
        )

    def delete(self, memory_id: str) -> DeleteResult:
        """Delete a memory via Mem0's documented DELETE /v1/memories/{id}/
        endpoint (docs.mem0.ai/platform/quickstart lists delete alongside
        add/search/update on the same REST surface this adapter targets).

        This is the primitive an eval needs to reproduce the real,
        merged mem0ai/mem0#5936 / #5970 bug class: a multi-entity delete
        whose client-side aggregation silently truncated to only the
        last response instead of all N. memtrust's own delete_many() in
        base.py is what constructs that N-entity scenario against this
        single-id delete() call.
        """
        timer = self._timed()
        try:
            resp = self._http.delete(f"/v1/memories/{memory_id}/")
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        return DeleteResult(
            success=True, memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=data
        )

    def close(self) -> None:
        self._http.close()


class Mem0SelfHostedAdapter(MemoryBackendAdapter):
    """Adapter for the self-hosted Mem0 OSS `server/` FastAPI wrapper.

    See this module's docstring for the confidence level and exactly what
    was, and was not, confirmed against the real `mem0ai/mem0` source.

    Unlike `Mem0Adapter`, configuration is gated on a base URL
    (`MEM0_SELFHOSTED_BASE_URL`), not an API key, because the self-hosted
    server ships with no auth by default -- the same reasoning
    docs/methodology.md gives for `MemPalaceAdapter`'s
    `MEMPALACE_STORAGE_PATH`. An optional `MEM0_SELFHOSTED_API_KEY` is
    sent as an `X-API-Key` header for deployments that do front the
    server with auth (env's `AUTH_DISABLED` unset).
    """

    name = "mem0_selfhosted"
    env_var = "MEM0_SELFHOSTED_BASE_URL"
    supports_update = True

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        resolved_base_url = base_url or os.environ.get(self.env_var)
        if not resolved_base_url:
            raise BackendNotConfiguredError(self.name, self.env_var)
        resolved_api_key = api_key or os.environ.get("MEM0_SELFHOSTED_API_KEY")
        headers = {"Content-Type": "application/json"}
        if resolved_api_key:
            headers["X-API-Key"] = resolved_api_key
        self._http = httpx.Client(base_url=resolved_base_url, headers=headers, timeout=timeout)

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        timer = self._timed()
        payload: dict[str, object] = {
            "messages": [{"role": "user", "content": content}],
            "user_id": session_id,
        }
        if metadata:
            payload["metadata"] = metadata
        try:
            resp = self._http.post("/memories", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        memory_id = _extract_memory_id(data)
        return StoreResult(memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=data)

    def query(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        run_id: str | None = None,
        agent_id: str | None = None,
        threshold: float | None = None,
    ) -> QueryResult:
        """Search self-hosted Mem0 memories.

        `run_id`/`agent_id` default to `None` (omitted from the filter
        entirely), but a caller may deliberately pass `""` -- this
        adapter always includes the key in `filters` when the value is
        not `None` (an `is not None` check, not a truthy check), so an
        empty string is preserved through to the server rather than
        silently dropped the way the server's own deprecated top-level
        `user_id`/`run_id`/`agent_id` merge path drops falsy values (see
        module docstring, mem0ai/mem0#5973). This is what lets the
        contradiction/entity-scoping evals construct the exact filter
        shape that issue describes and observe how the self-hosted
        server actually responds to it, instead of memtrust masking the
        input before it ever reaches the vendor.

        `threshold` is passed straight through to `SearchRequest.threshold`
        -- confirmed present on the self-hosted server's search model --
        which is what makes mem0ai/mem0#4453 (threshold inversion)
        reachable through this adapter; `Mem0Adapter` (hosted) has no
        equivalent parameter today.
        """
        timer = self._timed()
        filters: dict[str, str] = {"user_id": session_id}
        if run_id is not None:
            filters["run_id"] = run_id
        if agent_id is not None:
            filters["agent_id"] = agent_id
        payload: dict[str, object] = {"query": query, "filters": filters, "top_k": top_k}
        if threshold is not None:
            payload["threshold"] = threshold
        try:
            resp = self._http.post("/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc

        raw_results = data.get("results", data if isinstance(data, list) else [])
        records = [
            MemoryRecord(
                memory_id=str(item.get("id", "")),
                content=str(item.get("memory", item.get("text", ""))),
                score=item.get("score"),
                created_at=item.get("created_at"),
                metadata=item.get("metadata") or {},
                raw=item,
            )
            for item in raw_results
        ]
        # Same reasoning as Mem0Adapter.query() above: the self-hosted
        # search response has no documented per-result conflict marker
        # either, so this is recorded as NOT_APPLICABLE rather than
        # guessed. See evals/contradiction.py for how the eval derives
        # its own signal from the raw record set regardless.
        conflict_signal = ConflictSignal.NOT_APPLICABLE
        return QueryResult(
            records=records,
            conflict_signal=conflict_signal,
            latency_ms=timer.elapsed_ms(),
            raw=data,
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        timer = self._timed()
        try:
            resp = self._http.put(f"/memories/{memory_id}", json={"text": content})
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        return UpdateResult(
            memory_id=memory_id, acknowledged=True, latency_ms=timer.elapsed_ms(), raw=data
        )

    def delete(self, memory_id: str) -> DeleteResult:
        """Delete a memory via the self-hosted server's unprefixed
        `DELETE /memories/{id}` route -- same reasoning as `Mem0Adapter.delete()`
        above, against the self-hosted route shape instead of the hosted
        `/v1/...` one.
        """
        timer = self._timed()
        try:
            resp = self._http.delete(f"/memories/{memory_id}")
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        return DeleteResult(
            success=True, memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=data
        )

    def close(self) -> None:
        self._http.close()


def _extract_memory_id(data: object) -> str:
    if isinstance(data, dict):
        if "id" in data:
            return str(data["id"])
        results = data.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return str(results[0].get("id", ""))
    return ""
