"""Adapter for OpenViking (https://github.com/volcengine/OpenViking).

Confidence: MEDIUM on architecture, LOW on exact memory-write/query
endpoint paths.

OpenViking (ByteDance/Volcengine) is confirmed to organize agent context
as a virtual filesystem addressed by `viking://` URIs, with a REST server
mode listening on port 1933 and documented Python client classes
(`OpenViking` for embedded/local mode, `SyncHTTPClient`/`AsyncHTTPClient`
for remote server mode). The fetched API docs during this build covered
resource/skill ingestion (`add_resource`, `add_skill`) in detail but did
not surface a confirmed method name for writing or querying a
conversational *memory* entry specifically (as opposed to a resource or
skill file) -- OpenViking's memory layer is described as an automatic,
session-derived extraction process rather than a direct "store this fact"
call in the documentation surfaced here.

This adapter is written best-effort against the confirmed `viking://`
filesystem paradigm: store() writes a file under a session-scoped path,
query() greps/searches that path, update() overwrites the file at the
same path. If OpenViking's real memory API differs materially from a
filesystem write/search, this adapter's behavior should be corrected by
whoever verifies it against a live instance -- flagged explicitly here
and in docs/methodology.md rather than presented as confirmed.

Gated on OPENVIKING_API_KEY, matching the project's hosted "OpenViking
Studio" offering; OPENVIKING_BASE_URL may override the default host to
point at a self-hosted server instead.
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

DEFAULT_BASE_URL = "http://localhost:1933"


class OpenVikingAdapter(MemoryBackendAdapter):
    name = "openviking"
    env_var = "OPENVIKING_API_KEY"
    supports_update = True

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        api_key = os.environ.get(self.env_var)
        if not api_key:
            raise BackendNotConfiguredError(self.name, self.env_var)
        resolved_base_url = base_url or os.environ.get("OPENVIKING_BASE_URL") or DEFAULT_BASE_URL
        self._http = httpx.Client(
            base_url=resolved_base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def _path(self, session_id: str, key: str) -> str:
        return f"memory/{session_id}/{key}"

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        timer = self._timed()
        memory_key = _slugify(content)
        payload: dict[str, object] = {
            "path": f"viking://{self._path(session_id, memory_key)}",
            "content": content,
            "metadata": metadata or {},
        }
        try:
            resp = self._http.post("/v1/fs/write", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        memory_id = str(data.get("path", payload["path"]))
        return StoreResult(memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=data)

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        timer = self._timed()
        payload = {"path_prefix": f"viking://memory/{session_id}/", "query": query, "limit": top_k}
        try:
            resp = self._http.post("/v1/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc

        items = data.get("results", data.get("matches", []))
        records = [
            MemoryRecord(
                memory_id=str(item.get("path", "")),
                content=str(item.get("content", item.get("snippet", ""))),
                score=item.get("score"),
                created_at=item.get("updated_at"),
                metadata=item.get("metadata") or {},
                raw=item,
            )
            for item in items
        ]
        # OpenViking's filesystem paradigm has no documented conflict-
        # marker field surfaced in this build's research pass -- a
        # contradicting fact written to the same path simply overwrites
        # the file's content with no version history exposed through the
        # search API as documented. Recorded as NOT_APPLICABLE rather than
        # guessed as FLAGGED or SILENT_OVERWRITE; see docs/methodology.md.
        return QueryResult(
            records=records,
            conflict_signal=ConflictSignal.NOT_APPLICABLE,
            latency_ms=timer.elapsed_ms(),
            raw=data,
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        timer = self._timed()
        payload = {"path": memory_id, "content": content}
        try:
            resp = self._http.post("/v1/fs/write", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        return UpdateResult(
            memory_id=memory_id, acknowledged=True, latency_ms=timer.elapsed_ms(), raw=data
        )

    def delete(self, memory_id: str) -> DeleteResult:
        # Best-effort reconstruction against the same viking:// filesystem
        # paradigm store()/update() above are written against: this build's
        # research pass did not surface a confirmed "delete a memory"
        # method name (see module docstring), so this targets the plain
        # filesystem-delete symmetrical with /v1/fs/write. Whoever verifies
        # this adapter against a live instance should correct the path if
        # OpenViking's real client exposes something else.
        timer = self._timed()
        payload = {"path": memory_id}
        try:
            resp = self._http.post("/v1/fs/delete", json=payload)
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        return DeleteResult(
            success=True, memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=data
        )

    def close(self) -> None:
        self._http.close()


def _slugify(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
