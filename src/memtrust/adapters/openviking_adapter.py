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

list_resource_paths()/trigger_resync() (supports_resource_sync = True)
are written against the same `viking://` filesystem paradigm, guessing a
`/v1/fs/list` listing endpoint and a `/v1/fs/resync` resync-trigger
endpoint by analogy with the confirmed `/v1/fs/write` and `/v1/search`
paths above. Neither endpoint was confirmed in this build's research
pass -- confidence is LOW on the exact paths, same flag as store's
memory-write endpoint above. This is the capability that lets the
resource-sync-safety eval (evals/resource_sync_safety.py) exercise
OpenViking at all; it exists specifically because OpenViking's Feishu
resync mechanism has a reported data-loss bug (a resync silently
deleting user-owned files the ingestion watcher did not generate --
volcengine/OpenViking#3029) that the store/query/update model alone
cannot observe.

store() honors a `resource_path` metadata key (e.g.
"entities/people/jordan-lee.md") when the caller supplies one, writing
to that real nested `viking://` path instead of always falling back to
the flat `memory/{session_id}/{sha256(content)[:16]}` single-level
filename. Without `resource_path` in metadata, the flat-hash behavior
is unchanged (backward compatible for every existing caller). This
closes a structural gap found validating volcengine/OpenViking#1703
(index_resource() in OpenViking's embedding_utils.py skipped every
subdirectory during reindex, so nested-directory content was never
vectorized and searches over it silently returned nothing): before this
change, memtrust's own store() never actually constructed a real nested
directory tree against OpenViking, so a directory-indexing bug like
#1703 was structurally unreachable by this harness regardless of how
good the eval classification logic was. list_resource_paths() now does
a real recursive tree walk (bounded by an optional `max_depth`) so
nested paths a listing response reports as directories are actually
descended into and returned as leaf file paths, not just whatever a
single flat response happened to contain. See docs/methodology.md for
the honest scope of what this closes: it makes the #1703 bug class
reachable by this harness's storage layer, it does not reproduce
OpenViking's real server-side reindex bug without a live instance.

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
    supports_resource_sync = True

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
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        # OpenViking's filesystem paradigm has no documented operating-mode
        # variant -- accepted and ignored (no-op); see
        # MemoryBackendAdapter.supported_modes.
        del mode
        timer = self._timed()
        metadata = metadata or {}
        # A caller (e.g. evals/resource_sync_safety.py) that knows the real
        # nested path a piece of content belongs under -- "entities/people/
        # jordan-lee.md", "preferences/user-482/notification-settings.md" --
        # passes it via the `resource_path` metadata key, and store()
        # writes to that real path instead of flattening every write to a
        # single-level content-hash filename. Falls back to the flat hash
        # when no resource_path is supplied, so every existing caller that
        # never sets this key keeps its current behavior unchanged. See the
        # module docstring for why this matters (volcengine/OpenViking#1703).
        resource_path = metadata.get("resource_path")
        memory_key = resource_path.strip("/") if resource_path else _slugify(content)
        payload: dict[str, object] = {
            "path": f"viking://{self._path(session_id, memory_key)}",
            "content": content,
            "metadata": metadata,
        }
        try:
            resp = self._http.post("/v1/fs/write", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        memory_id = str(data.get("path", payload["path"]))
        return StoreResult(memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=data)

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        del mode  # no-op, see store() above
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

    def list_resource_paths(self, prefix: str, max_depth: int = 8) -> list[str]:
        """Recursively list every leaf file path under `prefix`.

        A single `/v1/fs/list` response is not assumed to already contain
        every nested file -- each returned entry is inspected, and any
        entry the listing endpoint reports as a directory (a dict with
        `is_dir`/`type: "directory"`, or a bare path string ending in `/`)
        is itself descended into with a follow-up `/v1/fs/list` call,
        rather than being dropped or treated as a leaf. `max_depth` bounds
        that recursion (default 8) so a misbehaving or cyclic listing
        response cannot recurse unboundedly.

        This exists because a flat, non-recursive read of whatever the
        first response happened to contain would make nested content
        structurally invisible to this harness regardless of what
        store() actually wrote -- see the module docstring and
        volcengine/OpenViking#1703.
        """
        return self._list_resource_paths_recursive(prefix, depth=0, max_depth=max_depth)

    def _list_resource_paths_recursive(self, prefix: str, depth: int, max_depth: int) -> list[str]:
        payload = {"path_prefix": f"viking://{prefix}"}
        try:
            resp = self._http.post("/v1/fs/list", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        entries = data.get("paths", data.get("entries", []))
        paths: list[str] = []
        for entry in entries:
            if isinstance(entry, dict):
                entry_path = str(entry.get("path", ""))
                is_dir = bool(entry.get("is_dir") or entry.get("type") == "directory")
            else:
                entry_path = str(entry)
                is_dir = entry_path.endswith("/")
            if not entry_path:
                continue
            if is_dir:
                if depth >= max_depth:
                    continue
                sub_prefix = entry_path
                if sub_prefix.startswith("viking://"):
                    sub_prefix = sub_prefix[len("viking://") :]
                sub_prefix = sub_prefix.rstrip("/")
                if not sub_prefix or sub_prefix == prefix.rstrip("/"):
                    # Guard against a listing entry that just echoes the
                    # queried prefix back as a "directory" -- recursing on
                    # it would loop forever at constant depth.
                    continue
                paths.extend(self._list_resource_paths_recursive(sub_prefix, depth + 1, max_depth))
            else:
                paths.append(entry_path)
        return paths

    def trigger_resync(self, prefix: str) -> None:
        payload = {"path_prefix": f"viking://{prefix}"}
        try:
            resp = self._http.post("/v1/fs/resync", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc

    def close(self) -> None:
        self._http.close()


def _slugify(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
