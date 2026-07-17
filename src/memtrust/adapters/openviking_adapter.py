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

On volcengine/OpenViking#1523 (contributor A0nameless0man: an embedder
migration silently degrades search quality mid-migration -- switching
embedding models overwrites vectors in place with no dimension/model
validation): the `/v1/search` response shape documented and confirmed
during this build (see the top of this docstring) surfaces `path`,
`content`/`snippet`, `score`, `updated_at`, and `metadata` per result --
nothing that identifies which embedding model or vector dimensionality
produced a given record. `query()` above therefore cannot report
`MemoryRecord.embedding_model`/`embedding_dims` (see adapters/base.py) for
real OpenViking responses; both stay at their default `None`. This is why
#1523 is scored at the harness level instead, against fake adapters
engineered to reproduce its exact bug shape -- see
evals/embedding_drift.py and docs/methodology.md for the eval and its
honest scope.

`supports_crash_recovery_simulation` is NOT set on this adapter (stays at
the base class default, False). volcengine/OpenViking#2644 (contributor
yeyitech) reports a local vectordb backend's `_recover()` silently
skipping index rebuild on server-process restart when index files are
missing but store data exists -- a real, cited bug this project's
crash-recovery eval (evals/crash_recovery.py) exists to catch. This
adapter is a pure HTTP client with no ability to start, kill, or restart
a live OpenViking server process, and no confirmed endpoint that reads
raw stored data bypassing the search/index layer `query()` above goes
through -- both are required for that eval to run against a real
backend. Until this adapter (or a future one) gains real
process-lifecycle control, the crash-recovery eval only ever runs
against a purpose-built fake adapter that models #2644's shape; see
evals/crash_recovery.py's module docstring and docs/methodology.md for
exactly what that does and does not prove.

`supports_stats = True`: get_stats() hits `GET /api/v1/stats/memories`
and reads its `total_memories` field. Confidence HIGH on this specific
endpoint and response shape -- both are quoted verbatim in volcengine/
OpenViking#1255's bug report (contributor SeeYangZhi), which is also the
motivating case: the endpoint silently returns an all-zero count even
when filesystem listing and `/v1/search` both independently confirm real
memories exist. See evals/stats_accuracy.py, the eval this feeds.
"""

from __future__ import annotations

import json
import os

import httpx

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
    StatsResult,
    StoreResult,
    UpdateResult,
)

DEFAULT_BASE_URL = "http://localhost:1933"

#: Approximate reranker batch-token budget a typical local reranker (e.g.
#: qwen3-reranker-0.6B) enforces, per hhspiny's own measured report on
#: volcengine/OpenViking#2880 (a `QMD_RERANK_CONTEXT_SIZE`-style default).
#: OpenViking's own rerank-provider config is not confirmed to expose this
#: value through any endpoint this adapter calls, so this is a
#: conservative, documented approximation -- not a value read from a live
#: OpenViking config. See `_rerank_fallback_risk()` below and
#: RankingSignal.RERANK_FALLBACK in base.py.
MAX_RERANK_TOKENS = 4096


def _error_detail(exc: httpx.HTTPError) -> str:
    """Build a BackendAPIError detail that includes the real response body
    when one is available, not just the generic status-line `str(exc)`.

    Motivating case: volcengine/OpenViking#1227, where a server-side
    Pydantic validation error (e.g. `"id" extra_forbidden`) was silently
    swallowed down to a useless status-line-only message like
    "Client error '400 Bad Request' for url ..." -- the actual validation
    detail explaining WHAT failed was discarded entirely. `exc.response`
    is `None` for request-level failures (a connection error, a timeout)
    that never got a response back at all, not just for
    `httpx.HTTPStatusError` -- handled explicitly rather than assumed
    present, since not every `httpx.HTTPError` subclass carries one.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    body = response.text
    if not body:
        return str(exc)
    return f"{exc} -- response body: {body}"


def _estimate_tokens(text: str) -> int:
    """Cheap, tokenizer-free token-count approximation (~4 chars/token, the
    same rough heuristic widely used for English text when no real
    tokenizer is available). Not a substitute for the reranker's own
    tokenizer -- see MAX_RERANK_TOKENS's docstring for the honesty caveat
    this approximation carries."""
    return max(1, len(text) // 4)


def _rerank_fallback_risk(records: list[MemoryRecord]) -> bool:
    """Whether this response's own candidate set carries the same input
    shape volcengine/OpenViking#1737 (an empty-string document) or
    #2739/#2880 (a batch exceeding the reranker's real token budget)
    report as silently degrading to a raw vector-score fallback with no
    caller-visible signal.

    This is a heuristic classification computed entirely from records this
    adapter already has in hand -- it cannot intercept OpenViking's actual
    server-side rerank-batch construction, so it flags the SHAPE a
    response's candidates carry (an empty document, or a total estimated
    token count over budget), not confirmed proof this specific live call
    fell back server-side. See RankingSignal.RERANK_FALLBACK in base.py
    for the full honesty caveat and upstream-fix citations.
    """
    if any(record.content == "" for record in records):
        return True
    total_tokens = sum(_estimate_tokens(record.content) for record in records)
    return total_tokens > MAX_RERANK_TOKENS


class OpenVikingAdapter(MemoryBackendAdapter):
    name = "openviking"
    env_var = "OPENVIKING_API_KEY"
    supports_update = True
    supports_resource_sync = True
    supports_stats = True
    supports_prefix_delete = True

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
            raise BackendAPIError(self.name, _error_detail(exc)) from exc
        except json.JSONDecodeError as exc:
            # volcengine/OpenViking#2966: a legacy uint16-length-truncated
            # record's `fields` JSON crashes internal delta-list conversion
            # (_convert_delta_list_for_index -> convert_fields_for_index ->
            # a bare json.loads(fields_json)) on the write/upsert path this
            # call exercises when it overwrites an existing key. This
            # adapter's own `resp.json()` call would otherwise let that raw
            # JSONDecodeError escape as an unclassified bare exception,
            # violating the "always raise BackendAPIError" contract every
            # adapter must honor -- see CrashSignal
            # .LEGACY_CORRUPT_RECORD_UNDELETABLE in base.py for the full
            # honesty caveat on what this classifies.
            raise BackendAPIError(
                self.name,
                f"response body is not valid JSON: {exc}",
                crash_signal=CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE,
            ) from exc
        memory_id = str(data.get("path", payload["path"]))
        # See ExtractionSignal in base.py -- volcengine/OpenViking#2751: a
        # store() call that hits OpenViking's own session-based extraction
        # pipeline can complete without raising and still report zero
        # memories actually extracted (the OpenAI VLM backend's hardcoded
        # max_tokens=32768 exceeds gpt-4o-mini's real 16384 cap; the
        # resulting API 400 gets swallowed inside compressor_v2, and the
        # write commit still returns 200/accepted with total_memories
        # staying 0, silent). This adapter's store() targets the
        # /v1/fs/write filesystem endpoint (see module docstring for the
        # confidence caveat on the real memory-extraction endpoint), which
        # is not confirmed to echo back a `total_memories`/
        # `memories_extracted` count on every deployment/version -- when
        # present and explicitly 0, that is the identical "response looks
        # like a normal 200 but nothing was extracted" shape #2751 reports.
        # Absent entirely, this defaults to FACTS_EXTRACTED whenever a
        # usable path/memory_id came back, the same "presence of an id is
        # a successful store" assumption store() already made before this
        # change -- see docs/methodology.md for the confidence caveat.
        total_memories = data.get("total_memories", data.get("memories_extracted"))
        if total_memories is not None and total_memories == 0:
            extraction_signal = ExtractionSignal.EMPTY_EXTRACTION
        elif memory_id:
            extraction_signal = ExtractionSignal.FACTS_EXTRACTED
        else:
            extraction_signal = ExtractionSignal.EMPTY_EXTRACTION
        return StoreResult(
            memory_id=memory_id,
            latency_ms=timer.elapsed_ms(),
            raw=data,
            extraction_signal=extraction_signal,
        )

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
            raise BackendAPIError(self.name, _error_detail(exc)) from exc

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
        #
        # Rerank-awareness (volcengine/OpenViking#1737, #2739, #2880): a
        # multi-candidate response carrying an empty-string document or an
        # oversized total token budget matches the shape both cited bug
        # reports describe as silently falling back to raw vector scores
        # -- see _rerank_fallback_risk()'s and RankingSignal
        # .RERANK_FALLBACK's own docstrings for the full honesty caveat
        # (both bugs are already fixed upstream; this flags the SHAPE, not
        # confirmed live server-side fallback).
        ranking_signal = (
            RankingSignal.RERANK_FALLBACK
            if len(records) > 1 and _rerank_fallback_risk(records)
            else RankingSignal.NOT_APPLICABLE
        )
        return QueryResult(
            records=records,
            conflict_signal=ConflictSignal.NOT_APPLICABLE,
            latency_ms=timer.elapsed_ms(),
            ranking_signal=ranking_signal,
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
            raise BackendAPIError(self.name, _error_detail(exc)) from exc
        except json.JSONDecodeError as exc:
            # Same volcengine/OpenViking#2966 shape as store() above --
            # update()'s /v1/fs/write call is this adapter's own upsert
            # path (it overwrites an existing key), the exact "upsert_data()"
            # write path #2966 reports as crashing on a legacy-corrupt
            # record. See CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE.
            raise BackendAPIError(
                self.name,
                f"response body is not valid JSON: {exc}",
                crash_signal=CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE,
            ) from exc
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
            raise BackendAPIError(self.name, _error_detail(exc)) from exc
        except json.JSONDecodeError as exc:
            # volcengine/OpenViking#2966: LocalIndex.delete_data() runs the
            # same _convert_delta_list_for_index() -> bare
            # json.loads(fields_json) path store()/update() above hit --
            # this is the literal "delete()" half of "delete/upsert" the
            # issue names. See CrashSignal
            # .LEGACY_CORRUPT_RECORD_UNDELETABLE for the full citation.
            raise BackendAPIError(
                self.name,
                f"response body is not valid JSON: {exc}",
                crash_signal=CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE,
            ) from exc
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
            raise BackendAPIError(self.name, _error_detail(exc)) from exc
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
            raise BackendAPIError(self.name, _error_detail(exc)) from exc

    def get_stats(self, session_id: str | None = None) -> StatsResult:
        """Read OpenViking's dedicated stats/dashboard endpoint.

        Confidence: HIGH on the endpoint path and response shape --
        `GET /api/v1/stats/memories` and its `total_memories` field are
        confirmed verbatim from volcengine/OpenViking#1255's real bug
        report (contributor SeeYangZhi), which pastes the exact JSON
        response shape this parses. Distinct from the LOW-confidence
        guessed `/v1/fs/write`/`/v1/search` endpoints elsewhere in this
        adapter (see the module docstring): this path was never confirmed
        against a live instance either, but the exact response shape comes
        from a real, reproduced report rather than being inferred by
        analogy.

        `session_id` is accepted for interface compatibility with
        MemoryBackendAdapter.get_stats() but ignored -- the real endpoint
        this targets is a tenant-wide/global counter (per #1255's report),
        not session-scoped, so there is nothing to pass it as.
        """
        del session_id
        timer = self._timed()
        try:
            resp = self._http.get("/api/v1/stats/memories")
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, _error_detail(exc)) from exc
        total = data.get("total_memories")
        total_memories = int(total) if isinstance(total, int | float) else None
        return StatsResult(total_memories=total_memories, latency_ms=timer.elapsed_ms(), raw=data)

    def delete_prefix(self, prefix: str, recursive: bool = True) -> DeletePrefixResult:
        """Recursively delete every resource path under `prefix`.

        Best-effort reconstruction against volcengine/OpenViking#3064's
        recommended fix direction: discover every child path via
        list_resource_paths() (this adapter's own recursive tree walk,
        see that method's docstring) rather than a single-level listing,
        then delete() each discovered path plus the prefix root itself.

        Honesty caveat (see VectorIntegritySignal in base.py and
        evals/orphan_cleanup.py for the eval this feeds): list_resource_paths()
        goes through the same `/v1/fs/list` AGFS-directory-listing endpoint
        #3064's root cause lives in. If a live OpenViking server still has
        the bug this issue reports -- `_ls_entries()` raising when the
        parent directory no longer exists in AGFS, silently swallowed by a
        bare `except: pass` -- this adapter's own listing call would
        suffer the identical limitation: it would enumerate zero children,
        this method would only delete the prefix root, and any child
        vector-index entries would survive undetected by anything that
        only checks list_resource_paths() again afterward. That is
        precisely why evals/orphan_cleanup.py's classification also
        re-queries for seeded content, not just re-lists paths -- see
        VectorIntegritySignal.ORPHANED_VECTOR_ENTRY.
        """
        timer = self._timed()
        child_paths: list[str] = []
        if recursive:
            try:
                child_paths = self.list_resource_paths(prefix)
            except BackendAPIError:
                child_paths = []
        deleted: list[str] = []
        failed: list[str] = []
        for path in child_paths:
            try:
                result = self.delete(path)
            except BackendAPIError:
                failed.append(path)
                continue
            (deleted if result.success else failed).append(path)
        root_uri = f"viking://{prefix.strip('/')}"
        try:
            root_result = self.delete(root_uri)
            (deleted if root_result.success else failed).append(root_uri)
        except BackendAPIError:
            failed.append(root_uri)
        return DeletePrefixResult(
            prefix=prefix,
            deleted_paths=deleted,
            failed_paths=failed,
            latency_ms=timer.elapsed_ms(),
        )

    def close(self) -> None:
        self._http.close()


def _slugify(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
