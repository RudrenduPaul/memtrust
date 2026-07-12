"""Adapter for MemPalace (https://github.com/mempalace/mempalace).

Confidence: MEDIUM on product behavior, LOW on exact Python method names.

MemPalace is confirmed local-first and explicitly requires no API key --
it stores verbatim conversation text in a local, SQLite-backed index
(chromadb for embeddings) organized as a "palace" of wings/rooms/drawers,
and separately ships a temporal entity-relationship graph documented as
supporting add/query/invalidate/timeline operations. It publishes a
`mempalace` package on PyPI and documents a Python API at
mempalaceofficial.com/reference/python-api, but that reference page's
exact class and method names were not confirmed during this build (the
page was not fetchable in this environment). Rather than guess a plausible-
looking API and ship it silently, this adapter is written against the
documented *concepts* (a Palace object scoped to a local storage path,
with store/query/invalidate operations) and every vendor-specific call is
isolated behind `_get_palace()` so it fails loudly and specifically
(BackendAPIError, not an unrelated AttributeError) if the real package's
surface differs. See docs/methodology.md for the full uncertainty note
and what a contributor should verify against the live package before
trusting this adapter's output.

Because MemPalace needs no cloud API key, this adapter's "configuration"
requirement is a local storage path (MEMPALACE_STORAGE_PATH) rather than
a secret -- a deliberate, documented deviation from the API-key pattern
the other three adapters use, not an oversight.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

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


class _PalaceProtocol(Protocol):
    """Shape this adapter expects from the real `mempalace` package.
    Defined as a Protocol (not imported from the package) so tests can
    inject a fake implementation without the real, chromadb-dependent
    package installed -- see tests/test_adapters.py.
    """

    def remember(self, room: str, content: str, metadata: dict[str, str]) -> str: ...

    def recall(self, room: str, query: str, top_k: int) -> list[dict[str, Any]]: ...

    def invalidate(self, room: str, memory_id: str, content: str) -> dict[str, Any]: ...


class MemPalaceAdapter(MemoryBackendAdapter):
    name = "mempalace"
    env_var = "MEMPALACE_STORAGE_PATH"
    supports_update = True

    def __init__(self, palace: _PalaceProtocol | None = None) -> None:
        storage_path = os.environ.get(self.env_var)
        if not storage_path and palace is None:
            raise BackendNotConfiguredError(self.name, self.env_var)
        self._storage_path = storage_path
        self._palace = palace

    def _get_palace(self) -> _PalaceProtocol:
        if self._palace is not None:
            return self._palace
        try:
            import mempalace  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendAPIError(
                self.name,
                "the `mempalace` package is not installed. Install it with "
                "`pip install mempalace` (see docs/methodology.md for the "
                "documented-vs-verified caveat on its Python API surface).",
            ) from exc
        try:
            self._palace = mempalace.Palace(storage_path=self._storage_path)
        except AttributeError as exc:
            raise BackendAPIError(
                self.name,
                "mempalace.Palace(storage_path=...) was not found on the "
                "installed package. This adapter was written against "
                "MemPalace's documented concepts, not a confirmed API "
                "reference -- see docs/methodology.md.",
            ) from exc
        return self._palace

    def store(
        self, session_id: str, content: str, metadata: dict[str, str] | None = None
    ) -> StoreResult:
        timer = self._timed()
        palace = self._get_palace()
        try:
            memory_id = palace.remember(room=session_id, content=content, metadata=metadata or {})
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        return StoreResult(memory_id=memory_id, latency_ms=timer.elapsed_ms())

    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult:
        timer = self._timed()
        palace = self._get_palace()
        try:
            results = palace.recall(room=session_id, query=query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc

        records = [
            MemoryRecord(
                memory_id=str(item.get("id", "")),
                content=str(item.get("content", "")),
                score=item.get("score"),
                created_at=item.get("created_at"),
                metadata=item.get("metadata") or {},
                raw=item,
            )
            for item in results
        ]
        invalidated = [r for r in records if r.metadata.get("invalidated") == "true"]
        # MemPalace's documented temporal graph exposes an explicit
        # invalidate operation, which is a stronger signal than either
        # Mem0's opaque pipeline or a plain vector store's overwrite:
        # a query result carrying an "invalidated" marker means the
        # backend itself flagged that fact as superseded.
        conflict_signal = ConflictSignal.FLAGGED if invalidated else ConflictSignal.NOT_APPLICABLE
        return QueryResult(
            records=records, conflict_signal=conflict_signal, latency_ms=timer.elapsed_ms()
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        timer = self._timed()
        palace = self._get_palace()
        try:
            result = palace.invalidate(room=session_id, memory_id=memory_id, content=content)
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        new_id = str(result.get("id", memory_id))
        return UpdateResult(
            memory_id=new_id, acknowledged=True, latency_ms=timer.elapsed_ms(), raw=result
        )

    def delete(self, memory_id: str) -> DeleteResult:
        """NOT IMPLEMENTED for MemPalace.

        Unlike store/query/update, which are written against MemPalace's
        documented *concepts* (remember/recall/invalidate) even though
        the exact Python method names are unverified (see module
        docstring), no delete/forget concept was surfaced in this build's
        research pass at all -- there is nothing to reconstruct a
        best-effort call against, not even an uncertain one. Rather than
        guess a `palace.forget(...)` call that may not exist on the real
        package, this raises a clear, typed error so callers (and the
        eval layer) can distinguish "not supported yet" from a network
        failure, same as every other BackendAPIError.

        A contributor who confirms MemPalace's real deletion API should
        implement this properly and remove this docstring/raise -- see
        docs/methodology.md for the uncertainty-tracking convention.
        """
        raise BackendAPIError(
            self.name,
            "delete() is not implemented for MemPalace: no documented "
            "delete/forget primitive was confirmed for the `mempalace` "
            "package during this adapter's build. See docs/methodology.md.",
        )
