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

Mode variants ("raw" vs "AAAK"): mempalace/mempalace#27 (cited in
README.md and docs/methodology.md as founding rationale for this project)
documents a "lossless" compression claim for MemPalace's default write
path that community testing showed is actually lossy -- a reported 12.4
percentage-point accuracy drop between an uncompressed mode and the
compressed mode the issue calls "AAAK". `supported_modes` below exposes
those two mode names so `evals/compression.py` can request each one via
`store()`/`query()`'s `mode` parameter and directly measure round-trip
fidelity per mode. The mode *names* come from that community issue, not
from a confirmed constructor/method parameter in the installed
`mempalace` package -- exactly the same LOW-confidence caveat that
already applies to every other method name in this file (see the module
confidence note above and docs/methodology.md's adapter-confidence
table). If the real package's `remember()`/`recall()` do not accept a
`mode` keyword, passing a non-None mode fails loudly as a
`BackendAPIError` (via the existing generic `except Exception` wrapping
below), not silently -- it never falls back to pretending the mode was
honored.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from memtrust.adapters.base import (
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
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

    def remember(
        self, room: str, content: str, metadata: dict[str, str], mode: str | None = None
    ) -> str: ...

    def recall(
        self, room: str, query: str, top_k: int, mode: str | None = None
    ) -> list[dict[str, Any]]: ...

    def invalidate(self, room: str, memory_id: str, content: str) -> dict[str, Any]: ...


class MemPalaceAdapter(MemoryBackendAdapter):
    name = "mempalace"
    env_var = "MEMPALACE_STORAGE_PATH"
    supports_update = True
    #: See the module docstring's "Mode variants" section above -- these
    #: names come from mempalace/mempalace#27, not a confirmed API
    #: reference.
    supported_modes = ("raw", "AAAK")

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
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        timer = self._timed()
        palace = self._get_palace()
        try:
            memory_id = palace.remember(
                room=session_id, content=content, metadata=metadata or {}, mode=mode
            )
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        return StoreResult(memory_id=memory_id, latency_ms=timer.elapsed_ms())

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        timer = self._timed()
        palace = self._get_palace()
        try:
            results = palace.recall(room=session_id, query=query, top_k=top_k, mode=mode)
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
