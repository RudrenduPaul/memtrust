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
    DeleteResult,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    RankingSignal,
    RetrievalWarning,
    StoreResult,
    UpdateResult,
)

#: Metadata keys that mempalace/mempalace#1733 (see RankingSignal's
#: docstring in adapters/base.py) identified as the fields
#: `mempalace/layers.py`'s `Layer1.generate()` sorts drawers by. Checked in
#: this priority order because `importance` is the field the issue's root
#: cause names directly (0/45,969 drawers on a real palace ever had it
#: written); `emotional_weight`/`weight` are the sort's other documented
#: keys, checked as fallbacks so this adapter still reports something
#: meaningful if a future MemPalace version populates one of those instead.
_RANKING_METADATA_KEYS = ("importance", "emotional_weight", "weight")


def _classify_ranking_signal(records: list[MemoryRecord]) -> RankingSignal:
    """Inspect a query response's records for a ranking-relevant metadata
    field and report whether a real per-record signal appears to exist.

    This is a coarse, adapter-level claim, not the full picture: it can
    say "this field is present and varies" (SIGNAL_DRIVEN) or "this field
    is absent or constant across every record" (MISSING_ORDERING_KEY), but
    it cannot by itself confirm the backend's returned order actually
    correlates with a varying field -- that requires ground truth about
    intended order that only a specific eval case has. See
    evals/ranking_quality.py's classify_ranking_case, which cross-checks
    this claim against the actual returned order before crediting a
    SIGNAL_DRIVEN report, exactly the way evals/contradiction.py's
    classify_case never trusts an adapter's bare conflict_signal claim
    outright either.

    Fewer than 2 records is treated as NOT_APPLICABLE -- there is nothing
    to compare an "identical across records" claim against with 0 or 1
    record.
    """
    if len(records) < 2:
        return RankingSignal.NOT_APPLICABLE

    for key in _RANKING_METADATA_KEYS:
        values = [r.metadata[key] for r in records if key in r.metadata]
        if not values:
            continue
        if len(values) < len(records):
            # The field is present on some records but not all -- exactly
            # as ambiguous as "every record shares the same value": there
            # is no complete, real per-record signal to point to.
            return RankingSignal.MISSING_ORDERING_KEY
        if len(set(values)) == 1:
            # Present everywhere but identical -- the exact
            # mempalace/mempalace#1733 shape: a field that silently
            # defaulted to one constant value for every drawer.
            return RankingSignal.MISSING_ORDERING_KEY
        return RankingSignal.SIGNAL_DRIVEN

    # None of the known ranking-relevant keys appeared on any record at
    # all. That is itself indistinguishable, from the caller's side, from
    # "this field silently defaults to a constant" -- a backend that never
    # writes the key produces the same observable symptom (no real
    # per-record signal) as one that writes a constant default. Flagging
    # both the same way is deliberate, not an oversight -- see
    # docs/methodology.md's honesty note on this eval's limits.
    return RankingSignal.MISSING_ORDERING_KEY


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
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Return either a bare list of record dicts (this adapter's
        original, still-unconfirmed guess at `Palace.recall()`'s shape),
        or a dict shaped like MemPalace/mempalace#1005's confirmed,
        merged `search_memories()` response:
        `{"results": [...], "warnings": [...], "available_in_scope": N}`.

        Both are unconfirmed guesses about what a real `Palace.recall()`
        method (if one exists under that name at all) actually returns --
        see the module docstring's confidence caveat. The dict shape is
        the one piece of *this* response body confirmed against real,
        merged vendor source (the #1005 diff), so `query()` below checks
        for it and parses `warnings`/`available_in_scope` when present,
        while still accepting the older bare-list shape unchanged so a
        wrong guess about which shape the real method uses doesn't break
        every existing caller -- see `query()` for the parsing and the
        loud `BackendAPIError` a dict missing a `results` key raises.
        """
        ...

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
        *,
        verify: bool = False,
    ) -> StoreResult:
        """Store a memory, optionally confirming it durably landed.

        `verify` is opt-in and defaults to False -- this adapter is the
        reference implementation for MemoryBackendAdapter.verify_store()
        specifically because MemPalace is the vendor whose silent-write
        bugs (NUL-byte checkpoint corruption, stale/self-deadlocked
        locks) motivated adding it: both bugs let `remember()` return
        normally while the write itself was dropped or corrupted, which
        looked identical to weaker model recall until now. Passing
        `verify=True` costs one extra `recall()` call per `store()` call
        -- see docs/methodology.md for why that stays opt-in rather than
        the default.
        """
        timer = self._timed()
        palace = self._get_palace()
        try:
            memory_id = palace.remember(
                room=session_id, content=content, metadata=metadata or {}, mode=mode
            )
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        result = StoreResult(memory_id=memory_id, latency_ms=timer.elapsed_ms())
        if verify:
            result.verified = self.verify_store(result, session_id, content)
        return result

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        timer = self._timed()
        palace = self._get_palace()
        try:
            response = palace.recall(room=session_id, query=query, top_k=top_k, mode=mode)
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc

        degraded_retrieval: RetrievalWarning | None = None
        if isinstance(response, dict):
            # MemPalace/mempalace#1005's confirmed search_memories() shape:
            # {"results": [...], "warnings": [...], "available_in_scope": N}.
            # A vector-query failure (HNSW/Chroma index drift) degrades into
            # this response instead of raising -- the backend still returns
            # whatever it could rank, plus warnings explaining the shortfall.
            # See _PalaceProtocol.recall()'s docstring for why both this
            # shape and the older bare-list shape are accepted.
            raw_results = response.get("results")
            if raw_results is None:
                raise BackendAPIError(
                    self.name,
                    "recall() returned a dict without a 'results' key -- "
                    "expected either a bare list of record dicts, or "
                    "MemPalace/mempalace#1005's confirmed search_memories() "
                    "shape ({'results': [...], 'warnings': [...], "
                    f"'available_in_scope': ...}}). Got keys: {sorted(response.keys())}.",
                )
            results = raw_results
            raw_warnings = response.get("warnings") or []
            if not isinstance(raw_warnings, list):
                raise BackendAPIError(
                    self.name,
                    "recall() response's 'warnings' field must be a list, "
                    f"got {type(raw_warnings).__name__}.",
                )
            warnings = [str(w) for w in raw_warnings]
            available_in_scope = response.get("available_in_scope")
            if not isinstance(available_in_scope, int) or isinstance(available_in_scope, bool):
                # Per mempalace/mempalace#1005, available_in_scope is None
                # when the backend couldn't compute a scope count (e.g. a
                # filter-planner error) -- treat anything else that isn't a
                # real int (a MagicMock test stub, a float, a string) the
                # same way: "unknown," never coerced into a misleading
                # number. `bool` is excluded explicitly since `bool` is a
                # subclass of `int` in Python and a stray True/False here
                # would silently pass isinstance(..., int).
                available_in_scope = None
            if warnings:
                degraded_retrieval = RetrievalWarning(
                    warnings=warnings, available_in_scope=available_in_scope
                )
        else:
            results = response

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
        ranking_signal = _classify_ranking_signal(records)
        return QueryResult(
            records=records,
            conflict_signal=conflict_signal,
            latency_ms=timer.elapsed_ms(),
            ranking_signal=ranking_signal,
            degraded_retrieval=degraded_retrieval,
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
