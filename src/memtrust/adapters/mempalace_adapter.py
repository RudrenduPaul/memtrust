"""Adapter for MemPalace (https://github.com/mempalace/mempalace).

Confidence: MEDIUM on product behavior, LOW on exact Python method names.

MemPalace requires no API key: authentication is not part of its
configuration surface, unlike the other three adapters in this repo (see
"Because MemPalace needs no cloud API key" below). That is a narrower
claim than "no network access required," and conflating the two is a
real, documented bug, not a hypothetical one. Motivating case:
mempalace/mempalace#524 (contributor gaby) found that `pip install
mempalace` succeeds entirely offline, but `mempalace mine .` then fails
on a network-restricted VM: chromadb's default embedder does not ship
its model, it downloads an ONNX model from an AWS S3 bucket the first
time it is actually used to embed something. So MemPalace is local-first
for storage -- it stores verbatim conversation text in a local,
SQLite-backed index (chromadb for embeddings) organized as a "palace" of
wings/rooms/drawers -- but it is NOT network-independent: a first-use
embedder call still requires outbound network access, API key or not.
MemPalace separately ships a temporal entity-relationship graph
documented as supporting add/query/invalidate/timeline operations. It
publishes a
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

Migration-rollback simulation (NOT implemented here, deliberately):
MemPalace ships its own `migrate.migrate()` function, which is documented
to perform an on-disk swap at the end of a storage migration. Community
report mempalace/mempalace#1028 (GitHub user eldar702) describes an
earlier, unguarded `shutil.rmtree()`-then-`shutil.move()` version of that
swap: the old backup was deleted FIRST, so a `move()` failure partway
through (e.g. a cross-device `EXDEV` error) could permanently lose the
palace directory. mempalace/mempalace#935 is the real upstream fix -- a
safer "rename-aside" swap. This adapter, like every other adapter in this
repo, has zero direct filesystem control over a live `mempalace` package's
internal `migrate.migrate()` call -- it only wraps `remember()`/
`recall()`/`invalidate()` through `_get_palace()` below (see the module
confidence note above). So `MemPalaceAdapter` does not set
`supports_migration_rollback_simulation = True` and does not override
`simulate_migration_failure()` -- the same "no real process/filesystem-
lifecycle control" reasoning that already applies to
`supports_crash_recovery_simulation` (see adapters/base.py). See
evals/migration_rollback.py and
tests/test_evals.py::MigrationRollbackFakeAdapter for the harness-side
simulation this gap is closed with instead, and docs/methodology.md for
the honesty caveat that applies here the same way it applies to the rest
of this adapter.

MCP metadata-tool coverage (mempalace_status/list_wings/list_rooms):
MemPalace/mempalace#1871 (contributor alionar) found that the MCP
server's metadata/histogram-listing tools -- `mempalace_status`,
`mempalace_list_wings`, `mempalace_list_rooms` -- did a full-collection
scan on every call against a Qdrant-backed palace, which is O(N^2)
against repeated calls and hung the server at 158K+ drawers (fixed by
server-side Qdrant faceting). Before this change, memtrust had zero
coverage of this code path at all: every method above this note is
written against the guessed `remember()`/`recall()`/`invalidate()`
library concepts, never the MCP-server tool surface.

Investigating this confirmed two things against the real, installed
`mempalace` package (PyPI, not the guessed API this module's other
methods are written against):

  1. There is no `Palace` class anywhere in the real package -- `store()`/
     `query()`/`update()` above are written against an unconfirmed guess
     (see this module's opening confidence note) that does not match the
     real package's actual surface, and will raise BackendAPIError
     against it today. That mismatch is a separate, pre-existing gap this
     change does not attempt to fix.
  2. `mempalace.mcp_server` DOES ship real, plain module-level functions
     -- `tool_status()`, `tool_list_wings()`, `tool_list_rooms(wing=None)`
     -- that are the actual implementation MCP tool calls dispatch to.
     They are callable directly, in-process, with no MCP stdio/HTTP
     transport involved: confirmed by importing the real package, seeding
     a local chromadb-backed palace directly via
     `mempalace.palace.get_backend_for_palace(...).get_collection(...)`,
     and calling these functions, which correctly report ground-truth
     wing/room counts. This is exactly the "library-level function that
     does the same underlying work" this build's investigation was asked
     to look for, and it is a genuinely confirmed finding, not a repeat of
     the LOW-confidence guessing this module's other methods carry.

`metadata_overview()`/`list_metadata_categories()`/
`list_metadata_subcategories()` below wrap those three real functions.
One more confirmed-real detail those methods depend on: `mempalace.
mcp_server`'s module-level `_config.palace_path` property reads the
`MEMPALACE_PALACE_PATH` environment variable (`MEMPAL_PALACE_PATH` as a
legacy alias) -- a *different* env var name than this adapter's own
`MEMPALACE_STORAGE_PATH`. `_sync_mcp_palace_path()` below bridges that gap
explicitly (mirrors `self._storage_path` into `MEMPALACE_PALACE_PATH`
before every call) rather than silently assuming the two names line up.

Scale coverage: evals/mempalace_metadata_scale.py exercises these three
methods against a real, locally seeded chromadb-backed palace at
increasing checkpoint sizes and checks both correctness (reported counts
match ground truth) and that per-call latency does not blow up
super-linearly as the corpus grows -- the same "recall/latency as a
function of scale" pattern evals/scale_stress.py already established, but
for this repo's first coverage of the metadata-listing code path instead
of store()/query(). See that module's docstring for the one part of
alionar's exact repro this could NOT be reproduced live in this
environment: MemPalace's Qdrant backend is REST-only against a live
external Qdrant server (no embedded/local mode), and this build's
environment has neither docker nor a local Qdrant binary available --
stated honestly there rather than fabricated.
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
    MetadataCategoryCountsResult,
    MetadataOverviewResult,
    QueryResult,
    RankingSignal,
    RetrievalWarning,
    StoreResult,
    UpdateResult,
)

#: Env var the real installed `mempalace` package's `mempalace.mcp_server`
#: module reads its palace location from (confirmed via
#: `mempalace.config.MempalaceConfig.palace_path`), distinct from this
#: adapter's own MEMPALACE_STORAGE_PATH -- see the module docstring's "MCP
#: metadata-tool coverage" section above for how the two get bridged.
_MCP_PALACE_PATH_ENV_VAR = "MEMPALACE_PALACE_PATH"

#: Metadata keys that mempalace/mempalace#1733 (see RankingSignal's
#: docstring in adapters/base.py) identified as the fields
#: `mempalace/layers.py`'s `Layer1.generate()` sorts drawers by. Checked in
#: this priority order because `importance` is the field the issue's root
#: cause names directly (0/45,969 drawers on a real palace ever had it
#: written); `emotional_weight`/`weight` are the sort's other documented
#: keys, checked as fallbacks so this adapter still reports something
#: meaningful if a future MemPalace version populates one of those instead.
#: `authored_at` was added by MemPalace's own merged PR#1890
#: (mempalace/mempalace) as a timestamp tie-breaker `_hybrid_rank` falls
#: back to when two drawers score identically on the other fields -- a
#: real, ranking-driving metadata field this adapter previously never
#: checked for at all, silently dropping it from `_classify_ranking_signal`
#: even when it was the only thing actually varying between records.
_RANKING_METADATA_KEYS = ("importance", "emotional_weight", "weight", "authored_at")


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


def _record_metadata(item: dict[str, Any]) -> dict[str, str]:
    """Build a query() result item's metadata dict, folding in a
    top-level `authored_at` field when the nested `metadata.authored_at`
    isn't already present.

    MemPalace's own merged PR#1890 added `authored_at` timestamp metadata
    (used as a `_hybrid_rank` tie-breaker), but its real API can surface
    that field at the TOP LEVEL of a response item instead of nested under
    `metadata` -- confirmed by gemini-code-assist's review comment on that
    same PR#1890 diff, flagging the exact same top-level-vs-nested
    inconsistency in MemPalace's own response-building code. Before this
    fix, this adapter only ever read `item.get("metadata")`, so a
    top-level `authored_at` was silently dropped and never reached
    `_classify_ranking_signal` or `MemoryRecord.metadata` at all -- an
    adapter-side instance of the identical bug shape, not a re-guess at
    a different problem. Nested `metadata.authored_at` (when present)
    always wins over a top-level value, so a response that (correctly, per
    the confirmed shape) nests it is never overridden by a stray top-level
    duplicate.
    """
    metadata = dict(item.get("metadata") or {})
    if "authored_at" not in metadata and item.get("authored_at") is not None:
        metadata["authored_at"] = str(item["authored_at"])
    return metadata


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


class _MCPMetadataToolsProtocol(Protocol):
    """Shape this adapter expects from `mempalace.mcp_server` (or a fake
    standing in for it in tests). Confirmed real against the installed
    `mempalace` package -- see the module docstring's "MCP metadata-tool
    coverage" section -- unlike `_PalaceProtocol` above, whose method
    names remain an unconfirmed guess.
    """

    def tool_status(self) -> dict[str, Any]: ...

    def tool_list_wings(self) -> dict[str, Any]: ...

    def tool_list_rooms(self, wing: str | None = None) -> dict[str, Any]: ...


class MemPalaceAdapter(MemoryBackendAdapter):
    name = "mempalace"
    env_var = "MEMPALACE_STORAGE_PATH"
    supports_update = True
    #: See the module docstring's "Mode variants" section above -- these
    #: names come from mempalace/mempalace#27, not a confirmed API
    #: reference.
    supported_modes = ("raw", "AAAK")
    #: See the module docstring's "MCP metadata-tool coverage" section --
    #: metadata_overview()/list_metadata_categories()/
    #: list_metadata_subcategories() below wrap the real, confirmed
    #: mempalace.mcp_server.tool_status/tool_list_wings/tool_list_rooms
    #: functions.
    supports_metadata_overview = True

    def __init__(
        self,
        palace: _PalaceProtocol | None = None,
        mcp_tools: _MCPMetadataToolsProtocol | None = None,
    ) -> None:
        storage_path = os.environ.get(self.env_var)
        if not storage_path and palace is None:
            raise BackendNotConfiguredError(self.name, self.env_var)
        self._storage_path = storage_path
        self._palace = palace
        #: See `_get_mcp_metadata_tools()` below -- lazily imported from
        #: the real `mempalace.mcp_server` module unless a fake is
        #: injected here (tests only; see tests/test_adapters.py).
        self._mcp_tools = mcp_tools

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

    def _get_mcp_metadata_tools(self) -> _MCPMetadataToolsProtocol:
        """Lazily import `mempalace.mcp_server`, mirroring `_get_palace()`
        above -- except this module IS confirmed real against the
        installed package (see the module docstring's "MCP metadata-tool
        coverage" section), so the only failure mode here is the package
        not being installed at all, not a guessed-wrong method name.
        """
        if self._mcp_tools is not None:
            return self._mcp_tools
        try:
            import mempalace.mcp_server as _mcp_server  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendAPIError(
                self.name,
                "the `mempalace` package is not installed, or its "
                "`mcp_server` submodule failed to import. Install it with "
                "`pip install mempalace` to exercise the "
                "mempalace_status/mempalace_list_wings/mempalace_list_rooms "
                "MCP-tool code path (see this module's docstring's 'MCP "
                "metadata-tool coverage' section).",
            ) from exc
        self._mcp_tools = _mcp_server
        return self._mcp_tools

    def _sync_mcp_palace_path(self) -> None:
        """Bridge this adapter's MEMPALACE_STORAGE_PATH into the real
        `mempalace.mcp_server` module's actual config env var,
        MEMPALACE_PALACE_PATH (confirmed different names -- see the module
        docstring). `mempalace.config.MempalaceConfig.palace_path` is a
        property that re-reads the env var on every access, so setting it
        immediately before each call (rather than once at __init__ time)
        is what lets a single process safely point this adapter at more
        than one storage path across its lifetime, and is a no-op when
        `_storage_path` was never set (fake-only unit tests that inject
        `palace`/`mcp_tools` directly without ever going through a real
        storage path).
        """
        if self._storage_path:
            os.environ[_MCP_PALACE_PATH_ENV_VAR] = self._storage_path

    def metadata_overview(self) -> MetadataOverviewResult:
        """Real, confirmed library-level equivalent of MemPalace's
        `mempalace_status` MCP tool -- see the module docstring's "MCP
        metadata-tool coverage" section for how this was confirmed against
        the installed package, and MemPalace/mempalace#1871 for the O(N^2)
        full-collection-scan bug that made this code path worth covering
        at all.
        """
        timer = self._timed()
        tools = self._get_mcp_metadata_tools()
        self._sync_mcp_palace_path()
        try:
            raw = tools.tool_status()
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if not isinstance(raw, dict):
            raise BackendAPIError(
                self.name,
                f"tool_status() returned {type(raw).__name__}, expected dict",
            )
        error = raw.get("error")
        return MetadataOverviewResult(
            total_records=raw.get("total_drawers"),
            categories=dict(raw.get("wings") or {}),
            subcategories=dict(raw.get("rooms") or {}),
            latency_ms=timer.elapsed_ms(),
            partial=bool(raw.get("partial")) or error is not None,
            error=str(error) if error is not None else None,
        )

    def list_metadata_categories(self) -> MetadataCategoryCountsResult:
        """Real, confirmed library-level equivalent of MemPalace's
        `mempalace_list_wings` MCP tool -- see metadata_overview() above.
        """
        timer = self._timed()
        tools = self._get_mcp_metadata_tools()
        self._sync_mcp_palace_path()
        try:
            raw = tools.tool_list_wings()
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if not isinstance(raw, dict):
            raise BackendAPIError(
                self.name,
                f"tool_list_wings() returned {type(raw).__name__}, expected dict",
            )
        error = raw.get("error")
        return MetadataCategoryCountsResult(
            counts=dict(raw.get("wings") or {}),
            scope=None,
            latency_ms=timer.elapsed_ms(),
            partial=bool(raw.get("partial")) or error is not None,
            error=str(error) if error is not None else None,
        )

    def list_metadata_subcategories(
        self, category: str | None = None
    ) -> MetadataCategoryCountsResult:
        """Real, confirmed library-level equivalent of MemPalace's
        `mempalace_list_rooms` MCP tool -- see metadata_overview() above.

        Args:
            category: restrict the listing to this wing name, or None to
                list rooms across every wing (mirrors the real
                `tool_list_rooms(wing=None)` signature exactly).
        """
        timer = self._timed()
        tools = self._get_mcp_metadata_tools()
        self._sync_mcp_palace_path()
        try:
            raw = tools.tool_list_rooms(wing=category)
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if not isinstance(raw, dict):
            raise BackendAPIError(
                self.name,
                f"tool_list_rooms() returned {type(raw).__name__}, expected dict",
            )
        error = raw.get("error")
        reported_scope = raw.get("wing")
        return MetadataCategoryCountsResult(
            counts=dict(raw.get("rooms") or {}),
            scope=str(reported_scope) if reported_scope not in (None, "all") else category,
            latency_ms=timer.elapsed_ms(),
            partial=bool(raw.get("partial")) or error is not None,
            error=str(error) if error is not None else None,
        )

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
                metadata=_record_metadata(item),
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
