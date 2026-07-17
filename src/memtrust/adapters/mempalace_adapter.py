"""Adapter for MemPalace (https://github.com/mempalace/mempalace), PyPI
package `mempalace` (installed and live-verified at version 3.5.0 during
this rewrite).

CONFIRMED-REAL API, NOT A GUESS
--------------------------------
Every previous version of this module was written against a fictional
`mempalace.Palace` class (a `Palace(storage_path=...)` object exposing
`remember()`/`recall()`/`invalidate()`) that was never confirmed against
the real, installed package -- and turned out not to exist at all.
Independently verified during this rewrite: `python3 -c "import mempalace;
hasattr(mempalace, 'Palace')"` -> `False`; every `^class ` in every file of
the installed package was grepped, and `mempalace/palace.py` only defines
two exception classes (`MineAlreadyRunning`, `MineValidationError`), no
`Palace` class anywhere. So `store()`/`query()`/`update()` in every prior
version of this file called a method that never existed, on an object that
was never constructible -- these paths have never worked against the real
package, ever, in this project's history. Every test that appeared to pass
was exercising a hand-written fake standing in for that guess, never the
real thing.

This rewrite instead calls the REAL, plain module-level functions in
`mempalace.mcp_server` -- the actual implementation the MCP server's tool
calls dispatch to -- exactly the same lazy `import mempalace.mcp_server as
_mcp_server` pattern this module already used, correctly, for
`metadata_overview()`/`list_metadata_categories()`/
`list_metadata_subcategories()` (see the "MCP metadata-tool coverage"
history below). `_get_mcp_tools()` is now the ONE lazy-import point for
every real vendor call this adapter makes -- `_get_palace()` and
`_PalaceProtocol` are gone; there is no fictional-API code path left to
silently fall back to.

Every return shape documented below was captured by calling the real
functions live, against a real local chromadb-backed palace (`pip install
mempalace`, `MEMPALACE_PALACE_PATH` pointed at a temp dir, `import
mempalace.mcp_server as mcp; mcp.tool_add_drawer(...)` etc., matching the
same seeding pattern `evals/mempalace_metadata_scale.py`'s
`build_chroma_metadata_seeder()` already established for this repo) -- not
read off a docstring and trusted. Exact captured shapes:

  tool_add_drawer(wing, room, content, source_file=None, added_by="mcp")
    -> {"success": True, "drawer_id": str, "wing": str, "room": str,
        "chunks": int, ["chunk_ids": [str, ...]]}
    -> {"success": True, "reason": "already_exists", "drawer_id": str}
       (idempotent re-add of byte-identical content under the same
       wing/room)
    -> {"success": False, "error": str}
       (sanitizer rejection, e.g. an invalid wing/room name)

  tool_search(query, limit=5, wing=None, room=None, source_file=None,
              max_distance=1.5, min_similarity=None, context=None)
    -> {"query": str, "filters": {"wing":..., "room":..., "source_file":...},
        "total_before_filter": int,
        "results": [
          {"text": str, "wing": str, "room": str, "source_file": str,
           "source_path": str, "created_at": str, "similarity": float,
           "distance": float, "effective_distance": float,
           "closet_boost": float, "matched_via": str, "bm25_score": float,
           ["closet_preview": str]},
          ...
        ]}
    -> {"error": str}  (sanitizer rejection; no "results" key at all)

  tool_update_drawer(drawer_id, content=None, wing=None, room=None)
    -> {"success": True, "drawer_id": str, "wing": str, "room": str}
    -> {"success": False, "error": f"Drawer not found: {drawer_id}"}

  tool_delete_drawer(drawer_id)
    -> {"success": True, "drawer_id": str, "deleted_ids": [str, ...],
        "chunks_deleted": int}
    -> {"success": False, "error": f"Drawer not found: {drawer_id}"}

  tool_kg_add(subject, predicate, object, valid_from=None, valid_to=None,
              source_closet=None, source_file=None, source_drawer_id=None)
    -> {"success": True, "triple_id": str, "fact": f"{subject} → {predicate} → {object}"}

  tool_kg_invalidate(subject, predicate, object, ended=None)
    -> {"success": True, "fact": str, "ended": str}
       (live-verified: this succeeds even when no matching fact was ever
       added -- the real function does not check prior existence before
       recording an invalidation; see kg_invalidate()'s docstring below)

  tool_kg_query(entity, as_of=None, direction="both")
    -> {"entity": str, "as_of": str | None,
        "facts": [
          {"direction": "outgoing"|"incoming", "subject": str,
           "predicate": str, "object": str, "valid_from": str | None,
           "valid_to": str | None, "confidence": float,
           "source_closet": str | None, "current": bool},
          ...
        ],
        "count": int}

  tool_status()/tool_list_wings()/tool_list_rooms(wing=None): unchanged
    from this module's pre-existing, already-confirmed "MCP metadata-tool
    coverage" work (see that section below) -- `{"error": ..., "details":
    ..., "hint": ..., "backend": ...}` when no palace/collection exists yet
    (no drawer has ever been written), the documented wing/room-count shape
    otherwise.

WING/ROOM MAPPING FOR store()/query()/update()/delete()
---------------------------------------------------------
`mempalace`'s real drawer API needs BOTH a `wing` and a `room` to file or
search content under -- there is no single-dimension scope the way the
fictional `Palace.remember(room=...)` guess assumed. This adapter maps
memtrust's single `session_id` parameter onto `wing` (not `room`): every
memtrust query() call already hard-scopes to one session, and `wing` is
the dimension `tool_search`'s own `wing=` filter uses to exclude every
other session's drawers from a result set, so `wing=session_id` reproduces
the old adapter's full per-session isolation. `room` -- the real API's
second, finer-grained scoping dimension, with no memtrust-side equivalent
-- defaults to `MemPalaceAdapter.DEFAULT_ROOM` ("memtrust") unless the
caller's `metadata` dict passed to `store()` supplies an explicit
`metadata["room"]` override, so a caller that wants finer scoping still
can, without it being required.

`store()`'s `metadata` PARAMETER NO LONGER CARRIES ARBITRARY KEY/VALUE
TAGS INTO THE VENDOR CALL -- A REAL, CONFIRMED LIMITATION
-------------------------------------------------------------------------
The old, fictional `Palace.remember(metadata=...)` guess accepted an
arbitrary string-keyed metadata dict per record (this is what let the old
test suite inject synthetic `importance`/`emotional_weight` values to
drive RankingSignal). The REAL, confirmed `tool_add_drawer(wing, room,
content, source_file=None, added_by="mcp")` has no such parameter at all
-- there is no way to attach arbitrary per-record metadata to a drawer
through this tool. `store()` here still accepts memtrust's generic
`metadata: dict[str, str] | None` parameter (required by
`MemoryBackendAdapter`'s abstract signature), but only reads two
recognized keys out of it for real vendor parameters (`"room"`,
`"source_file"`, `"added_by"`) -- any other key a caller passes is
silently dropped, not forwarded anywhere, and not an adapter bug: there is
genuinely nowhere in the real API to put it.

RANKING SIGNAL: RE-POINTED AT A REAL FIELD, NOT THE ORIGINAL BUG'S FIELD
-------------------------------------------------------------------------
The original `_classify_ranking_signal` (mempalace/mempalace#1733,
contributor Kartalops) checked `importance`/`emotional_weight`/`weight`/
`authored_at` -- fields `mempalace/layers.py`'s `Layer1.generate()`
("wake-up") sorts by. Two things are now confirmed that were not before:
(1) since `tool_add_drawer` has no metadata-injection parameter (see
above), this adapter can never cause those fields to exist on a drawer in
the first place; (2) `tool_search` -- the function this adapter actually
calls, confirmed live above -- sorts its own results by `effective_distance`
(vector cosine distance, optionally closet-boosted), not by
`importance`/`emotional_weight`/`weight` at all. Those are two different
code paths in the real package; `Layer1.generate()`'s "wake-up" sort was
never reachable from this adapter even under the old fictional API. So
checking for `importance` et al. was never a meaningful test of what this
adapter actually calls -- it was checking a field belonging to a method
this module never calls. `_classify_ranking_signal` below is re-pointed at
`similarity` -- the field `tool_search`'s own confirmed sort key
(`effective_distance`) directly determines, live-verified above (the
higher-similarity record came back first) -- so a MISSING_ORDERING_KEY
verdict now means "this response's own real ranking field was constant or
absent," the same concept the original signal was trying to express, just
pointed at a field this adapter's real vendor call actually produces.

CONFLICT SIGNAL: NOW HONESTLY NOT_APPLICABLE FOR DRAWER-BACKED query()
-------------------------------------------------------------------------
The old adapter set `ConflictSignal.FLAGGED` when a returned record's
`metadata.get("invalidated") == "true"`. That was an assumption about the
fictional `Palace.invalidate()`'s write-side effect on `recall()`'s
future output, never confirmed against a real response. The real
`tool_search` response items (shape confirmed live above) never carry an
`"invalidated"` key or any other conflict/invalidation marker -- drawers
have no such concept in the real package at all; that concept only exists
on the KG side (see below). `query()` below therefore always reports
`ConflictSignal.NOT_APPLICABLE` for drawer-backed queries -- an honest
downgrade from the old (fictional, unverified) FLAGGED path, not a
regression this rewrite introduced silently.

CONTRADICTION/STALENESS DETECTION MOVED TO THE KG API, AS A NEW,
ADDITIVE CAPABILITY -- NOT A REPLACEMENT FOR query()/update()
-------------------------------------------------------------------------
The real, installed `mempalace` package cleanly separates "drawer" (free-
text content storage, what store()/query()/update()/delete() below map
onto) from "knowledge graph" (subject-predicate-object facts with
temporal validity, `tool_kg_add`/`tool_kg_invalidate`/`tool_kg_query`).
The KG's `tool_kg_invalidate` writes a genuine, real `ended`/`valid_to`
timestamp and flips a real `current: bool` field `tool_kg_query` reports
back (both confirmed live above) -- exactly the "this fact was explicitly
invalidated" signal `ConflictSignal.FLAGGED` was designed to detect, and a
categorically stronger, more real signal than the old fictional drawer
`"invalidated"` metadata marker ever was.

Rather than force memtrust's generic `store()`/`query()`/`update()`
contract (free-text content, one call per memory) onto the structurally
different subject/predicate/object KG shape, this adapter keeps
store()/query()/update()/delete() mapped to the drawer API (the natural,
same-shape mapping `MemoryBackendAdapter`'s abstract signatures already
assume) and adds `kg_add()`/`kg_invalidate()`/`kg_query()` as new,
additive, MemPalace-specific methods -- not part of the shared
`MemoryBackendAdapter` contract, the same convention
`metadata_overview()`/`list_metadata_categories()`/
`list_metadata_subcategories()` already established for this adapter's
other MemPalace-specific, non-universal capabilities. A future
contradiction-detection eval that wants the KG's genuinely stronger signal
can call these three directly; see tests/test_adapters.py for an
end-to-end proof against the real package that `kg_invalidate()` after
`kg_add()` genuinely flips `current` to `False` and stamps a real `ended`
date on live re-query.

NO PER-RECORD ID IN query() RESPONSES -- A REAL, CONFIRMED GAP
-------------------------------------------------------------------------
`tool_search`'s confirmed response items (see the shape above) carry no
`"id"`/`"drawer_id"` field at all -- only `text`/`wing`/`room`/
`source_file`/`source_path`/`created_at`/scoring fields. This was not
obvious from the docstring; it required reading `mempalace/searcher.py`'s
result-building code directly (the only near-miss is an internal, always-
stripped `_parent_drawer_id` key used for chunk-neighbor lookups, never
returned to the caller). Recomputing an id client-side via
`mempalace.ids.make_drawer_id_from_content(wing, room, content)` was
considered and rejected: `update_drawer()` deliberately keeps a drawer's
ORIGINAL id stable across a content edit (live-verified above -- updating
a drawer's content still reports the pre-update drawer_id), so recomputing
from a query response's CURRENT content would silently produce the wrong
id for any drawer that has ever been updated. Every `MemoryRecord` a
`query()` call returns below therefore has `memory_id=""` -- an honest,
explicit "the real vendor response has nothing to put here" rather than a
guessed or recomputed value that would be wrong some of the time.
Consequence: `MemoryBackendAdapter.verify_store()`'s id-match branch can
never fire against this adapter (empty string never equals a real
drawer_id) -- verification always falls through to its content-substring
fallback branch instead, which still works correctly, since `text` in a
query response is the real stored content. This is a real behavior change
from the old, fictional adapter (whose fake `Palace.recall()` in tests
always echoed back a matching id), not a bug in this rewrite.

CHUNKED CONTENT: update()/delete() VERIFIED TO WORK CORRECTLY, DESPITE A
STALE VENDOR DOCSTRING CLAIMING OTHERWISE
-------------------------------------------------------------------------
`tool_add_drawer`'s own docstring in the installed package claims: "To
delete or fetch the underlying drawers, iterate `chunk_ids` or query by
`parent_drawer_id` -- `tool_get_drawer(drawer_id)` and
`tool_delete_drawer(drawer_id)` report 'not found' on the chunked path
because no row is stored under the logical group id." Content above the
real `chunk_size` (800 chars by default,
`mempalace.config.DEFAULT_CHUNK_SIZE`) is indeed split into multiple
physical chunk drawers on `store()`. But that docstring's claim does NOT
hold up: reading `_logical_drawer_record()` (used by both
`tool_update_drawer` and `tool_delete_drawer`) shows it explicitly
resolves a logical group id to its full chunk set via
`_logical_chunk_group()` when a direct single-row lookup misses -- and
live verification confirms it: storing 2000 characters of content (well
over the 800-char default), then calling `tool_update_drawer(drawer_id=
<the logical id>, content="replacement")` correctly de-chunks it back down
to a single row, and `tool_delete_drawer(drawer_id=<the logical id>)`
against a separate 2000-character record correctly deletes all 3
underlying chunk rows and the content becomes genuinely unsearchable
afterward. This adapter's `update()`/`delete()` therefore work correctly
against `store()`-returned ids regardless of content length -- an initial
draft of this module trusted the vendor's own (stale, contradicted-by-its-
own-code) docstring text here without verifying it live, which would have
shipped a false "confirmed" limitation; see
`test_real_mempalace_chunked_content_update_delete_round_trip` below for
the live proof this claim is based on, not `tool_add_drawer`'s docstring
text.

DEGRADED-RETRIEVAL WARNINGS: PARSING KEPT, BUT NEVER OBSERVED REAL
-------------------------------------------------------------------------
A previous version of this module claimed MemPalace/mempalace#1005 (a
`{"results": [...], "warnings": [...], "available_in_scope": N}` degraded-
response shape) was "confirmed against the real, merged PR diff." That
claim does not hold up against the actually-installed 3.5.0 package: a
grep for `"warnings"` and `available_in_scope` across
`mempalace/searcher.py` and `mempalace/mcp_server.py` returns zero matches
anywhere in the installed package -- no code path in this version ever
populates either key.
`query()` below still parses `warnings`/`available_in_scope` defensively
(harmless, forward-compatible if a future mempalace version reintroduces
this shape), but `degraded_retrieval` should be expected to always be
`None` against the real package as installed today; do not read a `None`
here as "this backend never degrades," only as "this specific installed
version's `tool_search` has no code path that reports degrading."

CROSS-PATH RELIABILITY GAP -- A REAL, LIVE-REPRODUCED VENDOR BUG, NOT AN
ADAPTER BUG, WITH A CONFIRMED MITIGATION APPLIED BELOW
-------------------------------------------------------------------------
Live-reproduced during this rewrite (not from a GitHub issue -- found by
exercising this adapter's own multi-instance test setup): within a single
long-lived Python process, pointing `MEMPALACE_PALACE_PATH` at a SECOND,
never-before-seen palace directory and then immediately calling
`tool_search()` there can spuriously return `{"error": "No palace found",
...}` even though `tool_add_drawer()` against that exact same path, one
call earlier, succeeded and reported a real `drawer_id`. Root cause,
confirmed by reading `mempalace/mcp_server.py`'s `_get_client()`: its
client-cache invalidation keys off the CURRENT palace_path's
`chroma.sqlite3` file's `(inode, mtime)` compared against the PREVIOUSLY
cached values, but the surrounding code that decides whether to probe
`_refresh_vector_disabled_flag()` at all does not appear to always
re-trigger cleanly across a same-process path switch -- the net effect,
confirmed by direct reproduction, is a stale-negative on the very first
search against a freshly-switched path. A further reproduction also
showed switching BACK to an already-used path can return leftover results
from whichever path was visited in between, i.e. this is a same-process
cross-path staleness problem in both directions, not only "new path
first-search fails." The real, confirmed `tool_reconnect()` function
reliably clears this when called once immediately after the path changes
(live-verified: reconnect-then-add-then-search against a brand-new path
returned exactly the right content, zero errors, across repeated tries).

`_sync_mcp_palace_path()` below now detects a real path change (comparing
the current `MEMPALACE_PALACE_PATH` env value against the one this call is
about to set) and calls `tool_reconnect()` once when it fires, rather than
on every call -- so a single long-lived `MemPalaceAdapter` instance
pointed at one fixed storage path for its whole life (memtrust's normal
usage pattern: one adapter, one env var, for the duration of one eval run)
pays zero extra vendor calls, while a process that constructs multiple
`MemPalaceAdapter` instances against different storage paths (this
module's own test suite does exactly this) gets a real, working fix
instead of the silent corruption `_sync_mcp_palace_path()`'s pre-existing
docstring claimed was already safe, but -- per this reproduction -- was
not. `tool_reconnect()` itself failing is swallowed (best-effort only,
matching a real, confirmed vendor function that can itself report
`{"success": False, "message": "Chroma database missing", ...}` for a
palace that doesn't exist yet) -- the very next real call still surfaces
its own error normally if the underlying problem persists.

KNOWLEDGE-GRAPH STORAGE IGNORES `MEMPALACE_PALACE_PATH` ENTIRELY WHEN
CALLED AS A LIBRARY -- A REAL, CONFIRMED, SEPARATE LIMITATION
-------------------------------------------------------------------------
Live-reproduced and root-caused by reading `mempalace/mcp_server.py`'s
`_resolve_kg_path()`: it returns
`os.path.join(_config.palace_path, "knowledge_graph.sqlite3")` ONLY when a
module-level `_palace_flag_given` flag is `True` -- and that flag is set
exactly once, at import time, from `bool(_args.palace)`, i.e. whether the
process's `sys.argv` carried an explicit CLI `--palace` flag. A library
caller (this adapter, or any other in-process Python caller that never
went through `mempalace`'s own CLI argument parser) always has
`_palace_flag_given = False`, so `_resolve_kg_path()` unconditionally
returns `mempalace.knowledge_graph.DEFAULT_KG_PATH` --
`~/.mempalace/knowledge_graph.sqlite3`, a SINGLE FIXED FILE IN THE REAL
CALLING USER'S HOME DIRECTORY, completely ignoring
`MEMPALACE_PALACE_PATH`/this adapter's `MEMPALACE_STORAGE_PATH` for the
knowledge-graph subsystem specifically. Confirmed live: two
`MemPalaceAdapter` instances constructed against two different
`MEMPALACE_STORAGE_PATH` values in the same process/environment both read
and write the exact same physical KG file -- `kg_add()`/`kg_invalidate()`/
`kg_query()` below carry NO per-instance storage isolation at all, unlike
every drawer method above them. This is a real, load-bearing limitation
for any caller (including this module's own test suite -- see
`test_real_mempalace_kg_*` below, which uses a randomized, per-run-unique
subject entity specifically because it cannot assume a clean KG slate) and
should be treated as environment-global, shared state, not scoped to any
one adapter instance or palace directory.

Mode variants ("raw"/"AAAK") REMOVED -- CONFIRMED ABSENT, NOT JUST
UNCONFIRMED
-------------------------------------------------------------------------
The previous `supported_modes = ("raw", "AAAK")` (from
mempalace/mempalace#27) was already flagged LOW confidence -- a guessed
constructor/method parameter, never confirmed. It is now definitively
confirmed ABSENT: neither `tool_add_drawer` nor `tool_search` accepts a
`mode` keyword at all. `supported_modes` is now `()`; `store()`/`query()`
below still accept a `mode` parameter (required by
`MemoryBackendAdapter`'s abstract signature) and silently ignore it, per
that signature's own documented contract for adapters with no mode
variants.

Migration-rollback simulation (still NOT implemented here, unchanged
reasoning): see this module's git history for the original "no adapter in
this repo has real process/filesystem-lifecycle control over a live
backend" reasoning -- nothing about this rewrite changes that; still
covered instead by evals/migration_rollback.py's harness-side simulation.

MCP metadata-tool coverage (mempalace_status/list_wings/list_rooms):
unchanged by this rewrite -- `metadata_overview()`/
`list_metadata_categories()`/`list_metadata_subcategories()` below still
wrap the same confirmed-real `tool_status()`/`tool_list_wings()`/
`tool_list_rooms()` functions this module already used correctly before
this change (MemPalace/mempalace#1871, contributor alionar). They now
share `_get_mcp_tools()` with every other method in this file instead of
their own separate `_get_mcp_metadata_tools()` -- one lazy-import point
for the whole confirmed-real `mempalace.mcp_server` surface this adapter
uses, since there is no longer a second, fictional-API code path to keep
separate from it.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

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
#: adapter's own MEMPALACE_STORAGE_PATH -- see `_sync_mcp_palace_path()`.
_MCP_PALACE_PATH_ENV_VAR = "MEMPALACE_PALACE_PATH"

#: The real, confirmed per-record field `tool_search`'s own ranking is
#: actually driven by (`effective_distance`, surfaced to callers as
#: `similarity`) -- see the module docstring's "RANKING SIGNAL" section
#: for why this replaces the old importance/emotional_weight/weight/
#: authored_at keys, which belong to a different method this adapter never
#: calls.
_RANKING_METADATA_KEYS = ("similarity",)


def _classify_ranking_signal(records: list[MemoryRecord]) -> RankingSignal:
    """Inspect a query response's records for a ranking-relevant metadata
    field and report whether a real per-record signal appears to exist.

    See the module docstring's "RANKING SIGNAL" section for why this
    checks `similarity` (the field the real, confirmed `tool_search`
    actually sorts by) rather than the fictional-API-era
    importance/emotional_weight/weight/authored_at keys.

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
            return RankingSignal.MISSING_ORDERING_KEY
        if len(set(values)) == 1:
            return RankingSignal.MISSING_ORDERING_KEY
        return RankingSignal.SIGNAL_DRIVEN

    return RankingSignal.MISSING_ORDERING_KEY


def _record_metadata(item: dict[str, Any]) -> dict[str, str]:
    """Build a query() result item's metadata dict from a real, confirmed
    `tool_search` response item -- every top-level field except `text`
    (which becomes `MemoryRecord.content`), stringified, with `None`
    values dropped. Deliberately unopinionated about which keys exist
    (rather than hardcoding the exact confirmed field list) so a future
    `mempalace` version that adds or renames a field still surfaces it
    here instead of silently dropping it.
    """
    return {
        str(key): str(value) for key, value in item.items() if key != "text" and value is not None
    }


class _MCPToolsProtocol(Protocol):
    """Shape this adapter expects from `mempalace.mcp_server` (or a fake
    standing in for it in tests). Every method here is confirmed real and
    live-verified against the installed `mempalace` package (see the
    module docstring) -- unlike the removed `_PalaceProtocol`, whose
    method names were never more than an unconfirmed guess at an API that
    turned out not to exist.
    """

    def tool_status(self) -> dict[str, Any]: ...

    def tool_list_wings(self) -> dict[str, Any]: ...

    def tool_list_rooms(self, wing: str | None = None) -> dict[str, Any]: ...

    def tool_add_drawer(
        self,
        wing: str,
        room: str,
        content: str,
        source_file: str | None = None,
        added_by: str = "mcp",
    ) -> dict[str, Any]: ...

    def tool_search(
        self,
        query: str,
        limit: int = 5,
        wing: str | None = None,
        room: str | None = None,
        source_file: str | None = None,
        max_distance: float = 1.5,
        min_similarity: float | None = None,
        context: str | None = None,
    ) -> dict[str, Any]: ...

    def tool_update_drawer(
        self,
        drawer_id: str,
        content: str | None = None,
        wing: str | None = None,
        room: str | None = None,
    ) -> dict[str, Any]: ...

    def tool_delete_drawer(self, drawer_id: str) -> dict[str, Any]: ...

    def tool_reconnect(self) -> dict[str, Any]: ...

    def tool_kg_add(
        self,
        subject: str,
        predicate: str,
        object: str,
        valid_from: str | None = None,
        valid_to: str | None = None,
        source_closet: str | None = None,
        source_file: str | None = None,
        source_drawer_id: str | None = None,
    ) -> dict[str, Any]: ...

    def tool_kg_invalidate(
        self, subject: str, predicate: str, object: str, ended: str | None = None
    ) -> dict[str, Any]: ...

    def tool_kg_query(
        self, entity: str, as_of: str | None = None, direction: str = "both"
    ) -> dict[str, Any]: ...


@dataclass
class KGFactResult:
    """Result of MemPalaceAdapter.kg_add() -- see the module docstring's
    "CONTRADICTION/STALENESS DETECTION MOVED TO THE KG API" section.
    MemPalace-specific, not part of the shared MemoryBackendAdapter
    contract."""

    success: bool
    triple_id: str | None
    fact: str | None
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class KGInvalidateResult:
    """Result of MemPalaceAdapter.kg_invalidate(). MemPalace-specific."""

    success: bool
    fact: str | None
    ended: str | None
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class KGFact:
    """One fact returned by MemPalaceAdapter.kg_query() -- mirrors the
    real, confirmed `tool_kg_query()` per-fact shape field-for-field."""

    direction: str
    subject: str
    predicate: str
    object: str
    valid_from: str | None
    valid_to: str | None
    confidence: float | None
    source_closet: str | None
    current: bool


@dataclass
class KGQueryResult:
    """Result of MemPalaceAdapter.kg_query(). MemPalace-specific."""

    entity: str
    as_of: str | None
    facts: list[KGFact]
    count: int
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)


class MemPalaceAdapter(MemoryBackendAdapter):
    name = "mempalace"
    env_var = "MEMPALACE_STORAGE_PATH"
    supports_update = True
    #: Confirmed empty -- see the module docstring's "Mode variants"
    #: section. Neither tool_add_drawer nor tool_search accepts a `mode`
    #: keyword in the real, installed package.
    supported_modes: tuple[str, ...] = ()
    supports_metadata_overview = True

    #: Default `room` for store()/query()/update() when the caller's
    #: `metadata` dict passed to store() doesn't supply `metadata["room"]`
    #: explicitly -- see the module docstring's "WING/ROOM MAPPING"
    #: section.
    DEFAULT_ROOM = "memtrust"

    def __init__(self, mcp_tools: _MCPToolsProtocol | None = None) -> None:
        storage_path = os.environ.get(self.env_var)
        if not storage_path and mcp_tools is None:
            raise BackendNotConfiguredError(self.name, self.env_var)
        self._storage_path = storage_path
        #: See `_get_mcp_tools()` below -- the ONE lazy-import point for
        #: every real `mempalace.mcp_server` call this adapter makes,
        #: unless a fake is injected here (tests only; see
        #: tests/test_adapters.py). Replaces the removed
        #: `_palace`/`_PalaceProtocol`/`_get_palace()` machinery, which
        #: called a `mempalace.Palace` class that was never real -- see
        #: the module docstring's opening section.
        self._mcp_tools = mcp_tools

    def _get_mcp_tools(self) -> _MCPToolsProtocol:
        if self._mcp_tools is not None:
            return self._mcp_tools
        try:
            import mempalace.mcp_server as _mcp_server  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendAPIError(
                self.name,
                "the `mempalace` package is not installed, or its "
                "`mcp_server` submodule failed to import. Install it with "
                "`pip install mempalace` (see this module's docstring for "
                "the confirmed-real API surface this adapter calls).",
            ) from exc
        self._mcp_tools = cast(_MCPToolsProtocol, _mcp_server)
        return self._mcp_tools

    def _sync_mcp_palace_path(self) -> None:
        """Bridge this adapter's MEMPALACE_STORAGE_PATH into the real
        `mempalace.mcp_server` module's actual config env var,
        MEMPALACE_PALACE_PATH (confirmed different names). Re-set
        immediately before every call (not once at __init__ time) so a
        single process can point this adapter at more than one storage
        path across its lifetime; a no-op when `_storage_path` was never
        set (fake-only unit tests that inject `mcp_tools` directly).

        See the module docstring's "CROSS-PATH RELIABILITY GAP" section:
        when the env var's value is actually about to change (not the
        common case -- a single adapter's `_storage_path` never changes
        after `__init__`, so this only fires when some OTHER code in the
        same process, e.g. a second `MemPalaceAdapter` instance, last
        pointed `MEMPALACE_PALACE_PATH` somewhere else), this calls the
        real, confirmed `tool_reconnect()` once to clear mempalace's own
        stale client/vector-disabled cache -- live-verified necessary and
        sufficient to avoid a spurious "No palace found" on the very next
        real call.
        """
        if not self._storage_path:
            return
        previous = os.environ.get(_MCP_PALACE_PATH_ENV_VAR)
        os.environ[_MCP_PALACE_PATH_ENV_VAR] = self._storage_path
        if previous is not None and previous != self._storage_path:
            # Best-effort: a real failure here still surfaces normally
            # from the actual vendor call that follows this sync.
            with contextlib.suppress(Exception):
                self._get_mcp_tools().tool_reconnect()

    def metadata_overview(self) -> MetadataOverviewResult:
        """Real, confirmed library-level equivalent of MemPalace's
        `mempalace_status` MCP tool -- see the module docstring and
        MemPalace/mempalace#1871 for the O(N^2) full-collection-scan bug
        that made this code path worth covering at all.
        """
        timer = self._timed()
        tools = self._get_mcp_tools()
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
        tools = self._get_mcp_tools()
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
        tools = self._get_mcp_tools()
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

        Maps onto the real, confirmed `tool_add_drawer(wing=session_id,
        room=..., content=content, source_file=..., added_by=...)` -- see
        the module docstring's "WING/ROOM MAPPING" and "`store()`'s
        `metadata` PARAMETER" sections for exactly which `metadata` keys
        are read (`room`, `source_file`, `added_by`) and why the rest are
        not forwarded anywhere.

        `mode` is accepted (required by the abstract signature) and
        silently ignored -- see the module docstring's "Mode variants"
        section; the real API has no mode concept.

        `verify` behaves exactly as MemoryBackendAdapter.verify_store()
        documents, with one adapter-specific caveat: because the real
        `tool_search` response carries no per-record id (see the module
        docstring), verification here always resolves through
        verify_store()'s content-substring fallback branch, never its
        id-match branch.
        """
        timer = self._timed()
        tools = self._get_mcp_tools()
        self._sync_mcp_palace_path()
        metadata = metadata or {}
        room = metadata.get("room", self.DEFAULT_ROOM)
        source_file = metadata.get("source_file")
        added_by = metadata.get("added_by", "memtrust")
        try:
            raw = tools.tool_add_drawer(
                wing=session_id,
                room=room,
                content=content,
                source_file=source_file,
                added_by=added_by,
            )
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if not isinstance(raw, dict):
            raise BackendAPIError(
                self.name,
                f"tool_add_drawer() returned {type(raw).__name__}, expected dict",
            )
        if not raw.get("success"):
            raise BackendAPIError(
                self.name, raw.get("error") or "tool_add_drawer() reported failure"
            )
        memory_id = str(raw.get("drawer_id", ""))
        result = StoreResult(memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=raw)
        if verify:
            result.verified = self.verify_store(result, session_id, content)
        return result

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        """Maps onto the real, confirmed `tool_search(query=query,
        limit=top_k, wing=session_id)`. See the module docstring's
        "NO PER-RECORD ID", "CONFLICT SIGNAL", and "RANKING SIGNAL"
        sections for the honest, confirmed limitations this method's
        output now carries relative to the old, fictional-API adapter.
        """
        timer = self._timed()
        tools = self._get_mcp_tools()
        self._sync_mcp_palace_path()
        try:
            raw = tools.tool_search(query=query, limit=top_k, wing=session_id)
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if not isinstance(raw, dict):
            raise BackendAPIError(
                self.name,
                f"tool_search() returned {type(raw).__name__}, expected dict",
            )
        raw_results = raw.get("results")
        if raw_results is None:
            detail = f" (error: {raw['error']})" if "error" in raw else ""
            raise BackendAPIError(
                self.name,
                "tool_search() returned no 'results' key -- expected the "
                "confirmed real response shape "
                "{'query':..., 'filters':..., 'total_before_filter':..., "
                f"'results': [...]}}. Got keys: {sorted(raw.keys())}.{detail}",
            )

        records = [
            MemoryRecord(
                # See the module docstring's "NO PER-RECORD ID" section --
                # the real tool_search response never carries an id, and
                # this adapter deliberately does not guess one.
                memory_id="",
                content=str(item.get("text", "")),
                score=item.get("similarity"),
                created_at=item.get("created_at"),
                metadata=_record_metadata(item),
                raw=item,
            )
            for item in raw_results
        ]

        degraded_retrieval: RetrievalWarning | None = None
        raw_warnings = raw.get("warnings")
        if raw_warnings:
            if not isinstance(raw_warnings, list):
                raise BackendAPIError(
                    self.name,
                    "tool_search() response's 'warnings' field must be a "
                    f"list, got {type(raw_warnings).__name__}.",
                )
            warnings = [str(w) for w in raw_warnings]
            available_in_scope = raw.get("available_in_scope")
            if not isinstance(available_in_scope, int) or isinstance(available_in_scope, bool):
                available_in_scope = None
            degraded_retrieval = RetrievalWarning(
                warnings=warnings, available_in_scope=available_in_scope
            )

        ranking_signal = _classify_ranking_signal(records)
        return QueryResult(
            records=records,
            # See the module docstring's "CONFLICT SIGNAL" section -- the
            # real tool_search response has no invalidation marker at all;
            # this is now always NOT_APPLICABLE for drawer-backed queries.
            conflict_signal=ConflictSignal.NOT_APPLICABLE,
            latency_ms=timer.elapsed_ms(),
            ranking_signal=ranking_signal,
            degraded_retrieval=degraded_retrieval,
            raw=raw,
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        """Maps onto the real, confirmed `tool_update_drawer(drawer_id=
        memory_id, content=content)`. `session_id` is accepted (required
        by the abstract signature) and unused -- the real
        `tool_update_drawer` addresses a drawer by id alone, with no wing/
        room/session scoping parameter at all.

        `acknowledged=False` (not a raised BackendAPIError) reports a
        vendor-acknowledged "no such drawer" -- e.g. a memory_id that was
        already deleted, or never real to begin with. This does NOT
        happen for a legitimately chunked `store()`-returned memory_id --
        see the module docstring's "CHUNKED CONTENT" section for why that
        case works correctly despite `tool_add_drawer`'s own stale
        docstring claiming otherwise. A genuine transport/vendor exception
        still raises BackendAPIError, same as every other method here.
        """
        del session_id  # unused: the real tool_update_drawer has no scoping param
        timer = self._timed()
        tools = self._get_mcp_tools()
        self._sync_mcp_palace_path()
        try:
            raw = tools.tool_update_drawer(drawer_id=memory_id, content=content)
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if not isinstance(raw, dict):
            raise BackendAPIError(
                self.name,
                f"tool_update_drawer() returned {type(raw).__name__}, expected dict",
            )
        acknowledged = bool(raw.get("success"))
        new_id = str(raw.get("drawer_id", memory_id)) if acknowledged else memory_id
        return UpdateResult(
            memory_id=new_id, acknowledged=acknowledged, latency_ms=timer.elapsed_ms(), raw=raw
        )

    def delete(self, memory_id: str) -> DeleteResult:
        """Maps onto the real, confirmed `tool_delete_drawer(drawer_id=
        memory_id)`. Previously always raised BackendAPIError ("no
        documented delete/forget primitive was confirmed") -- that gap is
        closed now that the real package's actual delete tool is known.

        `success=False` (not a raised BackendAPIError) reports a vendor-
        acknowledged "no such drawer," matching DeleteResult's own
        documented convention ("success here reports the vendor's own
        acknowledgement shape... not whether the HTTP call itself
        succeeded") -- e.g. a memory_id that was already deleted. This
        does NOT happen for a legitimately chunked `store()`-returned
        memory_id -- see the module docstring's "CHUNKED CONTENT" section.
        """
        timer = self._timed()
        tools = self._get_mcp_tools()
        self._sync_mcp_palace_path()
        try:
            raw = tools.tool_delete_drawer(drawer_id=memory_id)
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if not isinstance(raw, dict):
            raise BackendAPIError(
                self.name,
                f"tool_delete_drawer() returned {type(raw).__name__}, expected dict",
            )
        success = bool(raw.get("success"))
        return DeleteResult(
            success=success, memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=raw
        )

    # -------------------------------------------------------------------
    # Knowledge-graph API -- MemPalace-specific, additive capability, NOT
    # part of the shared MemoryBackendAdapter contract. See the module
    # docstring's "CONTRADICTION/STALENESS DETECTION MOVED TO THE KG API"
    # section for why these exist alongside (not instead of) store()/
    # query()/update()/delete() above.
    # -------------------------------------------------------------------

    def kg_add(
        self,
        subject: str,
        predicate: str,
        object: str,  # noqa: A002 - mirrors the real tool_kg_add() parameter name exactly
        *,
        valid_from: str | None = None,
        valid_to: str | None = None,
        source_closet: str | None = None,
        source_file: str | None = None,
        source_drawer_id: str | None = None,
    ) -> KGFactResult:
        """Maps onto the real, confirmed `tool_kg_add(...)`."""
        timer = self._timed()
        tools = self._get_mcp_tools()
        self._sync_mcp_palace_path()
        try:
            raw = tools.tool_kg_add(
                subject=subject,
                predicate=predicate,
                object=object,
                valid_from=valid_from,
                valid_to=valid_to,
                source_closet=source_closet,
                source_file=source_file,
                source_drawer_id=source_drawer_id,
            )
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if not isinstance(raw, dict):
            raise BackendAPIError(
                self.name, f"tool_kg_add() returned {type(raw).__name__}, expected dict"
            )
        return KGFactResult(
            success=bool(raw.get("success")),
            triple_id=raw.get("triple_id"),
            fact=raw.get("fact"),
            latency_ms=timer.elapsed_ms(),
            raw=raw,
            error=raw.get("error"),
        )

    def kg_invalidate(
        self,
        subject: str,
        predicate: str,
        object: str,  # noqa: A002 - mirrors the real tool_kg_invalidate() parameter name exactly
        *,
        ended: str | None = None,
    ) -> KGInvalidateResult:
        """Maps onto the real, confirmed `tool_kg_invalidate(...)`.

        Live-verified: the real function succeeds even when no matching
        fact was ever added via kg_add() -- it does not check prior
        existence before recording an invalidation. Callers that need to
        distinguish "invalidated a real fact" from "invalidated a fact
        that was never there" must check that themselves (e.g. via
        kg_query() before calling this); this adapter does not
        second-guess the vendor's own accepted result.
        """
        timer = self._timed()
        tools = self._get_mcp_tools()
        self._sync_mcp_palace_path()
        try:
            raw = tools.tool_kg_invalidate(
                subject=subject, predicate=predicate, object=object, ended=ended
            )
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if not isinstance(raw, dict):
            raise BackendAPIError(
                self.name,
                f"tool_kg_invalidate() returned {type(raw).__name__}, expected dict",
            )
        return KGInvalidateResult(
            success=bool(raw.get("success")),
            fact=raw.get("fact"),
            ended=raw.get("ended"),
            latency_ms=timer.elapsed_ms(),
            raw=raw,
            error=raw.get("error"),
        )

    def kg_query(
        self, entity: str, *, as_of: str | None = None, direction: str = "both"
    ) -> KGQueryResult:
        """Maps onto the real, confirmed `tool_kg_query(...)`."""
        timer = self._timed()
        tools = self._get_mcp_tools()
        self._sync_mcp_palace_path()
        try:
            raw = tools.tool_kg_query(entity=entity, as_of=as_of, direction=direction)
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if not isinstance(raw, dict):
            raise BackendAPIError(
                self.name, f"tool_kg_query() returned {type(raw).__name__}, expected dict"
            )
        if "facts" not in raw:
            raise BackendAPIError(
                self.name,
                "tool_kg_query() returned no 'facts' key -- expected the "
                f"confirmed real response shape. Got keys: {sorted(raw.keys())}.",
            )
        facts = [
            KGFact(
                direction=str(f.get("direction", "")),
                subject=str(f.get("subject", "")),
                predicate=str(f.get("predicate", "")),
                object=str(f.get("object", "")),
                valid_from=f.get("valid_from"),
                valid_to=f.get("valid_to"),
                confidence=f.get("confidence"),
                source_closet=f.get("source_closet"),
                current=bool(f.get("current")),
            )
            for f in raw["facts"]
        ]
        return KGQueryResult(
            entity=str(raw.get("entity", entity)),
            as_of=raw.get("as_of"),
            facts=facts,
            count=int(raw.get("count", len(facts))),
            latency_ms=timer.elapsed_ms(),
            raw=raw,
        )
