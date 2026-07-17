"""Adapter for self-hosted `graphiti-core` (https://github.com/getzep/graphiti),
the open-source temporal-knowledge-graph engine that also powers Zep Cloud.

Confidence: MEDIUM on wire-level shape, LOW on live end-to-end behavior.

`zep_graphiti_adapter.py`'s `ZepGraphitiAdapter` targets Zep Cloud's hosted
REST API. This adapter is the second, separately-configured adapter
docs/methodology.md already calls for under "Why Zep targets the hosted
Cloud API, not self-hosted Graphiti": *"If self-hosted Graphiti support is
wanted later, it should be a second adapter (e.g.
`zep_graphiti_selfhosted_adapter.py`) with its own configuration story, not
a silent branch inside this one."* That is exactly what this file is.

## Why this adapter exists

Ten real, independently-verified graphiti-core bugs live entirely in the
self-hosted library layer that `ZepGraphitiAdapter` (hosted REST) cannot
reach at all, because Zep Cloud's REST surface never exposes graphiti-core's
internals to a caller:

  * **getzep/graphiti#1302** (open as of this build): `lucene_sanitize()`
    in `graphiti_core/helpers.py` builds a `str.translate()` escape map
    that includes every uppercase `O`, `R`, `N`, `T`, `A`, `D` character,
    intending to escape the Lucene boolean operators `AND`/`OR`/`NOT` --
    but the map operates per-character, not per-token, so it backslash-
    escapes those letters *anywhere* they appear in a query string (e.g.
    "ORION" becomes "\\O\\RION"), silently degrading BM25 full-text
    ranking for any query containing those letters. No exception, no
    error -- just worse-ranked (or unranked) results. Confirmed by fetching
    `graphiti_core/helpers.py` from `getzep/graphiti`'s `main` branch
    directly on 2026-07-16; the six single-letter translate entries
    (`'O': r'\\O', 'R': r'\\R', 'N': r'\\N', 'T': r'\\T', 'A': r'\\A',
    'D': r'\\D'`) are present verbatim, confirming the bug is still live
    upstream as of this adapter's build.
  * **getzep/graphiti#836** (open as of this build): `add_episode(...,
    update_communities=True)` raises `ValueError: too many values to
    unpack` (or "not enough values to unpack") whenever the episode's
    extracted node count is not exactly 2. Confirmed by fetching
    `graphiti_core/graphiti.py` and
    `graphiti_core/utils/maintenance/community_operations.py` from
    `main` on 2026-07-16: `add_episode()`'s community-update branch does
        `communities, community_edges = await semaphore_gather(
            *[update_community(...) for node in nodes], ...
        )`
    but `semaphore_gather` returns a plain `list` with one 2-tuple
    `(list[CommunityNode], list[CommunityEdge])` per node in `nodes` --
    so this line only succeeds when `len(nodes) == 2`. Any episode that
    extracts 0, 1, or 3+ entities (the overwhelmingly common case) raises
    `ValueError` from this unpack, not from anything resembling
    documented, expected behavior.
  * **getzep/graphiti#920** (open as of this build, contributor
    markwkiehl): `add_episode(..., reference_time=<tz-aware datetime>)`
    raises `TypeError: can't compare offset-naive and offset-aware
    datetimes` from `resolve_edge_contradictions()` in
    `graphiti_core/utils/maintenance/edge_operations.py` (line ~371,
    `edge.valid_at < resolved_edge.valid_at`), reached via
    `add_episode()` -> `resolve_extracted_edges()` ->
    `resolve_extracted_edge()` -> `resolve_edge_contradictions()`.
    Confirmed via the issue's own filed traceback (`gh issue view 920
    --repo getzep/graphiti`, fetched 2026-07-16): a caller who passes a
    timezone-*aware* `reference_time` (e.g. `datetime.now(timezone.utc)`,
    which is exactly what Python best practice recommends, and exactly
    what this adapter's own `store()` passes -- see `datetime.now(UTC)`
    below) can hit a stored edge whose `valid_at` was persisted as
    timezone-*naive*, and the bare `<` comparison between the two raises
    instead of normalizing either side first.
  * **getzep/graphiti#1013** (merged/fixed upstream): the Neo4j bulk
    edge-save Cypher query used to build its `SET` clause from an
    enumerated field list that omitted `EntityEdge.attributes` and
    `reference_time`, so a bulk-saved edge could silently come back
    missing properties a single-edge `save()` call would have written.
    Confirmed fixed on `main` as of 2026-07-16:
    `graphiti_core/models/edges/edge_db_queries.py`'s
    `get_entity_edge_save_bulk_query()` now emits `SET e = edge` for the
    Neo4j provider (the default case; FalkorDB's own branch of the same
    function uses `SET r = edge`, same shape, different Cypher variable
    name) -- an unconditional property copy, not an enumerated list --
    which is the documented fix shape for #1013.
  * **getzep/graphiti#1001** (closed via #1013): FalkorDB's older
    `add_triplet()`-era edge-creation path created edges via `MATCH`
    (which silently no-ops if either endpoint node is absent) and never
    set `source_node_uuid`/`target_node_uuid` as edge properties at all.
    Confirmed closed: `add_triplet()` no longer exists anywhere in
    `graphiti_core/driver/falkordb_driver.py` on `main` as of 2026-07-16
    -- the FalkorDB driver was rewritten around a
    `graphiti_core/driver/falkordb/operations/` module structure that
    post-dates this bug report.
  * **getzep/graphiti#1222** (closed, superseded by #1475): a query whose
    sanitized/stopword-filtered token list comes back empty -- e.g. an
    empty query string, the exact shape `explore_node` hits when called
    with only a `node_uuid` and no `node_name` -- makes
    `build_fulltext_query()` in `graphiti_core/driver/falkordb_driver.py`
    append empty parentheses to the group filter, producing the invalid
    RediSearch syntax `(@group_id:"...") ()`, raised as `RediSearch:
    Syntax error at offset 22 near my_graph`. Confirmed via the PR's own
    filed reproduction (`gh issue view 1222 --repo getzep/graphiti`,
    fetched 2026-07-16); closed as a duplicate of #1475 ("taking as the
    FalkorDB fulltext query-sanitization fix"), which is the PR that
    actually closes this gap upstream, not #1222 itself.
  * **getzep/graphiti#1183** (merged): before this fix, `sanitize()`'s
    character-replacement map in the same `falkordb_driver.py` omitted
    pipe (`|`), slash (`/`), and backslash (`\\`) -- episode text
    containing those characters (e.g. `"install.sh | bash"`, the PR's own
    reported production trigger) survived sanitization, tokenized on
    whitespace into a stray `|` token, and got rejoined with `" | "` as a
    RediSearch OR separator, producing an adjacent-pipe malformed query
    (`"sh | | | bash"`) RediSearch rejects as `RediSearch: Syntax error at
    offset 178 near sh`. Confirmed via the merged PR's own filed
    reproduction and diff (`gh pr view 1183 --repo getzep/graphiti`,
    fetched 2026-07-16): the fix added those three characters to
    `sanitize()`'s map and filters empty tokens before pipe-joining in
    `build_fulltext_query()`.

  * **getzep/graphiti#1467** (open as of this build, contributor
    elimydlarz): `GeminiEmbedder.create_batch()`
    (`graphiti_core/embedder/gemini.py`) special-cases `batch_size=1` only
    for `"gemini-embedding-001"` -- confirmed by fetching that file from
    `main` on 2026-07-16, the `elif batch_size is None:` branch falls
    straight to `DEFAULT_BATCH_SIZE = 100` for every other model name,
    including `"gemini-embedding-2-preview"`/`"gemini-embedding-2"`. Those
    two models return exactly one embedding per `embed_content()` call
    regardless of how many strings are in the batch, and `create_batch()`
    never checks `len(result.embeddings) != len(batch)` before returning,
    so it silently hands back a too-short list. graphiti-core's own dedup
    pipeline (`_semantic_candidate_search()` in
    `graphiti_core/utils/maintenance/node_operations.py`, reached from
    `add_episode()` once 2+ entities are extracted) then zips the
    extracted-node list against this too-short embeddings list with
    `strict=True`, raising `ValueError: zip() argument 2 is shorter than
    argument 1`. This adapter has no code path to select a Gemini embedder
    at all before this build -- see "GeminiEmbedder support" below.
  * **getzep/graphiti#1525** (closed, contributor maui314159, fixed
    upstream via merged PR#1531): a NUL byte (`\\x00`) embedded in episode
    content -- common in text extracted from PDFs/PPTX -- survives
    FalkorDB's client-side query-parameter serialization
    (`falkordb/helpers.py::quote_string`, which only escapes backslash and
    double-quote) and makes FalkorDB's parser reject the entire bulk
    episode-save query with `redis.exceptions.ResponseError: Failed to
    parse query parameter '<name>' value`, silently dropping every episode
    in that call. Confirmed via the issue's own filed reproduction and
    PR#1531's diff (`gh issue view 1525`/`gh pr view 1531 --repo
    getzep/graphiti`, fetched 2026-07-16); already fixed on `main`, so this
    signal exists to classify the shape if reproduced against an
    older/unpinned graphiti-core version, same role every other
    already-fixed `CrashSignal` member here plays.

  * **getzep/graphiti#1625** (open as of this build, contributor pcy06):
    FalkorDB's Cypher query for `EpisodeNodeOperations.retrieve_episodes()`
    intends to filter with `e.valid_at <= $reference_time`, but FalkorDB
    can return rows for which that same expression evaluates `False` when
    projected as a column -- a future-dated episode can leak into a
    point-in-time query. Confirmed via the issue's own filed reproduction
    (`gh issue view 1625 --repo getzep/graphiti`, fetched 2026-07-16),
    which demonstrates a `valid_at=2024-03-01` episode returned for a
    `reference_time=2024-02-01` query. This adapter's `query()` never
    calls `retrieve_episodes()` at all -- it calls `Graphiti.search()`,
    which returns edges, never episodes -- so this bug was previously
    entirely unreachable through this adapter. See `retrieve_episodes()`
    below and `evals/episode_temporal_leak.py`, the new eval this build
    added specifically to detect (not fix -- the real bug lives inside
    graphiti-core's FalkorDB driver's Cypher query construction) this
    leak shape.

## GeminiEmbedder support

Set `GRAPHITI_EMBEDDER_PROVIDER=gemini` (or pass `embedder_provider="gemini"`
to the constructor) plus `GRAPHITI_GEMINI_API_KEY` (or `gemini_api_key=`) to
construct a real `graphiti_core.embedder.gemini.GeminiEmbedder` and pass it
as `Graphiti(embedder=...)` -- confirmed against the real, current
`Graphiti.__init__` signature on `main` (2026-07-16), which accepts an
optional `embedder: EmbedderClient | None` and falls back to its own
default `OpenAIEmbedder()` when `None`, exactly this adapter's prior,
unconfigurable behavior. `GRAPHITI_GEMINI_EMBEDDING_MODEL` (or
`gemini_embedding_model=`) optionally overrides the embedding model name
(e.g. `"gemini-embedding-2-preview"`, the model getzep/graphiti#1467
concerns); left unset, `GeminiEmbedderConfig`'s own default
(`"text-embedding-001"`) applies. Leaving `GRAPHITI_EMBEDDER_PROVIDER`
unset (the default) is fully backward compatible: `_build_embedder()`
returns `None`, and `Graphiti(embedder=None)` behaves identically to never
passing the keyword at all. This is what makes getzep/graphiti#1467's bug
class reachable through this adapter at all -- see `CrashSignal
.EMBEDDING_BATCH_COUNT_MISMATCH` in base.py and `_classify_crash()` below
for the classification this build added alongside it. Requires the
optional `graphiti-core[google-genai]` extra; a missing `google-genai`
install raises `BackendAPIError` naming that extra, the same "fail loudly
and specifically" convention `_get_client()`'s FalkorDB-extra check
already establishes.

None of the above nine confirmations came from running graphiti-core
against a live Neo4j or FalkorDB instance in this environment -- seven came
from fetching the real source files from `getzep/graphiti`'s `main` branch
on GitHub (`raw.githubusercontent.com/getzep/graphiti/main/...`) during
this build (five on 2026-07-16, two more -- #1467, #1525 -- also on
2026-07-16), and reading them directly; #1222 and #1183 came from reading
each issue/PR's own filed reproduction and diff via `gh issue view`/`gh pr
view` the same day. That is stronger evidence than documentation, but it is
still a snapshot of an unpinned, moving branch (or, for #1222/#1183/#1525,
a point-in-time reading of the issue tracker), and it is not a substitute
for exercising this adapter against a real deployment. See "What this
adapter does NOT prove" below.

## Design: why a separate class/file, not a flag on ZepGraphitiAdapter

Same reasoning `Mem0SelfHostedAdapter` documents in `mem0_adapter.py`:
hosted Zep Cloud and self-hosted graphiti-core are materially different
deployment shapes (REST vs. a direct Python client against a graph
driver), with different configuration stories (one API key vs. a database
connection) and different confidence levels. Collapsing them into one
class with an if/else would hide that from callers and from
docs/methodology.md's confidence table.

## Configuration: two backends, one env-var contract

Self-hosted graphiti-core has no single API key -- it needs a running
graph database (Neo4j by default, or FalkorDB) plus its own LLM/embedder
credentials for entity extraction. This adapter is gated on
`GRAPHITI_NEO4J_URI` (its primary, class-level `env_var`, consistent with
the "one env var, or SKIPPED" contract every other adapter follows) OR
`GRAPHITI_FALKORDB_URL` as an alternate. `BackendNotConfiguredError` is
only raised when *neither* is set. Supporting env vars:

  * `GRAPHITI_NEO4J_URI` -- e.g. `bolt://localhost:7687`
  * `GRAPHITI_NEO4J_USER` -- defaults to `"neo4j"` if unset
  * `GRAPHITI_NEO4J_PASSWORD`
  * `GRAPHITI_FALKORDB_URL` -- e.g. `redis://localhost:6379`; when set,
    this takes precedence over the Neo4j variables and a `FalkorDriver`
    is constructed instead of the default `Neo4jDriver`.
  * `GRAPHITI_UPDATE_COMMUNITIES` -- `"1"`/`"true"`/`"yes"` (case
    insensitive) threads `update_communities=True` through every
    `add_episode()` call this adapter makes, which is the toggle that
    would reach getzep/graphiti#836's code path. Defaults to `False`.
    Can also be set per-instance via the `update_communities` constructor
    kwarg, which takes precedence over the env var when explicitly passed
    (not `None`).

graphiti-core's real public API (`Graphiti.add_episode()`,
`Graphiti.search()`, `Graphiti.remove_episode()`) is entirely `async def`.
`MemoryBackendAdapter`'s `store()`/`query()`/`update()`/`delete()` are
synchronous, matching every other adapter in this repo -- this adapter
bridges the two with `asyncio.run(...)` inside each sync method, the same
pattern any synchronous caller of an async-only library has to use.

## What this adapter does NOT prove

This adapter was built and unit-tested (see tests/test_adapters.py)
entirely against a `graphiti_core`-shaped Protocol double -- the real
`graphiti-core` package is not installed in this build environment
(`ModuleNotFoundError: No module named 'graphiti_core'`), and no Neo4j or
FalkorDB instance was started or reached during this build. That means:

  * The exact constructor/method signatures this file calls
    (`Graphiti(uri=, user=, password=)`, `Graphiti(graph_driver=)`,
    `FalkorDriver(host=, port=, username=, password=)`,
    `add_episode(name=, episode_body=, source_description=,
    reference_time=, group_id=, update_communities=)`,
    `search(query, group_ids=, num_results=)`,
    `remove_episode(episode_uuid)`) were confirmed by reading the real
    source on GitHub, not by importing and calling the real package.
  * The `update_communities` toggle demonstrably threads through to
    `add_episode()` (see `store()` below and its unit test), but this
    build has no way to demonstrate it actually *triggers*
    getzep/graphiti#836's `ValueError` without a live Neo4j instance and
    real LLM credentials to run entity extraction against -- the bug
    lives inside `semaphore_gather`'s interaction with however many
    entities the LLM extraction step returns, which nothing in this
    adapter's mocked test path exercises.
  * `lucene_sanitize()` (getzep/graphiti#1302) is exercised nowhere in
    this adapter's own code -- it is an internal helper graphiti-core's
    search pipeline calls on the caller's query string before it ever
    reaches this adapter. This adapter cannot patch, bypass, or directly
    unit-test that function; it can only route eval queries containing
    the trigger characters (see tests/fixtures/contradiction_cases.json)
    at a live self-hosted instance, so a contributor who runs this
    adapter against a real deployment can compare BM25 ranking with and
    without those letters and observe the degradation directly.
  * `EDGE_INTEGRITY_VIOLATION` (getzep/graphiti#1013/#1001) is checked at
    the harness level, in `evals/contradiction.py`'s `classify_case()`,
    against whatever `source_node_uuid`/`target_node_uuid` values this
    adapter's `query()` puts in `MemoryRecord.raw`. Since both bugs are
    confirmed fixed/closed on the version of graphiti-core this adapter
    was built against, a live run today should not actually trigger this
    signal -- it exists so the check is *available* if this adapter is
    ever pinned to, or someone's self-hosted deployment happens to run,
    an older graphiti-core version where the bug is still present.
  * `store()`'s `CrashSignal` classification (`CrashSignal.UNPACK_ERROR`
    for #836's shape, `CrashSignal.TYPE_COMPARISON_ERROR` for #920's
    shape -- see `_classify_crash()` below and `CrashSignal` in
    `base.py`) is unit-tested against fake clients that raise the exact
    `ValueError`/`TypeError` message strings each real issue's filed
    traceback reports (confirmed via `gh issue view` for both issues,
    2026-07-16). That proves this adapter correctly recognizes those two
    message shapes once graphiti-core has already raised them -- it does
    NOT prove, and does not attempt to prove, that this adapter's own
    calls into `add_episode()` actually trigger either crash against a
    live instance (same limitation as the `update_communities` bullet
    above), and it does nothing to prevent either crash or fix
    graphiti-core itself. It is a diagnostic classifier applied after the
    fact to whatever exception a live instance (if reached) raises, nothing
    more.
  * `query()`'s `CrashSignal` classification (`CrashSignal
    .QUERY_SANITIZATION_ERROR` for #1222's/#1183's shared "RediSearch:
    Syntax error ..." shape -- same `_classify_crash()` and `CrashSignal`
    referenced above) is unit-tested against fake clients that raise the
    exact `RediSearch: Syntax error at offset N near ...` message strings
    each PR's own filed reproduction reports (`gh issue view 1222`/
    `gh pr view 1183 --repo getzep/graphiti`, fetched 2026-07-16), for
    both an empty query string (#1222's trigger) and a query containing
    pipe/slash characters (#1183's trigger). Same honesty caveat as the
    `store()` bullet immediately above: this proves the classifier
    recognizes the shared crash *shape* once a live FalkorDB instance has
    already raised it, not that this adapter's own `query()` calls
    actually trigger a RediSearch syntax error against a live instance --
    the fixtures this adapter's own mocked test double uses never
    exercise FalkorDB's real `sanitize()`/`build_fulltext_query()`
    functions at all, the same limitation the `lucene_sanitize()` bullet
    above already states for #1302.

Anyone relying on this adapter to reproduce any of the seven issues above
against a real deployment should verify against a live Neo4j/FalkorDB
instance first, exactly the same caveat docs/methodology.md already
states for `Mem0SelfHostedAdapter`, `MemPalaceAdapter`, and
`OpenVikingAdapter`.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlsplit

from memtrust.adapters.base import (
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
    CrashSignal,
    DeleteResult,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    StoreResult,
    UpdateResult,
)

DEFAULT_NEO4J_USER = "neo4j"


class _GraphitiProtocol(Protocol):
    """Shape this adapter expects from a real `graphiti_core.Graphiti`
    instance (Neo4j- or FalkorDB-backed). Defined as a Protocol, not
    imported from the package, so tests can inject a fake/mock
    implementation without the real `graphiti-core` package (and its
    Neo4j/FalkorDB driver dependencies) installed -- the same convention
    `mempalace_adapter.py`'s `_PalaceProtocol` already establishes in this
    repo.
    """

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        group_id: str | None = None,
        update_communities: bool = False,
    ) -> Any: ...

    async def search(
        self, query: str, group_ids: list[str] | None = None, num_results: int = 10
    ) -> list[Any]: ...

    async def remove_episode(self, episode_uuid: str) -> None: ...

    async def close(self) -> None: ...

    driver: Any
    """The real `Graphiti` instance's own `.driver` attribute (a
    `GraphDriver` -- `Neo4jDriver` or `FalkorDriver`), confirmed present on
    every real `Graphiti` instance by reading `graphiti_core/graphiti.py`'s
    `__init__` on 2026-07-16 (`self.driver = graph_driver or Neo4jDriver(...)`).
    Only accessed by `retrieve_episodes()` below, via
    `driver.episode_node_ops` -- a real, confirmed property on `GraphDriver`
    (`graphiti_core/driver/driver.py`) exposing an `EpisodeNodeOperations`
    instance. See `retrieve_episodes()` and
    `evals/episode_temporal_leak.py` for why this needs the raw driver
    rather than going through `Graphiti.search()` (which only ever returns
    edges -- `EntityEdge` -- never episodes)."""


def _to_plain_dict(obj: object) -> dict[str, object]:
    """Best-effort conversion of a graphiti_core Pydantic model (or a test
    double standing in for one) to a plain dict, without importing
    pydantic or graphiti_core here -- this adapter only duck-types against
    whatever `model_dump()`/dict shape it's handed, matching how
    `_GraphitiProtocol` above never imports the real package either."""
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        result = model_dump()
        return result if isinstance(result, dict) else {}
    if isinstance(obj, dict):
        return obj
    try:
        return dict(vars(obj))
    except TypeError:
        return {}


def _classify_crash(exc: Exception) -> CrashSignal:
    """Pattern-match a caught vendor exception's type and message against
    the three known graphiti-core crash shapes `CrashSignal` (base.py)
    documents, and fall back to `CrashSignal.UNKNOWN` for everything else.

    This is deliberately narrow: the first two checks look at the
    exception's Python type first (ValueError vs. TypeError), then a
    specific substring within that type's message, so an unrelated
    ValueError (say, a malformed UUID) or an unrelated TypeError (say, a
    wrong-argument-count bug) is never miscategorized as getzep/graphiti
    #836 or #920. Matching a substring of `str(exc)` is the only surface
    available here -- once a vendor library has already raised, this
    adapter has no access to the original traceback's source location,
    only what the exception itself reports. See CrashSignal's docstring in
    base.py for the full honesty caveat: this recognizes a known crash
    *shape*, it does not prove (or require) that the crash occurred at the
    exact upstream line this adapter's module docstring cites.

    The third and fourth checks (RediSearch `Syntax error`, getzep/graphiti
    #1222/#1183; "failed to parse query parameter", getzep/graphiti#1525)
    are message-only, with no `isinstance` gate: the real exceptions both
    FalkorDB's RediSearch fulltext-query path and its query-parameter
    serialization raise are typically a `redis`-client `ResponseError`,
    and this adapter deliberately does not import the optional
    `redis`/`falkordb` packages just to `isinstance`-check against them
    (see `_get_client()` above -- graphiti-core's FalkorDB extra is
    optional and may not be installed). Requiring both "redisearch" and
    "syntax error" as substrings (not "syntax error" alone) keeps the
    third check from ever matching an unrelated Python `SyntaxError` or a
    different vendor's syntax-error message; requiring both "failed to
    parse query parameter" and "value" in the fourth keeps it from
    matching an unrelated parse failure that merely mentions "value".
    """
    message = str(exc).lower()
    if isinstance(exc, ValueError) and "values to unpack" in message:
        # getzep/graphiti#836: add_episode(update_communities=True)'s
        # `communities, community_edges = await semaphore_gather(...)`
        # unpack only succeeds when semaphore_gather returns exactly 2
        # items -- covers both "too many values to unpack" and "not
        # enough values to unpack" phrasings of the same shape.
        return CrashSignal.UNPACK_ERROR
    if isinstance(exc, TypeError) and "offset-naive and offset-aware" in message:
        # getzep/graphiti#920: comparing a tz-naive stored timestamp
        # against a tz-aware datetime in edge-contradiction resolution.
        return CrashSignal.TYPE_COMPARISON_ERROR
    if isinstance(exc, ValueError) and "zip()" in message and "shorter than" in message:
        # getzep/graphiti#1467: GeminiEmbedder.create_batch() silently
        # returns fewer vectors than inputs for the gemini-embedding-2*
        # model family, later tripping a strict-zip ValueError several
        # frames away in graphiti-core's own dedup pipeline. See
        # CrashSignal.EMBEDDING_BATCH_COUNT_MISMATCH's docstring in
        # base.py for the full citation trail.
        return CrashSignal.EMBEDDING_BATCH_COUNT_MISMATCH
    if "redisearch" in message and "syntax error" in message:
        # getzep/graphiti#1222 (empty sanitized query -> "(@group_id:...)
        # ()") / #1183 (unescaped |, /, \ in episode text -> an empty
        # token between RediSearch OR-pipe delimiters) -- both raise the
        # identical "RediSearch: Syntax error at offset N near ..." shape
        # from FalkorDB's fulltext-query path. See CrashSignal
        # .QUERY_SANITIZATION_ERROR's docstring in base.py for the full
        # citation trail.
        return CrashSignal.QUERY_SANITIZATION_ERROR
    if "failed to parse query parameter" in message and "value" in message:
        # getzep/graphiti#1525 (maui314159, fixed upstream via merged
        # PR#1531): a NUL byte embedded in episode content (commonly from
        # PDF/PPTX extraction) survives FalkorDB's client-side parameter
        # serialization and makes FalkorDB's parser reject the entire
        # bulk episode-save query with "Failed to parse query parameter
        # '<name>' value" -- silently dropping every episode in that
        # call, not just the offending one. See CrashSignal
        # .QUERY_PARAMETER_PARSE_ERROR's docstring in base.py for the
        # full citation trail.
        return CrashSignal.QUERY_PARAMETER_PARSE_ERROR
    return CrashSignal.UNKNOWN


def _parse_falkordb_url(url: str) -> tuple[str, int, str | None, str | None]:
    """Parse `GRAPHITI_FALKORDB_URL` into the (host, port, username,
    password) tuple `graphiti_core.driver.falkordb_driver.FalkorDriver`'s
    constructor accepts. Accepts a bare `host:port` as well as a full
    `redis://user:pass@host:port` URL -- FalkorDB is Redis-protocol
    compatible and its own docs use `redis://` URLs, but a caller who just
    writes `localhost:6379` should not be forced to add a scheme.
    """
    candidate = url if "://" in url else f"redis://{url}"
    parsed = urlsplit(candidate)
    return (parsed.hostname or "localhost", parsed.port or 6379, parsed.username, parsed.password)


class ZepGraphitiSelfHostedAdapter(MemoryBackendAdapter):
    """Adapter for self-hosted `graphiti-core` (Neo4j by default, optional
    FalkorDB). See this module's docstring for the confidence level, the
    five bug classes this adapter exists to make reachable, and exactly
    what was -- and was not -- confirmed against real source vs. a live
    instance.
    """

    name = "graphiti_selfhosted"
    env_var = "GRAPHITI_NEO4J_URI"
    supports_update = True

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        falkordb_url: str | None = None,
        update_communities: bool | None = None,
        embedder_provider: str | None = None,
        gemini_api_key: str | None = None,
        gemini_embedding_model: str | None = None,
        graphiti_client: _GraphitiProtocol | None = None,
    ) -> None:
        resolved_neo4j_uri = neo4j_uri or os.environ.get(self.env_var)
        resolved_falkordb_url = falkordb_url or os.environ.get("GRAPHITI_FALKORDB_URL")
        if graphiti_client is None and not resolved_neo4j_uri and not resolved_falkordb_url:
            # Either GRAPHITI_NEO4J_URI or GRAPHITI_FALKORDB_URL configures
            # this adapter -- env_var names the primary one for the error
            # message the same way every other adapter's does, but the
            # constructor itself accepts either. See module docstring.
            raise BackendNotConfiguredError(self.name, self.env_var)
        if update_communities is None:
            update_communities = os.environ.get(
                "GRAPHITI_UPDATE_COMMUNITIES", ""
            ).strip().lower() in ("1", "true", "yes")
        self._update_communities = update_communities
        self._client = graphiti_client
        self._neo4j_uri = resolved_neo4j_uri
        self._neo4j_user = neo4j_user or os.environ.get("GRAPHITI_NEO4J_USER", DEFAULT_NEO4J_USER)
        self._neo4j_password = neo4j_password or os.environ.get("GRAPHITI_NEO4J_PASSWORD")
        self._falkordb_url = resolved_falkordb_url
        # Embedder selection -- see "GeminiEmbedder support" in this
        # module's docstring for getzep/graphiti#1467's motivation.
        # `None`/unset (the default) leaves graphiti-core's own default
        # OpenAIEmbedder() untouched, same backward-compatible-default
        # convention every other optional field on this adapter follows.
        self._embedder_provider = embedder_provider or os.environ.get("GRAPHITI_EMBEDDER_PROVIDER")
        self._gemini_api_key = gemini_api_key or os.environ.get("GRAPHITI_GEMINI_API_KEY")
        self._gemini_embedding_model = gemini_embedding_model or os.environ.get(
            "GRAPHITI_GEMINI_EMBEDDING_MODEL"
        )
        gemini_requested = self._embedder_provider == "gemini"
        if gemini_requested and graphiti_client is None and not self._gemini_api_key:
            raise BackendNotConfiguredError(self.name, "GRAPHITI_GEMINI_API_KEY")

    def _build_embedder(self) -> Any:
        """Construct the `EmbedderClient` to pass as `Graphiti(embedder=...)`,
        or `None` to let graphiti-core fall back to its own default
        `OpenAIEmbedder()` -- see `__init__`'s `embedder_provider` param and
        this module's docstring, "GeminiEmbedder support" section.

        Only `"gemini"` is wired up today -- the one provider
        getzep/graphiti#1467 concerns. An unrecognized `embedder_provider`
        value raises BackendAPIError naming exactly what this adapter
        supports, rather than silently falling back to the OpenAI default
        a caller who set it clearly did not intend.
        """
        if self._embedder_provider is None:
            return None
        if self._embedder_provider != "gemini":
            raise BackendAPIError(
                self.name,
                f"unsupported embedder_provider {self._embedder_provider!r}; this adapter "
                "only wires up 'gemini' today -- the provider getzep/graphiti#1467 concerns. "
                "See module docstring.",
            )
        try:
            from graphiti_core.embedder.gemini import (  # type: ignore[import-not-found]
                GeminiEmbedder,
                GeminiEmbedderConfig,
            )
        except ImportError as exc:
            raise BackendAPIError(
                self.name,
                "graphiti-core is installed without Gemini embedder support. Install "
                "`pip install graphiti-core[google-genai]`, or unset "
                "GRAPHITI_EMBEDDER_PROVIDER to use graphiti-core's default OpenAIEmbedder.",
            ) from exc
        config_kwargs: dict[str, Any] = {"api_key": self._gemini_api_key}
        if self._gemini_embedding_model:
            config_kwargs["embedding_model"] = self._gemini_embedding_model
        return GeminiEmbedder(GeminiEmbedderConfig(**config_kwargs))

    def _get_client(self) -> _GraphitiProtocol:
        if self._client is not None:
            return self._client
        try:
            from graphiti_core import Graphiti  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendAPIError(
                self.name,
                "the `graphiti-core` package is not installed. Install it "
                "with `pip install graphiti-core` (add the `[falkordb]` "
                "extra for FalkorDB support, or `[google-genai]` for Gemini "
                "embedder support). See docs/methodology.md for this "
                "adapter's confidence level and what was confirmed against "
                "the real graphiti_core source vs. a live instance.",
            ) from exc

        embedder = self._build_embedder()

        if self._falkordb_url:
            try:
                from graphiti_core.driver.falkordb_driver import (  # type: ignore[import-not-found]
                    FalkorDriver,
                )
            except ImportError as exc:
                raise BackendAPIError(
                    self.name,
                    "graphiti-core is installed without FalkorDB support. "
                    "Install `pip install graphiti-core[falkordb]`, or unset "
                    "GRAPHITI_FALKORDB_URL and configure GRAPHITI_NEO4J_URI "
                    "instead.",
                ) from exc
            host, port, username, password = _parse_falkordb_url(self._falkordb_url)
            driver = FalkorDriver(host=host, port=port, username=username, password=password)
            self._client = Graphiti(graph_driver=driver, embedder=embedder)
        else:
            self._client = Graphiti(
                uri=self._neo4j_uri,
                user=self._neo4j_user,
                password=self._neo4j_password,
                embedder=embedder,
            )
        return self._client

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
        *,
        verify: bool = False,
    ) -> StoreResult:
        """Store an episode via `Graphiti.add_episode()`.

        `mode` is a no-op, same convention as every other adapter (see
        `MemoryBackendAdapter.supported_modes`). `metadata` is also
        accepted-and-ignored: graphiti-core's real `add_episode()`
        signature (confirmed against `graphiti_core/graphiti.py` on
        `main`, 2026-07-16) has no generic key/value metadata parameter --
        unlike Mem0/MemPalace, there is nothing to thread it into without
        fabricating a parameter the real API doesn't accept. This is
        stated here explicitly rather than silently dropping the value
        with no comment.
        """
        del mode
        del metadata
        timer = self._timed()
        client = self._get_client()
        try:
            result = asyncio.run(
                client.add_episode(
                    name=f"memtrust-{session_id}",
                    episode_body=content,
                    source_description="memtrust-eval",
                    reference_time=datetime.now(UTC),
                    group_id=session_id,
                    update_communities=self._update_communities,
                )
            )
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            # A single except-Exception handler, routed through
            # _classify_crash() for every failure shape -- not just the two
            # (ValueError, TypeError) shapes this adapter's own module
            # docstring calls out by name (getzep/graphiti#836/#920).
            # _classify_crash() itself still narrows #836/#920 by exact
            # type+message before falling through to message-only checks
            # (see that function's docstring), so this single handler loses
            # no classification precision -- it's what lets a non-
            # ValueError/TypeError crash shape (e.g. getzep/graphiti#1525's
            # `redis.exceptions.ResponseError` "Failed to parse query
            # parameter ... value", raised from this same add_episode()
            # bulk-episode-save call) also get classified instead of
            # silently collapsing to a hardcoded UNKNOWN default the way an
            # earlier version of this method did. See CrashSignal.UNKNOWN's
            # docstring for why "not classified" (None, never used here)
            # and "classified as UNKNOWN" (this handler's honest fallback
            # for anything _classify_crash() doesn't recognize) are
            # different facts.
            raise BackendAPIError(self.name, str(exc), crash_signal=_classify_crash(exc)) from exc

        result_dict = _to_plain_dict(result)
        episode = getattr(result, "episode", None) if not isinstance(result, dict) else None
        if episode is not None:
            episode_uuid = str(getattr(episode, "uuid", "") or "")
        else:
            episode_field = result_dict.get("episode")
            episode_uuid = (
                str(episode_field.get("uuid", "")) if isinstance(episode_field, dict) else ""
            )

        store_result = StoreResult(
            memory_id=episode_uuid, latency_ms=timer.elapsed_ms(), raw=result_dict
        )
        if verify:
            store_result.verified = self.verify_store(store_result, session_id, content)
        return store_result

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        """Search via `Graphiti.search()`, which returns `list[EntityEdge]`
        directly (confirmed against `graphiti_core/graphiti.py` on `main`,
        2026-07-16) -- unlike the hosted `ZepGraphitiAdapter`, there is no
        REST envelope to unwrap.

        Note on getzep/graphiti#1302 (`lucene_sanitize`): the `query`
        string passed here reaches graphiti-core's own internal sanitizer
        before it ever hits the database -- this adapter has no surface to
        intercept or bypass that. Fixture cases containing uppercase
        O/R/N/T/A/D (tests/fixtures/contradiction_cases.json) are what let
        an eval run against a *live* self-hosted instance demonstrate the
        BM25 ranking degradation; against the mocked test double used in
        this repo's own test suite, the sanitizer never runs at all -- see
        module docstring's "What this adapter does NOT prove."

        Note on getzep/graphiti#1222/#1183 (RediSearch `Syntax error`):
        unlike #1302 above, this adapter DOES classify this crash shape --
        see `_classify_crash()` below and `CrashSignal
        .QUERY_SANITIZATION_ERROR` in base.py. A single `except Exception`
        handler (rather than store()'s `(ValueError, TypeError)`
        pre-filter) is the right shape here: the real exception FalkorDB's
        RediSearch fulltext-query path raises for this bug class is
        neither a `ValueError` nor a `TypeError`, so narrowing the catch
        by type first -- the way store() does for #836/#920 -- would skip
        right past it. `_classify_crash()` itself still falls back to
        `CrashSignal.UNKNOWN` for anything that doesn't match a known
        shape, same convention store() uses.
        """
        del mode
        timer = self._timed()
        client = self._get_client()
        try:
            edges = asyncio.run(client.search(query, group_ids=[session_id], num_results=top_k))
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc), crash_signal=_classify_crash(exc)) from exc

        records: list[MemoryRecord] = []
        any_invalidated = False
        for edge in edges:
            edge_dict = _to_plain_dict(edge)
            invalid_at = edge_dict.get("invalid_at")
            if invalid_at:
                any_invalidated = True
            created_at = edge_dict.get("valid_at") or edge_dict.get("created_at")
            attributes = edge_dict.get("attributes")
            records.append(
                MemoryRecord(
                    memory_id=str(edge_dict.get("uuid", "")),
                    content=str(edge_dict.get("fact", "")),
                    score=None,
                    created_at=str(created_at) if created_at else None,
                    metadata={"invalid_at": str(invalid_at)} if invalid_at else {},
                    attributes=dict(attributes) if isinstance(attributes, dict) else {},
                    raw=edge_dict,
                )
            )

        # graphiti-core's bi-temporal invalid_at is the same documented
        # signal ZepGraphitiAdapter (hosted) reads -- see that module's
        # docstring for the citation. No invalidated edge in the result
        # set does not prove no contradiction occurred, only that this
        # call alone cannot tell FLAGGED from an invisible resolution --
        # same disambiguation note as the hosted adapter.
        conflict_signal = ConflictSignal.FLAGGED if any_invalidated else ConflictSignal.SERVED_STALE
        return QueryResult(
            records=records,
            conflict_signal=conflict_signal,
            latency_ms=timer.elapsed_ms(),
            raw={"edges": [r.raw for r in records]},
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        # graphiti-core has no separate "update" verb, same as hosted Zep
        # Cloud -- a contradicting fact is submitted through add_episode()
        # again and the extraction pipeline resolves the contradiction
        # itself. See ZepGraphitiAdapter.update() for the identical
        # reasoning against the hosted API.
        timer = self._timed()
        result = self.store(session_id, content)
        return UpdateResult(
            memory_id=result.memory_id,
            acknowledged=True,
            latency_ms=timer.elapsed_ms(),
            raw=result.raw,
        )

    def delete(self, memory_id: str) -> DeleteResult:
        """Delete via `Graphiti.remove_episode(episode_uuid)`, confirmed
        against `graphiti_core/graphiti.py` on `main` (2026-07-16): it
        looks up the episode, deletes entity edges/nodes exclusively
        mentioned by it, and deletes the episode itself.
        """
        timer = self._timed()
        client = self._get_client()
        try:
            asyncio.run(client.remove_episode(memory_id))
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=timer.elapsed_ms())

    def retrieve_episodes(
        self,
        reference_time: datetime,
        group_ids: list[str] | None = None,
        last_n: int = 10,
    ) -> list[dict[str, object]]:
        """Call graphiti-core's real, driver-level
        `EpisodeNodeOperations.retrieve_episodes()` directly, bypassing
        `Graphiti.search()`/`query()` entirely -- `search()` only ever
        returns edges (`EntityEdge`), never episodes (`EpisodicNode`), so
        there is no way to reach point-in-time episode retrieval through
        this adapter's normal `query()` path at all. This is the primitive
        `evals/episode_temporal_leak.py` needs to reproduce
        getzep/graphiti#1625 (contributor pcy06, open as of this build):
        FalkorDB's Cypher query for `retrieve_episodes()` intends to filter
        by `e.valid_at <= $reference_time`, but FalkorDB can return rows
        for which that same expression evaluates `False` when projected --
        confirmed via the issue's own filed reproduction (`gh issue view
        1625 --repo getzep/graphiti`, fetched 2026-07-16), which shows a
        future-dated episode (`valid_at=2024-03-01`) returned for a
        `reference_time=2024-02-01` query. Real signature confirmed by
        reading `graphiti_core/driver/operations/episode_node_ops.py` on
        `main`, 2026-07-16:
        `EpisodeNodeOperations.retrieve_episodes(self, executor,
        reference_time, last_n=3, group_ids=None, source=None,
        saga=None) -> list[EpisodicNode]` -- `executor` is the driver
        itself (confirmed against the issue's own repro code, which calls
        `driver.episode_node_ops.retrieve_episodes(driver,
        reference_time, ...)`).

        This is detection, not resolution -- the bug this classifies lives
        entirely inside graphiti-core's FalkorDB driver's Cypher query
        construction (pcy06 already proposed the real upstream fix: project
        the temporal comparison first, then filter on that boolean), not in
        this adapter's own code. memtrust's role is surfacing whether a
        given self-hosted deployment's `retrieve_episodes()` call actually
        exhibits the leak, not fixing graphiti-core itself.

        Raises:
            BackendAPIError: if this graphiti_core driver has no
                `episode_node_ops` surface at all (e.g. an injected test
                double that doesn't model one), or if the real call itself
                fails -- classified via `_classify_crash()` the same as
                every other vendor call this adapter wraps.
        """
        client = self._get_client()
        driver = getattr(client, "driver", None)
        episode_ops = getattr(driver, "episode_node_ops", None) if driver is not None else None
        if episode_ops is None:
            raise BackendAPIError(
                self.name,
                "this graphiti_core driver has no episode_node_ops surface -- "
                "cannot call retrieve_episodes() directly. See "
                "evals/episode_temporal_leak.py's module docstring.",
            )
        try:
            episodes = asyncio.run(
                episode_ops.retrieve_episodes(
                    driver, reference_time, last_n=last_n, group_ids=group_ids
                )
            )
        except Exception as exc:  # noqa: BLE001 - vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc), crash_signal=_classify_crash(exc)) from exc
        return [_to_plain_dict(ep) for ep in episodes]

    def close(self) -> None:
        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            asyncio.run(close())
