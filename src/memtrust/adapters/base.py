"""Shared interface every backend adapter implements.

MemTrust scores backends by running the same eval logic against every
tracked vendor through this one interface. If an adapter needs a special
case to pass an eval, that is a bug in the adapter, not a feature -- the
whole point of a standardized harness is that scoring logic never changes
per vendor.

Every adapter reads its credentials from an environment variable. If the
variable is missing, the adapter raises BackendNotConfiguredError instead
of crashing or silently no-oping. This is what lets `memtrust run` work in
a fresh clone with zero API keys: unconfigured backends print SKIPPED and
the run continues.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum


class BackendNotConfiguredError(Exception):
    """Raised when a backend adapter is missing required configuration.

    Callers (the CLI, eval runners) must catch this specifically and treat
    it as "skip this backend," never as a fatal error for the whole run.
    """

    def __init__(self, backend_name: str, missing_env_var: str) -> None:
        self.backend_name = backend_name
        self.missing_env_var = missing_env_var
        super().__init__(
            f"{backend_name} is not configured: environment variable "
            f"{missing_env_var} is not set. Skipping this backend. "
            f"See docs/methodology.md for setup instructions."
        )


class CrashSignal(StrEnum):
    """WHY a store()/query()/update()/delete() call raised BackendAPIError,
    classified by pattern-matching the caught exception's type and message
    against known internal-library crash shapes -- distinct from *whether*
    the call failed (that's BackendAPIError itself, always raised on any
    failure) and distinct from ConflictSignal/RankingSignal/CorruptionSignal
    above, which all classify the *content* of a successful response, never
    an exception. Without this, every internal crash of a vendor library --
    a known, previously-reported bug class vs. some unrelated transient
    error -- surfaces as an identical opaque `BackendAPIError(detail=str(exc))`
    with no way to tell them apart.

    This is pattern-matching on an exception's type and a substring of its
    message, not proof the crash reproduces a specific upstream line of code
    every time this signal is assigned -- a message string is the only
    surface an adapter has to work with once the vendor library has already
    raised. See zep_graphiti_selfhosted_adapter.py's `_classify_crash()` and
    docs/methodology.md for the honesty caveat: this classifies a known
    crash *shape* by pattern-matching the exception, it does not prevent the
    crash or fix the underlying library itself.
    """

    UNPACK_ERROR = "unpack_error"
    """A `ValueError` raised from a Python tuple/list unpacking count
    mismatch (message contains "values to unpack", covering both "too many
    values to unpack" and "not enough values to unpack"). Matches the shape
    of getzep/graphiti#836: `add_episode(update_communities=True)`'s
    community-update branch does
    `communities, community_edges = await semaphore_gather(*[...])`, which
    only succeeds when `semaphore_gather` returns exactly 2 items -- any
    episode whose extracted node count is not exactly 2 raises this
    `ValueError` shape from graphiti_core's own internals, not from
    anything this adapter's own code does."""

    TYPE_COMPARISON_ERROR = "type_comparison_error"
    """A `TypeError` raised from comparing incompatible types, specifically
    a timezone-naive vs. timezone-aware `datetime` comparison (message
    contains "offset-naive and offset-aware"). Matches the shape of
    getzep/graphiti#920: an edge-contradiction-resolution code path
    compares a stored edge's (possibly tz-naive) timestamp against a
    tz-aware `datetime` value without normalizing both to the same
    awareness first, raising this exact `TypeError` from Python's own
    datetime comparison, surfaced through graphiti_core's internals."""

    QUERY_SANITIZATION_ERROR = "query_sanitization_error"
    """A RediSearch `Syntax error` raised from FalkorDB's fulltext-query
    path (message contains both "redisearch" and "syntax error", e.g.
    `"RediSearch: Syntax error at offset 22 near my_graph"`). Matches the
    shape of two independently-documented graphiti-core bugs, both in
    `graphiti_core.driver.falkordb_driver.FalkorDriver`'s `sanitize()`/
    `build_fulltext_query()`:

      * getzep/graphiti#1222 (closed, superseded by #1475): when a query's
        sanitized/stopword-filtered token list comes back empty (e.g. an
        empty query string, such as `explore_node` called with only a
        `node_uuid` and no `node_name`), `build_fulltext_query()` appends
        empty parentheses to the group filter, producing invalid RediSearch
        syntax of the exact shape `(@group_id:"...") ()`.
      * getzep/graphiti#1183 (merged): before this fix, `sanitize()`'s
        character-replacement map omitted pipe (`|`), slash (`/`), and
        backslash (`\\`) -- episode text containing those characters (e.g.
        `"install.sh | bash"`) survived sanitization, tokenized on
        whitespace into a stray `|` token, and got rejoined with `" | "`
        as a RediSearch OR separator, producing an adjacent-pipe malformed
        query (`"sh | | | bash"`) with an empty token between delimiters.
        The merged fix added those three characters to `sanitize()`'s map
        and filters empty tokens before joining.

    Both bugs raise the identical `RediSearch: Syntax error ...` message
    shape once FalkorDB's RediSearch engine parses the malformed query --
    this signal does not distinguish which of the two produced a given
    crash, only that the shape matches. The real exception here is
    typically a `redis`-client `ResponseError`; see
    zep_graphiti_selfhosted_adapter.py's `_classify_crash()` for the exact
    substring match (message-only, not `isinstance`, since this adapter
    does not import the optional `redis`/`falkordb` packages directly) and
    docs/methodology.md for the honesty caveat."""

    LEGACY_CORRUPT_RECORD_UNDELETABLE = "legacy_corrupt_record_undeletable"
    """A `json.JSONDecodeError` raised while a write-path call (`store()`'s
    overwrite/upsert path, `update()`, or `delete()`) attempted to parse a
    response body, matching the shape of volcengine/OpenViking#2966
    (contributor lRoccoon): a legacy uint16-length-truncated record
    (written by a pre-#2171 OpenViking image whose buffer-size wraparound
    bug silently truncated an oversized `fields` blob) can never be
    deleted or overwritten through normal APIs, because
    `LocalIndex.delete_data()`/`upsert_data()` both run
    `_convert_delta_list_for_index()` ->
    `FieldTypeConverter.convert_fields_for_index()`, which calls a bare
    `json.loads(fields_json)` on the corrupt record's `fields`/`old_fields`
    with no error handling -- `json.decoder.JSONDecodeError` propagates on
    every call, so the record becomes a permanent "ghost" (visible to
    search/startup-recovery mitigations that already tolerate it, but
    structurally undeletable). This adapter has no access to OpenViking's
    internal traceback once the vendor's own HTTP layer has already
    responded -- it can only classify this shape from a malformed/
    non-JSON response body its own `resp.json()` call fails to parse,
    which is the observable symptom on the client side of the same
    underlying bug. See openviking_adapter.py's `store()`/`update()`/
    `delete()` and `docs/methodology.md` for the honesty caveat: this
    classifies the shape by pattern-matching the client-visible exception,
    it does not reproduce OpenViking's internal RocksDB/CandsTable state
    or prove this specific call hit a legacy-truncated record versus some
    other cause of a malformed response body."""

    EMBEDDING_BATCH_COUNT_MISMATCH = "embedding_batch_count_mismatch"
    """A `ValueError` raised from Python's `zip(..., strict=True)` count
    mismatch (message contains both "zip()" and "shorter than"). Matches
    the shape of getzep/graphiti#1467 (contributor elimydlarz, open as of
    this build): `GeminiEmbedder.create_batch()` (`graphiti_core/embedder/
    gemini.py`) only special-cases `batch_size=1` for the
    `"gemini-embedding-001"` model -- confirmed by reading the real,
    current `main`-branch source, 2026-07-16 -- so for
    `"gemini-embedding-2-preview"`/`"gemini-embedding-2"`, a single
    `embed_content(contents=batch)` call can silently return ONE embedding
    for an N-item batch (these models do not batch the way `-001` does),
    with no exception raised inside `create_batch()` itself. The caller
    that ends up with a too-short embeddings list is graphiti-core's own
    dedup pipeline -- `_semantic_candidate_search()`
    (`graphiti_core/utils/maintenance/node_operations.py`) or
    `create_entity_node_embeddings()` (`graphiti_core/nodes.py`), reached
    from `add_episode()` once an episode extracts 2+ entities -- which
    zips the extracted-node list against the returned embeddings list with
    `strict=True`, raising `ValueError: zip() argument 2 is shorter than
    argument 1` (or the reverse-argument-order phrasing, depending on
    which list came up short) once the count mismatch is finally observed,
    several call frames away from where the embedder actually returned
    the wrong count. See
    zep_graphiti_selfhosted_adapter.py's `_build_embedder()` (the Gemini
    embedder wiring this signal's classification depends on being
    reachable at all) and `_classify_crash()`."""

    QUERY_PARAMETER_PARSE_ERROR = "query_parameter_parse_error"
    """A FalkorDB `redis.exceptions.ResponseError` raised with a message
    matching `"failed to parse query parameter '<name>' value"` (message
    contains both "failed to parse query parameter" and "value"). Matches
    the shape of getzep/graphiti#1525 (contributor maui314159, fixed
    upstream via merged PR#1531): a NUL byte (`\\x00`) embedded in a string
    value -- commonly present in text extracted from PDFs/PPTX -- makes
    FalkorDB's client-side query-parameter serialization
    (`falkordb/helpers.py::quote_string`, which only escapes backslash and
    double-quote) emit the NUL byte verbatim into the `CYPHER <key>=<value>
    ...` parameter header FalkorDB's own client builds. FalkorDB's parser
    then rejects the ENTIRE query -- including a bulk episode save shaped
    like `UNWIND $episodes AS e MERGE (n:Episodic {uuid:e.uuid})
    SET n.content=e.content` -- with this exact message, silently dropping
    every episode in that call, not just the one containing the NUL byte.
    PR#1531's real fix strips `\\x00` recursively from FalkorDB query
    parameters in `FalkorDriver.execute_query`/`FalkorDriverSession.run`
    before they reach the client, closing the bug upstream as of that
    merge -- this signal exists so the harness can still *classify* the
    failure shape if it ever reproduces against an older/unpatched
    graphiti-core version, the same "diagnostic classifier, not a live
    reproduction" role every other `CrashSignal` member plays. The real
    exception is a `redis`-client `ResponseError`; see
    zep_graphiti_selfhosted_adapter.py's `_classify_crash()` for the exact
    substring match (message-only, not `isinstance`, same reasoning
    `QUERY_SANITIZATION_ERROR` above documents: this adapter does not
    import the optional `redis`/`falkordb` packages just to `isinstance`-
    check against them)."""

    UNKNOWN = "unknown"
    """The caught exception's type/message did not match any known crash
    shape this enum classifies. This is the honest, expected outcome for
    the overwhelming majority of failures -- network errors, auth
    failures, database unavailability, and any vendor bug other than the
    specific shapes above all land here. `UNKNOWN` is not a gap in this
    classification; it is what "not one of the specific bug classes we
    know how to recognize" looks like."""


class BackendAPIError(Exception):
    """Raised when a configured backend's API call fails (network, auth,
    5xx, malformed response). Distinct from BackendNotConfiguredError so
    callers can tell "never had credentials" apart from "had credentials,
    the call still failed."
    """

    def __init__(
        self, backend_name: str, detail: str, crash_signal: CrashSignal | None = None
    ) -> None:
        self.backend_name = backend_name
        self.detail = detail
        self.crash_signal = crash_signal
        """Which known internal-library crash shape (see CrashSignal above)
        this failure's exception matched, or None when the adapter that
        raised this error did not attempt classification at all (most
        adapters -- classification is opt-in per adapter, the same
        "None means not attempted" convention StoreResult.verified uses).
        A non-None value that equals CrashSignal.UNKNOWN means the adapter
        DID attempt classification and the exception simply didn't match
        any known shape -- that is a meaningfully different fact from
        "this adapter never classifies," which is why the default is None,
        not UNKNOWN."""
        super().__init__(f"{backend_name} API error: {detail}")


class ConflictSignal(StrEnum):
    """How a backend responded when a query touched a fact that had been
    contradicted by a later store()/update() call.

    This is the classification the contradiction-detection eval reads off
    every query() response -- see evals/contradiction.py.
    """

    FLAGGED = "flagged"
    """The backend surfaced the conflict: it returned both versions, an
    explicit conflict marker, or otherwise made the contradiction visible
    to the caller instead of silently resolving it."""

    SILENT_OVERWRITE = "silent_overwrite"
    """The backend replaced the old fact with the new one and gave no
    signal in the query response that a prior, different value ever
    existed."""

    SERVED_STALE = "served_stale"
    """The backend returned the old, now-contradicted fact and gave no
    signal that a newer, conflicting fact had been stored since."""

    NOT_APPLICABLE = "not_applicable"
    """The backend has no update/contradiction-relevant primitive to
    evaluate here (see MemoryBackendAdapter.supports_update). Recorded
    explicitly rather than silently dropping the backend from the eval
    table."""

    EMPTY_OR_LOST = "empty_or_lost"
    """The backend DOES have an update/contradiction-relevant primitive
    (MemoryBackendAdapter.supports_update is True), the store()/update()/
    query() calls all completed without raising BackendAPIError, but the
    query response came back with zero records -- no exception, no error,
    just nothing. This is distinct from NOT_APPLICABLE, which means the
    backend structurally cannot be evaluated here at all: EMPTY_OR_LOST
    means the backend *should* have had something to say and silently
    didn't. This is the "call succeeded but produced nothing" failure mode
    the benchmark exists to catch -- see evals/contradiction.py's
    classify_case for exactly when this is assigned, never a vendor's own
    self-report."""

    EDGE_INTEGRITY_VIOLATION = "edge_integrity_violation"
    """A returned record is edge-shaped (its `raw` fragment carries both a
    `source_node_uuid` and a `target_node_uuid` key -- the property names
    graphiti_core's `EntityEdge` writes on every relationship) but at
    least one of those two values is missing or falsy. This is a
    structural integrity check, not a value-classification one: it exists
    because two independently root-caused, real graphiti_core bugs
    produced exactly this shape --

      * getzep/graphiti#1013: the Neo4j bulk edge-save Cypher query built
        its SET clause from an enumerated field list that omitted
        `EntityEdge.attributes` and `reference_time`, so a bulk-saved edge
        could come back missing properties a single-edge save() would
        have written. Confirmed fixed upstream (merged) by switching the
        bulk query to `SET e = edge` (Neo4j's default-case Cypher
        variable; FalkorDB's branch of the same function uses `SET r =
        edge`) -- see
        graphiti_core/models/edges/edge_db_queries.py's
        `get_entity_edge_save_bulk_query()` on the `main` branch as of
        2026-07-16.
      * getzep/graphiti#1001: FalkorDB's `add_triplet()`-era edge-creation
        path used a `MATCH` (not `MERGE`/`CREATE`) for the endpoint nodes,
        which silently no-ops if either node is absent, and never set
        `source_node_uuid`/`target_node_uuid` as edge properties at all.
        Closed via the same #1013 refactor; the FalkorDB driver has since
        been rewritten (see `graphiti_core/driver/falkordb/operations/`)
        and no longer exposes the old `add_triplet()` method.

    Both are described upstream as already fixed/closed on `main` as of
    this adapter's build -- this signal exists so the harness can *detect*
    the failure shape if it ever reproduces against an older/pinned
    graphiti-core version, or against a different backend that has the
    same class of bug, not because memtrust has reproduced either bug
    live against a running Neo4j/FalkorDB instance. See
    docs/methodology.md and zep_graphiti_selfhosted_adapter.py for the
    honesty caveat on what this repo has and has not actually run."""


class RankingSignal(StrEnum):
    """Whether a backend's claimed result ordering is actually driven by a
    real per-record ranking signal, or silently degenerates to something
    else (typically insertion order) while still returning fully correct
    *content*.

    This is a distinct taxonomy from ConflictSignal above, not a variant of
    it. ConflictSignal classifies whether returned content is correct after
    a contradiction; RankingSignal classifies whether correct content came
    back in the right order. A backend can score perfectly on every
    ConflictSignal case and still be silently broken here -- the returned
    records are the right records, just wrongly ordered, which
    ConflictSignal structurally cannot see because there was never a
    content conflict to classify.

    Motivating case: mempalace/mempalace#1733 (GitHub user Kartalops).
    `mempalace/layers.py`'s `Layer1.generate()` sorts drawers by
    `importance`/`emotional_weight`/`weight`, but no ingest path in the
    real package ever writes those keys -- confirmed 0/45,969 drawers on a
    real palace. `importance` silently defaults to a constant, so the
    "ranked by importance" sort degenerates to plain insertion order, and
    `wake-up` returns the oldest moments instead of the documented "high
    importance, recent" ones. This is the classification the
    ranking-quality eval reads off every query() response -- see
    evals/ranking_quality.py.
    """

    SIGNAL_DRIVEN = "signal_driven"
    """A ranking-relevant metadata field (importance/emotional_weight/
    weight or whatever the adapter checks) carries genuinely varied values
    across the returned records -- a real per-record signal exists that
    could plausibly be driving the order. This is a claim about the
    presence of a signal, not proof the backend actually sorted by it; see
    evals/ranking_quality.py's classify_case-equivalent, which cross-checks
    this claim against the actual returned order before crediting it."""

    MISSING_ORDERING_KEY = "missing_ordering_key"
    """Every returned record shares the same value for a ranking-relevant
    metadata field -- including the case where the field is absent from
    every record entirely, which is indistinguishable from "silently
    defaults to a constant" from the caller's side. No real per-record
    signal is driving order, whatever the backend's documentation claims.
    This is the exact mempalace/mempalace#1733 shape: `importance` present
    (or absent) and identical across every record because no ingest path
    ever wrote a real value."""

    ORDER_INCONSISTENT = "order_inconsistent"
    """A ranking-relevant metadata field carries genuinely varied values
    across the returned records, but the actual returned order does not
    correlate with those values (not sorted descending by the field). A
    real signal exists and the backend still isn't using it to order
    results -- an adjacent but distinct bug from MISSING_ORDERING_KEY."""

    NOT_APPLICABLE = "not_applicable"
    """No ranking-relevant metadata field was present on any returned
    record and the result set gave no other basis (e.g. too few records)
    to evaluate ordering at all. Recorded explicitly, never silently
    dropped from the results table -- same convention as
    ConflictSignal.NOT_APPLICABLE."""

    RERANK_FALLBACK = "rerank_fallback"
    """This response's own candidate set carries the same input shape two
    independent, real OpenViking bug reports document as silently
    degrading a reranker-backed query to raw vector-similarity scores with
    no caller-visible signal: an empty-string document
    (volcengine/OpenViking#1737, contributor wychosenone --
    `RerankClient.rerank_batch` given an empty-string document makes the
    VikingDB rerank API return a null response, which a swallowed
    `TypeError` silently falls back from) or a candidate batch whose total
    estimated token count exceeds the reranker's real input budget
    (volcengine/OpenViking#2739/#2880, contributor hhspiny -- the
    hierarchical retriever sends unbounded L2 abstracts, some 10k+ tokens,
    into a single rerank batch, blowing past a typical local reranker's
    ~4096-token limit; hhspiny's own production logs showed 280 failed
    requests in this exact shape).

    Both cited bugs are already fixed upstream (#1737 via merged #1933;
    #2739/#2880 via merged PR#3289) by the time this signal was built, so
    this adapter cannot reproduce either against a live current-version
    OpenViking instance -- this signal exists so memtrust can *detect* the
    same failure shape if it ever reproduces against an older/self-hosted/
    pinned OpenViking version, the same honesty convention
    ConflictSignal.EDGE_INTEGRITY_VIOLATION and
    CorruptionSignal.CONFIG_REJECTED already establish above for
    already-fixed-upstream bug classes. See openviking_adapter.py's
    `_rerank_fallback_risk()` for the exact heuristic (empty-content
    check, then an approximate ~4-chars/token budget estimate) and its own
    honesty caveat: this flags the SHAPE a response's candidate set
    carries, it does not confirm this specific live call actually fell
    back to vector scores server-side, since this adapter has no way to
    observe OpenViking's internal rerank-provider call at all."""


class CorruptionSignal(StrEnum):
    """How a backend's *write path* (construction-time config, or a store()/
    update() call) handled a failure mode that ConflictSignal cannot express.

    ConflictSignal above classifies how a *query response* handled a
    contradiction between two stored facts -- it is inherently a read-side,
    post-store observation. The bug classes this enum exists for are
    different in kind, not just in vendor: they either (a) happen at
    construction time, before any store()/query() call is ever made, when a
    backend rejects an invalid configuration (e.g. mem0's Kuzu graph store
    historically raised ValueError from MemoryGraph.__init__ when
    embedding_dims was None/<=0 -- mem0ai/mem0#3558), or (b) happen silently
    *during* a write, corrupting data with no exception and no query-side
    signal that anything went wrong until a much later, unrelated read fails
    to find a match (e.g. mem0's Valkey/Redis vector stores historically
    called `np.array(None, dtype=np.float32)` on a metadata-only update,
    silently writing a 4-byte garbage vector over a real embedding --
    mem0ai/mem0#4362). Both are init/write-path phenomena a query-response
    enum has no vocabulary for, which is why this is a separate enum rather
    than new ConflictSignal members.

    Adapters that have no surface to observe either failure mode (i.e. every
    adapter except ones like Mem0DirectAdapter that hold a direct, in-process
    handle to the vendor library rather than talking to it over HTTP) should
    report NOT_APPLICABLE rather than guessing.
    """

    CONFIG_REJECTED = "config_rejected"
    """Backend construction (or a construction-time-equivalent call inside
    store()/query()) raised ValueError/pydantic.ValidationError for an
    invalid configuration, and the adapter caught it and classified it
    instead of letting it propagate as an unhandled crash. This is the
    signal a caller should see for the mem0ai/mem0#3558 bug *class*: "bad
    embedding_dims config is rejected before it can silently corrupt a
    graph DB," not a literal re-run of that exact removed code path -- see
    Mem0DirectAdapter's module docstring for why the original Kuzu code
    this bug lived in no longer exists in the installed mem0ai package."""

    VECTOR_ZEROED = "vector_zeroed"
    """A write call completed without raising, but the adapter's own
    inspection of what was actually persisted (not the vendor's normal
    read API, which -- per mem0ai/mem0#4336 -- does not surface this) shows
    the embedding was replaced with a zero-length or wrong-dimensionality
    vector instead of being left untouched. This is what
    mem0ai/mem0#4362 fixed for Valkey/Redis; a backend still exhibiting it
    would report this signal, a fixed one reports CLEAN for the same
    operation."""

    CLEAN = "clean"
    """The write/config-validation path completed with no corruption or
    config-rejection detected for this operation."""

    NOT_APPLICABLE = "not_applicable"
    """This adapter has no surface to observe either failure mode (most
    REST-based adapters: a vendor's HTTP API gives no way to inspect
    construction-time config validation separately from a network error, or
    to inspect raw stored vector bytes independently of its own search/get
    responses). Recorded explicitly rather than silently omitting the
    field."""


class ExtractionSignal(StrEnum):
    """Whether a store() call that completed without raising actually found
    something to persist, or silently extracted zero facts from the input.

    This is a distinct taxonomy from ConflictSignal/RankingSignal/
    CorruptionSignal above, not a variant of any of them. Those three all
    classify *query()* responses or write-path *corruption*; this one
    classifies the one failure mode none of them can see: a store() call
    that raises nothing, returns a normal-shaped 200/dict response, and yet
    the backend's own LLM-based fact-extraction step found nothing worth
    keeping -- so the response has no usable `id` (and no `results[0].id`)
    for the adapter to report back to the caller. Byte-for-byte, that
    response is indistinguishable from a genuine successful store unless an
    adapter explicitly checks for it, which is exactly what
    `mem0_adapter.py`'s `_extract_memory_id()` historically did not do: it
    silently returned `""` and let a normal-looking StoreResult mask the
    difference between "stored" and "extracted nothing."

    Motivating case: mem0ai/mem0#5178 (GitHub user thalesfsp) -- `add()`
    can return a response with zero extracted memories for input the caller
    clearly intended to be remembered, with no exception and no other
    signal that anything different happened versus a normal store. This
    backlog item originally also referenced mem0ai/mem0#5878/#5901/#5903
    (GitHub user Bartok9) as adjacent reports of the same "store()
    succeeded but the fact silently never made it in" shape; #5178 is the
    one this signal is scoped against directly, since it is the row this
    change closes.

    This is the "LLM extraction silently swallowed" failure class: a vendor
    fact-extraction pass (which can legitimately decide "there is nothing
    worth remembering here," e.g. for chit-chat with no factual content)
    and a vendor bug that drops a real fact are indistinguishable from the
    response shape alone. This signal does not attempt to tell those two
    apart -- see EMPTY_EXTRACTION's docstring -- it exists so that
    distinction is at least visible to a caller instead of silently
    collapsed into an ordinary-looking StoreResult.
    """

    FACTS_EXTRACTED = "facts_extracted"
    """store() completed without raising and the response carried a usable
    memory id (a top-level `id`, or `results[0].id`) -- the backend's
    extraction step found and persisted at least one fact."""

    EMPTY_EXTRACTION = "empty_extraction"
    """store() completed without raising, but the response carried no
    usable memory id anywhere this adapter knows to look. This is the exact
    mem0ai/mem0#5178 shape: a call that looks identical to a successful
    store from its return value alone, except that nothing was actually
    extracted. Recorded explicitly rather than silently returning a
    StoreResult with memory_id="" and no other trace anything unusual
    happened -- a caller (or an eval) that only checks "did store() raise"
    has no way to see this otherwise."""

    NOT_APPLICABLE = "not_applicable"
    """This adapter has no extraction concept to observe here -- either the
    backend has no LLM-extraction step at all (most non-mem0 adapters,
    which persist exactly what they're given rather than deciding what's
    worth keeping), or this specific store() call failed for a different,
    already-classified reason (e.g. Mem0DirectAdapter's CONFIG_REJECTED
    corruption_signal path below, where memory_id="" means "construction
    was rejected before any extraction could run," not "extraction ran and
    found nothing" -- conflating the two would misattribute a config error
    to this signal). Recorded explicitly, same convention as
    ConflictSignal.NOT_APPLICABLE/RankingSignal.NOT_APPLICABLE/
    CorruptionSignal.NOT_APPLICABLE above."""


class EmbeddingDriftSignal(StrEnum):
    """Whether a record stored before an embedding-model migration remained
    retrievable after the migration, or silently stopped being findable --
    a structurally distinct failure mode from every enum above. ConflictSignal
    classifies a query response after a *content* contradiction;
    RankingSignal classifies whether correct content came back in the right
    *order*; CorruptionSignal classifies a single write call's own
    construction/write-path failure. None of those can express "this record
    was fine, a completely unrelated later store() call for a *different*
    record silently broke it" -- which is exactly the shape of the
    motivating bug.

    Motivating case: volcengine/OpenViking#1523 (contributor A0nameless0man).
    An embedder migration silently degrades search quality mid-migration:
    switching embedding models overwrites previously-stored vectors in
    place with no dimension/model validation, so records embedded under the
    old model can stop being retrievable once new-model writes start
    landing -- with no exception, no error, and no signal in any single
    query response that this happened. A single query() call has no way to
    see this: the record that broke wasn't touched by the query, it was
    broken by a *different*, earlier store() call that happened to migrate
    embedding models. This is why detecting it requires an eval that
    controls two store() calls under two different fixture-level model
    labels and compares retrievability before vs. after, not a per-response
    classification like ConflictSignal/RankingSignal/CorruptionSignal
    above -- see evals/embedding_drift.py for the harness-level eval this
    signal is scored by, and its module docstring plus docs/methodology.md
    for the honest scope of what this can and cannot prove without a real
    backend that exposes embedding-model metadata (none in this repo do,
    as of this writing).
    """

    EMBEDDING_DRIFT = "embedding_drift"
    """A record confirmed retrievable (by its own content, as a substring
    of a returned record's content) immediately after being stored under
    one fixture-level embedding-model label became unretrievable after
    later records were stored under a *different* fixture-level
    embedding-model label in the same session -- the exact volcengine/
    OpenViking#1523 shape. Only assigned when the record was provably
    retrievable BEFORE the migration step, so a record that was never
    retrievable in the first place (a generic recall miss, unrelated to
    any migration) is never misattributed to drift -- see
    evals/embedding_drift.py's classify_embedding_drift_record()."""

    CLEAN = "clean"
    """The record was confirmed retrievable before the migration step and
    remained retrievable (same content match) afterward -- no drift
    observed for this record."""

    NOT_APPLICABLE = "not_applicable"
    """The record's pre-migration retrievability could not be confirmed at
    all (it was never observed retrievable even before any migration step
    ran), so there is no valid baseline to compare against -- recorded
    explicitly rather than guessed at either way, same convention as every
    other NOT_APPLICABLE member in this module."""


class ExtractionQualitySignal(StrEnum):
    """Whether a backend's write path retained content it should have
    filtered, correctly filtered content it should have discarded, and
    whether re-storing previously-recalled content silently fans out into
    duplicate records.

    This is a distinct taxonomy from ConflictSignal, RankingSignal, and
    CorruptionSignal above, not a variant of any of them. Those three all
    presuppose a *single, specific* stored fact -- one that gets
    contradicted, one whose order matters, one whose write path corrupts
    it. This taxonomy instead asks a corpus-scale question that none of
    them can express: across many independently stored items, does the
    backend retain content indiscriminately regardless of whether it was
    ever worth storing at all? `evals/contradiction.py`'s own module
    docstring is explicit that it is a *single-case* contradiction-
    detection eval; this is the distinct "extraction quality at scale"
    counterpart the backlog asked for -- see evals/extraction_quality.py.

    Motivating case: mem0ai/mem0#4573 (GitHub user jamebobob). A 32-day
    real production audit found 97.8% of 10,134 stored mem0 entries were
    junk the extraction pipeline should never have persisted: boot-file
    restating (~52.7% -- the agent's own startup/config text re-stored as
    if it were a new memory every session), cron/heartbeat noise (~11.5%
    -- routine liveness-check output with no durable content), system
    dumps (~8.2% -- raw tracebacks/error payloads), and hallucinated
    profiles (~5.2% -- attributes about the user the model invented, never
    actually stated). On top of the base junk-retention problem, jamebobob
    also documented a feedback-loop case: a single hallucinated memory,
    once recalled back into context, got re-extracted and re-stored as
    "new" input -- and that single re-store fanned out into 808 duplicate
    records, not one.

    Honesty caveat (see evals/extraction_quality.py and
    docs/methodology.md for the full write-up): this enum and the eval
    that classifies against it are structurally validated against
    hand-written fake adapters in this repo's own test suite, proving the
    *classification logic* correctly tells retained-junk apart from
    rejected-junk and detects unexpected record-count growth after a
    recall-then-re-store sequence. Neither this enum nor
    evals/extraction_quality.py has been run against a live mem0 instance
    at jamebobob's real 10,000+ entry scale -- no live number this eval
    could report should be read as a reproduction of the 97.8%/808 figures
    above, only as a harness now capable of measuring the same shape of
    failure if pointed at a real backend.
    """

    RETAINED_JUNK = "retained_junk"
    """A case whose ground-truth `should_be_stored` is False (boot-file
    restating, cron/heartbeat noise, a system dump, or a hallucinated
    profile) was still retrievable via query() after being stored -- the
    backend has no effective extraction-quality gate, or the gate it has
    did not catch this item. This is the exact failure mode jamebobob's
    audit measured at 97.8% against real mem0."""

    REJECTED_JUNK = "rejected_junk"
    """A case whose ground-truth `should_be_stored` is False was NOT
    retrievable via query() after being stored -- something in the write
    path (an extraction-quality gate, a dedup/relevance filter, or
    equivalent) correctly kept it out. The good outcome for a junk case."""

    RETAINED_VALID = "retained_valid"
    """A case whose ground-truth `should_be_stored` is True was
    retrievable via query() after being stored -- genuinely valuable
    content survived the write path. The good outcome for a valid case."""

    LOST_VALID = "lost_valid"
    """A case whose ground-truth `should_be_stored` is True was NOT
    retrievable via query() after being stored -- an overly aggressive
    filter (or an unrelated write-path bug) dropped content that should
    have been kept. The bad outcome for a valid case, and the necessary
    counterweight to RETAINED_JUNK: a backend that filters everything
    would score perfectly on junk-rejection while failing every real user
    here, which is why this eval scores both axes independently rather
    than only measuring junk retention."""

    FEEDBACK_LOOP_DUPLICATE = "feedback_loop_duplicate"
    """After storing a seed item, querying it back (simulating an agent
    recalling it into context), and re-storing that exact recalled text
    as if it were new input, the number of matching records grew by more
    than the single store() call that re-storage represents. This is the
    generalized shape of jamebobob's exact 808-duplicate finding: one
    piece of recalled content, re-extracted, fanning out into many stored
    copies instead of one (or zero, if the backend dedups). See
    evals/extraction_quality.py's `classify_feedback_loop_case` for the
    exact growth threshold."""

    NO_UNEXPECTED_GROWTH = "no_unexpected_growth"
    """The feedback-loop sequence above completed and the record count
    grew by at most what the single re-store() call should have added --
    no runaway duplication observed. The good outcome for a feedback-loop
    case."""

    NOT_APPLICABLE = "not_applicable"
    """A case could not be classified at all -- the store()/query() call
    sequence raised BackendAPIError. Recorded explicitly, never silently
    dropped from the results table, same convention as every other signal
    enum's NOT_APPLICABLE above."""


class VectorIntegritySignal(StrEnum):
    """Whether a delete_prefix() call actually removed every vector-index
    entry it addressed, or left child entries permanently orphaned while
    reporting the prefix as gone.

    This is the opposite polarity of ResourceSyncSignal (evals/
    resource_sync_safety.py), which classifies files a resync wrongly
    DELETED. This taxonomy classifies the reverse failure: entries a
    delete wrongly KEPT. Distinct from CorruptionSignal above, which
    classifies a single write call's own construction/write-path failure,
    not a multi-entry recursive-delete's completeness.

    Motivating case: volcengine/OpenViking#3064 (contributor AcTiveXXX).
    `viking_fs.rm()`'s orphan-cleanup path, reached when a target path no
    longer exists in AGFS (e.g. files were deleted directly from the
    filesystem, bypassing OpenViking's own API), tries to discover child
    URIs to delete via `_collect_uris()` -- which walks AGFS directory
    listings and wraps that walk in a bare `except Exception: pass`. When
    the directory itself is already gone, the listing call raises, the
    bare except silently swallows it, and `_collect_uris()` returns an
    empty list. Only the root URI then reaches `delete_uris()`, whose
    filter is an exact-match `Eq("uri", ...)` with no prefix/recursive
    semantics -- so every child vector-index entry beneath the root
    survives, permanently orphaned (AcTiveXXX measured ~9% orphan rate in
    a real deployment: ~100 orphans out of ~1,000 total entries). The
    listing-based discovery this adapter's own `delete_prefix()` uses (see
    openviking_adapter.py) is subject to the identical AGFS-listing
    limitation when pointed at a live OpenViking server exhibiting this
    bug -- this taxonomy exists so that shape is at least visible to a
    caller instead of silently reported as a clean delete.
    """

    ORPHANED_VECTOR_ENTRY = "orphaned_vector_entry"
    """After delete_prefix(), list_resource_paths() confirms every path
    under the deleted prefix is gone (an AGFS-listing-level "deleted"
    verdict), but a subsequent query() for content seeded under that
    prefix still returns a matching record -- the vector index kept an
    entry the filesystem-level listing already reports as removed. The
    exact volcengine/OpenViking#3064 shape: `delete_uris()`'s exact-match
    filter only ever removed the root URI, so child vector entries survive
    the delete undetected by anything that only checks AGFS listings."""

    CLEAN = "clean"
    """After delete_prefix(), neither list_resource_paths() nor query()
    surfaces any trace of content seeded under the deleted prefix -- the
    delete was actually complete, not just reported complete."""

    NOT_APPLICABLE = "not_applicable"
    """Either the adapter has no prefix-delete primitive to exercise at
    all (MemoryBackendAdapter.supports_prefix_delete is False), or a
    case's before/after state could not be established (e.g. seeding
    itself failed). Recorded explicitly, same convention as every other
    NOT_APPLICABLE member in this module."""


class ConsistencySignal(StrEnum):
    """Whether an identical query, repeated against unchanged stored data,
    returns the same result set every time.

    Every other taxonomy in this module classifies a SINGLE query()
    response in isolation (correctness after a contradiction, ordering,
    write-path corruption, extraction). None of them can see a backend
    that is internally non-deterministic: each individual response can
    look perfectly well-formed -- real records, real content, no error --
    while the *set* of records returned for the exact same query changes
    from call to call with nothing in the fixture data ever having
    changed. This taxonomy exists specifically to classify that
    consistency question, which requires issuing the same query() call
    multiple times and comparing result sets against each other, not
    against any ground truth a single response could carry.

    Motivating case: volcengine/OpenViking#204 (contributor ponsde, a
    repeat contributor to this project). `search()`/`find()` returned
    non-deterministic result sets for identical repeated queries -- 5 runs
    of the same query returned an average pairwise Jaccard similarity of
    0.11 over the returned memory/resource URIs, and in ponsde's own
    later, more rigorous 3-query x 5-run test, some query/method
    combinations shared ZERO common URIs across all 5 runs. ponsde
    self-diagnosed two candidate root causes through extensive follow-up
    (source-tracing `viking_fs.py`'s no-session code path, and a clean-vs-
    production A/B test): a production vector collection created at
    `Dimension:3072` while the active embedding config specified `1024`
    (an embedding-dimension mismatch corrupting the underlying ANN index),
    and/or non-deterministic graph traversal in the HNSW ANN search
    implementation itself, amplified by `HierarchicalRetriever`'s
    recursive search mechanism. Neither root cause is something this
    adapter can fix or directly observe (both live inside OpenViking's own
    C++ engine layer) -- this taxonomy exists so memtrust can at least
    *detect* the same non-deterministic-repeated-query shape ponsde
    documented, using the identical Jaccard-similarity methodology his own
    hand-built repro script already used. See
    evals/result_consistency.py for the eval this taxonomy is scored by.
    """

    CONSISTENT = "consistent"
    """N repeated, identical query() calls against unchanged fixture data
    produced an average pairwise (consecutive-run) Jaccard similarity at
    or above this eval's configured threshold -- the backend's result set
    for this query is stable run to run, within the tolerance the
    threshold allows for a genuinely ranked-and-truncated top_k result
    that could reasonably reorder near a score tie."""

    INCONSISTENT = "inconsistent"
    """N repeated, identical query() calls against unchanged fixture data
    produced an average pairwise Jaccard similarity below the configured
    threshold -- the exact volcengine/OpenViking#204 shape: the same
    query, against data that never changed between calls, returns a
    meaningfully different result set from one call to the next."""

    NOT_APPLICABLE = "not_applicable"
    """Fewer than 2 of the N repeated query() calls completed without
    raising BackendAPIError, so there is no valid pair of result sets to
    compare -- recorded explicitly rather than guessed at either way, same
    convention as every other NOT_APPLICABLE member in this module."""


class LanguageDegradationSignal(StrEnum):
    """Whether a backend's hybrid retrieval pipeline's non-semantic
    signals (BM25 keyword matching, entity-based boosting) genuinely fired
    for a given query, or silently degraded to semantic-only retrieval
    with no error surfaced -- a language-conditioned failure mode none of
    `ExtractionQualitySignal`/`RankingSignal`/`EmbeddingDriftSignal` above
    classify (all three concern content/order/drift, never whether a
    *language-dependent* pipeline stage ran at all).

    Motivating case: wangjiawei-vegetable (rank 147, mem0ai/mem0#4884,
    open, merge-ready companion PR #4943 as of this build). mem0 v3's
    hybrid retrieval pipeline hardcodes spaCy's English model
    `en_core_web_sm` for BOTH BM25 lemmatization
    (`mem0/utils/lemmatization.py::lemmatize_for_bm25`) and entity
    extraction (`mem0/utils/spacy_models.py::get_nlp_full`) -- confirmed
    by reading the installed `mem0ai==2.0.12` package's own
    `mem0/utils/spacy_models.py` directly: `get_nlp_full()`/
    `get_nlp_lemma()` both call `spacy.load("en_core_web_sm", ...)`
    unconditionally, with no language parameter or per-language model
    selection anywhere in the module. For non-Latin-script text (CJK,
    Arabic, Thai, Hindi, etc.), the English pipeline's tokenization/
    lemmatization does not produce keyword/entity signals that
    meaningfully overlap between query time and store time, so mem0's own
    `mem0/memory/main.py::_search_vector_store()` can end up with an
    empty `bm25_scores` dict and/or an empty `entity_boosts` dict for a
    query where either would fire for equivalent English text -- and
    `mem0/utils/scoring.py::score_and_rank()`'s own `has_bm25 =
    bool(bm25_scores)`/`has_entity = bool(entity_boosts)` gates mean the
    combined score silently falls back to semantic-only weighting, with
    no exception and no field in the normal (non-`explain`) response
    indicating this happened.

    This signal is derived from `Memory.search(explain=True)`'s real,
    installed `score_details` per-result breakdown (`bm25_score`,
    `entity_boost` -- confirmed by reading `mem0/utils/scoring.py::
    score_and_rank()` directly), not guessed or inferred from content
    alone -- see `mem0_direct_adapter.py`'s `query()` for exactly how.
    """

    HYBRID_SIGNALS_ACTIVE = "hybrid_signals_active"
    """At least one returned record's `score_details` showed a nonzero
    `bm25_score` or `entity_boost` -- the hybrid pipeline's keyword/
    entity-matching signals genuinely contributed to at least one result,
    not just semantic similarity."""

    SEMANTIC_ONLY_DEGRADED = "semantic_only_degraded"
    """`explain=True` was requested and every returned record's
    `score_details` showed `bm25_score == 0.0` AND `entity_boost == 0.0`
    -- the exact mem0ai/mem0#4884 shape: hybrid retrieval silently
    degraded to semantic-only, with no error or warning anywhere in the
    normal response. Only assigned when records were actually returned to
    inspect -- see NOT_APPLICABLE below for the zero-records case, which
    is a different failure mode (EMPTY_OR_LOST-shaped, not this one)."""

    NOT_APPLICABLE = "not_applicable"
    """`explain` was not requested, no records were returned to inspect,
    or this adapter has no `score_details`-equivalent surface to observe
    at all. Recorded explicitly rather than silently defaulting to either
    signal above, same convention every other signal enum's
    NOT_APPLICABLE member in this package follows."""


@dataclass
class MemoryRecord:
    """One stored memory as returned by a backend's query response."""

    memory_id: str
    content: str
    score: float | None = None
    created_at: str | None = None
    embedding_model: str | None = None
    """Identifier of the embedding model that produced this record's stored
    vector, IF the backend's query response exposes that information.
    Defaults to `None` ("unknown/not reported") for every adapter -- most
    real backends' search APIs do not surface per-record embedding-model
    provenance at all (confirmed absent from OpenViking's documented
    `/v1/search` response shape during this build; see
    openviking_adapter.py's module docstring). This field exists so an
    adapter that CAN report it has somewhere typed to put it, and so
    evals/embedding_drift.py's fixture-level model-label tracking has a
    real adapter-native field to cross-check against on the rare backend
    that does expose this -- it is not itself proof any adapter populates
    it. See EmbeddingDriftSignal above and evals/embedding_drift.py for the
    eval this supports."""
    embedding_dims: int | None = None
    """Dimensionality of this record's stored vector, IF the backend's
    query response exposes it. Same default-`None`, same-rarity caveat as
    `embedding_model` above -- see its docstring."""
    metadata: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, object] = field(default_factory=dict)
    """Structured, non-string-coerced properties this record's backend
    attaches to it -- e.g. graphiti_core's `EntityEdge.attributes` dict,
    which can hold arbitrary typed values (not just strings) describing
    the edge/entity. Distinct from `metadata` above (harness-derived,
    string-only markers like `invalid_at`) and from `raw` below (the
    entire unmodified response fragment, kept for audit purposes only).
    `attributes` exists specifically so a backend's own structured
    per-record properties survive the adapter boundary in a typed-enough
    shape for an eval to inspect programmatically, rather than forcing
    every consumer to dig through `raw`. Defaults to empty for every
    adapter that has no such concept -- most backends don't, and an empty
    dict here is a normal, expected value, not a gap.
    """
    raw: dict[str, object] = field(default_factory=dict)
    """Unmodified vendor response fragment for this record, kept for
    audit/raw-log purposes. Never used for scoring -- scoring only reads
    the typed fields above so every backend is judged by the same rules.
    """


@dataclass
class RetrievalWarning:
    """A backend's own signal that a query() response under-delivered
    without treating that as a hard failure -- confirmed against the real,
    merged MemPalace/mempalace#1005 PR diff (`feat(searcher): warnings +
    sqlite BM25 top-up when vector underdelivers`): when its vector index
    (HNSW/Chroma) drifts or a query raises, `search_memories()` no longer
    hard-fails -- it returns whatever it *could* rank, plus a `warnings`
    list explaining why, plus `available_in_scope` (a sqlite-authoritative
    count of how many records actually match the query's scope,
    independent of how many the vector path could rank).

    This is a distinct failure mode from ConflictSignal.EMPTY_OR_LOST,
    which only fires when a query response comes back with zero records.
    A backend can return, say, 3 of 50 available records -- non-empty,
    so EMPTY_OR_LOST never fires -- while still silently shortchanging the
    caller on the other 47. Before this field existed, that shape was
    indistinguishable from "the backend genuinely only found 3 relevant
    records," which misattributes a backend retrieval bug to content
    relevance. See mempalace_adapter.py's query() for where this is
    populated (the one adapter that currently sets it) and
    docs/methodology.md's adapter-confidence table for the confirmed-
    against-diff-but-not-run-against-a-live-instance caveat that applies
    here the same way it applies to the rest of that adapter."""

    warnings: list[str]
    """Verbatim warning strings the backend attached to this response
    (e.g. "vector search unavailable: ...", "5243 drawers match this
    scope in sqlite; vector ranked 1 ..."). Kept exactly as the vendor
    returned them, never rewritten or summarized, so a report reader can
    see the backend's own explanation."""
    available_in_scope: int | None
    """The backend's own count of how many records exist in the queried
    scope, independent of how many it was actually able to rank/return
    for this call. None when the backend didn't report a count, or
    reported a value that wasn't a real int (see mempalace_adapter.py's
    parsing -- a non-int value is treated as "unknown," never coerced)."""


@dataclass
class QueryResult:
    """Result of MemoryBackendAdapter.query()."""

    records: list[MemoryRecord]
    conflict_signal: ConflictSignal
    """How this query response handled any contradiction relevant to the
    query. Adapters that cannot detect this default to NOT_APPLICABLE and
    the eval scores it as a gap, not a pass or fail -- see
    evals/contradiction.py for the classification logic."""
    latency_ms: float
    ranking_signal: RankingSignal = RankingSignal.NOT_APPLICABLE
    """Whether a real per-record ranking signal appears to be driving this
    response's result order, distinct from ConflictSignal above -- see
    RankingSignal's docstring and evals/ranking_quality.py. Adapters that
    do not inspect their own response for a ranking-relevant field default
    to NOT_APPLICABLE, same convention as conflict_signal defaulting to
    NOT_APPLICABLE for adapters that skip contradiction detection."""
    degraded_retrieval: RetrievalWarning | None = None
    """Set when the backend's own response signaled it under-delivered on
    this query without treating it as a hard failure -- see
    RetrievalWarning's docstring for the confirmed MemPalace/mempalace#1005
    provenance. `None` (the default) means either the backend reported no
    such signal, or the adapter doesn't inspect its response for one --
    the same backward-compatible-default convention `ranking_signal` and
    `conflict_signal` already establish, so every existing adapter's
    QueryResult construction keeps working unchanged. Distinct from
    `conflict_signal == ConflictSignal.EMPTY_OR_LOST`: that only fires on
    zero records, this fires whenever the backend itself says "I
    under-delivered," including when it still returned some records --
    see RetrievalWarning's docstring for exactly why that distinction
    matters."""
    language_degradation_signal: LanguageDegradationSignal = (
        LanguageDegradationSignal.NOT_APPLICABLE
    )
    """Whether this query's hybrid retrieval signals (BM25/entity-boost)
    genuinely fired, or silently degraded to semantic-only -- see
    LanguageDegradationSignal's docstring for the mem0ai/mem0#4884
    provenance and evals/language_degradation.py. Defaults to
    NOT_APPLICABLE, same backward-compatible-default convention every
    other signal field on this dataclass follows -- only
    `Mem0DirectAdapter.query(explain=True)` currently sets this to
    something else."""
    raw: dict[str, object] = field(default_factory=dict)


@dataclass
class StoreResult:
    """Result of MemoryBackendAdapter.store()."""

    memory_id: str
    latency_ms: float
    raw: dict[str, object] = field(default_factory=dict)
    corruption_signal: CorruptionSignal = CorruptionSignal.NOT_APPLICABLE
    """See CorruptionSignal above. Defaults to NOT_APPLICABLE so every
    existing adapter's StoreResult construction keeps working unchanged --
    only an adapter with a genuine construction-time-config or raw-write
    inspection surface (see Mem0DirectAdapter) should ever set this to
    something else."""
    extraction_signal: ExtractionSignal | None = None
    """See ExtractionSignal above. `None` (not NOT_APPLICABLE) is the
    default so every existing adapter's StoreResult construction keeps
    working unchanged, mirroring `verified`'s None-vs-False distinction
    immediately below: `None` means "this adapter does not report this
    signal at all," which is different from NOT_APPLICABLE meaning "this
    adapter reports the signal and this particular call had nothing to
    classify it against." Only the mem0-backed adapters (Mem0Adapter,
    Mem0SelfHostedAdapter, Mem0DirectAdapter), which are the ones with a
    real LLM-extraction step to observe, set this to something other than
    None -- see mem0_adapter.py's and mem0_direct_adapter.py's store()."""
    verified: bool | None = None
    """Whether a post-write read-back confirmed the content is actually
    retrievable. `None` means the adapter did not attempt verification
    (the default -- see MemoryBackendAdapter.verify_store) -- this is
    deliberately distinct from `False`. `None` is "we don't know," and
    must never be treated as "verified passed" by scoring or reporting
    code. `True`/`False` mean an adapter actually called verify_store()
    (or equivalent) and got a definitive answer.

    Why this field exists at all: store() raising no exception has never
    been proof that a write was durable -- a vendor can return 200 OK (or
    a fake in-process success) while silently dropping or corrupting the
    write server-side. Two independently root-caused MemPalace bug
    classes did exactly this (checkpoint corruption via NUL bytes;
    stale/self-deadlocked locks silently no-oping a write). Without this
    field, that failure mode is indistinguishable from "the model just
    didn't recall the fact," which misattributes a backend durability bug
    to model quality. See docs/methodology.md for why verification is
    opt-in rather than automatic.
    """


@dataclass
class UpdateResult:
    """Result of MemoryBackendAdapter.update()."""

    memory_id: str
    acknowledged: bool
    latency_ms: float
    raw: dict[str, object] = field(default_factory=dict)
    corruption_signal: CorruptionSignal = CorruptionSignal.NOT_APPLICABLE
    """See CorruptionSignal above and StoreResult.corruption_signal -- same
    default-NOT_APPLICABLE backward-compatibility reasoning. This is the
    field a metadata-only update() variant sets to VECTOR_ZEROED/CLEAN when
    an adapter can actually inspect what a vector-store update wrote (see
    Mem0DirectAdapter.update_metadata_only)."""


@dataclass
class RawFilterProbeResult:
    """Result of MemoryBackendAdapter.probe_raw_filter() -- a single,
    caller-controlled filter dict submitted directly to this backend's
    underlying vector-store filter-query-building layer, bypassing the
    normal session-scoped query()'s hardcoded `{"user_id": session_id}`
    filter. This is the primitive evals/filter_injection.py needs to
    reproduce mem0ai/mem0#5980's exact injection shape (a dict/list-valued
    filter value that could embed arbitrary Elasticsearch query operators
    into a `term` query) against a real backend's own filter-building code,
    not a memtrust reimplementation of it.
    """

    accepted: bool
    """True if the underlying call completed without raising -- the
    backend's filter-building layer accepted this filter value as given,
    whatever its type. False if it raised (a validation rejection, a
    malformed-query error from the vendor's own client, or any other
    exception) -- see `error` for detail. This is a raw pass/fail
    observation only, not itself a judgment of "safe" vs "vulnerable":
    evals/filter_injection.py's FilterInjectionSignal makes that judgment
    by cross-referencing `accepted` against each case's known-malicious/
    known-benign ground truth, the same way ExtractionQualitySignal
    cross-references a case's `should_be_stored` ground truth rather than
    treating "retrievable" as inherently good or bad."""
    error: str | None = None
    """The caught exception's message when accepted is False, else None."""
    applicable: bool = True
    """False when the probe never actually reached the vector store's
    filter-building call at all -- e.g. a construction-time config
    rejection (a missing embedding-dimension config, a missing
    Elasticsearch credential) failed before `filters` was ever submitted
    to anything. Distinct from `accepted=False`, which means the call DID
    reach the filter-building layer and that layer rejected the value.
    Without this distinction, a case that never got a real filter-
    validation verdict (construction failed) would be indistinguishable
    from one the backend genuinely rejected on the filter's own merits --
    evals/filter_injection.py's classify_filter_injection_case() checks
    this first and reports FilterInjectionSignal.NOT_APPLICABLE whenever
    it is False, before ever looking at `accepted`."""
    raw: dict[str, object] = field(default_factory=dict)
    """Best-effort raw detail from a successful call (e.g. a truncated
    repr of the vendor response), kept for audit purposes only -- never
    used for scoring."""


@dataclass
class MetadataOverviewResult:
    """Result of MemoryBackendAdapter.metadata_overview() -- the
    library-level equivalent of a vendor's MCP metadata/overview tool
    (MemPalace's `mempalace_status`, see mempalace_adapter.py). Optional
    capability, same convention as RawFilterProbeResult above -- only
    meaningful for a backend whose native surface exposes an aggregate
    record-count-plus-grouping overview; see supports_metadata_overview.

    MemPalace/mempalace#1871 (contributor alionar) found that this exact
    class of tool -- `mempalace_status`, `mempalace_list_wings`,
    `mempalace_list_rooms` -- did a full-collection scan on every call,
    O(N^2) against repeated calls, hanging the MCP server at 158K+
    records. memtrust had zero coverage of this code path before this
    type existed: evals/scale_stress.py only measures store()/query()
    latency, never a metadata/histogram-listing call.
    """

    total_records: int | None
    """Total record count the backend reports, or None if the response
    didn't include one (see `partial`/`error`)."""
    categories: dict[str, int]
    """Top-level grouping breakdown (MemPalace: wing name -> drawer
    count). Named generically, not `wings`, so this type stays usable by
    a future adapter whose native grouping concept isn't wings/rooms at
    all -- see list_metadata_categories()/list_metadata_subcategories()
    below for the same naming choice."""
    subcategories: dict[str, int]
    """Second-level grouping breakdown (MemPalace: room name -> drawer
    count), unscoped (across every category) when this came from
    metadata_overview() rather than list_metadata_subcategories()."""
    latency_ms: float
    partial: bool = False
    """True when the backend itself reported a partial/degraded result
    (e.g. a metadata fetch that errored partway through) -- see `error`.
    Never inferred from "the call didn't raise"; only set when the
    backend's own response says so, the same rule every other *Result
    type in this module follows."""
    error: str | None = None
    """The backend's own reported error string when `partial` is True,
    else None."""


@dataclass
class MetadataCategoryCountsResult:
    """Result of MemoryBackendAdapter.list_metadata_categories() /
    list_metadata_subcategories() -- see MetadataOverviewResult above for
    the motivating bug and the generic category/subcategory naming."""

    counts: dict[str, int]
    scope: str | None
    """The category this subcategory listing was scoped to (MemPalace:
    the `wing` argument), or None for an unscoped listing / a top-level
    category listing that has no scope concept at all."""
    latency_ms: float
    partial: bool = False
    error: str | None = None


@dataclass
class DeleteResult:
    """Result of MemoryBackendAdapter.delete().

    On failure, adapters raise BackendAPIError instead of returning a
    DeleteResult with success=False -- `success` here reports the
    vendor's own acknowledgement shape (e.g. "deleted" vs "already
    gone"), not whether the HTTP call itself succeeded.
    """

    success: bool
    memory_id: str
    latency_ms: float
    raw: dict[str, object] = field(default_factory=dict)
    corruption_signal: CorruptionSignal = CorruptionSignal.NOT_APPLICABLE
    """See CorruptionSignal above and StoreResult.corruption_signal/
    UpdateResult.corruption_signal -- same default-NOT_APPLICABLE
    backward-compatibility reasoning, added here so DeleteResult mirrors
    every other write-path result type rather than being the one write
    primitive with no corruption-inspection surface at all. On this
    adapter's real HTTP backends, a delete() call that hits a write-path
    corruption shape (e.g. volcengine/OpenViking#2966's legacy uint16-
    truncated records, see CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE)
    raises BackendAPIError instead of returning a DeleteResult at all --
    per this dataclass's own docstring above, adapters raise on failure
    rather than returning success=False -- so this field stays at its
    default for every adapter in this repo today. It exists for the same
    forward-compatibility reason StoreResult/UpdateResult's fields do:
    only a future adapter with a genuine direct-handle inspection surface
    (see Mem0DirectAdapter's precedent for those two fields) could ever
    set this to something other than NOT_APPLICABLE."""


@dataclass
class DeletePrefixResult:
    """Result of MemoryBackendAdapter.delete_prefix() -- a recursive
    directory/prefix delete, distinct from delete()/delete_many()'s
    single-`memory_id`-at-a-time model. This is the primitive
    evals/orphan_cleanup.py needs to reproduce the volcengine/
    OpenViking#3064 shape: a prefix delete that reports success while
    leaving child vector-index entries orphaned -- something delete() and
    delete_many() cannot construct at all, since both require the caller
    to already know every individual memory_id up front, while a real
    orphan-cleanup delete targets a whole subtree by prefix without
    necessarily knowing every leaf id in advance (exactly the scenario
    #3064 describes: files removed directly from the backing filesystem,
    so their ids/paths are unknown to the caller issuing the cleanup
    delete).
    """

    prefix: str
    deleted_paths: list[str]
    """Every path (leaf files discovered via list_resource_paths(), plus
    the prefix root itself) this call successfully deleted, in the order
    delete() was called on them. Does NOT include paths this call never
    discovered in the first place (e.g. a child the underlying listing
    call silently omitted) -- see VectorIntegritySignal.ORPHANED_VECTOR_ENTRY
    for the classification of exactly that gap."""
    failed_paths: list[str]
    """Every path this call discovered but whose delete() call either
    raised BackendAPIError or returned success=False. Never silently
    dropped -- same one-result-per-discovered-path accounting principle
    delete_many() already establishes for delete()."""
    latency_ms: float


@dataclass
class StatsResult:
    """Result of MemoryBackendAdapter.get_stats() -- a backend's own
    self-reported count of how many memories it currently holds, read from
    a dedicated stats/dashboard endpoint rather than derived from a
    store()/query() call.

    This exists to reproduce volcengine/OpenViking#1255 (contributor
    SeeYangZhi): `GET /api/v1/stats/memories` returns an all-zero count
    even when filesystem listing (`list_resource_paths()`) and semantic
    search (`query()`) both independently confirm the memories genuinely
    exist -- a separate metrics/counting code path that was never wired up
    to the real write path. `total_memories` here is deliberately the
    backend's own self-report, exactly the number a caller would see if it
    trusted this endpoint alone; evals/stats_accuracy.py is what
    cross-checks it against an independent ground-truth count."""

    total_memories: int | None
    """The backend's self-reported total memory count, or `None` if the
    response had no field this adapter knows how to read (distinct from a
    real `0`, which is a valid, meaningful count that may still turn out
    to be wrong -- see StatsSignal.STATS_UNDERCOUNTED in
    evals/stats_accuracy.py)."""
    latency_ms: float = 0.0
    raw: dict[str, object] = field(default_factory=dict)
    """Unmodified vendor response, kept for audit purposes only -- never
    used for scoring directly (see StatsResult.total_memories above)."""


@dataclass
class MigrationFailureResult:
    """Result of MemoryBackendAdapter.simulate_migration_failure().

    Reports whether the ORIGINAL pre-migration data survived a simulated
    mid-migration failure -- the primitive
    MemoryBackendAdapter.supports_migration_rollback_simulation gates and
    evals/migration_rollback.py drives. Modeled on the exact shape
    MemPalace/mempalace#1028 (GitHub user eldar702) reports: an unguarded
    `shutil.rmtree()`-then-`shutil.move()` swap at the end of
    MemPalace's own `migrate.migrate()` function deletes the old backup
    FIRST, so if the `move()` step fails partway (e.g. a cross-device
    `EXDEV` error), the palace directory is permanently lost -- there is
    no backup left to fall back to. MemPalace/mempalace#935 is the real
    upstream fix: a safer "rename-aside" swap that renames the new data
    into place first, keeps the old backup renamed-aside, and only
    deletes the backup after confirming the swap succeeded, so a failure
    mid-swap leaves the original data recoverable.
    """

    session_id: str
    memory_id: str
    content: str
    """The exact content simulate_migration_failure() stored as the
    ORIGINAL pre-migration data before simulating the failure."""
    original_data_recoverable: bool
    """This adapter's OWN observation (via its normal query() path) of
    whether `content` is still retrievable after the simulated failure.
    Kept as a diagnostic/cross-check field only -- evals/
    migration_rollback.py's run_migration_rollback_eval() does not trust
    this self-report as the classification's ground truth. It makes its
    own independent query() call after simulate_migration_failure()
    returns and classifies from that instead, the same "never trust a
    single response" principle every eval in this package follows (see
    evals/crash_recovery.py's classify_crash_recovery_case and
    evals/ranking_quality.py's classify_ranking_case for precedent)."""


class MemoryBackendAdapter(ABC):
    """Abstract base every backend adapter must implement.

    Contract for implementers:
      * __init__ must read credentials from an environment variable and
        raise BackendNotConfiguredError immediately if missing -- never
        defer the check to the first method call.
      * store()/query()/update()/delete() must raise BackendAPIError (not
        a bare vendor exception) on any network/API failure, so the
        harness can report a uniform error shape across all backends.
      * Never mutate eval logic per backend. If a backend cannot support
        an operation (see supports_update), report that fact through
        supports_update / ConflictSignal.NOT_APPLICABLE rather than
        faking a response.
    """

    #: Human-readable vendor name, used in CLI output and reports.
    name: str = "unknown"

    #: Environment variable this adapter reads its credential/config from.
    #: Subclasses must set this so BackendNotConfiguredError messages are
    #: accurate and so `memtrust run` can pre-check configuration status
    #: without constructing the adapter.
    env_var: str = ""

    #: Whether this backend exposes an update/invalidate primitive the
    #: contradiction-detection eval can meaningfully exercise. If False,
    #: the eval records ConflictSignal.NOT_APPLICABLE instead of running
    #: the eval against this backend and silently dropping it from the
    #: results table.
    supports_update: bool = True

    #: Whether this backend exposes a directory/resource mirror that a
    #: multi-file resync operation can act on (list_resource_paths /
    #: trigger_resync below). Defaults to False -- the store/query/update
    #: model most adapters implement has no concept of a resync, so most
    #: backends genuinely cannot be exercised here. If False, the
    #: resource-sync-safety eval records the backend as skipped instead of
    #: calling list_resource_paths/trigger_resync and crashing on the
    #: default NotImplementedError below.
    supports_resource_sync: bool = False

    #: Named operating modes this backend exposes that change how content
    #: is stored/retrieved (e.g. a vendor's "compressed"/"lossless" write
    #: path vs. its raw path). Empty by default -- most adapters have no
    #: mode variants at all, and the compression/round-trip-fidelity eval
    #: (evals/compression.py) treats an empty tuple as "run once under a
    #: single synthetic 'default' mode" rather than skipping the backend.
    #: An adapter that declares a non-empty tuple here is asserting that
    #: passing each of those strings as `mode=` to store()/query() below
    #: actually selects a different vendor-side code path -- see
    #: mempalace_adapter.py for the one adapter that currently does this,
    #: and the honesty caveat attached to its mode names.
    supported_modes: tuple[str, ...] = ()

    #: Whether this adapter can simulate a "server process crashed and
    #: restarted, losing its in-memory search index while the underlying
    #: store data survives" event via simulate_crash_restart() /
    #: raw_store_contains() below. Defaults to False -- every real,
    #: HTTP-only adapter in this repo talks to a backend over the network
    #: with zero ability to start, kill, or restart the vendor's own
    #: server process, and has no API surface that bypasses the vendor's
    #: search index to inspect raw stored data directly. Only a
    #: purpose-built in-memory fake adapter (see
    #: tests/test_evals.py::CrashRecoveryFakeAdapter) can genuinely model
    #: both halves of this -- see evals/crash_recovery.py for the eval
    #: this gates, and docs/methodology.md for why that eval cannot prove
    #: anything about a real backend's process-lifecycle behavior.
    supports_crash_recovery_simulation: bool = False

    #: Whether this adapter can simulate a storage-migration's final swap
    #: step failing partway through, via simulate_migration_failure()
    #: below. Defaults to False -- same "no adapter in this repo has real
    #: process/filesystem-lifecycle control over a live backend" reasoning
    #: as supports_crash_recovery_simulation above: no adapter here holds
    #: a handle to a live vendor package's internal migration code path it
    #: could actually interrupt partway through. Only a purpose-built
    #: in-memory fake adapter (see
    #: tests/test_evals.py::MigrationRollbackFakeAdapter) can genuinely
    #: model both the buggy unguarded-rmtree()-then-move() swap and the
    #: fixed rename-aside swap -- see evals/migration_rollback.py for the
    #: eval this gates, and that module's docstring plus
    #: mempalace_adapter.py's module docstring for why this cannot prove
    #: anything about a real MemPalace deployment's migrate.migrate()
    #: behavior.
    supports_migration_rollback_simulation: bool = False

    #: Whether this adapter can submit an arbitrary, caller-controlled
    #: filter dict directly to the underlying vector store's own
    #: filter-query-building layer, bypassing query()'s hardcoded
    #: `{"user_id": session_id}` filter, via probe_raw_filter() below.
    #: Defaults to False -- every real, HTTP-only adapter in this repo has
    #: no documented way to submit a raw, unvalidated filter value outside
    #: its own normal session-scoped query() path. Only Mem0DirectAdapter,
    #: which holds a direct, in-process handle to the vendor library
    #: including its constructed vector_store, can genuinely reach into
    #: that layer -- see evals/filter_injection.py, the eval this gates,
    #: and mem0_direct_adapter.py's probe_raw_filter() override.
    supports_raw_filter_probe: bool = False

    #: Whether this backend exposes a dedicated stats/dashboard endpoint
    #: this adapter can read via get_stats() below, distinct from deriving
    #: a count from query()/list_resource_paths() results. Defaults to
    #: False -- most adapters in this repo have no such endpoint documented
    #: at all. See evals/stats_accuracy.py, the eval this gates, and
    #: openviking_adapter.py's get_stats() for the one adapter that
    #: currently sets this True (volcengine/OpenViking#1255).
    supports_stats: bool = False

    #: Whether this adapter exposes a recursive prefix/directory delete
    #: primitive via delete_prefix() below, distinct from the single-
    #: memory_id delete()/delete_many() every adapter must implement.
    #: Defaults to False -- most adapters have no directory/resource-mirror
    #: concept at all (the same store/query/update model that gates
    #: supports_resource_sync above). Only an adapter with a real
    #: directory-listing primitive to discover child paths by prefix (see
    #: OpenVikingAdapter.list_resource_paths()) can implement this
    #: meaningfully -- see evals/orphan_cleanup.py, the eval this gates,
    #: and VectorIntegritySignal in this module for the failure shape
    #: (volcengine/OpenViking#3064) it exists to detect.
    supports_prefix_delete: bool = False

    #: Whether this adapter exposes a library-level equivalent of a
    #: vendor's MCP metadata/overview tool surface (MemPalace's
    #: `mempalace_status`/`mempalace_list_wings`/`mempalace_list_rooms`)
    #: via metadata_overview()/list_metadata_categories()/
    #: list_metadata_subcategories() below. Defaults to False -- most
    #: adapters in this repo have no such surface at all (store/query/
    #: update/delete is the entire interface). Only MemPalaceAdapter,
    #: which holds a direct in-process handle to the real
    #: `mempalace.mcp_server` module's `tool_status`/`tool_list_wings`/
    #: `tool_list_rooms` functions (confirmed real and callable without
    #: spinning up the actual MCP stdio/HTTP transport -- see
    #: mempalace_adapter.py's module docstring), can genuinely exercise
    #: this. See evals/mempalace_metadata_scale.py, the eval this gates,
    #: for the motivating MemPalace/mempalace#1871 O(N^2) full-collection-
    #: scan bug this closes coverage for.
    supports_metadata_overview: bool = False

    @abstractmethod
    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        """Store a new memory under the given session/user scope.

        Args:
            session_id: logical conversation/user scope for this memory.
            content: the text to store.
            metadata: optional vendor-agnostic key/value tags.
            mode: optional operating-mode selector (see `supported_modes`).
                Adapters that don't expose mode variants MUST accept this
                parameter and ignore it (no-op) rather than raising, so
                that callers written against the shared interface can pass
                `mode=` uniformly across every backend without special-
                casing the ones that don't support it. This preserves
                backward compatibility: any pre-existing call site that
                never passes `mode` continues to behave identically.

        Raises:
            BackendAPIError: on any network or vendor-side failure.

        Note on durability: returning without raising is *not* proof the
        write is durable or even retrievable -- a vendor can silently
        drop or corrupt a write server-side and still return a normal
        response (this is exactly what happened in two independently
        root-caused MemPalace bugs: NUL-byte checkpoint corruption and
        stale/self-deadlocked locks). Implementers that want to guard
        against this should accept an opt-in `verify: bool = False`
        keyword-only parameter and, when True, call `self.verify_store()`
        after the write succeeds and set `StoreResult.verified` from its
        return value. This is intentionally NOT part of every adapter's
        required signature (existing callers that never pass `verify`
        must keep working unchanged against any adapter), and it is
        intentionally NOT on by default anywhere -- see verify_store()
        and docs/methodology.md for why (it doubles API calls per store()
        when enabled).
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        """Retrieve memories relevant to `query` within `session_id`.

        Args:
            mode: optional operating-mode selector, same contract as
                `store()`'s `mode` parameter above -- ignored (no-op) by
                adapters with no mode variants.

        Raises:
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        """Store a fact that may contradict a previously stored one.

        Implementers should call whatever the vendor's native mechanism is
        for "this may supersede an existing memory" (an explicit update
        call, a second store() that the vendor's own pipeline resolves,
        etc.) and report what actually happened in UpdateResult -- do not
        synthesize agreement with the request.

        Raises:
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, memory_id: str) -> DeleteResult:
        """Delete a single stored memory/entity by id.

        This is the primitive an eval needs to reproduce vendor bugs in
        the "delete N entities" class (e.g. a client whose batch-delete
        code silently keeps only the last response instead of aggregating
        all N) -- see delete_many() below, which is what an eval actually
        calls to construct that scenario.

        Implementers that genuinely cannot delete (no documented/verified
        vendor endpoint) must still define this method and raise
        BackendAPIError with a clear "not implemented for this backend"
        detail rather than omitting the method -- the eval layer needs a
        uniform call shape across all adapters, same as store/query/
        update, even when the honest answer is "not supported yet."

        Raises:
            BackendAPIError: on any network/vendor-side failure, or when
                this backend has no verified delete endpoint.
        """
        raise NotImplementedError

    def delete_many(self, memory_ids: list[str]) -> list[DeleteResult]:
        """Delete several memories, one at a time, via delete().

        Default implementation for every adapter: a plain per-id loop
        that appends each DeleteResult (or a failure record, if an
        individual delete() call raises) to a single results list sized
        exactly len(memory_ids). This is deliberately naive -- it exists
        so the eval layer has one aggregation path to trust, rather than
        each adapter rolling its own batch logic that could silently
        drop or overwrite results the way a buggy vendor client might
        (see mem0ai/mem0#5936, #5970: a multi-entity delete that kept
        only the last response instead of all N). Adapters with a real
        vendor batch-delete endpoint may override this for efficiency,
        but must preserve the same one-result-per-input-id contract.

        Does not raise: a per-id BackendAPIError is caught and recorded
        as a DeleteResult(success=False, ...) at that id's position so
        one failure in the middle of a batch cannot truncate or drop the
        results for the ids after it.
        """
        results: list[DeleteResult] = []
        for memory_id in memory_ids:
            try:
                results.append(self.delete(memory_id))
            except BackendAPIError as exc:
                results.append(
                    DeleteResult(
                        success=False,
                        memory_id=memory_id,
                        latency_ms=0.0,
                        raw={"error": str(exc)},
                    )
                )
        return results

    def list_resource_paths(self, prefix: str) -> list[str]:
        """List resource/file paths currently present under `prefix`.

        Optional capability -- only meaningful for backends that model a
        directory/resource mirror (see supports_resource_sync). Unlike
        store()/query()/update(), this is NOT an abstract method: most
        adapters have no resource-mirror concept at all, so the default
        implementation raises NotImplementedError rather than forcing
        every adapter to stub it out. Implementers that set
        supports_resource_sync = True must override this.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_resource_sync is False).
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError(
            f"{self.name} does not implement list_resource_paths() "
            f"(supports_resource_sync={self.supports_resource_sync})"
        )

    def trigger_resync(self, prefix: str) -> None:
        """Trigger whatever native mechanism this backend uses to reconcile
        its resource mirror under `prefix` against the source it ingests
        from (e.g. a directory watcher's resync/rescan pass).

        Optional capability, same convention as list_resource_paths()
        above -- default raises NotImplementedError, only backends with
        supports_resource_sync = True are expected to override it.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_resource_sync is False).
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError(
            f"{self.name} does not implement trigger_resync() "
            f"(supports_resource_sync={self.supports_resource_sync})"
        )

    def simulate_crash_restart(self) -> None:
        """Simulate a "server process crashed and restarted" event: the
        in-memory search index is lost, but whatever the backend's
        underlying store actually persisted survives.

        This is deliberately NOT a real process kill/restart -- no
        adapter in this repo holds a handle to a live vendor server
        process it could actually terminate and relaunch (every real
        adapter here is a pure HTTP client; see the module docstring at
        the top of this file). It is an explicit, named simulation
        primitive an in-memory fake adapter implements to model the
        specific failure shape volcengine/OpenViking#2644 (contributor
        yeyitech) reports: a local vectordb's `_recover()` silently skips
        rebuilding the index on process restart when index files are
        missing but store data exists, so post-restart queries silently
        return nothing even though the data was never actually lost.

        Optional capability, same convention as list_resource_paths()/
        trigger_resync() above -- default raises NotImplementedError,
        only adapters with supports_crash_recovery_simulation = True are
        expected to override it. See evals/crash_recovery.py.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_crash_recovery_simulation is False).
        """
        raise NotImplementedError(
            f"{self.name} does not implement simulate_crash_restart() "
            f"(supports_crash_recovery_simulation={self.supports_crash_recovery_simulation})"
        )

    def raw_store_contains(self, session_id: str, memory_id: str) -> bool:
        """Check whether the underlying store still holds `memory_id`,
        bypassing whatever search/index layer query() goes through.

        This is the primitive that lets evals/crash_recovery.py tell
        "the index is gone but the data survived" (the volcengine/
        OpenViking#2644 bug shape) apart from "the data itself is gone
        too" -- query() alone cannot distinguish these, since a lost
        index and lost data both make query() return nothing.

        Optional capability, same convention as list_resource_paths()/
        trigger_resync() above -- most real adapters have no vendor API
        that reads underlying stored data independently of the vendor's
        own search index, so the default raises NotImplementedError.
        Only adapters with supports_crash_recovery_simulation = True are
        expected to override it.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_crash_recovery_simulation is False).
        """
        raise NotImplementedError(
            f"{self.name} does not implement raw_store_contains() "
            f"(supports_crash_recovery_simulation={self.supports_crash_recovery_simulation})"
        )

    def simulate_migration_failure(self, session_id: str, content: str) -> MigrationFailureResult:
        """Store `content` as the ORIGINAL pre-migration data, then
        simulate a storage migration whose final swap step is interrupted
        before the commit step completes, and report whether `content` is
        still recoverable afterward.

        This is deliberately NOT a real migration or a real filesystem
        fault injection -- no adapter in this repo holds a handle to a
        live vendor package's actual migrate() code path it could
        genuinely interrupt partway through (every real adapter here is
        either a pure HTTP client or, for MemPalaceAdapter, a thin wrapper
        around whatever the installed `mempalace` package's own
        remember()/recall()/invalidate() do internally -- see
        mempalace_adapter.py's module docstring). It is an explicit, named
        simulation primitive an in-memory fake adapter implements to model
        the specific failure shape MemPalace/mempalace#1028 (GitHub user
        eldar702) reports: MemPalace's own `migrate.migrate()` function had
        an unguarded `shutil.rmtree()`-then-`shutil.move()` swap at the end
        of a migration -- if the `move()` step failed partway (e.g. a
        cross-device `EXDEV` error), the palace directory could be
        permanently lost, since the old backup was already deleted first.
        MemPalace/mempalace#935 is the real upstream fix this primitive
        exists to let an eval verify the CONCEPT of (not a specific merged
        diff): a "rename-aside" swap that renames the new data into place
        first, keeps the old backup renamed-aside, and only deletes it
        after confirming the swap succeeded, so a failure mid-swap leaves
        the original data recoverable.

        Optional capability, same convention as simulate_crash_restart()/
        raw_store_contains() above -- default raises NotImplementedError,
        only adapters with supports_migration_rollback_simulation = True
        are expected to override it. See evals/migration_rollback.py.

        Args:
            session_id: logical conversation/user scope to store the
                original pre-migration content under.
            content: the exact text to treat as the original pre-migration
                data whose survival across the simulated failure is being
                tested.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_migration_rollback_simulation is False).
        """
        raise NotImplementedError(
            f"{self.name} does not implement simulate_migration_failure() "
            f"(supports_migration_rollback_simulation="
            f"{self.supports_migration_rollback_simulation})"
        )

    def probe_raw_filter(self, filters: dict[str, object]) -> RawFilterProbeResult:
        """Submit `filters` directly to this backend's underlying
        vector-store filter-query-building layer, bypassing the normal
        session-scoped query()'s hardcoded `{"user_id": session_id}`
        filter -- the primitive evals/filter_injection.py needs to probe
        whether a dict/list-valued filter value (the mem0ai/mem0#5980
        injection shape) is validated before being embedded into a query.

        Optional capability, same convention as list_resource_paths()/
        trigger_resync()/simulate_crash_restart() above -- default raises
        NotImplementedError, only adapters with supports_raw_filter_probe
        = True are expected to override it.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_raw_filter_probe is False).
        """
        raise NotImplementedError(
            f"{self.name} does not implement probe_raw_filter() "
            f"(supports_raw_filter_probe={self.supports_raw_filter_probe})"
        )

    def get_stats(self, session_id: str | None = None) -> StatsResult:
        """Read this backend's own dedicated stats/dashboard endpoint --
        e.g. a `total_memories` counter maintained by a separate
        metrics/aggregation code path, not derived from a store()/query()
        call this adapter makes itself.

        This is the primitive evals/stats_accuracy.py needs to reproduce
        volcengine/OpenViking#1255's exact shape: a stats endpoint that
        returns an all-zero (or otherwise undercounted) result even though
        the memories genuinely exist and are independently confirmed via
        `list_resource_paths()`/`query()`.

        Optional capability, same convention as list_resource_paths()/
        trigger_resync()/probe_raw_filter() above -- default raises
        NotImplementedError, only adapters with supports_stats = True are
        expected to override it.

        Args:
            session_id: optional scope to read stats for, if this
                backend's stats endpoint supports scoping. Adapters whose
                real endpoint has no such concept accept and ignore it
                (no-op), the same backward-compatible convention store()'s
                `mode` parameter establishes.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_stats is False).
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError(
            f"{self.name} does not implement get_stats() (supports_stats={self.supports_stats})"
        )

    def delete_prefix(self, prefix: str, recursive: bool = True) -> DeletePrefixResult:
        """Recursively delete every resource path under `prefix`, distinct
        from delete()/delete_many()'s single-known-memory_id model -- the
        primitive an eval needs to reproduce the volcengine/OpenViking#3064
        orphan-cleanup bug class: a prefix delete that discovers child
        paths via a directory-listing call and can therefore miss entries
        the listing call itself fails to enumerate (e.g. because the
        parent directory no longer exists in the backing filesystem),
        leaving those entries as permanently orphaned vector-index records
        even though the delete reported success.

        Optional capability, same convention as list_resource_paths()/
        trigger_resync()/probe_raw_filter() above -- default raises
        NotImplementedError, only adapters with supports_prefix_delete =
        True are expected to override it. See evals/orphan_cleanup.py and
        VectorIntegritySignal in this module for the eval and signal this
        gates.

        Args:
            prefix: the resource-path prefix to delete everything under.
            recursive: when True (the default), discover and delete every
                nested child path under `prefix`, not just paths directly
                at the top level. Implementers that cannot distinguish
                depth at all may treat this as always-recursive.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_prefix_delete is False).
            BackendAPIError: on any network/vendor-side failure that
                prevents the delete from being attempted at all (a
                per-child delete() failure is instead recorded in the
                returned DeletePrefixResult.failed_paths, not raised).
        """
        raise NotImplementedError(
            f"{self.name} does not implement delete_prefix() "
            f"(supports_prefix_delete={self.supports_prefix_delete})"
        )

    def metadata_overview(self) -> MetadataOverviewResult:
        """Fetch this backend's aggregate record-count-plus-grouping
        overview -- the library-level equivalent of MemPalace's
        `mempalace_status` MCP tool.

        Optional capability, same convention as list_resource_paths()/
        probe_raw_filter() above -- default raises NotImplementedError,
        only adapters with supports_metadata_overview = True are expected
        to override it.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_metadata_overview is False).
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError(
            f"{self.name} does not implement metadata_overview() "
            f"(supports_metadata_overview={self.supports_metadata_overview})"
        )

    def list_metadata_categories(self) -> MetadataCategoryCountsResult:
        """List every top-level group this backend's records are
        organized under, with a record count per group -- the
        library-level equivalent of MemPalace's `mempalace_list_wings`
        MCP tool.

        Optional capability, same convention as metadata_overview()
        above -- default raises NotImplementedError, only adapters with
        supports_metadata_overview = True are expected to override it.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_metadata_overview is False).
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError(
            f"{self.name} does not implement list_metadata_categories() "
            f"(supports_metadata_overview={self.supports_metadata_overview})"
        )

    def list_metadata_subcategories(
        self, category: str | None = None
    ) -> MetadataCategoryCountsResult:
        """List every second-level group this backend's records are
        organized under, optionally scoped to one top-level `category` --
        the library-level equivalent of MemPalace's `mempalace_list_rooms`
        MCP tool.

        Optional capability, same convention as metadata_overview()
        above -- default raises NotImplementedError, only adapters with
        supports_metadata_overview = True are expected to override it.

        Args:
            category: restrict the listing to this top-level group
                (MemPalace: a wing name), or None to list across every
                category.

        Raises:
            NotImplementedError: if the adapter does not implement this
                (i.e. supports_metadata_overview is False).
            BackendAPIError: on any network or vendor-side failure.
        """
        raise NotImplementedError(
            f"{self.name} does not implement list_metadata_subcategories() "
            f"(supports_metadata_overview={self.supports_metadata_overview})"
        )

    @staticmethod
    def _timed() -> _Timer:
        return _Timer()

    def verify_store(self, store_result: StoreResult, session_id: str, content: str) -> bool:
        """Opt-in read-after-write check: query() immediately after a
        store() call and confirm the just-written content is actually
        retrievable, instead of trusting "store() didn't raise" as proof
        of a durable write.

        This is a helper, not something store() calls automatically --
        no adapter's default `store()` behavior changes by this method
        existing. An adapter opts in by accepting a `verify: bool = False`
        keyword-only parameter on its own store() and calling this
        explicitly when `verify=True` (see mempalace_adapter.py for the
        reference implementation). Left off by default because it costs
        one extra vendor API call (a query()) for every store() call made
        with verify=True -- turning that on unconditionally for every eval
        run would silently double memtrust's own API/latency cost against
        every backend under test. See docs/methodology.md.

        Args:
            store_result: the StoreResult just returned by this adapter's
                own store() call, used for its memory_id.
            session_id: the same session/scope the content was stored
                under.
            content: the exact text that was just stored.

        Returns:
            True only if a query() call for `session_id` returns a record
            whose *content* actually contains `content` -- either the
            record matching `store_result.memory_id` by id (checked
            first, and its content must still match: a record that comes
            back under the right id but with corrupted content is a
            failed verification, not a pass) or, if no record shares that
            id (some backends don't echo a stable id on the query path),
            any returned record whose content contains `content`. False
            covers both "no matching record at all" (dropped write) and
            "a record came back but the content doesn't match" (corrupted
            write) -- both are reported as a normal `False` return, never
            raised as an error.

        Raises:
            BackendAPIError: if the verification query() call itself
                fails (a real network/vendor error is a different failure
                mode than "the write was silently dropped," and should
                still surface as an error rather than being swallowed
                into `False`).
        """
        result = self.query(session_id, content)
        for record in result.records:
            if record.memory_id and record.memory_id == store_result.memory_id:
                return bool(content) and content in record.content
        return any(content and content in record.content for record in result.records)


class _Timer:
    """Tiny context-manager-free stopwatch so adapters can report latency
    without importing timing boilerplate in every subclass."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000
