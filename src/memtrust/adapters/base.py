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


class BackendAPIError(Exception):
    """Raised when a configured backend's API call fails (network, auth,
    5xx, malformed response). Distinct from BackendNotConfiguredError so
    callers can tell "never had credentials" apart from "had credentials,
    the call still failed."
    """

    def __init__(self, backend_name: str, detail: str) -> None:
        self.backend_name = backend_name
        self.detail = detail
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
