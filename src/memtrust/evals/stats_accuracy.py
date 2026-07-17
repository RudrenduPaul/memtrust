"""MemTrust's stats/dashboard-accuracy eval.

Every other eval in this package treats a backend's own store()/query()
responses as the only source of truth about what is actually stored (with
appropriate independent cross-checks -- see crash_recovery.py's
raw_store_contains(), resource_sync_safety.py's list_resource_paths()).
None of them look at a backend's *self-reported summary statistics* at
all. This eval closes that specific gap: it checks whether a backend's own
"how many memories do you have" endpoint agrees with an independently
verified count of what is actually retrievable.

Motivating case: volcengine/OpenViking#1255 (contributor SeeYangZhi).
`GET /api/v1/stats/memories` returned an all-zero count
(`{"total_memories": 0, "by_category": {...all zero...}, ...}`) on a fresh
instance immediately after session-commit-triggered memory extraction had
already completed -- confirmed still present via two independent
observations quoted in the bug report itself: Docker logs showing
"Extracted 10 candidate memories" / "Created memory file: ...", and a
`POST /v1/search/find` call that successfully returned those same memory
files with their extracted abstracts. The reporter's own root-cause read:
the stats endpoint reads from a dedicated stats/metadata index populated
only by a synchronous "content/write" code path, never by the async
extraction pipeline that actually wrote the memories -- so the counter and
the real data structurally diverge, with no exception anywhere to signal
the mismatch.

This eval reproduces that shape generically: store N records, independently
verify how many of them are actually retrievable via the adapter's normal
query() path (the same surface #1255's own report used `/v1/search/find`
for), then call get_stats() and compare its self-reported count against
that independently verified number.

**Why a new StatsSignal enum rather than a new ConflictSignal member.**
`ConflictSignal.EMPTY_OR_LOST` (adapters/base.py) fires when a single
query() call succeeds but returns zero records -- it is about *retrieval*
silently coming back empty. This eval's failure mode is different in kind:
query() itself may work perfectly (as it did in #1255's own report --
`/v1/search/find` DID return the memories), while a *separate,
independently-maintained* stats/dashboard endpoint reports a number that
disagrees with that working retrieval path. Folding this into
ConflictSignal would blur "retrieval failed" and "retrieval works, but a
different, unrelated counter is wrong" into the same signal -- the same
"a cross-call/cross-endpoint comparison is a structurally different kind
of classification than a single QueryResult" reasoning that already
justifies CrashRecoverySignal, ResourceSyncSignal, and RankingSignal each
being their own enum (see their docstrings in adapters/base.py and
evals/crash_recovery.py).

**Honest scope.** Like every other eval added without a live, configured
vendor instance available (see evals/scale_stress.py, evals/
embedding_drift.py, docs/methodology.md), this eval's classification logic
is proven correct against purpose-built fake adapters in
tests/test_evals.py -- one reproducing #1255's exact "always reports zero
regardless of real stored count" shape, one reporting an accurate count as
the negative control. It has not been run against a live OpenViking
instance; running it there requires OPENVIKING_API_KEY (or a self-hosted
OPENVIKING_BASE_URL) to be configured, which is not the case in the
environment this eval was built in.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter

#: Small by default (matches every other bundled fixture's single-digit
#: scale, see scale_stress.py's DEFAULT_N_RECORDS docstring for why that
#: is a deliberate, fast-in-CI choice) -- this eval only needs enough
#: records to tell "the endpoint reports a real number" apart from "the
#: endpoint always reports zero/undercounts," not to exercise scale.
DEFAULT_N_RECORDS = 5
DEFAULT_SESSION_ID = "stats-accuracy-session"


class StatsSignal(StrEnum):
    """Whether a backend's self-reported stats/dashboard count agrees with
    an independently verified count of what is actually retrievable.

    Defined locally in this module rather than in adapters/base.py,
    following the same precedent evals/scale_stress.py's ScaleSignal and
    evals/resource_sync_safety.py's ResourceSyncSignal already set: this is
    a harness-computed classification derived from ground truth (records
    this eval itself stored and independently re-verified), not a signal
    any adapter self-reports.
    """

    STATS_MATCH = "stats_match"
    """get_stats()'s reported count is greater than or equal to the
    independently verified retrievable count. The good outcome -- the
    stats endpoint is not undercounting relative to what this eval could
    confirm is actually there. (A report strictly higher than verified is
    still STATS_MATCH, not a separate "overcounted" signal -- this eval
    exists to catch #1255's undercounting shape specifically; a backend
    reporting MORE than this eval's own narrow verification found has no
    known real-world motivating bug case to justify inventing a
    classification for it, and would likely just mean the backend's own
    count includes records this eval's own query-based verification
    under-samples for unrelated reasons.)"""

    STATS_UNDERCOUNTED = "stats_undercounted"
    """get_stats()'s reported count is strictly less than the independently
    verified retrievable count -- the exact volcengine/OpenViking#1255
    shape: the memories are demonstrably there (this eval could retrieve
    them via the normal query() path) but the dedicated stats/dashboard
    endpoint reports fewer of them, including the reported #1255 case of
    reporting zero when the real count is nonzero."""

    NOT_APPLICABLE = "not_applicable"
    """Either the adapter has no stats endpoint at all
    (MemoryBackendAdapter.supports_stats is False -- the eval is skipped,
    not run), or nothing was successfully stored/verified in the first
    place, so there is no independently verified count to compare
    get_stats() against."""


@dataclass
class StatsAccuracyEvalResult:
    backend_name: str
    n_records_requested: int
    records_stored: int = 0
    verified_count: int | None = None
    """Independently verified count of the stored records that are
    actually retrievable via adapter.query() -- the ground truth this
    eval compares get_stats() against. `None` if nothing was ever stored
    successfully."""
    reported_count: int | None = None
    """adapter.get_stats().total_memories, verbatim -- the number under
    test. `None` if get_stats() reported no usable count."""
    signal: StatsSignal = StatsSignal.NOT_APPLICABLE
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None

    @property
    def undercount_gap(self) -> int | None:
        """verified_count - reported_count, i.e. how many retrievable
        records the stats endpoint failed to count. `None` unless both
        counts are known. Never negative in practice under STATS_MATCH
        (see StatsSignal.STATS_MATCH's docstring on why an over-report
        is not itself flagged), but not clamped here -- this is a raw
        diagnostic value, not the classification itself."""
        if self.verified_count is None or self.reported_count is None:
            return None
        return self.verified_count - self.reported_count


def classify_stats_accuracy(verified_count: int | None, reported_count: int | None) -> StatsSignal:
    """Classify one run's outcome from its two independently obtained
    counts. Never trusts get_stats() alone -- the same "never trust a
    single response" principle every eval in this package follows (see
    evals/crash_recovery.py's classify_crash_recovery_case and
    evals/resource_sync_safety.py's classify_resource_sync_file).
    """
    if verified_count is None or reported_count is None:
        return StatsSignal.NOT_APPLICABLE
    if reported_count < verified_count:
        return StatsSignal.STATS_UNDERCOUNTED
    return StatsSignal.STATS_MATCH


def _verify_stored_count(
    adapter: MemoryBackendAdapter, session_id: str, markers: list[str], top_k: int
) -> int:
    """Independently confirm how many of `markers` are actually
    retrievable via the adapter's normal query() path -- the same surface
    #1255's own bug report used (`POST /v1/search/find`) to prove the
    memories existed despite the stats endpoint reporting zero.

    Never trusts "query() didn't raise" as proof a record came back --
    each marker must actually appear in the joined content of the
    response, the same rule evals/scale_stress.py's _query_needle applies.
    """
    found = 0
    for marker in markers:
        try:
            query_result = adapter.query(session_id, marker, top_k=top_k)
        except BackendAPIError:
            continue
        if any(marker in record.content for record in query_result.records):
            found += 1
    return found


def run_stats_accuracy_eval(
    adapter: MemoryBackendAdapter,
    n_records: int = DEFAULT_N_RECORDS,
    session_id: str = DEFAULT_SESSION_ID,
    top_k: int = 5,
) -> StatsAccuracyEvalResult:
    """Store `n_records` uniquely-markered records, independently verify
    how many are actually retrievable via query(), then compare that
    ground truth against adapter.get_stats()'s self-reported count.

    Args:
        adapter: the backend under test.
        n_records: how many synthetic records to store and verify.
        session_id: session/scope every record is stored under.
        top_k: top_k passed to each verification query() call.
    """
    result = StatsAccuracyEvalResult(backend_name=adapter.name, n_records_requested=n_records)

    if not adapter.supports_stats:
        result.skipped = True
        result.skip_reason = (
            f"{adapter.name} does not implement get_stats() (supports_stats=False) -- "
            "skipped, not run. See adapters/base.py's MemoryBackendAdapter.get_stats() "
            "and evals/stats_accuracy.py's module docstring."
        )
        return result

    markers = [f"stats-accuracy-marker-{i}-{uuid4().hex[:8]}" for i in range(n_records)]
    stored_markers: list[str] = []
    for index, marker in enumerate(markers):
        try:
            adapter.store(session_id, f"Fact #{index}: unique marker {marker}.")
            stored_markers.append(marker)
            result.records_stored += 1
        except BackendAPIError as exc:
            if result.error is None:
                result.error = f"store failed for marker {marker}: {exc}"

    if not stored_markers:
        result.signal = StatsSignal.NOT_APPLICABLE
        return result

    result.verified_count = _verify_stored_count(adapter, session_id, stored_markers, top_k)

    try:
        stats = adapter.get_stats(session_id)
    except BackendAPIError as exc:
        result.error = f"get_stats failed: {exc}"
        result.signal = StatsSignal.NOT_APPLICABLE
        return result

    result.reported_count = stats.total_memories
    result.signal = classify_stats_accuracy(result.verified_count, result.reported_count)
    return result
