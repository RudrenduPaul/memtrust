"""MemTrust's scale/volume stress-testing eval.

Every other eval in this package (contradiction, ranking_quality,
resource_sync_safety, compression) runs against a bundled fixture of 4-7
hand-written cases. That is sufficient to exercise a backend's correctness
logic, but it structurally cannot exercise the class of bug that only shows
up once a corpus grows large -- two real, documented, still-open vendor
reports are exactly this shape:

  * volcengine/OpenViking#2850 (lg320531124): BM25 search silently returns
    empty results once a corpus grows large enough. README.md and
    docs/methodology.md have both already noted, honestly, that memtrust's
    EMPTY_OR_LOST/NOT_APPLICABLE signals can distinguish "empty response"
    from "no response," but that nothing in this repo could actually
    *reproduce the scale condition* that triggers it -- every fixture
    tops out at single-digit record counts.
  * getzep/graphiti#1275 (rafaelreis-r): O(n) entity-resolution context
    growth causes episodes to be silently dropped once ingestion passes
    roughly 300 episodes. Same shape of gap: nothing in this repo ever
    ingests more than a handful of episodes.

This eval closes that gap. It uses `evals/scale_fixtures.py`'s
`generate_scale_corpus()` to build a corpus of N synthetic records (N
configurable, default small enough to run fast in CI, architected to scale
to 10K+ against a real backend), stores them incrementally, and at a series
of checkpoints along the way re-queries a sample of already-stored records
by their unique marker token to see whether they are still recoverable.

The specific thing this is built to detect is a backend that works fine at
N=5 (every bundled fixture's scale) and silently breaks somewhere between
N=50 and N=1000+, with no exception raised anywhere -- exactly what both
#2850 and #1275 describe: a call that completes normally and returns
nothing (or returns something, but not the thing that was asked for),
which is indistinguishable from "the record was never there" unless a
harness specifically tracks recall *as a function of corpus size*.

Two distinct signals are tracked at each checkpoint, because #2850 and
#1275 are not quite the same failure shape:

  * A fixed **anchor** record (the very first one ever stored) is
    re-queried at every checkpoint. If it was recoverable at N=5 and
    becomes unrecoverable once N grows, that is the #1275 shape: older
    content silently evicted/dropped as volume grows, while recent
    content is still fine.
  * A **sample** of records spread across everything stored so far is
    re-queried at every checkpoint. If the sample's overall recall rate
    holds steady at small N and collapses at large N, that is the #2850
    shape: search itself degrades (or goes silently empty) as the corpus
    grows, independent of which specific record is being asked for.

Design principle (same as every other eval in this package): classification
never trusts "the call didn't raise" as proof anything worked. A record is
only counted as recoverable if its unique marker token is actually present
in the joined text of the query response -- an empty response, a response
containing unrelated records, or a response missing just this one marker
are all scored identically as "not recoverable," the same "silent
empty-success is not a pass" rule evals/contradiction.py's EMPTY_OR_LOST
signal establishes.

**Honest limitation, stated plainly (see docs/methodology.md for the full
write-up).** This is a NEW capability as of this change. It has not been run
against any live backend at real scale (10K+) -- doing so requires live
vendor credentials and a real run, neither of which happened during this
build. What this file's own test suite (tests/test_scale_stress.py) proves
is narrower and just as real: the eval's classification logic correctly
distinguishes a fake adapter engineered to degrade at volume from one that
scales cleanly. It does not, and cannot, prove that OpenViking's or
Graphiti's real production systems currently exhibit #2850 or #1275 --
only that memtrust's harness is now structurally capable of detecting that
shape of bug *if* a live backend exhibits it, the same honest boundary
already drawn around resource_sync_safety.py's NESTED_CONTENT_UNINDEXED
signal and ranking_quality.py's MISSING_ORDERING_KEY signal.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import StrEnum

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter
from memtrust.evals.scale_fixtures import ScaleFixtureRecord, generate_scale_corpus

#: Fast-by-default record count. Large enough to move well past every
#: bundled fixture's single-digit scale and actually exercise volume
#: behavior, small enough to run in CI in a few seconds against a fake
#: in-memory adapter. Passing a larger n_records (the harness itself has
#: no upper bound besides scale_fixtures.generate_scale_corpus's 999,999
#: cap) is how this same code path reaches the 10K+ regime the real vendor
#: bugs manifest at -- see this module's docstring.
DEFAULT_N_RECORDS = 500

#: Recall drop (in percentage points) between the first and last scoreable
#: checkpoint, at or above which a run is classified SILENTLY_DEGRADED_AT_
#: SCALE rather than merely PARTIAL_DEGRADATION. Chosen well above normal
#: run-to-run noise for a fake/real adapter that is genuinely working (a
#: healthy backend's recall at N=5 and at N=500 should be nearly
#: identical, not just "close").
DEGRADATION_THRESHOLD_PP = 15.0

#: Minimum acceptable recall rate at the largest checkpoint for a run to be
#: classified WORKED_AT_SCALE outright, even when the pp-drop threshold
#: above isn't crossed (e.g. a backend that was already imperfect at N=5
#: and stays exactly as imperfect at N=500 has zero degradation, but is
#: still not "working").
MIN_ACCEPTABLE_FINAL_RECALL = 0.9

#: How many non-anchor records to sample per checkpoint for the general
#: recall signal (see this module's docstring). Fixed rather than
#: proportional to checkpoint size so the query cost of a checkpoint stays
#: bounded even at N=10,000.
SAMPLE_SIZE_PER_CHECKPOINT = 5


class ScaleSignal(StrEnum):
    """How a backend's recall behaved as the corpus it was queried against
    grew from a handful of records to the full requested N.

    Defined locally in this module rather than in adapters/base.py,
    following the same precedent evals/resource_sync_safety.py's
    ResourceSyncSignal already sets: this is a harness-computed
    classification derived from ground truth (which records were actually
    stored, and in what order), not a signal any adapter self-reports.
    """

    WORKED_AT_SCALE = "worked_at_scale"
    """Recall stayed high (>= MIN_ACCEPTABLE_FINAL_RECALL) at the largest
    checkpoint and did not drop by more than DEGRADATION_THRESHOLD_PP
    between the smallest and largest scoreable checkpoint, and the anchor
    (first-ever-stored) record was still recoverable at every checkpoint.
    No evidence of scale-dependent degradation was observed."""

    SILENTLY_DEGRADED_AT_SCALE = "silently_degraded_at_scale"
    """Either the anchor record -- recoverable at a small checkpoint --
    became unrecoverable at a larger one (the getzep/graphiti#1275 shape:
    old content silently dropped as volume grows), or the general sample's
    recall rate fell by at least DEGRADATION_THRESHOLD_PP percentage
    points between the smallest and largest checkpoint (the
    volcengine/OpenViking#2850 shape: search itself degrades at volume).
    Every store()/query() call involved completed without raising
    BackendAPIError -- this is a silent failure, not a crash, which is
    exactly the failure mode both cited issues describe and exactly why a
    harness needs to track recall as a function of scale to see it at
    all."""

    PARTIAL_DEGRADATION = "partial_degradation"
    """Final-checkpoint recall fell short of MIN_ACCEPTABLE_FINAL_RECALL,
    but not because of a scale-correlated drop large enough to meet
    SILENTLY_DEGRADED_AT_SCALE's threshold -- e.g. a backend that already
    missed some records at the smallest checkpoint and stayed equally
    imperfect at the largest one. Worth flagging (recall is genuinely
    incomplete), but distinct from a volume-triggered collapse: this
    could just as easily be an ordinary indexing miss unrelated to
    corpus size."""

    ERROR = "error"
    """A BackendAPIError was raised during the run that this eval could
    not route around (see ScaleTestResult.error for detail). Distinct
    from SILENTLY_DEGRADED_AT_SCALE -- an explicit error is a different,
    more honest failure mode than a silent one, and conflating the two
    would credit a backend that at least raised an exception with the
    same verdict as one that returned an empty success."""

    NOT_APPLICABLE = "not_applicable"
    """Fewer than 2 checkpoints produced a scoreable recall rate (e.g.
    n_records too small to generate more than one checkpoint, or every
    checkpoint's queries all errored) -- there is nothing to compare
    "small scale" against "large scale" with. Recorded explicitly, never
    silently dropped, matching every other *Signal enum's NOT_APPLICABLE
    convention in this package."""


@dataclass
class NeedleQueryResult:
    """The outcome of re-querying for one specific previously-stored
    record by its unique marker token."""

    index: int
    marker: str
    found: bool
    latency_ms: float | None
    error: str | None = None


@dataclass
class ScaleCheckpointResult:
    """A snapshot of recall taken after `checkpoint_n` records had been
    attempted (stored or store-failed) in the corpus."""

    checkpoint_n: int
    records_stored_so_far: int
    """How many of the first `checkpoint_n` records in generation order
    actually succeeded their store() call (<= checkpoint_n; less than
    checkpoint_n only if some stores raised BackendAPIError)."""
    needle_queries: list[NeedleQueryResult] = field(default_factory=list)
    anchor_recall: bool | None = None
    """Whether the very first record ever stored (index 0) was still
    recoverable at this checkpoint. `None` if index 0 itself failed to
    store (so there is nothing to check recall of)."""
    recall_rate: float | None = None
    """Fraction of `needle_queries` that were found, across both the
    anchor and the general sample. `None` if no needle queries were
    attempted at this checkpoint (should not happen once checkpoint_n
    >= 1 and at least one record stored, but guarded rather than
    assumed)."""
    latency_p50_ms: float | None = None
    latency_p99_ms: float | None = None


@dataclass
class ScaleTestResult:
    """Result of `run_scale_stress_eval()` -- distinguishes "worked at
    scale" from "silently degraded at scale" (see ScaleSignal) for one
    backend, one (n_records, seed) corpus."""

    backend_name: str
    n_records_requested: int
    seed: int
    checkpoints: list[ScaleCheckpointResult] = field(default_factory=list)
    records_stored: int = 0
    """Total successful store() calls across the entire run (not just at
    checkpoints)."""
    records_store_errors: int = 0
    records_checked: int = 0
    """Total needle queries attempted across every checkpoint."""
    records_recoverable: int = 0
    """Total needle queries, across every checkpoint, that found their
    target record."""
    recall_degradation_pct: float | None = None
    """(first scoreable checkpoint's recall_rate - last scoreable
    checkpoint's recall_rate) * 100, in percentage points. Positive means
    recall got worse as the corpus grew; `None` if fewer than 2
    checkpoints produced a scoreable recall_rate."""
    anchor_lost_at_n: int | None = None
    """The smallest checkpoint_n at which the anchor record (index 0)
    stopped being recoverable, having been recoverable at an earlier
    checkpoint. `None` if the anchor was recoverable at every checkpoint
    it was checked at, or was never successfully stored in the first
    place."""
    latency_p99_ms: float | None = None
    """p99 latency across every needle query issued during the entire
    run (not per-checkpoint -- see ScaleCheckpointResult.latency_p99_ms
    for the per-checkpoint breakdown)."""
    signal: ScaleSignal = ScaleSignal.NOT_APPLICABLE
    error: str | None = None


def _default_checkpoints(n: int) -> list[int]:
    """Pick a small, ascending set of checkpoint sizes spanning from "the
    same scale every other bundled fixture runs at" up to the full
    requested N, so a degradation between "small" and "large" has
    somewhere concrete to show up.

    For a typical n_records=500 this yields [5, 50, 250, 500] -- the first
    checkpoint deliberately mirrors the ~5-example scale every other eval's
    fixture already runs at, so a comparison against this eval's own
    smallest checkpoint is an apples-to-apples "does this backend still
    work at the scale every other eval already proved it works at."
    """
    candidates = [5, max(1, n // 10), max(1, n // 2), n]
    return sorted({c for c in candidates if 1 <= c <= n})


def _percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile, pure stdlib (no numpy dependency
    anywhere else in this repo -- see pyproject.toml). `p` in [0, 100].
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    lower_weight = ordered[lower] * (upper - rank)
    upper_weight = ordered[upper] * (rank - lower)
    return lower_weight + upper_weight


def _query_needle(
    adapter: MemoryBackendAdapter,
    record: ScaleFixtureRecord,
    session_id: str,
    top_k: int,
) -> NeedleQueryResult:
    """Re-query for one specific previously-stored record by its unique
    marker token, and check whether the marker is actually present in the
    returned text -- never trusts "query() didn't raise" as proof the
    record came back, same rule every other eval in this package applies
    to its own adapter calls.
    """
    try:
        query_result = adapter.query(session_id, record.marker, top_k=top_k)
    except BackendAPIError as exc:
        return NeedleQueryResult(
            index=record.index, marker=record.marker, found=False, latency_ms=None, error=str(exc)
        )
    content = " ".join(r.content for r in query_result.records)
    found = record.marker.lower() in content.lower()
    return NeedleQueryResult(
        index=record.index,
        marker=record.marker,
        found=found,
        latency_ms=query_result.latency_ms,
    )


def _run_checkpoint_queries(
    adapter: MemoryBackendAdapter,
    corpus: list[ScaleFixtureRecord],
    stored_indices: set[int],
    checkpoint_n: int,
    session_id: str,
    rng: random.Random,
    top_k: int,
) -> ScaleCheckpointResult:
    available = [i for i in range(checkpoint_n) if i in stored_indices]
    needle_results: list[NeedleQueryResult] = []
    anchor_recall: bool | None = None

    if 0 in stored_indices:
        anchor_result = _query_needle(adapter, corpus[0], session_id, top_k)
        needle_results.append(anchor_result)
        anchor_recall = anchor_result.found

    sample_pool = [i for i in available if i != 0]
    sample_size = min(SAMPLE_SIZE_PER_CHECKPOINT, len(sample_pool))
    sample_indices = sorted(rng.sample(sample_pool, sample_size)) if sample_size else []
    for idx in sample_indices:
        needle_results.append(_query_needle(adapter, corpus[idx], session_id, top_k))

    found_count = sum(1 for r in needle_results if r.found)
    recall_rate = found_count / len(needle_results) if needle_results else None
    latencies = [r.latency_ms for r in needle_results if r.latency_ms is not None]

    return ScaleCheckpointResult(
        checkpoint_n=checkpoint_n,
        records_stored_so_far=len(available),
        needle_queries=needle_results,
        anchor_recall=anchor_recall,
        recall_rate=recall_rate,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p99_ms=_percentile(latencies, 99),
    )


def classify_scale_result(
    checkpoint_results: list[ScaleCheckpointResult],
    recall_degradation_pct: float | None,
    anchor_lost_at_n: int | None,
) -> ScaleSignal:
    """Classify a completed run's checkpoints into a single ScaleSignal.

    Never a blind pass/fail on "did every store() call succeed" -- the
    classification is driven entirely by recomputed recall at each
    checkpoint (see ScaleCheckpointResult.recall_rate), the same
    ground-truth-driven pattern every other eval's classify_* function in
    this package follows (evals/contradiction.py's classify_case,
    evals/ranking_quality.py's classify_ranking_case,
    evals/resource_sync_safety.py's classify_resource_sync_file).
    """
    scoreable = [c for c in checkpoint_results if c.recall_rate is not None]
    if len(scoreable) < 2:
        return ScaleSignal.NOT_APPLICABLE

    final_recall = scoreable[-1].recall_rate
    if anchor_lost_at_n is not None:
        return ScaleSignal.SILENTLY_DEGRADED_AT_SCALE
    if recall_degradation_pct is not None and recall_degradation_pct >= DEGRADATION_THRESHOLD_PP:
        return ScaleSignal.SILENTLY_DEGRADED_AT_SCALE
    if final_recall is not None and final_recall < MIN_ACCEPTABLE_FINAL_RECALL:
        return ScaleSignal.PARTIAL_DEGRADATION
    return ScaleSignal.WORKED_AT_SCALE


def run_scale_stress_eval(
    adapter: MemoryBackendAdapter,
    n_records: int = DEFAULT_N_RECORDS,
    seed: int = 42,
    session_id: str = "scale-stress-session",
    checkpoints: list[int] | None = None,
    top_k: int = 10,
) -> ScaleTestResult:
    """Store `n_records` synthetic records into `adapter` incrementally,
    and at each checkpoint re-query a sample of already-stored records
    (plus the fixed first-ever-stored "anchor" record) by their unique
    marker token to measure recall as a function of corpus size.

    Args:
        adapter: the backend under test.
        n_records: how many records to generate and store. Default
            (DEFAULT_N_RECORDS=500) is deliberately fast-in-CI, not the
            10K+ scale the motivating vendor bugs manifest at -- pass a
            larger value to actually reach that regime against a live,
            configured backend. The harness itself places no additional
            cap beyond scale_fixtures.generate_scale_corpus's 999,999.
        seed: forwarded to generate_scale_corpus() for a reproducible
            corpus, and used to seed this function's own sampling RNG
            (kept separate from any RNG the adapter itself might use).
        session_id: session/scope every record is stored under.
        checkpoints: explicit checkpoint sizes to snapshot recall at.
            Defaults to `_default_checkpoints(n_records)` (deliberately
            includes a checkpoint at the same ~5-record scale every other
            bundled fixture already runs at, so this eval's own smallest
            checkpoint is directly comparable to "does this backend work
            at all," and a checkpoint at the full n_records).
        top_k: top_k passed to every needle query() call.

    Raises:
        ValueError: if n_records < 1 (mirrors
            scale_fixtures.generate_scale_corpus's own validation).
    """
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}")

    resolved_checkpoints = (
        sorted({c for c in checkpoints if 1 <= c <= n_records})
        if checkpoints is not None
        else _default_checkpoints(n_records)
    )

    corpus = generate_scale_corpus(n_records, seed=seed, session_id=session_id)
    rng = random.Random(seed)

    result = ScaleTestResult(backend_name=adapter.name, n_records_requested=n_records, seed=seed)
    stored_indices: set[int] = set()
    next_checkpoint_idx = 0
    all_latencies: list[float] = []
    anchor_ever_recovered = False

    for record in corpus:
        try:
            adapter.store(
                session_id,
                record.content,
                metadata={"scale_index": str(record.index)},
            )
            stored_indices.add(record.index)
            result.records_stored += 1
        except BackendAPIError as exc:
            result.records_store_errors += 1
            if result.error is None:
                result.error = f"first store failure at index={record.index}: {exc}"

        attempted_so_far = record.index + 1
        while (
            next_checkpoint_idx < len(resolved_checkpoints)
            and resolved_checkpoints[next_checkpoint_idx] == attempted_so_far
        ):
            checkpoint_n = resolved_checkpoints[next_checkpoint_idx]
            checkpoint_result = _run_checkpoint_queries(
                adapter, corpus, stored_indices, checkpoint_n, session_id, rng, top_k
            )
            result.checkpoints.append(checkpoint_result)

            result.records_checked += len(checkpoint_result.needle_queries)
            result.records_recoverable += sum(
                1 for q in checkpoint_result.needle_queries if q.found
            )
            all_latencies.extend(
                q.latency_ms for q in checkpoint_result.needle_queries if q.latency_ms is not None
            )

            if checkpoint_result.anchor_recall is True:
                anchor_ever_recovered = True
            elif (
                checkpoint_result.anchor_recall is False
                and anchor_ever_recovered
                and result.anchor_lost_at_n is None
            ):
                result.anchor_lost_at_n = checkpoint_n

            next_checkpoint_idx += 1

    scoreable = [c for c in result.checkpoints if c.recall_rate is not None]
    if len(scoreable) >= 2:
        first_rate = scoreable[0].recall_rate
        last_rate = scoreable[-1].recall_rate
        if first_rate is not None and last_rate is not None:
            result.recall_degradation_pct = (first_rate - last_rate) * 100

    result.latency_p99_ms = _percentile(all_latencies, 99)
    result.signal = classify_scale_result(
        result.checkpoints, result.recall_degradation_pct, result.anchor_lost_at_n
    )
    if (
        result.signal == ScaleSignal.NOT_APPLICABLE
        and result.records_stored == 0
        and result.records_store_errors > 0
    ):
        # Every single store() call failed -- there is nothing to compute
        # recall over, but "nothing to compute" (a genuinely empty corpus,
        # or a corpus too small to have >=2 checkpoints) and "every write
        # to the backend errored" are different findings. The latter is a
        # concrete, explicit failure and deserves its own signal rather
        # than being folded into the generic "not enough data" bucket.
        result.signal = ScaleSignal.ERROR
    return result
