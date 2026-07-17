"""memtrust's episode-level temporal-leak detection eval for self-hosted
graphiti-core's FalkorDB driver.

Motivating case: getzep/graphiti#1625 (contributor pcy06, open as of this
build). FalkorDB's Cypher query for
`EpisodeNodeOperations.retrieve_episodes(reference_time=...)` intends to
filter episodes with `e.valid_at <= $reference_time`, but FalkorDB can
return rows for which that same expression evaluates `False` when
projected as a result column -- a future-dated episode can leak into a
point-in-time query that is documented to return only past-or-present
episodes. pcy06's own filed reproduction (`gh issue view 1625 --repo
getzep/graphiti`) demonstrates this concretely: a `valid_at=2024-03-01`
episode returned for a `reference_time=2024-02-01` query, with no
exception raised -- the query simply returns rows outside the requested
temporal boundary.

This is a structurally distinct failure class from
`ConflictSignal.EDGE_INTEGRITY_VIOLATION`/`FLAGGED`'s existing
`invalid_at`-based temporal handling in `evals/contradiction.py`:
episodes (`EpisodicNode`) and edges (`EntityEdge`) are different
graphiti-core node types with independent temporal-integrity properties.
An edge's `invalid_at` marks bi-temporal invalidation of a *fact*;
an episode's `valid_at` marks when the *source document* was created, and
`retrieve_episodes()`'s point-in-time contract is about the latter, not
the former. Nothing in `contradiction.py`'s existing classification logic
observes episodes at all -- `ZepGraphitiSelfHostedAdapter.query()` calls
`Graphiti.search()`, which returns only edges. This eval calls the new
`ZepGraphitiSelfHostedAdapter.retrieve_episodes()` primitive instead,
which reaches graphiti-core's driver-level `EpisodeNodeOperations`
directly, bypassing `search()`/`query()` entirely -- see that adapter
module's docstring for the real, source-confirmed method signature.

Honest scope: this is detection, not resolution. The bug this eval
classifies lives entirely inside graphiti-core's FalkorDB driver's Cypher
query construction -- pcy06 already proposed the real upstream fix
(project the temporal comparison first, then filter on that boolean).
memtrust's role is surfacing whether a given self-hosted deployment's
`retrieve_episodes()` call actually exhibits the leak, never fixing
graphiti-core itself. Like every other eval in this package that has never
been run against a live backend (see
`zep_graphiti_selfhosted_adapter.py`'s own "What this adapter does NOT
prove" section), this eval's own test suite only proves the
*classification logic* is correct against a fake driver double that
reproduces the exact reported shape -- it has not been run against a live
FalkorDB instance in this environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from memtrust.adapters.base import BackendAPIError
from memtrust.adapters.zep_graphiti_selfhosted_adapter import ZepGraphitiSelfHostedAdapter


class EpisodeTemporalSignal(StrEnum):
    """WHETHER a `retrieve_episodes(reference_time=...)` call returned any
    episode whose `valid_at` is after `reference_time` -- distinct from
    `ConflictSignal`'s edge-level `invalid_at` handling in
    `evals/contradiction.py` (see module docstring above for why episodes
    and edges need independent temporal-integrity taxonomies). Defined
    locally in this eval module rather than added to `base.py`, following
    the precedent `evals/scale_stress.py`'s own local `ScaleSignal` enum
    already establishes for a capability that is not part of every
    adapter's shared `MemoryBackendAdapter` interface.
    """

    NO_LEAK = "no_leak"
    """Every returned episode's `valid_at` is less than or equal to
    `reference_time` -- the documented, correct point-in-time contract."""

    TEMPORAL_LEAK = "temporal_leak"
    """At least one returned episode's `valid_at` is strictly after
    `reference_time` -- the exact getzep/graphiti#1625 shape: a
    point-in-time query leaked a future-dated episode into its result set
    with no exception raised."""

    NOT_APPLICABLE = "not_applicable"
    """The `retrieve_episodes()` call itself failed (see
    `ZepGraphitiSelfHostedAdapter.retrieve_episodes()` -- raised as
    `BackendAPIError`, e.g. because this driver has no
    `episode_node_ops` surface at all), so there is nothing to classify.
    Recorded explicitly rather than silently treated as NO_LEAK, same
    "never let a failed call read as a passing result" convention every
    other eval in this package follows."""


@dataclass
class EpisodeTemporalLeakResult:
    backend_name: str
    reference_time: datetime
    signal: EpisodeTemporalSignal
    leaked_episode_names: list[str] = field(default_factory=list)
    """Name (falling back to uuid, then the literal string "unknown") of
    every episode this eval found with `valid_at > reference_time`. Empty
    for NO_LEAK/NOT_APPLICABLE."""
    total_episodes_returned: int = 0
    error: str | None = None


def classify_episode_temporal_leak(
    episodes: list[dict[str, Any]], reference_time: datetime
) -> tuple[EpisodeTemporalSignal, list[str]]:
    """Classify a batch of episode dicts (the shape
    `ZepGraphitiSelfHostedAdapter.retrieve_episodes()` returns -- each a
    plain dict via `_to_plain_dict()`, carrying at least `valid_at` and
    `name`/`uuid`) against `reference_time`.

    Returns (signal, leaked_episode_names). An episode is only ever
    counted as leaked if its `valid_at` is present and strictly after
    `reference_time` -- a missing or unparseable `valid_at` is skipped
    (not counted either way), since this eval's job is detecting a
    confirmed leak, not penalizing a driver for reporting a field this
    adapter doesn't recognize the shape of.

    `valid_at` may arrive as a real `datetime` (test doubles / a
    `model_dump()`-free double) or as an ISO-8601 string (the shape
    `_to_plain_dict()`'s `model_dump()` path produces for a real,
    installed `EpisodicNode` -- pydantic serializes `datetime` fields to
    ISO strings by default). Both are handled; a naive-vs-aware mismatch
    when comparing against `reference_time` is deliberately NOT
    normalized here -- see `run_episode_temporal_leak_eval()`'s docstring
    for why callers must pass a timezone-aware `reference_time` when
    comparing against a real graphiti-core deployment.
    """
    leaked: list[str] = []
    for episode in episodes:
        valid_at = episode.get("valid_at")
        if isinstance(valid_at, str):
            try:
                valid_at = datetime.fromisoformat(valid_at)
            except ValueError:
                continue
        if not isinstance(valid_at, datetime):
            continue
        if valid_at > reference_time:
            name = episode.get("name") or episode.get("uuid") or "unknown"
            leaked.append(str(name))
    signal = EpisodeTemporalSignal.TEMPORAL_LEAK if leaked else EpisodeTemporalSignal.NO_LEAK
    return signal, leaked


def run_episode_temporal_leak_eval(
    adapter: ZepGraphitiSelfHostedAdapter,
    reference_time: datetime,
    group_ids: list[str] | None = None,
    last_n: int = 10,
) -> EpisodeTemporalLeakResult:
    """Call `adapter.retrieve_episodes(reference_time, group_ids, last_n)`
    and classify the result via `classify_episode_temporal_leak()` above.

    `reference_time` should be timezone-aware when run against a real
    graphiti-core deployment -- confirmed via
    `zep_graphiti_selfhosted_adapter.py`'s own module docstring and
    `store()`'s `datetime.now(UTC)` convention, real `EpisodicNode.valid_at`
    values are timezone-aware. A naive `reference_time` compared against an
    aware `valid_at` raises `TypeError` from Python's own datetime
    comparison, which this function does not catch -- the same "let a
    genuine caller error surface, don't silently swallow it" principle
    `verify_store()` documents elsewhere in this package.
    """
    try:
        episodes = adapter.retrieve_episodes(reference_time, group_ids=group_ids, last_n=last_n)
    except BackendAPIError as exc:
        return EpisodeTemporalLeakResult(
            backend_name=adapter.name,
            reference_time=reference_time,
            signal=EpisodeTemporalSignal.NOT_APPLICABLE,
            error=str(exc),
        )

    signal, leaked = classify_episode_temporal_leak(episodes, reference_time)
    return EpisodeTemporalLeakResult(
        backend_name=adapter.name,
        reference_time=reference_time,
        signal=signal,
        leaked_episode_names=leaked,
        total_episodes_returned=len(episodes),
    )
