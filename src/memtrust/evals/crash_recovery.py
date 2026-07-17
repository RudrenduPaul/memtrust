"""MemTrust's process-lifecycle/crash-recovery eval.

None of the other evals in this package can see what happens to a backend
across a server-process crash and restart -- they all exercise a single,
continuously-running adapter instance. This eval closes that specific gap.

Motivating case: volcengine/OpenViking#2644 (contributor yeyitech). A local
vectordb backend's `_recover()` routine, run on server-process startup,
silently skips rebuilding the search index when the on-disk index files
are missing but the underlying store data is still present -- e.g. the
process crashed mid-write, or the index files were deleted/corrupted
independently of the store. No exception is raised anywhere: the process
starts up cleanly, accepts queries normally, and simply returns nothing
for data that is still sitting in the store, unindexed. A caller has no
way to distinguish this from "the data was genuinely never stored" short
of inspecting the store directly.

**Honest scope of what this eval can and cannot prove.** memtrust's
adapters are pure HTTP clients (see adapters/base.py's module docstring)
with zero ability to start, kill, or restart a real vendor server
process -- there is no live OpenViking binary this harness manages, and
building real subprocess lifecycle control was out of scope for the
environment this eval was built in. So this eval does not, and cannot,
reproduce #2644 against a live OpenViking instance. Instead it targets
the STRUCTURAL failure shape directly: it stores data via an adapter,
calls an explicit `adapter.simulate_crash_restart()` (a named simulation
primitive, never a real process kill -- see
MemoryBackendAdapter.supports_crash_recovery_simulation), queries the
same data afterward, and independently checks whether the underlying
store still holds it via `adapter.raw_store_contains()` (which bypasses
the search index query() goes through). Only a purpose-built in-memory
fake adapter can genuinely model both halves of this today -- see
tests/test_evals.py::CrashRecoveryFakeAdapter. Adapters without this
capability (every real adapter in this repo, including
OpenVikingAdapter) report NOT_APPLICABLE / skipped, never a guessed
result. See docs/methodology.md for the full write-up of what this does
and does not close.

Classification produces one of:

  * RECOVERED                 -- post-restart query() still returns the
                                  record. The index survived (or was
                                  correctly rebuilt).
  * INDEX_LOST_DATA_SURVIVED  -- post-restart query() returns nothing for
                                  this record, but raw_store_contains()
                                  confirms the underlying store still has
                                  it. This is the exact #2644 shape: data
                                  intact, index lost, queries silently
                                  return nothing. Distinct from
                                  ConflictSignal.EMPTY_OR_LOST (adapters/
                                  base.py), which is about a single query
                                  returning nothing with no crash/restart
                                  context at all -- this signal exists
                                  specifically for the post-recovery
                                  symptom, where the eval has independent
                                  evidence the data itself was never lost.
  * DATA_LOST                 -- post-restart query() returns nothing AND
                                  raw_store_contains() confirms the data
                                  itself is gone too. A different, more
                                  severe failure than #2644's shape (this
                                  is real data loss, not an unrebuilt
                                  index), kept as its own signal rather
                                  than folded into INDEX_LOST_DATA_SURVIVED.
  * NOT_APPLICABLE             -- either the adapter has no crash-recovery
                                  simulation capability at all
                                  (supports_crash_recovery_simulation is
                                  False -- the eval is skipped, not run),
                                  or the record was never confirmed
                                  queryable before the simulated crash in
                                  the first place, so there is nothing
                                  meaningful to classify about recovery.

Design principle (same as evals/resource_sync_safety.py's
classify_resource_sync_file and evals/ranking_quality.py's
classify_ranking_case): classification never blindly trusts a single
query() response. It cross-checks query() against an independent
raw_store_contains() observation, the same way resource_sync_safety.py
cross-checks list_resource_paths() (existence) against query()
(searchability) to tell "never indexed" apart from "deleted".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "crash_recovery_cases.json"
)


class CrashRecoverySignal(StrEnum):
    """How one stored record fared across a simulated crash + restart."""

    RECOVERED = "recovered"
    """Present and searchable via query() both before and after the
    simulated crash/restart -- the index survived or was correctly
    rebuilt on startup."""

    INDEX_LOST_DATA_SURVIVED = "index_lost_data_survived"
    """query() returns nothing for this record after the simulated crash/
    restart, but raw_store_contains() independently confirms the
    underlying store still has the data. The exact volcengine/
    OpenViking#2644 shape: `_recover()` skipped rebuilding the index
    while store data survived, so queries silently return nothing for
    data that was never actually lost."""

    DATA_LOST = "data_lost"
    """query() returns nothing after the simulated crash/restart AND
    raw_store_contains() confirms the data itself is also gone -- a
    different, more severe failure than #2644's shape (real data loss,
    not merely an unrebuilt index)."""

    NOT_APPLICABLE = "not_applicable"
    """Either the adapter has no crash-recovery-simulation capability
    (MemoryBackendAdapter.supports_crash_recovery_simulation is False --
    the eval is skipped entirely, not run per-case), or the record was
    never confirmed present/queryable before the simulated crash, so
    there is nothing meaningful to classify about recovery."""


@dataclass
class CrashRecoveryCase:
    case_id: str
    session_id: str
    content: str


@dataclass
class CrashRecoveryCaseResult:
    case: CrashRecoveryCase
    signal: CrashRecoverySignal
    present_before_crash: bool
    queryable_after_crash: bool
    raw_store_contains_after_crash: bool | None
    """None when the adapter has no crash-recovery-simulation capability
    (raw_store_contains() was never called) or when the eval could not
    reach the point of calling it (e.g. a BackendAPIError before the
    simulated crash). True/False otherwise."""
    error: str | None = None


@dataclass
class CrashRecoveryEvalResult:
    backend_name: str
    dataset_path: str
    case_results: list[CrashRecoveryCaseResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def scored_cases(self) -> list[CrashRecoveryCaseResult]:
        return [c for c in self.case_results if c.error is None]

    def _fraction(self, signal: CrashRecoverySignal) -> float | None:
        scored = self.scored_cases
        if not scored:
            return None
        matching = sum(1 for c in scored if c.signal == signal)
        return matching / len(scored)

    @property
    def recovered_rate(self) -> float | None:
        return self._fraction(CrashRecoverySignal.RECOVERED)

    @property
    def index_lost_data_survived_rate(self) -> float | None:
        """Fraction of cases where the simulated crash/restart lost the
        index while the underlying data survived -- the headline metric
        this eval exists to surface, and the exact volcengine/
        OpenViking#2644 shape."""
        return self._fraction(CrashRecoverySignal.INDEX_LOST_DATA_SURVIVED)

    @property
    def data_lost_rate(self) -> float | None:
        return self._fraction(CrashRecoverySignal.DATA_LOST)


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[CrashRecoveryCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        CrashRecoveryCase(
            case_id=c["case_id"],
            session_id=c["session_id"],
            content=c["content"],
        )
        for c in cases
    ]


def classify_crash_recovery_case(
    present_before_crash: bool,
    queryable_after_crash: bool,
    raw_store_contains_after_crash: bool | None,
) -> CrashRecoverySignal:
    """Classify a single case's outcome from its before/after observations.

    Never trusts query() alone to mean "the data is gone" -- that is
    exactly the ambiguity volcengine/OpenViking#2644 exploits (a lost
    index and lost data both make query() return nothing). Only
    raw_store_contains(), an independent observation that bypasses the
    search index, can tell the two apart.
    """
    if not present_before_crash:
        # Never confirmed present/queryable before the simulated crash --
        # nothing meaningful to say about "recovery" for this case.
        return CrashRecoverySignal.NOT_APPLICABLE
    if queryable_after_crash:
        return CrashRecoverySignal.RECOVERED
    if raw_store_contains_after_crash is True:
        return CrashRecoverySignal.INDEX_LOST_DATA_SURVIVED
    if raw_store_contains_after_crash is False:
        return CrashRecoverySignal.DATA_LOST
    # raw_store_contains_after_crash is None: the eval never got an
    # independent read on the underlying store (e.g. it errored out) --
    # not enough evidence to call this either INDEX_LOST_DATA_SURVIVED or
    # DATA_LOST.
    return CrashRecoverySignal.NOT_APPLICABLE


def run_crash_recovery_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> CrashRecoveryEvalResult:
    cases = load_dataset(dataset_path)
    result = CrashRecoveryEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    if not adapter.supports_crash_recovery_simulation:
        result.skipped = True
        result.skip_reason = (
            f"{adapter.name} does not support crash-recovery simulation "
            "(supports_crash_recovery_simulation=False) -- skipped, not run. "
            "No adapter in this repo has real process-lifecycle control over "
            "a live backend server; see evals/crash_recovery.py's module "
            "docstring and docs/methodology.md."
        )
        return result

    for case in cases:
        try:
            store_result = adapter.store(case.session_id, case.content)
            pre_crash_query = adapter.query(case.session_id, case.content, top_k=5)
            present_before_crash = any(
                case.content.lower() in r.content.lower() for r in pre_crash_query.records
            )

            adapter.simulate_crash_restart()

            post_crash_query = adapter.query(case.session_id, case.content, top_k=5)
            queryable_after_crash = any(
                case.content.lower() in r.content.lower() for r in post_crash_query.records
            )
            raw_store_contains_after_crash: bool | None = None
            if not queryable_after_crash:
                raw_store_contains_after_crash = adapter.raw_store_contains(
                    case.session_id, store_result.memory_id
                )
        except BackendAPIError as exc:
            result.case_results.append(
                CrashRecoveryCaseResult(
                    case=case,
                    signal=CrashRecoverySignal.NOT_APPLICABLE,
                    present_before_crash=False,
                    queryable_after_crash=False,
                    raw_store_contains_after_crash=None,
                    error=str(exc),
                )
            )
            continue

        signal = classify_crash_recovery_case(
            present_before_crash, queryable_after_crash, raw_store_contains_after_crash
        )
        result.case_results.append(
            CrashRecoveryCaseResult(
                case=case,
                signal=signal,
                present_before_crash=present_before_crash,
                queryable_after_crash=queryable_after_crash,
                raw_store_contains_after_crash=raw_store_contains_after_crash,
            )
        )

    return result
