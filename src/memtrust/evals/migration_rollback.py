"""MemTrust's migration-rollback-safety eval.

None of the other evals in this package can see what happens to a
backend's own data across a storage-format/version MIGRATION -- they all
exercise a single, already-migrated adapter instance talking to a single,
stable storage layout. This eval closes that specific, different gap.

Motivating case: MemPalace/mempalace#1028 (GitHub user eldar702). MemPalace
ships its own `migrate.migrate()` function that, at the end of a migration,
swaps a newly-written palace directory into place over the old one. The
reported version of that swap was unguarded: `shutil.rmtree()` deleted the
old backup FIRST, then `shutil.move()` moved the new data into place. If
the `move()` step failed partway through -- e.g. a cross-device `EXDEV`
error, which `shutil.move()` can raise when the source and destination
straddle a filesystem boundary -- the palace directory could be
permanently lost: the old backup was already gone, and the new data never
finished landing either. MemPalace/mempalace#935 is the real upstream fix:
a "rename-aside" swap that renames the new data into place first, keeps
the old backup renamed-aside (not deleted), and only deletes the backup
after independently confirming the swap succeeded. This eval exists to
verify the CONCEPT that fix pattern establishes -- rename-aside preserves
recoverability across a failed swap, unguarded rmtree-then-move does not
-- not to reproduce PR #935's specific merged diff line-for-line.

**Honest scope of what this eval can and cannot prove.** memtrust's
adapters have zero direct filesystem control over a live vendor package's
internal migration code path -- MemPalaceAdapter, the one adapter this
concept is scoped to (MemPalace is the only local-storage-path backend in
this repo; mem0/zep/openviking have no on-disk "migrate" concept a
migration-swap eval could exercise at all), only wraps whatever the
installed `mempalace` package's own remember()/recall()/invalidate()
methods do internally via `_get_palace()` -- see mempalace_adapter.py's
module docstring. There is no live MemPalace instance this harness runs a
real migration against, and no way to interrupt a real `shutil.move()`
call mid-flight from outside the vendor's own process. So this eval does
not, and cannot, reproduce #1028 against a live MemPalace instance.
Instead it targets the STRUCTURAL failure shape directly: it calls an
explicit `adapter.simulate_migration_failure(session_id, content)` (a
named simulation primitive, never a real filesystem fault injection -- see
MemoryBackendAdapter.supports_migration_rollback_simulation), which stores
`content` as the original pre-migration data and simulates the swap being
interrupted before its final commit step, then independently re-queries
for that same content afterward to see whether it survived. Only a
purpose-built in-memory fake adapter can genuinely model both the buggy
shape and the fixed shape today -- see
tests/test_evals.py::MigrationRollbackFakeAdapter and
MigrationRollbackRenameAsideFakeAdapter. MemPalaceAdapter itself does NOT
set supports_migration_rollback_simulation = True and reports
NOT_APPLICABLE / skipped, same as every real adapter in this repo reports
for crash_recovery.py's equivalent capability flag. See
docs/methodology.md for the full write-up of what this does and does not
close.

Classification produces one of:

  * RESTORED        -- an independent post-failure query() call (made by
                        this eval, not the adapter's own self-report)
                        still finds the original pre-migration content.
                        The safe outcome: MemPalace/mempalace#935's
                        rename-aside pattern, where the old backup is kept
                        renamed-aside and only deleted after the new data
                        is confirmed in place, so a failure partway
                        through the swap leaves the original data intact.
  * DATA_LOST        -- the same independent post-failure query() call
                        finds nothing. The exact MemPalace/mempalace#1028
                        shape: the old backup was already deleted before
                        the move step failed, so there is nothing left to
                        recover.
  * NOT_APPLICABLE   -- either the adapter has no migration-rollback
                        simulation capability at all
                        (supports_migration_rollback_simulation is False
                        -- the eval is skipped, not run), or a
                        BackendAPIError was raised before the post-failure
                        observation could be made for this case.

Design principle (same as evals/crash_recovery.py's
classify_crash_recovery_case and evals/ranking_quality.py's
classify_ranking_case): classification never blindly trusts a single
self-reported value. MigrationFailureResult.original_data_recoverable
(adapters/base.py) is the adapter's OWN observation and is kept in each
case result purely as a diagnostic/cross-check field -- the actual
RESTORED/DATA_LOST classification is always derived from this eval's own,
independent query() call made after simulate_migration_failure() returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "migration_rollback_cases.json"
)


class MigrationRollbackSignal(StrEnum):
    """How one piece of ORIGINAL pre-migration data fared across a
    simulated mid-migration failure.

    Defined locally in this module rather than in adapters/base.py,
    following the same precedent evals/crash_recovery.py's
    CrashRecoverySignal and evals/scale_stress.py's ScaleSignal already
    set: this is a harness-computed classification derived from ground
    truth (an independent post-failure query() observation), not a signal
    any adapter self-reports.
    """

    RESTORED = "restored"
    """An independent post-failure query() call (made by this eval, not
    the adapter's own self-report) still finds the original pre-migration
    content. The safe outcome -- MemPalace/mempalace#935's rename-aside
    swap pattern: the old backup is kept renamed-aside and only deleted
    after the new data is confirmed in place, so a failure partway
    through the swap (e.g. a cross-device EXDEV error on the move step)
    leaves the original data recoverable."""

    DATA_LOST = "data_lost"
    """The same independent post-failure query() call finds nothing. The
    exact MemPalace/mempalace#1028 shape (GitHub user eldar702): an
    unguarded shutil.rmtree()-then-shutil.move() swap deletes the old
    backup FIRST, so if the move step fails partway the palace directory
    is permanently lost -- there is no backup left to fall back to."""

    NOT_APPLICABLE = "not_applicable"
    """Either the adapter has no migration-rollback-simulation capability
    (MemoryBackendAdapter.supports_migration_rollback_simulation is False
    -- the eval is skipped entirely, not run per-case), or a
    BackendAPIError was raised before the post-failure observation could
    be made for this case."""


@dataclass
class MigrationRollbackCase:
    case_id: str
    session_id: str
    content: str


@dataclass
class MigrationRollbackCaseResult:
    case: MigrationRollbackCase
    signal: MigrationRollbackSignal
    original_data_recoverable: bool | None
    """Ground truth this eval's classification is actually based on: the
    result of this eval's own independent query() call, made after
    simulate_migration_failure() returns, for the original content. None
    only when a BackendAPIError was raised before this observation could
    be made."""
    adapter_reported_recoverable: bool | None = None
    """The adapter's own self-reported
    MigrationFailureResult.original_data_recoverable flag (adapters/
    base.py), kept for diagnostic/cross-check purposes only -- never used
    as the classification's ground truth. Same convention as
    evals/ranking_quality.py's adapter_reported_signal field: threaded
    through for transparency/comparison, not trusted outright. None when
    a BackendAPIError was raised before simulate_migration_failure() could
    return a result."""
    error: str | None = None


@dataclass
class MigrationRollbackEvalResult:
    backend_name: str
    dataset_path: str
    case_results: list[MigrationRollbackCaseResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def scored_cases(self) -> list[MigrationRollbackCaseResult]:
        return [c for c in self.case_results if c.error is None]

    def _fraction(self, signal: MigrationRollbackSignal) -> float | None:
        scored = self.scored_cases
        if not scored:
            return None
        matching = sum(1 for c in scored if c.signal == signal)
        return matching / len(scored)

    @property
    def restored_rate(self) -> float | None:
        return self._fraction(MigrationRollbackSignal.RESTORED)

    @property
    def data_lost_rate(self) -> float | None:
        """Fraction of cases where the simulated mid-migration failure
        permanently lost the original data -- the headline metric this
        eval exists to surface, and the exact MemPalace/mempalace#1028
        shape."""
        return self._fraction(MigrationRollbackSignal.DATA_LOST)


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[MigrationRollbackCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        MigrationRollbackCase(
            case_id=c["case_id"],
            session_id=c["session_id"],
            content=c["content"],
        )
        for c in cases
    ]


def classify_migration_rollback_case(
    original_data_recoverable_after_failure: bool,
) -> MigrationRollbackSignal:
    """Classify a single case's outcome from an independent post-failure
    observation.

    Takes only the eval's own independently-observed boolean (never the
    adapter's self-reported MigrationFailureResult.original_data_recoverable
    directly) -- see this module's docstring and
    run_migration_rollback_eval() below, which is the caller responsible
    for making that independent query() observation before calling this.
    """
    if original_data_recoverable_after_failure:
        return MigrationRollbackSignal.RESTORED
    return MigrationRollbackSignal.DATA_LOST


def run_migration_rollback_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> MigrationRollbackEvalResult:
    cases = load_dataset(dataset_path)
    result = MigrationRollbackEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    if not adapter.supports_migration_rollback_simulation:
        result.skipped = True
        result.skip_reason = (
            f"{adapter.name} does not support migration-rollback simulation "
            "(supports_migration_rollback_simulation=False) -- skipped, not run. "
            "No adapter in this repo has real filesystem control over a live "
            "backend's internal migration code path; see "
            "evals/migration_rollback.py's module docstring and "
            "docs/methodology.md."
        )
        return result

    for case in cases:
        try:
            failure_result = adapter.simulate_migration_failure(case.session_id, case.content)
            post_failure_query = adapter.query(case.session_id, case.content, top_k=5)
            original_data_recoverable = any(
                case.content.lower() in r.content.lower() for r in post_failure_query.records
            )
        except BackendAPIError as exc:
            result.case_results.append(
                MigrationRollbackCaseResult(
                    case=case,
                    signal=MigrationRollbackSignal.NOT_APPLICABLE,
                    original_data_recoverable=None,
                    adapter_reported_recoverable=None,
                    error=str(exc),
                )
            )
            continue

        signal = classify_migration_rollback_case(original_data_recoverable)
        result.case_results.append(
            MigrationRollbackCaseResult(
                case=case,
                signal=signal,
                original_data_recoverable=original_data_recoverable,
                adapter_reported_recoverable=failure_result.original_data_recoverable,
            )
        )

    return result
