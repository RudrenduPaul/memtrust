"""MemTrust's prefix-delete/orphan-cleanup eval.

`evals/resource_sync_safety.py` classifies files a resync operation
wrongly DELETES. This eval classifies the opposite polarity: vector-index
entries a prefix delete wrongly KEEPS -- a distinct failure mode neither
`resource_sync_safety.py` nor `delete()`/`delete_many()` (both single-
`memory_id`-at-a-time primitives) can observe at all.

Motivating case: volcengine/OpenViking#3064 (contributor AcTiveXXX).
`viking_fs.rm()`'s orphan-cleanup path, reached when a target directory no
longer exists in AGFS (e.g. files were deleted directly from the backing
filesystem, bypassing OpenViking's own API), discovers child URIs to
delete via a directory-listing walk wrapped in a bare `except: pass`. When
the directory itself is already gone, that walk silently returns an empty
list, so only the root URI reaches the vector store's delete call --
child vector-index entries beneath the root survive, permanently orphaned
(AcTiveXXX measured ~9% orphan rate in a real deployment).

This eval seeds nested content under one prefix via `store()`, calls
`adapter.delete_prefix(prefix, recursive=True)` (see
MemoryBackendAdapter.delete_prefix and MemoryBackendAdapter
.supports_prefix_delete in adapters/base.py), then classifies each seeded
file from two INDEPENDENT observations, never trusting either alone:

  * `list_resource_paths(prefix)` -- an AGFS-listing-level "is this path
    still there" check, the same primitive resource_sync_safety.py already
    uses.
  * `query(prefix, seed_content)` -- an index-level "does search still
    surface this content" check.

A file whose path is gone from list_resource_paths() AND whose content no
longer surfaces via query() is genuinely clean (VectorIntegritySignal
.CLEAN). A file whose path is gone from list_resource_paths() but whose
content STILL surfaces via query() is the exact #3064 shape: the
filesystem-level view says "deleted," the vector index disagrees
(VectorIntegritySignal.ORPHANED_VECTOR_ENTRY). This is the same
design principle evals/resource_sync_safety.py's classify_resource_sync_file
and evals/crash_recovery.py's classify_crash_recovery_case already
establish: classification never blindly trusts a single observation when
an independent cross-check is available.

Optional/flagged capability: this eval only runs against adapters that set
MemoryBackendAdapter.supports_prefix_delete = True. Adapters without a
prefix-delete primitive are skipped cleanly -- reported as skipped, never
silently dropped from the results table and never crashed by calling an
unimplemented method.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter, VectorIntegritySignal

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "orphan_cleanup_cases.json"
)


@dataclass
class OrphanCleanupSeedFile:
    path_suffix: str
    content: str


@dataclass
class OrphanCleanupCase:
    case_id: str
    prefix: str
    seed_files: list[OrphanCleanupSeedFile]


@dataclass
class OrphanCleanupFileResult:
    case_id: str
    path_suffix: str
    stored_path: str | None
    present_after_delete: bool
    """Whether list_resource_paths(prefix) still reports this path present
    after delete_prefix() -- an AGFS-listing-level observation."""
    queryable_after_delete: bool
    """Whether query(prefix, seed_content) still surfaces a record whose
    content matches this file's seeded content after delete_prefix() -- an
    index-level observation, independent of present_after_delete."""
    signal: VectorIntegritySignal
    error: str | None = None


@dataclass
class OrphanCleanupEvalResult:
    backend_name: str
    dataset_path: str
    file_results: list[OrphanCleanupFileResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def scored_files(self) -> list[OrphanCleanupFileResult]:
        return [f for f in self.file_results if f.error is None]

    def _fraction(self, signal: VectorIntegritySignal) -> float | None:
        scored = self.scored_files
        if not scored:
            return None
        matching = sum(1 for f in scored if f.signal == signal)
        return matching / len(scored)

    @property
    def orphaned_vector_entry_rate(self) -> float | None:
        """Fraction of seeded files whose vector-index entry survived a
        delete_prefix() call that reported the path itself as gone -- the
        headline metric this eval exists to surface, and the exact
        volcengine/OpenViking#3064 shape."""
        return self._fraction(VectorIntegritySignal.ORPHANED_VECTOR_ENTRY)

    @property
    def clean_rate(self) -> float | None:
        return self._fraction(VectorIntegritySignal.CLEAN)


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[OrphanCleanupCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        OrphanCleanupCase(
            case_id=c["case_id"],
            prefix=c["prefix"],
            seed_files=[
                OrphanCleanupSeedFile(path_suffix=sf["path_suffix"], content=sf["content"])
                for sf in c["seed_files"]
            ],
        )
        for c in cases
    ]


def classify_orphan_cleanup_file(
    present_after_delete: bool,
    queryable_after_delete: bool,
) -> VectorIntegritySignal:
    """Classify a single seeded file's outcome from its post-delete
    presence (AGFS-listing-level) and searchability (index-level)
    observations -- both taken AFTER delete_prefix() has already run.

    Returns ORPHANED_VECTOR_ENTRY whenever the listing says the path is
    gone but a query still surfaces the content -- the vector index
    disagreeing with the filesystem-level view is exactly the #3064 shape,
    regardless of whether the path itself somehow still lists as present
    too (that combination should not occur from a well-behaved adapter,
    but if it does, still-queryable content is the more severe signal and
    wins).
    """
    if queryable_after_delete:
        return VectorIntegritySignal.ORPHANED_VECTOR_ENTRY
    if not present_after_delete:
        return VectorIntegritySignal.CLEAN
    # Path still listed as present and no query match -- delete_prefix()
    # did not actually remove the path from the filesystem-level listing
    # either. Not the #3064 shape (that is specifically an index/listing
    # DISAGREEMENT), but still not a clean delete -- conservatively
    # classified as an orphan since the caller asked for this content gone
    # and list_resource_paths() says it is not.
    return VectorIntegritySignal.ORPHANED_VECTOR_ENTRY


def run_orphan_cleanup_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> OrphanCleanupEvalResult:
    cases = load_dataset(dataset_path)
    result = OrphanCleanupEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    if not adapter.supports_prefix_delete:
        result.skipped = True
        result.skip_reason = (
            f"{adapter.name} does not support prefix delete "
            "(supports_prefix_delete=False) -- skipped, not run."
        )
        return result

    for case in cases:
        stored_paths: dict[str, str] = {}
        try:
            for seed in case.seed_files:
                store_result = adapter.store(
                    case.prefix, seed.content, metadata={"resource_path": seed.path_suffix}
                )
                stored_paths[seed.path_suffix] = store_result.memory_id

            adapter.delete_prefix(case.prefix, recursive=True)
            paths_after = set(adapter.list_resource_paths(case.prefix))
        except BackendAPIError as exc:
            for seed in case.seed_files:
                result.file_results.append(
                    OrphanCleanupFileResult(
                        case_id=case.case_id,
                        path_suffix=seed.path_suffix,
                        stored_path=stored_paths.get(seed.path_suffix),
                        present_after_delete=False,
                        queryable_after_delete=False,
                        signal=VectorIntegritySignal.NOT_APPLICABLE,
                        error=str(exc),
                    )
                )
            continue

        for seed in case.seed_files:
            stored_path = stored_paths[seed.path_suffix]
            present_after = stored_path in paths_after

            queryable_after = False
            try:
                query_result = adapter.query(case.prefix, seed.content, top_k=5)
                queryable_after = any(
                    seed.content.lower() in r.content.lower() for r in query_result.records
                )
            except BackendAPIError:
                queryable_after = False

            signal = classify_orphan_cleanup_file(present_after, queryable_after)
            result.file_results.append(
                OrphanCleanupFileResult(
                    case_id=case.case_id,
                    path_suffix=seed.path_suffix,
                    stored_path=stored_path,
                    present_after_delete=present_after,
                    queryable_after_delete=queryable_after,
                    signal=signal,
                )
            )

    return result
