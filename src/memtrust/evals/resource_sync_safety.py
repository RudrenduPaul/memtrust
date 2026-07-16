"""MemTrust's directory/resource-sync safety eval.

The store/query/update model the other evals in this package exercise
only covers single-key memory operations. It has no concept of a
multi-file directory mirror/resync operation, so it cannot observe a
resync mechanism silently deleting user-owned files that a backend's
ingestion watcher did not itself generate.

This eval closes that specific gap. It is modeled directly on a real,
high-severity bug report: volcengine/OpenViking#3029, where OpenViking's
Feishu resync mechanism silently deleted user-owned files sitting
alongside the files its own ingestion watcher had generated. The eval
seeds a mix of "generated" files (standing in for what a watcher itself
would produce) and "user" files (standing in for files a person added to
the same resource path independently of the watcher) under one resource
prefix, triggers a resync, and re-lists the prefix to see what survived:

  * PRESERVED             -- the file is still present after the resync,
                              and (where content could be re-verified)
                              its content is unchanged.
  * DELETED_USER_FILE      -- the file was present before the resync and
                              is gone afterward, with no error and no
                              signal to the caller. This is the exact
                              #3029 failure mode: a resync silently
                              destroying data it never wrote.
  * OVERWRITTEN_UNCHANGED  -- the path itself still exists after the
                              resync (existence "unchanged"), but its
                              content no longer matches what was seeded,
                              i.e. the resync silently overwrote it
                              rather than deleting it outright.
  * NESTED_CONTENT_UNINDEXED -- the path itself still exists after the
                              resync (list_resource_paths() reports it
                              present) and a query() was actually issued,
                              but the search index never returned a
                              record for that path at all -- distinct
                              from OVERWRITTEN_UNCHANGED, which means a
                              record for the path *was* found but its
                              content had changed. This is the
                              volcengine/OpenViking#1703 failure mode:
                              index_resource() skipped every subdirectory
                              during reindex, so nested-directory content
                              was never vectorized in the first place --
                              "never indexed," not "deleted" or
                              "overwritten." A file at a top-level path
                              (no "/" in its resource path) failing this
                              way would just look like a generic search
                              miss; this signal is specifically for paths
                              that are nested, where "never indexed" has
                              a concrete, reproducible mechanism to point
                              to.

Optional/flagged capability: this eval only runs against adapters that
set MemoryBackendAdapter.supports_resource_sync = True. Adapters without
a directory/resource-mirror concept (the store/query/update-only model)
are skipped cleanly -- reported as skipped, never silently dropped from
the results table and never crashed by calling an unimplemented method.

Design principle (see evals/contradiction.py's classify_case for the same
pattern applied to contradiction detection): classification never blindly
trusts that a seeded file is "fine" just because the adapter didn't error. Each file's
final signal is derived from the actual before/after path listings (and,
where possible, a re-query of the file's own content) -- not from any
self-report the adapter makes about the resync succeeding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "resource_sync_cases.json"
)


class ResourceSyncSignal(StrEnum):
    """How one seeded file fared across a trigger_resync() call."""

    PRESERVED = "preserved"
    """Present after the resync; content re-verified as unchanged (or the
    adapter offered no way to re-verify content, in which case presence
    alone is treated as preserved)."""

    DELETED_USER_FILE = "deleted_user_file"
    """Present before the resync, gone afterward, with no error raised.
    The exact volcengine/OpenViking#3029 failure mode."""

    OVERWRITTEN_UNCHANGED = "overwritten_unchanged"
    """Still present after the resync, but its content no longer matches
    what was seeded -- silently overwritten rather than deleted."""

    NESTED_CONTENT_UNINDEXED = "nested_content_unindexed"
    """Still present after the resync (the path itself was not deleted or
    overwritten), but no query() call ever returned a record for that
    path at all -- the content exists on disk/in the filesystem mirror
    but was never indexed for search. This is "never indexed," not
    "deleted": the exact volcengine/OpenViking#1703 failure mode, where
    a reindex's directory walk skipped every subdirectory and left
    nested content permanently unsearchable with no error raised."""

    NOT_APPLICABLE = "not_applicable"
    """Either the backend has no resource-sync primitive this eval can
    exercise (MemoryBackendAdapter.supports_resource_sync is False), or a
    file's before/after state could not be established at all (e.g. it
    was never observed present before the resync in the first place)."""


@dataclass
class ResourceSyncSeedFile:
    path_suffix: str
    origin: str  # "generated" | "user"
    content: str


@dataclass
class ResourceSyncCase:
    case_id: str
    prefix: str
    seed_files: list[ResourceSyncSeedFile]


@dataclass
class ResourceSyncFileResult:
    case_id: str
    path_suffix: str
    origin: str
    stored_path: str | None
    present_before_resync: bool
    present_after_resync: bool
    content_matches_after_resync: bool | None
    indexed_after_resync: bool | None
    signal: ResourceSyncSignal
    error: str | None = None


@dataclass
class ResourceSyncEvalResult:
    backend_name: str
    dataset_path: str
    file_results: list[ResourceSyncFileResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def scored_files(self) -> list[ResourceSyncFileResult]:
        return [f for f in self.file_results if f.error is None]

    def _fraction(self, signal: ResourceSyncSignal, origin: str | None = None) -> float | None:
        scored = self.scored_files
        if origin is not None:
            scored = [f for f in scored if f.origin == origin]
        if not scored:
            return None
        matching = sum(1 for f in scored if f.signal == signal)
        return matching / len(scored)

    @property
    def user_file_deletion_rate(self) -> float | None:
        """Fraction of user-origin seed files silently deleted by a
        resync -- the headline metric this eval exists to surface."""
        return self._fraction(ResourceSyncSignal.DELETED_USER_FILE, origin="user")

    @property
    def preserved_rate(self) -> float | None:
        return self._fraction(ResourceSyncSignal.PRESERVED)

    @property
    def overwritten_unchanged_rate(self) -> float | None:
        return self._fraction(ResourceSyncSignal.OVERWRITTEN_UNCHANGED)

    @property
    def nested_content_unindexed_rate(self) -> float | None:
        """Fraction of seed files that survived the resync on disk but
        were never returned by any query() call -- the volcengine/
        OpenViking#1703 signal, distinct from deletion or overwrite."""
        return self._fraction(ResourceSyncSignal.NESTED_CONTENT_UNINDEXED)


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[ResourceSyncCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        ResourceSyncCase(
            case_id=c["case_id"],
            prefix=c["prefix"],
            seed_files=[
                ResourceSyncSeedFile(
                    path_suffix=sf["path_suffix"],
                    origin=sf["origin"],
                    content=sf["content"],
                )
                for sf in c["seed_files"]
            ],
        )
        for c in cases
    ]


def classify_resource_sync_file(
    present_before: bool,
    present_after: bool,
    content_matches_after_resync: bool | None,
    indexed_after_resync: bool | None = None,
) -> ResourceSyncSignal:
    """Classify a single seeded file's outcome from its before/after
    presence and (if re-verifiable) content/index match.

    Returns DELETED_USER_FILE whenever a file that was confirmed present
    before the resync is gone afterward -- deliberately independent of
    whether the caller passes a "generated" or "user" origin here, since
    the eval only computes rates *by* origin afterward (see
    ResourceSyncEvalResult.user_file_deletion_rate); a generated file
    disappearing unexpectedly is just as real a signal, it is simply not
    the metric #3029 was about.

    Returns NESTED_CONTENT_UNINDEXED whenever a file is still present
    after the resync but `indexed_after_resync` is explicitly False --
    meaning a query() call was actually made and returned no record at
    all for this path, as opposed to returning a record with stale
    content (OVERWRITTEN_UNCHANGED). `indexed_after_resync=None` (the
    default) means the caller never attempted to distinguish "no record
    found" from "record found with wrong content" -- existing callers
    that only pass the first three arguments keep the prior behavior
    unchanged (OVERWRITTEN_UNCHANGED wins on a content mismatch).

    A file never observed present before the resync has nothing
    meaningful to classify (its "before" state is unknown), so it is
    recorded as NOT_APPLICABLE rather than guessed at either way.
    """
    if present_before and not present_after:
        return ResourceSyncSignal.DELETED_USER_FILE
    if present_after and indexed_after_resync is False:
        return ResourceSyncSignal.NESTED_CONTENT_UNINDEXED
    if present_after and content_matches_after_resync is False:
        return ResourceSyncSignal.OVERWRITTEN_UNCHANGED
    if present_after:
        return ResourceSyncSignal.PRESERVED
    return ResourceSyncSignal.NOT_APPLICABLE


def run_resource_sync_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> ResourceSyncEvalResult:
    cases = load_dataset(dataset_path)
    result = ResourceSyncEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    if not adapter.supports_resource_sync:
        result.skipped = True
        result.skip_reason = (
            f"{adapter.name} does not support resource-sync operations "
            "(supports_resource_sync=False) -- skipped, not run."
        )
        return result

    for case in cases:
        stored_paths: dict[str, str] = {}
        try:
            for seed in case.seed_files:
                store_result = adapter.store(
                    case.prefix,
                    seed.content,
                    metadata={"resource_path": seed.path_suffix, "origin": seed.origin},
                )
                stored_paths[seed.path_suffix] = store_result.memory_id

            paths_before = set(adapter.list_resource_paths(case.prefix))
            adapter.trigger_resync(case.prefix)
            paths_after = set(adapter.list_resource_paths(case.prefix))
        except BackendAPIError as exc:
            for seed in case.seed_files:
                result.file_results.append(
                    ResourceSyncFileResult(
                        case_id=case.case_id,
                        path_suffix=seed.path_suffix,
                        origin=seed.origin,
                        stored_path=stored_paths.get(seed.path_suffix),
                        present_before_resync=False,
                        present_after_resync=False,
                        content_matches_after_resync=None,
                        indexed_after_resync=None,
                        signal=ResourceSyncSignal.NOT_APPLICABLE,
                        error=str(exc),
                    )
                )
            continue

        for seed in case.seed_files:
            stored_path = stored_paths[seed.path_suffix]
            present_before = stored_path in paths_before
            present_after = stored_path in paths_after

            content_matches: bool | None = None
            indexed_after: bool | None = None
            if present_after:
                try:
                    query_result = adapter.query(case.prefix, seed.content, top_k=5)
                    # Two distinct questions, not one: did the search index
                    # return *any* record for this exact path at all
                    # (indexed_after -- the #1703 signal), and separately,
                    # among whatever records did come back, does any one's
                    # content match what was seeded (content_matches -- the
                    # #3029-era overwrite signal). Collapsing these into one
                    # boolean would make "never indexed" indistinguishable
                    # from "indexed with stale content."
                    indexed_after = any(r.memory_id == stored_path for r in query_result.records)
                    content_matches = any(
                        seed.content.lower() in r.content.lower() for r in query_result.records
                    )
                except BackendAPIError:
                    content_matches = None
                    indexed_after = None

            signal = classify_resource_sync_file(
                present_before, present_after, content_matches, indexed_after
            )
            result.file_results.append(
                ResourceSyncFileResult(
                    case_id=case.case_id,
                    path_suffix=seed.path_suffix,
                    origin=seed.origin,
                    stored_path=stored_path,
                    present_before_resync=present_before,
                    present_after_resync=present_after,
                    content_matches_after_resync=content_matches,
                    indexed_after_resync=indexed_after,
                    signal=signal,
                )
            )

    return result
