"""MemPalace MCP metadata-tool scale eval (mempalace_status/list_wings/
list_rooms).

MemPalace/mempalace#1871 (contributor alionar) found that the MCP
server's metadata/histogram-listing tools -- `mempalace_status`,
`mempalace_list_wings`, `mempalace_list_rooms` -- did a full-collection
scan on every call against a Qdrant-backed palace: O(N^2) against
repeated calls, hanging the MCP server at 158K+ drawers (fixed by
server-side Qdrant faceting; see PR#1871's diff for the merged fix).

Before this eval, evals/scale_stress.py was memtrust's only scale-oriented
coverage, and it only ever measures store()/query() latency -- it has no
path at all to the metadata/histogram-listing code path alionar's bug
lives in, because MemPalaceAdapter never invoked the MCP-server surface,
only the guessed remember()/recall()/invalidate() library concepts.

adapters/mempalace_adapter.py's module docstring ("MCP metadata-tool
coverage" section) documents what this build's investigation confirmed
against the real, installed `mempalace` package: `mempalace.mcp_server`
ships real, plain module-level functions (`tool_status()`,
`tool_list_wings()`, `tool_list_rooms(wing=None)`) that are the actual
implementation the MCP tool calls dispatch to, callable directly with no
MCP stdio/HTTP transport involved. `MemPalaceAdapter.metadata_overview()`/
`list_metadata_categories()`/`list_metadata_subcategories()` wrap those
three real functions -- this eval drives them at increasing checkpoint
sizes, the same "measure the real thing as a function of corpus size"
shape evals/scale_stress.py established for store()/query().

Two things this eval checks at every checkpoint, neither trusting "the
call didn't raise" as proof anything worked (same rule every other eval
in this package applies):

  * Correctness: the wing/room counts `metadata_overview()`/
    `list_metadata_categories()`/`list_metadata_subcategories()` report
    must match the seeder's own ground-truth counts for however many
    records have actually been seeded so far. A backend that responds
    quickly but reports the wrong counts is not "working at scale" --
    see MetadataScaleSignal.INCORRECT_COUNTS_AT_SCALE.
  * Latency growth shape: per-call latency at the largest checkpoint is
    compared against the smallest checkpoint's latency, scaled by how
    much the record count itself grew. A latency ratio that outpaces the
    record-count ratio by more than a generous superlinear margin is
    flagged MetadataScaleSignal.SUPERLINEAR_LATENCY_GROWTH -- a
    regression guard for the exact repeated-full-collection-scan shape
    alionar's bug report describes, not a claim that this eval reproduces
    that bug's specific 158K-drawer failure.

**Honest limitation, stated plainly (mirrors evals/scale_stress.py's own
"Honest limitation" section).** `build_chroma_metadata_seeder()` below
seeds a real, local, chromadb-backed palace via the real, installed
`mempalace` package (confirmed working: seeding 20,000 records and
re-querying `tool_status()`/`tool_list_wings()` against them, in this
build's own investigation, returned correct ground-truth wing/room counts
in well under 100ms). That genuinely exercises the real metadata-listing
code path this bug lives in, on the real chroma backend.

What it does NOT do is reproduce alionar's *exact* repro shape. Re-reading
PR#1871's own review thread (gemini-code-assist's review comment)
confirms the merged fix has two parts: (1) server-side Qdrant faceting,
replacing the O(N^2) client-side scan for the Qdrant backend specifically,
and (2) a secondary optimization to the *Chroma fallback path* in
`_fetch_all_metadata` (making it reuse already-fetched records). Neither
part is what this eval's live seeder exercises: `mempalace.mcp_server`'s
`_sqlite_taxonomy()` fast path (a separate, already-merged optimization,
#1748/#1379) intercepts every `tool_status()`/`tool_list_wings()`/
`tool_list_rooms()` call against a standard-layout chroma-backed palace
*before* `_fetch_all_metadata` is ever reached -- confirmed in this
build's own investigation (sub-100ms responses at N=20,000, and the
source comment on `_sqlite_taxonomy()` itself: "fast wing->room tally
straight from chroma.sqlite3... to signal the caller to fall back to the
ChromaDB client pagination path"). So the live scale run this eval
performs is real, but it currently measures the fast path staying fast at
scale (a legitimate, previously-uncovered regression guard), not the
`_fetch_all_metadata` full-collection-scan path PR#1871 actually patches.

Two things could not be reproduced live in this build's environment,
for two different reasons:

  * PR#1871's fix is specifically for MemPalace's Qdrant backend
    (`mempalace/backends/qdrant.py`), which is REST-only against a live
    external Qdrant server (`http://localhost:6333` by default) with no
    embedded/local-storage mode at all -- confirmed by reading that
    backend's source. This build's environment has neither `docker` nor a
    local Qdrant binary available (both checked and confirmed absent), so
    a live, 158K-drawer, Qdrant-backed repro of alionar's exact bug could
    not be run here. A contributor with a live Qdrant instance available
    can point `MEMPALACE_BACKEND=qdrant` at one and run this eval
    unchanged -- the adapter methods and this eval's checkpoint/
    classification logic make no chroma-specific assumption.
  * Even on the chroma backend, deliberately forcing `_sqlite_taxonomy()`
    to fail/decline (so a run actually exercises `_fetch_all_metadata`
    instead of short-circuiting past it) would require constructing a
    non-standard palace layout or a corrupted sqlite index -- not
    attempted here, since a deliberately-broken palace is a different,
    narrower scenario than "this backend at real scale," and getting it
    wrong risks a misleading result more than a missing one.

Both are the same class of environmental gap this repo already draws
honestly elsewhere (see docs/methodology.md and
evals/crash_recovery.py's process-lifecycle caveat): this eval's
checkpoint/classification harness and the adapter methods it drives are
real, working, and verified end-to-end against a real backend at real
scale -- what has not happened yet is pointing that real harness at the
one specific code path (Qdrant faceting, or a forced chroma fallback)
alionar's report and PR#1871's fix actually touch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter

#: Fast-by-default record count -- mirrors scale_stress.py's
#: DEFAULT_N_RECORDS reasoning: large enough to move past every bundled
#: fixture's single-digit scale, small enough to seed and check quickly
#: in CI against a real local chromadb-backed palace (seeding 20,000
#: records took ~3s in this build's own investigation).
DEFAULT_N_RECORDS = 2000

#: How many non-anchor top-level categories (MemPalace: wings) the
#: default seeder spreads records across. Kept small and fixed so every
#: checkpoint has a meaningful per-category count to check, rather than
#: scaling with n_records the way real palaces rarely would either.
DEFAULT_N_CATEGORIES = 8

#: Minimum latency ratio (largest checkpoint / smallest checkpoint)
#: before growth is even considered for the superlinear check below --
#: guards against noise on two calls that are both already sub-millisecond
#: (a healthy backend's status() call at N=5 and N=50 can easily differ by
#: 2-3x on pure measurement noise; that is not evidence of anything).
_MIN_LATENCY_RATIO_TO_FLAG = 3.0

#: Exponent applied to the record-count ratio when deciding whether an
#: observed latency ratio counts as "superlinear." A perfectly linear
#: scan scores exactly 1.0 here; the true O(N^2) shape alionar reported
#: (repeated full-collection re-scans) would score close to 2.0. 1.5 is a
#: deliberately generous middle threshold: high enough that ordinary
#: linear-scaling noise does not false-positive, low enough to still catch
#: a real quadratic-ish regression before it reaches production scale.
_SUPERLINEAR_EXPONENT = 1.5


class MetadataScaleSignal(StrEnum):
    """How a backend's metadata-overview calls (status/list_categories/
    list_subcategories) behaved as the corpus grew, per
    run_mempalace_metadata_scale_eval()."""

    WORKED_AT_SCALE = "worked_at_scale"
    """Every checkpoint's reported counts matched ground truth, and
    latency did not grow superlinearly relative to record count."""

    INCORRECT_COUNTS_AT_SCALE = "incorrect_counts_at_scale"
    """At least one checkpoint's reported category/subcategory counts
    did not match the seeder's ground truth -- a correctness failure,
    checked independently of and before any latency judgment."""

    SUPERLINEAR_LATENCY_GROWTH = "superlinear_latency_growth"
    """Counts were correct everywhere, but latency grew far faster than
    the record count did between the smallest and largest checkpoint --
    the regression shape alionar's MemPalace/mempalace#1871 report
    describes (repeated full-collection scans behind the MCP metadata
    tools)."""

    ERROR = "error"
    """A BackendAPIError was raised during the run that this eval could
    not route around."""

    NOT_APPLICABLE = "not_applicable"
    """Fewer than 2 checkpoints produced a scoreable result (e.g.
    n_records too small), or the adapter under test does not implement
    this capability at all (see MetadataScaleResult.skipped)."""


@dataclass
class MetadataScaleCheckpointResult:
    """A snapshot taken after `checkpoint_n` records had been seeded."""

    checkpoint_n: int
    overview_latency_ms: float | None
    categories_latency_ms: float | None
    subcategories_latency_ms: float | None
    total_records_reported: int | None
    categories_reported: dict[str, int]
    categories_expected: dict[str, int]
    subcategories_reported: dict[str, int]
    subcategories_expected: dict[str, int]
    counts_correct: bool
    error: str | None = None


@dataclass
class MetadataScaleResult:
    """Result of `run_mempalace_metadata_scale_eval()` for one backend,
    one (n_records, seed) corpus."""

    backend_name: str
    n_records_requested: int
    checkpoints: list[MetadataScaleCheckpointResult] = field(default_factory=list)
    signal: MetadataScaleSignal = MetadataScaleSignal.NOT_APPLICABLE
    latency_ratio: float | None = None
    """(largest checkpoint's overview_latency_ms) / (smallest checkpoint's
    overview_latency_ms), or None if fewer than 2 checkpoints produced a
    latency to compare."""
    record_ratio: float | None = None
    """(largest checkpoint_n) / (smallest checkpoint_n), the baseline
    `latency_ratio` above is judged against."""
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None


class MetadataSeeder(Protocol):
    """Populates a MemPalace-backed collection so it holds exactly
    `total_n` synthetic drawers (cumulative -- calling this again with a
    larger `total_n` adds only the delta, never re-seeds from scratch),
    and returns the resulting ground-truth (categories, subcategories)
    count dicts for all `total_n` records.

    `build_chroma_metadata_seeder()` below is the real, live
    implementation this eval is meant to be run with. Tests inject a
    lightweight in-memory fake implementing this same Protocol so the
    checkpoint/classification logic can be verified without chromadb
    installed -- see tests/test_mempalace_metadata_scale.py.
    """

    def __call__(self, total_n: int) -> tuple[dict[str, int], dict[str, int]]: ...


def _default_checkpoints(n: int) -> list[int]:
    """Same shape as scale_stress.py's `_default_checkpoints`: a small,
    ascending set of checkpoint sizes from "the same scale every bundled
    fixture already runs at" up to the full requested N.
    """
    candidates = [5, max(1, n // 10), max(1, n // 2), n]
    return sorted({c for c in candidates if 1 <= c <= n})


def _run_checkpoint(
    adapter: MemoryBackendAdapter, seeder: MetadataSeeder, checkpoint_n: int
) -> MetadataScaleCheckpointResult:
    try:
        categories_expected, subcategories_expected = seeder(checkpoint_n)
    except BackendAPIError as exc:
        return MetadataScaleCheckpointResult(
            checkpoint_n=checkpoint_n,
            overview_latency_ms=None,
            categories_latency_ms=None,
            subcategories_latency_ms=None,
            total_records_reported=None,
            categories_reported={},
            categories_expected={},
            subcategories_reported={},
            subcategories_expected={},
            counts_correct=False,
            error=f"seeding failed: {exc}",
        )

    try:
        overview = adapter.metadata_overview()
        categories = adapter.list_metadata_categories()
        subcategories = adapter.list_metadata_subcategories()
    except BackendAPIError as exc:
        return MetadataScaleCheckpointResult(
            checkpoint_n=checkpoint_n,
            overview_latency_ms=None,
            categories_latency_ms=None,
            subcategories_latency_ms=None,
            total_records_reported=None,
            categories_reported={},
            categories_expected=categories_expected,
            subcategories_reported={},
            subcategories_expected=subcategories_expected,
            counts_correct=False,
            error=str(exc),
        )

    call_error = overview.error or categories.error or subcategories.error
    counts_correct = (
        call_error is None
        and overview.total_records == checkpoint_n
        and overview.categories == categories_expected
        and categories.counts == categories_expected
        and subcategories.counts == subcategories_expected
    )

    return MetadataScaleCheckpointResult(
        checkpoint_n=checkpoint_n,
        overview_latency_ms=overview.latency_ms,
        categories_latency_ms=categories.latency_ms,
        subcategories_latency_ms=subcategories.latency_ms,
        total_records_reported=overview.total_records,
        categories_reported=categories.counts,
        categories_expected=categories_expected,
        subcategories_reported=subcategories.counts,
        subcategories_expected=subcategories_expected,
        counts_correct=counts_correct,
        error=call_error,
    )


def classify_metadata_scale_result(
    checkpoints: list[MetadataScaleCheckpointResult],
) -> tuple[MetadataScaleSignal, float | None, float | None]:
    """Classify a completed run's checkpoints. Returns
    (signal, latency_ratio, record_ratio) -- never a blind pass/fail on
    "did every call succeed," the same ground-truth-driven pattern every
    other eval's classify_* function in this package follows.
    """
    if any(c.error is not None for c in checkpoints):
        return MetadataScaleSignal.ERROR, None, None

    scoreable = [c for c in checkpoints if c.error is None]
    if len(scoreable) < 2:
        return MetadataScaleSignal.NOT_APPLICABLE, None, None

    if any(not c.counts_correct for c in scoreable):
        return MetadataScaleSignal.INCORRECT_COUNTS_AT_SCALE, None, None

    first, last = scoreable[0], scoreable[-1]
    if (
        first.overview_latency_ms is None
        or last.overview_latency_ms is None
        or first.checkpoint_n >= last.checkpoint_n
        or first.overview_latency_ms <= 0
    ):
        return MetadataScaleSignal.WORKED_AT_SCALE, None, None

    record_ratio = last.checkpoint_n / first.checkpoint_n
    latency_ratio = last.overview_latency_ms / first.overview_latency_ms

    if (
        latency_ratio > _MIN_LATENCY_RATIO_TO_FLAG
        and latency_ratio > record_ratio**_SUPERLINEAR_EXPONENT
    ):
        return MetadataScaleSignal.SUPERLINEAR_LATENCY_GROWTH, latency_ratio, record_ratio
    return MetadataScaleSignal.WORKED_AT_SCALE, latency_ratio, record_ratio


def run_mempalace_metadata_scale_eval(
    adapter: MemoryBackendAdapter,
    seeder: MetadataSeeder,
    n_records: int = DEFAULT_N_RECORDS,
    checkpoints: list[int] | None = None,
) -> MetadataScaleResult:
    """Seed `seeder` up to `n_records` synthetic drawers at a series of
    checkpoints, and at each one check `adapter`'s metadata_overview()/
    list_metadata_categories()/list_metadata_subcategories() for both
    correctness and latency-growth shape.

    Args:
        adapter: the backend under test. Skipped (not run) unless
            `adapter.supports_metadata_overview` is True -- same
            convention evals/filter_injection.py already established for
            supports_raw_filter_probe.
        seeder: populates the backend with ground-truth-tracked synthetic
            records -- see MetadataSeeder above and
            build_chroma_metadata_seeder() for the real implementation.
        n_records: how many records to seed by the final checkpoint.
        checkpoints: explicit checkpoint sizes. Defaults to
            `_default_checkpoints(n_records)`.

    Raises:
        ValueError: if n_records < 1.
    """
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}")

    result = MetadataScaleResult(backend_name=adapter.name, n_records_requested=n_records)

    if not adapter.supports_metadata_overview:
        result.skipped = True
        result.skip_reason = (
            f"{adapter.name} does not support a metadata overview "
            "(supports_metadata_overview=False) -- skipped, not run. Only an adapter that "
            "holds a direct, in-process handle to a vendor's own MCP metadata-tool "
            "implementation (or an equivalent library-level function) can genuinely "
            "exercise this; see adapters/base.py's supports_metadata_overview and this "
            "module's docstring."
        )
        return result

    resolved_checkpoints = (
        sorted({c for c in checkpoints if 1 <= c <= n_records})
        if checkpoints is not None
        else _default_checkpoints(n_records)
    )

    result.checkpoints = [_run_checkpoint(adapter, seeder, cp) for cp in resolved_checkpoints]
    signal, latency_ratio, record_ratio = classify_metadata_scale_result(result.checkpoints)
    result.signal = signal
    result.latency_ratio = latency_ratio
    result.record_ratio = record_ratio

    if signal == MetadataScaleSignal.ERROR:
        first_error = next((c.error for c in result.checkpoints if c.error), None)
        result.error = first_error

    return result


def build_chroma_metadata_seeder(
    storage_path: str,
    *,
    collection_name: str = "mempalace_drawers",
    n_categories: int = DEFAULT_N_CATEGORIES,
    n_subcategories: int = 40,
    embedding_dim: int = 8,
) -> MetadataSeeder:
    """Real, live `MetadataSeeder` implementation: seeds a local,
    chromadb-backed palace at `storage_path` directly through the real,
    installed `mempalace` package's backend layer
    (`mempalace.palace.get_backend_for_palace` +
    `ChromaCollection.add(..., embeddings=...)`), bypassing
    MemPalaceAdapter.store() deliberately -- see
    adapters/mempalace_adapter.py's module docstring: `store()` is written
    against an unconfirmed `mempalace.Palace(...)` guess that does not
    exist in the real package, so it cannot be used to populate a real
    corpus at all. This seeder instead uses the confirmed-real backend
    API the real `mine`/CLI ingestion path itself is built on, passing
    explicit dummy embeddings so seeding never depends on network access
    (MemPalace's default embedder otherwise downloads an ONNX model on
    first use -- see mempalace_adapter.py's module docstring).

    Every record gets deterministic `wing`/`room` metadata
    (`wing_{i % n_categories}` / `room_{i % n_subcategories}`), matching
    the same "deterministic, ground-truth-trackable" design
    evals/scale_fixtures.py's generate_scale_corpus() already uses for
    scale_stress.py.

    Raises:
        BackendAPIError: if the `mempalace` package (or its chromadb
            dependency) is not installed, or a seeding call itself fails.
    """
    try:
        from mempalace.palace import get_backend_for_palace  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BackendAPIError(
            "mempalace",
            "the `mempalace` package is not installed. Install it with "
            "`pip install mempalace` to run a real, live metadata-scale "
            "stress test -- see this module's docstring for what "
            "classification-logic-only tests can still verify without it.",
        ) from exc

    try:
        backend = get_backend_for_palace(storage_path, explicit="chroma")
        collection = backend.get_collection(
            storage_path, collection_name=collection_name, create=True
        )
    except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
        raise BackendAPIError("mempalace", f"opening seed collection failed: {exc}") from exc

    seeded_so_far = 0

    def _seed(total_n: int) -> tuple[dict[str, int], dict[str, int]]:
        nonlocal seeded_so_far
        if total_n < seeded_so_far:
            raise ValueError(
                "metadata seeder is append-only: cannot shrink from "
                f"{seeded_so_far} already-seeded records down to {total_n}"
            )
        if total_n > seeded_so_far:
            documents = []
            ids = []
            metadatas = []
            embeddings = []
            for i in range(seeded_so_far, total_n):
                wing = f"wing_{i % n_categories}"
                room = f"room_{i % n_subcategories}"
                documents.append(f"synthetic metadata-scale drawer {i} (marker SCALEMETA{i:06d})")
                ids.append(f"scale-drawer-{i}")
                metadatas.append({"wing": wing, "room": room})
                embeddings.append([float((i * 7 + j) % 11) / 11.0 for j in range(embedding_dim)])
            try:
                collection.add(
                    documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings
                )
            except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
                raise BackendAPIError("mempalace", f"seeding collection failed: {exc}") from exc
            seeded_so_far = total_n

        categories_expected: dict[str, int] = {}
        subcategories_expected: dict[str, int] = {}
        for i in range(total_n):
            w = f"wing_{i % n_categories}"
            r = f"room_{i % n_subcategories}"
            categories_expected[w] = categories_expected.get(w, 0) + 1
            subcategories_expected[r] = subcategories_expected.get(r, 0) + 1
        return categories_expected, subcategories_expected

    return _seed
