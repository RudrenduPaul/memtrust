"""Compression / round-trip fidelity eval.

Neither LongMemEval, LoCoMo, nor memtrust's own contradiction eval measures
a vendor's "lossless" or compression-ratio claim directly. The closest
existing signal would be an aggregate accuracy delta between two operating
modes on one of those evals -- but that requires a way to tell the adapter
which mode to use, which is exactly what
`MemoryBackendAdapter.supported_modes` and the `mode` parameter on
`store()`/`query()` (see adapters/base.py) now provide.

This eval is deliberately narrow and NOT an LLM-judge eval: it stores a
piece of content, retrieves it back, and scores how much of the original
text survived the round trip using a direct, deterministic text-similarity
metric (`fidelity_ratio`, a character-level `difflib.SequenceMatcher`
ratio -- see its docstring). A vendor's "lossless compression" claim is
a factual claim about byte/character fidelity, not about semantic
equivalence, so grading it with an LLM judge would be both more expensive
and a worse fit than a direct string comparison: an LLM judge might rate a
paraphrased, information-dropping reconstruction as "close enough," which
is precisely the kind of leniency this eval exists to avoid.

Origin: mempalace/mempalace#27 (cited in README.md and docs/methodology.md
as founding rationale for this project) documents a "lossless" compression
claim that a 12.4 percentage-point accuracy drop in practice contradicts.
`run_compression_eval()` is what would let a contributor with a live
MemPalace instance reproduce that "raw vs AAAK" comparison directly, by
running this eval against `MemPalaceAdapter` (whose `supported_modes` is
`("raw", "AAAK")` -- see adapters/mempalace_adapter.py for the exact
provenance and confidence caveat on those two mode names).

**This eval has not been run against any live backend as of this file's
creation.** No fidelity number below or in any report this eval produces
should be read as measured until it has actually executed against a
configured, live adapter -- see docs/methodology.md's "What requires a
live vendor API key" table, which this eval should be added to under the
same rules as every other eval runner.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memtrust.adapters.base import BackendAPIError, MemoryBackendAdapter, QueryResult, StoreResult

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "compression_cases.json"
)

#: Mode label used when an adapter declares no `supported_modes` at all
#: (an empty tuple). The eval still runs exactly once per case in this
#: case, rather than being silently skipped -- see docs/methodology.md's
#: "no result silently dropped from the table" convention, applied here
#: the same way evals/contradiction.py applies it to NOT_APPLICABLE.
DEFAULT_MODE_LABEL = "default"


@dataclass
class CompressionCase:
    case_id: str
    content: str
    description: str = ""


@dataclass
class CompressionCaseResult:
    case_id: str
    mode: str
    fidelity_score: float
    content_length: int
    retrieved_length: int
    retrieved_content: str = ""
    error: str | None = None


@dataclass
class CompressionModeResult:
    """Per-mode results for one backend -- one of these per string in
    `adapter.supported_modes`, or a single one labeled DEFAULT_MODE_LABEL
    for adapters that report no mode variants."""

    mode: str
    case_results: list[CompressionCaseResult] = field(default_factory=list)

    @property
    def scored_cases(self) -> list[CompressionCaseResult]:
        return [c for c in self.case_results if c.error is None]

    @property
    def mean_fidelity(self) -> float | None:
        scored = self.scored_cases
        if not scored:
            return None
        return sum(c.fidelity_score for c in scored) / len(scored)


@dataclass
class CompressionEvalResult:
    backend_name: str
    dataset_path: str
    modes: list[str] = field(default_factory=list)
    mode_results: dict[str, CompressionModeResult] = field(default_factory=dict)

    def mean_fidelity_by_mode(self) -> dict[str, float | None]:
        return {mode: result.mean_fidelity for mode, result in self.mode_results.items()}

    @property
    def fidelity_drop_pp(self) -> float | None:
        """Percentage-point gap between the best- and worst-scoring mode.

        This is the number that would reproduce a "96.6% raw vs 84.2%
        AAAK, 12.4pp drop" style comparison once this eval has actually
        been run against a live backend that reports more than one mode.
        `None` if fewer than two modes produced a scoreable mean (e.g. a
        single-mode adapter, or a run where every case errored).
        """
        means = [m for m in self.mean_fidelity_by_mode().values() if m is not None]
        if len(means) < 2:
            return None
        return (max(means) - min(means)) * 100


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[CompressionCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        CompressionCase(
            case_id=c["case_id"],
            content=c["content"],
            description=c.get("description", ""),
        )
        for c in cases
    ]


def fidelity_ratio(original: str, retrieved: str) -> float:
    """Direct, deterministic round-trip fidelity score in [0.0, 1.0].

    Character-level similarity via `difflib.SequenceMatcher.ratio()`
    (Ratcliff/Obershelp), NOT an LLM judge -- this eval measures literal
    reconstruction fidelity, the thing a "lossless" claim is actually
    about, not semantic equivalence. 1.0 means the retrieved text is
    character-for-character identical to what was stored (a genuinely
    lossless round trip). Lower values mean measurable information loss:
    a truncated, reordered, or otherwise mangled reconstruction scores
    strictly below a perfect one.

    Two special-cased edges, both to keep the metric meaningful rather
    than mathematically degenerate:
      * both strings empty -> 1.0 (trivially identical, not a loss).
      * original non-empty but nothing was retrieved -> 0.0 (total loss,
        rather than SequenceMatcher's undefined/zero-length behavior).
    """
    if not original and not retrieved:
        return 1.0
    if not retrieved:
        return 0.0
    return difflib.SequenceMatcher(None, original, retrieved).ratio()


def _select_retrieved_content(store_result: StoreResult, query_result: QueryResult) -> str:
    """Pick the retrieved text to score against the original.

    Prefers the record whose memory_id matches what store() returned (the
    same memory we just wrote), falling back to the top-ranked result --
    query() is the only read primitive the shared adapter interface
    exposes (there is no get-by-id), so a round-trip eval has to go
    through search like any other caller would.
    """
    for record in query_result.records:
        if record.memory_id == store_result.memory_id:
            return record.content
    if query_result.records:
        return query_result.records[0].content
    return ""


def run_compression_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> CompressionEvalResult:
    """Run the same store+retrieve round trip for every case, once per
    mode the adapter reports supporting (see
    MemoryBackendAdapter.supported_modes), and score fidelity per mode.

    An adapter with no mode variants (`supported_modes == ()`) still runs
    once, under the synthetic label `DEFAULT_MODE_LABEL` -- it is not
    skipped or silently dropped from the results table, matching the
    convention `evals/contradiction.py` uses for NOT_APPLICABLE.
    """
    cases = load_dataset(dataset_path)
    modes = list(adapter.supported_modes) if adapter.supported_modes else [DEFAULT_MODE_LABEL]
    result = CompressionEvalResult(
        backend_name=adapter.name, dataset_path=str(dataset_path), modes=modes
    )

    for mode in modes:
        effective_mode = None if mode == DEFAULT_MODE_LABEL else mode
        mode_result = CompressionModeResult(mode=mode)
        for case in cases:
            session_id = f"compression-{case.case_id}-{mode}"
            try:
                store_result = adapter.store(session_id, case.content, mode=effective_mode)
                query_result = adapter.query(session_id, case.content, top_k=5, mode=effective_mode)
            except BackendAPIError as exc:
                mode_result.case_results.append(
                    CompressionCaseResult(
                        case_id=case.case_id,
                        mode=mode,
                        fidelity_score=0.0,
                        content_length=len(case.content),
                        retrieved_length=0,
                        retrieved_content="",
                        error=str(exc),
                    )
                )
                continue

            retrieved = _select_retrieved_content(store_result, query_result)
            mode_result.case_results.append(
                CompressionCaseResult(
                    case_id=case.case_id,
                    mode=mode,
                    fidelity_score=fidelity_ratio(case.content, retrieved),
                    content_length=len(case.content),
                    retrieved_length=len(retrieved),
                    retrieved_content=retrieved,
                )
            )
        result.mode_results[mode] = mode_result

    return result
