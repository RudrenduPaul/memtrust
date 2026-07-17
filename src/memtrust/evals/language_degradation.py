"""memtrust's non-Latin-script i18n retrieval-degradation eval for
`Mem0DirectAdapter`.

Motivating case: wangjiawei-vegetable (rank 147, mem0ai/mem0#4884, open,
merge-ready companion PR #4943 as of this build). mem0 v3's hybrid
retrieval pipeline (semantic + BM25 keyword + entity-based boosting)
hardcodes spaCy's English model `en_core_web_sm` for both BM25
lemmatization (`mem0/utils/lemmatization.py::lemmatize_for_bm25`) and
entity extraction (`mem0/utils/spacy_models.py::get_nlp_full`/
`get_nlp_lemma`) -- confirmed by reading the installed `mem0ai==2.0.12`
package's own `mem0/utils/spacy_models.py` directly, both loader
functions call `spacy.load("en_core_web_sm", ...)` unconditionally, with
no language parameter anywhere in the module. For non-Latin-script text
(Chinese, Japanese, Arabic, Thai, Hindi, ...), the English pipeline's
tokenization/lemmatization does not reliably produce keyword/entity
signals that overlap between query time and store time, so
`mem0/memory/main.py::_search_vector_store()`'s own `bm25_scores`/
`entity_boosts` dicts can come back empty for a query where they would
fire for equivalent English text -- and `mem0/utils/scoring.py::
score_and_rank()`'s `has_bm25 = bool(bm25_scores)`/`has_entity =
bool(entity_boosts)` gates mean the combined score silently falls back to
semantic-only weighting. No exception is raised and nothing in the
normal (non-`explain`) response indicates this happened -- exactly the
"silently degrades... users see no error" shape the issue itself
describes.

None of memtrust's other four signal taxonomies (`ExtractionQualitySignal`
/`RankingSignal`/`EmbeddingDriftSignal`/`CorruptionSignal`) classify this:
all four concern content/order/drift/write-path corruption, never whether
a language-dependent retrieval-pipeline STAGE ran at all for a given
query. See `LanguageDegradationSignal` in `adapters/base.py` for the full
taxonomy this eval scores against, and `mem0_direct_adapter.py`'s
`query(explain=True)` for exactly how each result's real, installed
`score_details` breakdown (`bm25_score`, `entity_boost`) is read -- this
eval never guesses degradation from content alone, it reads mem0's own
per-result diagnostic fields.

Honest scope: this eval is `Mem0DirectAdapter`-specific (only that
adapter holds a direct, in-process handle to `mem0.Memory` and can pass
`explain=True` through to it) -- REST-facing adapters
(`Mem0Adapter`/`Mem0SelfHostedAdapter`) have no equivalent parameter on
their vendor HTTP surface to make this observable at all. This eval and
its fixture (`tests/fixtures/language_degradation_cases.json`) prove the
classification logic is correct given a `score_details` response shape;
it has not been run against a live mem0 instance with real spaCy/embedder
credentials configured in this environment -- same "structurally capable,
not live-run" caveat every other eval in this package that depends on a
real LLM/embedding backend already carries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memtrust.adapters.base import BackendAPIError, LanguageDegradationSignal
from memtrust.adapters.mem0_direct_adapter import Mem0DirectAdapter

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "language_degradation_cases.json"
)


@dataclass
class LanguageDegradationCase:
    case_id: str
    session_id: str
    script: str
    """Which non-Latin script family this case models -- e.g. "chinese",
    "japanese", "arabic", "thai", "hindi". Used to compute a per-script
    degradation rate, not just one aggregate number, since the issue's own
    root cause (an English-only spaCy pipeline) could plausibly affect
    different scripts differently depending on how spaCy's English
    tokenizer happens to segment each one."""
    content: str
    query: str


@dataclass
class LanguageDegradationCaseResult:
    case: LanguageDegradationCase
    signal: LanguageDegradationSignal
    error: str | None = None


@dataclass
class LanguageDegradationEvalResult:
    backend_name: str
    dataset_path: str
    case_results: list[LanguageDegradationCaseResult] = field(default_factory=list)

    @property
    def scored_cases(self) -> list[LanguageDegradationCaseResult]:
        return [c for c in self.case_results if c.error is None]

    def _fraction(self, signal: LanguageDegradationSignal) -> float | None:
        scored = self.scored_cases
        if not scored:
            return None
        matching = sum(1 for c in scored if c.signal == signal)
        return matching / len(scored)

    @property
    def semantic_only_degraded_rate(self) -> float | None:
        """The headline metric this eval exists to surface -- fraction of
        non-Latin-script cases where BM25/entity-boost silently never
        fired. This is what would reproduce mem0ai/mem0#4884's finding if
        run against a live, unpatched mem0 instance."""
        return self._fraction(LanguageDegradationSignal.SEMANTIC_ONLY_DEGRADED)

    @property
    def hybrid_signals_active_rate(self) -> float | None:
        return self._fraction(LanguageDegradationSignal.HYBRID_SIGNALS_ACTIVE)

    def rate_by_script(self, signal: LanguageDegradationSignal) -> dict[str, float | None]:
        """Per-script breakdown of `signal`'s rate -- e.g.
        `rate_by_script(LanguageDegradationSignal.SEMANTIC_ONLY_DEGRADED)`
        returns `{"chinese": 1.0, "arabic": 0.5, ...}`, letting a caller
        see whether degradation is uniform across scripts or concentrated
        in specific ones, rather than only one aggregate number."""
        scripts = sorted({c.case.script for c in self.scored_cases})
        result: dict[str, float | None] = {}
        for script in scripts:
            cases = [c for c in self.scored_cases if c.case.script == script]
            if not cases:
                result[script] = None
                continue
            matching = sum(1 for c in cases if c.signal == signal)
            result[script] = matching / len(cases)
        return result


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[LanguageDegradationCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        LanguageDegradationCase(
            case_id=c["case_id"],
            session_id=c["session_id"],
            script=c["script"],
            content=c["content"],
            query=c["query"],
        )
        for c in cases
    ]


def run_language_degradation_eval(
    adapter: Mem0DirectAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> LanguageDegradationEvalResult:
    """Store each case's non-Latin-script content, then query() it back
    with `explain=True` -- reading `QueryResult.language_degradation_signal`
    for the verdict `Mem0DirectAdapter.query()` already derived from mem0's
    real `score_details` response. This eval does no classification of its
    own beyond aggregating what `query()` reports -- the actual
    `bm25_score`/`entity_boost` inspection lives in
    `mem0_direct_adapter.py::_classify_language_degradation()`, so a
    single, real code path backs both a direct `query(explain=True)` call
    and this eval's aggregate numbers.
    """
    cases = load_dataset(dataset_path)
    result = LanguageDegradationEvalResult(
        backend_name=adapter.name, dataset_path=str(dataset_path)
    )

    for case in cases:
        try:
            adapter.store(case.session_id, case.content)
            query_result = adapter.query(case.session_id, case.query, top_k=5, explain=True)
        except BackendAPIError as exc:
            result.case_results.append(
                LanguageDegradationCaseResult(
                    case=case,
                    signal=LanguageDegradationSignal.NOT_APPLICABLE,
                    error=str(exc),
                )
            )
            continue

        result.case_results.append(
            LanguageDegradationCaseResult(
                case=case, signal=query_result.language_degradation_signal
            )
        )

    return result
