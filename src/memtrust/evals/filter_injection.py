"""MemTrust's filter-injection / access-control-bypass eval.

None of the other evals in this package submit an adversarial, caller-
controlled filter value directly to a backend's underlying vector-store
filter-query-building layer -- they all exercise the normal, harness-
constructed `{"user_id": session_id}` filter every adapter's `query()`
sends. This eval closes that specific gap: it asks whether a backend's
filter-building code *validates* a filter value's type before embedding it
into a query, or trusts it unconditionally.

Motivating case: mem0ai/mem0#5980 (merged 2026-07-02, GitHub user
HrushiYadav; closes #5976; part of a coordinated 5-backend
injection-prevention series covering elasticsearch/neptune/azure/
opensearch/databricks -- this eval only concerns the elasticsearch fix).
`ElasticsearchVectorStore` (`mem0/vector_stores/elasticsearch.py` in the
installed package) previously embedded caller-supplied filter values
directly into Elasticsearch `term` queries -- `{"term": {f"metadata.{key}":
value}}` -- with no check that `value` was actually a scalar. A dict/list
value (e.g. `{"user_id": {"$ne": ""}}`, reproducing Elasticsearch's own
query-DSL operator syntax) could therefore inject arbitrary query
behavior into what was meant to be a single-user scope filter, enabling
access-control bypass / cross-user memory enumeration -- a caller who
should only ever see their own `user_id`'s memories could construct a
filter value that matches every user's.

**Confirmed fixed in the installed `mem0ai==2.0.12` package, by reading
`mem0/vector_stores/elasticsearch.py` source directly, not by trusting
PR #5980's "merged" status.** The installed file defines a module-level
`_validate_filter(key, value)` helper (a `_SAFE_FILTER_KEY` regex
allowlist for the key, `isinstance(value, (str, int, float, bool))` for
the value) and calls it immediately before every `{"term": ...}` clause is
built, in all three places `ElasticsearchDB` constructs one: `search()`
(the KNN pre-filter path), `keyword_search()` (the BM25 path), and
`list()`. A dict/list-valued filter fails the `isinstance` check and
raises `ValueError` before a query is ever built or sent to the
Elasticsearch client. `gh pr view 5980 --repo mem0ai/mem0` shows the real,
merged PR description matches this line for line (see
mem0_direct_adapter.py's module docstring, "Elasticsearch support"
section, for the full citation and confirmation detail) -- this is not a
case where the issue says "merged" but the pinned package version
predates the fix.

**What this eval actually does.** It calls
`MemoryBackendAdapter.probe_raw_filter(filters)` -- a new optional
capability (see `adapters/base.py`'s `supports_raw_filter_probe` and
`RawFilterProbeResult`) that submits a filter dict directly to a backend's
underlying vector-store filter-building layer, bypassing the normal
session-scoped `query()` path entirely (that path always sends a
harness-constructed `{"user_id": session_id}` filter and gives a caller no
way to submit an adversarial value through it). Only `Mem0DirectAdapter`
implements this today, because it is the one adapter in this repo that
holds a direct, in-process handle to the vendor library's own
`vector_store` object rather than talking to a backend over HTTP -- see
`mem0_direct_adapter.py::probe_raw_filter()`. Every other adapter reports
`supports_raw_filter_probe = False` and this eval records
`FilterInjectionSignal.NOT_APPLICABLE` (skipped) for it, same convention
`evals/crash_recovery.py` and `evals/resource_sync_safety.py` already
establish for their own opt-in capability flags.

**Honest scope of what this eval can and cannot prove.** This eval's
fixture cases (`tests/fixtures/filter_injection_cases.json`) are
submitted to whichever vector store `Mem0DirectAdapter` is configured
against with a real, mocked-at-the-wire-client `mem0.vector_stores.*`
class in the test suite (see `tests/test_mem0_direct_adapter.py`), so a
run against `vector_store_provider="elasticsearch"` genuinely exercises
the real, installed `mem0.vector_stores.elasticsearch.ElasticsearchDB`
class -- proving what this module's docstring claims about #5980. This
eval has never been run against a live Elasticsearch cluster (or a live
Redis/Valkey/Qdrant server) in this environment -- every test mocks the
vendor SDK/wire-client boundary, the same convention every other adapter
in this repo follows (see `docs/methodology.md`). Running this eval
against `vector_store_provider="redis"/"valkey"/"qdrant"` is also
possible (the probe itself is provider-agnostic -- see
`Mem0DirectAdapter.supports_raw_filter_probe`'s docstring) but this build
did not inspect whether those three installed vector-store classes
validate filter values the same way Elasticsearch's fixed code now does;
a result for one of those providers should be read as "what the installed
package's `list(filters=...)` actually did when given this value," not as
a claim this build confirmed against those providers' source the way it
did for Elasticsearch.

Classification produces one of:

  * FILTER_REJECTED       -- the raw filter-probe call raised before
                              completing. For a malicious case (a
                              dict/list-valued filter reproducing the
                              #5980 shape) this is the safe, correct
                              outcome. For a benign, scalar-valued control
                              case, this same signal instead means the
                              backend incorrectly rejected a legitimate
                              filter -- see
                              FilterInjectionEvalResult.benign_false_positive_rate,
                              which is what tells the two apart in the
                              aggregate numbers (the signal alone,
                              consistent with FilterInjectionSignal's own
                              docstring, is a raw pass/fail observation,
                              not a verdict -- a verdict requires the
                              case's `malicious` ground truth too).
  * FILTER_ACCEPTED_SAFELY -- the raw filter-probe call for a *benign*
                              case completed without raising -- the
                              expected, correct outcome for a legitimate
                              filter.
  * INJECTION_SUCCEEDED    -- the raw filter-probe call for a *malicious*
                              case completed without raising: the
                              vector-store's filter-building layer
                              accepted a non-scalar value with no type
                              validation, exactly the vulnerable pre-fix
                              #5980 code path. Never observed against the
                              installed, real `ElasticsearchDB` in this
                              build's tests (see module docstring above --
                              confirmed fixed); reserved for a
                              provider/version whose filter-building layer
                              has no `_validate_filter()` equivalent.
  * NOT_APPLICABLE         -- either the adapter has no raw-filter-probe
                              capability at all
                              (`supports_raw_filter_probe` is False -- the
                              eval is skipped, not run per-case), or the
                              probe call raised for a reason unrelated to
                              filter-value validation (e.g. a
                              construction-time config rejection -- see
                              `RawFilterProbeResult`'s docstring) before
                              any filter value could be meaningfully
                              classified.

Design principle (same as evals/extraction_quality.py's
`classify_case`-equivalent functions): a raw pass/fail observation
(`RawFilterProbeResult.accepted`) is never itself treated as "safe" or
"vulnerable" -- classification always cross-references it against the
case's own ground truth (`FilterInjectionCase.malicious`), the same way
`ExtractionQualitySignal` cross-references `should_be_stored` rather than
treating "retrievable" as inherently good or bad.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from memtrust.adapters.base import MemoryBackendAdapter, RawFilterProbeResult

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "filter_injection_cases.json"
)


class FilterInjectionSignal(StrEnum):
    """How a single filter value fared when submitted directly to a
    backend's vector-store filter-building layer. See this module's
    docstring for the full classification write-up and the honest
    "signal alone is not a verdict" caveat -- a verdict requires cross-
    referencing this against the case's `malicious` ground truth.
    """

    FILTER_REJECTED = "filter_rejected"
    """The raw filter-probe call raised before completing -- the filter
    value was rejected outright by the backend's own filter-building code
    (or the adapter's construction/connection layer; see NOT_APPLICABLE
    below for when that distinction matters). Assigned regardless of
    whether the case was malicious or benign -- see
    FilterInjectionEvalResult.malicious_rejected_rate (good outcome) vs.
    benign_false_positive_rate (bad outcome) for how the two are told
    apart."""

    FILTER_ACCEPTED_SAFELY = "filter_accepted_safely"
    """The raw filter-probe call completed without raising for a *benign*,
    scalar-valued control case -- the expected, correct outcome for a
    legitimate filter. Only ever assigned to a case whose `malicious` is
    False; a malicious case that is accepted without raising is classified
    INJECTION_SUCCEEDED instead (see below), never this value, so a reader
    scanning for this signal never has to cross-reference ground truth to
    know whether "accepted" was actually safe."""

    INJECTION_SUCCEEDED = "injection_succeeded"
    """The raw filter-probe call for a *malicious*, dict/list-valued
    filter payload (the exact mem0ai/mem0#5980 shape) completed without
    raising -- the vector store's filter-building layer accepted a
    non-scalar value with no type validation, exactly the vulnerable
    pre-fix code path #5980 describes. This is the headline failure
    signal this eval exists to catch; see
    FilterInjectionEvalResult.injection_succeeded_rate."""

    NOT_APPLICABLE = "not_applicable"
    """Either the adapter has no raw-filter-probe capability at all
    (MemoryBackendAdapter.supports_raw_filter_probe is False -- the eval
    is skipped entirely, not run per-case), or the probe raised for a
    reason unrelated to filter-value validation (e.g. a construction-time
    config rejection such as a missing embedding-dimension config or
    missing Elasticsearch credential -- see RawFilterProbeResult's
    docstring) before any filter value could be meaningfully classified.
    Recorded explicitly, never silently dropped from the results table,
    same convention as every other NOT_APPLICABLE signal in this repo."""


@dataclass
class FilterInjectionCase:
    case_id: str
    malicious: bool
    filter_key: str
    filter_value: object
    description: str = ""


@dataclass
class FilterInjectionCaseResult:
    case: FilterInjectionCase
    signal: FilterInjectionSignal
    probe_accepted: bool | None
    """Mirrors RawFilterProbeResult.accepted for this case, or None if the
    probe was never attempted (adapter has no capability, see `signal`
    NOT_APPLICABLE)."""
    error: str | None = None


@dataclass
class FilterInjectionEvalResult:
    backend_name: str
    dataset_path: str
    case_results: list[FilterInjectionCaseResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def malicious_case_results(self) -> list[FilterInjectionCaseResult]:
        return [c for c in self.case_results if c.case.malicious]

    @property
    def benign_case_results(self) -> list[FilterInjectionCaseResult]:
        return [c for c in self.case_results if not c.case.malicious]

    def _fraction(
        self, case_results: list[FilterInjectionCaseResult], signal: FilterInjectionSignal
    ) -> float | None:
        if not case_results:
            return None
        matching = sum(1 for c in case_results if c.signal == signal)
        return matching / len(case_results)

    @property
    def injection_succeeded_rate(self) -> float | None:
        """Fraction of malicious cases where the malicious filter value was
        accepted without rejection -- the headline metric this eval exists
        to catch, the exact mem0ai/mem0#5980 shape. `None` when there are
        no malicious cases to score. `0.0` against the real, installed
        Elasticsearch vector store (see module docstring: confirmed
        fixed)."""
        return self._fraction(
            self.malicious_case_results, FilterInjectionSignal.INJECTION_SUCCEEDED
        )

    @property
    def malicious_rejected_rate(self) -> float | None:
        """Fraction of malicious cases correctly rejected -- the good
        outcome, and the complement of injection_succeeded_rate among
        scoreable malicious cases (they do not have to sum to 1.0 if any
        malicious case landed on NOT_APPLICABLE instead)."""
        return self._fraction(self.malicious_case_results, FilterInjectionSignal.FILTER_REJECTED)

    @property
    def benign_accepted_rate(self) -> float | None:
        """Fraction of benign, scalar-valued control cases correctly
        accepted -- the good outcome for a legitimate filter."""
        return self._fraction(
            self.benign_case_results, FilterInjectionSignal.FILTER_ACCEPTED_SAFELY
        )

    @property
    def benign_false_positive_rate(self) -> float | None:
        """Fraction of benign, scalar-valued control cases the backend
        incorrectly rejected. Not itself a security issue -- the opposite
        failure mode -- but a signal a fix (or this eval's own probe
        mechanics) is overly strict, since a real caller with a legitimate
        scalar filter should never be rejected."""
        return self._fraction(self.benign_case_results, FilterInjectionSignal.FILTER_REJECTED)


def load_dataset(path: Path | str = DEFAULT_FIXTURE_PATH) -> list[FilterInjectionCase]:
    data = json.loads(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return [
        FilterInjectionCase(
            case_id=c["case_id"],
            malicious=c["malicious"],
            filter_key=c["filter_key"],
            filter_value=c["filter_value"],
            description=c.get("description", ""),
        )
        for c in cases
    ]


def classify_filter_injection_case(
    case: FilterInjectionCase, probe: RawFilterProbeResult
) -> FilterInjectionSignal:
    """Classify a single case's outcome, cross-referencing the raw
    accept/reject observation against the case's own `malicious` ground
    truth -- never treating "accepted" or "rejected" alone as inherently
    good or bad. See module docstring for the full write-up.

    Checks `probe.applicable` first: a probe that never reached the
    vector store's filter-building layer at all (a construction-time
    config rejection, unrelated to filter validation) is NOT_APPLICABLE
    regardless of `malicious` -- it would otherwise be misclassified as a
    "correct rejection" for a malicious case or a "false positive" for a
    benign one, neither of which is true since the filter value itself was
    never actually evaluated.
    """
    if not probe.applicable:
        return FilterInjectionSignal.NOT_APPLICABLE
    if case.malicious:
        return (
            FilterInjectionSignal.INJECTION_SUCCEEDED
            if probe.accepted
            else FilterInjectionSignal.FILTER_REJECTED
        )
    return (
        FilterInjectionSignal.FILTER_ACCEPTED_SAFELY
        if probe.accepted
        else FilterInjectionSignal.FILTER_REJECTED
    )


def run_filter_injection_eval(
    adapter: MemoryBackendAdapter,
    dataset_path: Path | str = DEFAULT_FIXTURE_PATH,
) -> FilterInjectionEvalResult:
    cases = load_dataset(dataset_path)
    result = FilterInjectionEvalResult(backend_name=adapter.name, dataset_path=str(dataset_path))

    if not adapter.supports_raw_filter_probe:
        result.skipped = True
        result.skip_reason = (
            f"{adapter.name} does not support a raw filter probe "
            "(supports_raw_filter_probe=False) -- skipped, not run. Only an adapter that "
            "holds a direct, in-process handle to a vendor library's own vector-store "
            "filter-building layer can genuinely submit an adversarial filter value "
            "outside its normal query() path; see adapters/base.py's "
            "supports_raw_filter_probe and evals/filter_injection.py's module docstring."
        )
        return result

    for case in cases:
        try:
            probe = adapter.probe_raw_filter({case.filter_key: case.filter_value})
        except NotImplementedError as exc:
            # Should not happen given the supports_raw_filter_probe check
            # above, but guarded the same defensive way every other
            # optional-capability eval in this repo guards its own call.
            result.case_results.append(
                FilterInjectionCaseResult(
                    case=case,
                    signal=FilterInjectionSignal.NOT_APPLICABLE,
                    probe_accepted=None,
                    error=str(exc),
                )
            )
            continue

        signal = classify_filter_injection_case(case, probe)
        result.case_results.append(
            FilterInjectionCaseResult(
                case=case,
                signal=signal,
                probe_accepted=probe.accepted,
                error=probe.error,
            )
        )

    return result
