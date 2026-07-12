# Contributing

The easiest and most useful way to contribute is adding a new backend adapter. This document
covers that path in detail, plus the general workflow for everything else.

## Adding a new backend adapter

Every adapter implements `memtrust.adapters.base.MemoryBackendAdapter`, defined in
`src/memtrust/adapters/base.py`. Read that file first; it is short and it is the actual contract,
not a summary of one.

### The interface

```python
class MemoryBackendAdapter(ABC):
    name: str
    env_var: str
    supports_update: bool = True

    def store(self, session_id: str, content: str, metadata: dict[str, str] | None = None) -> StoreResult: ...
    def query(self, session_id: str, query: str, top_k: int = 5) -> QueryResult: ...
    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult: ...
```

### Step by step

1. **Pick your `env_var`.** Every adapter reads exactly one environment variable in `__init__` and
   raises `BackendNotConfiguredError(self.name, self.env_var)` immediately if it's missing --
   never on the first method call. This is what lets `memtrust run` report SKIPPED instead of
   crashing when a backend isn't configured. If your backend genuinely needs no secret (like
   MemPalace, which is local-first), gate on whatever configuration value it does need instead --
   see `mempalace_adapter.py` for the pattern and `docs/methodology.md` for why.

2. **Implement `store()`, `query()`, `update()`** against the vendor's real API. Wrap every
   network/vendor failure in `BackendAPIError(self.name, detail)` -- never let a raw
   `httpx.HTTPError` or vendor SDK exception escape the adapter. `query()` must return a
   `ConflictSignal` (see below).

3. **Report `ConflictSignal` honestly.** This is what the contradiction-detection eval reads.
   - `FLAGGED` if your backend's response makes a contradiction visible (returns both old and new
     values, an explicit conflict marker, an invalidation timestamp, etc.)
   - `NOT_APPLICABLE` if you cannot determine this from the response -- do **not** guess `FLAGGED`
     or `SILENT_OVERWRITE` to make a number look better. The contradiction eval independently
     cross-checks your reported signal against the actual retrieved content (see
     `evals/contradiction.py::classify_case`), so an inflated self-report gets caught and
     downgraded, not rewarded.
   - If your backend has no update/contradiction-relevant primitive at all, set
     `supports_update = False` on the class. The eval then records `NOT_APPLICABLE` for every case
     without calling your adapter, and that gap is shown explicitly in results tables -- it is
     never silently dropped.

4. **Document your confidence level.** At the top of your adapter file, write a docstring stating
   what you verified against real vendor documentation and what you built as best-effort. Add a
   row to the confidence table in `docs/methodology.md`. If you are not confident about an exact
   endpoint path or method signature, say so in the code comment at the point of use, the same way
   `mempalace_adapter.py` and `openviking_adapter.py` do. A wrong guess that's labeled is useful; a
   wrong guess presented as confirmed is a bug that will mislead every leaderboard reader.

5. **Register it.** Add your adapter class to `ADAPTER_REGISTRY` in
   `src/memtrust/adapters/__init__.py`, keyed by the name users will pass to `--backends`.

6. **Write tests.** Every adapter test mocks the HTTP layer (`pytest-httpx`) or injects a fake
   object matching your adapter's expected vendor interface -- see `tests/test_adapters.py` for
   the pattern used by all four existing adapters. No test may make a real network call. Cover at
   minimum: `BackendNotConfiguredError` when the env var is missing, a successful `store`/`query`/
   `update` round trip against a mocked response, and a `BackendAPIError` on a failed HTTP call.

7. **Run the full check before opening a PR:**
   ```bash
   ruff check . && ruff format --check .
   mypy --strict src/memtrust
   pytest --cov=memtrust --cov-report=term-missing --cov-fail-under=80
   pip-audit
   ```

### What a PR adding an adapter should include

- The adapter file, following the pattern above.
- Its registration in `ADAPTER_REGISTRY`.
- Tests in `tests/test_adapters.py`.
- A confidence-level entry in `docs/methodology.md`'s adapter table.
- A one-line addition to the README's backend coverage table.

## Adding or extending an eval

The three eval families live in `src/memtrust/evals/`. Each is a plain function taking a
configured `MemoryBackendAdapter` (and an `LLMJudge` for the two that need semantic grading) and
returning a dataclass of results -- there is no plugin system to learn, just a function signature
to match. See `evals/contradiction.py` for the simplest example (no LLM judge needed) and
`evals/longmemeval.py` for the LLM-judged pattern.

To extend the contradiction-detection eval's case set, add entries to
`tests/fixtures/contradiction_cases.json`. Read `docs/methodology.md`'s note on how the
`contradicting_fact` field should be phrased before adding a case -- a correction that restates
the old value inside its own text can produce a misleading classification (this happened once
during the initial build and is documented there in detail).

To run the harness against the real, full LongMemEval or LoCoMo datasets instead of the bundled
synthetic samples, see the "to run against the real dataset" note under each eval in
`docs/methodology.md` -- both loaders accept a `dataset_path` argument already; only a format
conversion (or a second loader function) is needed.

## General workflow

1. Fork, branch, make your change.
2. Run the full check list above locally before pushing.
3. Keep PRs scoped to one adapter, one eval change, or one clearly-described fix -- easier to
   review, easier to bisect if something regresses.
4. Every claim in a PR description about a score or benchmark number must be reproducible from a
   command someone else can run. "I ran X and got Y" needs the X.

## Code of conduct

Be direct, be specific, assume good faith. Disagreement about a methodology choice is welcome and
expected -- open an issue with the specific flaw, not a vague complaint. This project exists
because vague, unverifiable claims about agent-memory backends are the problem it's trying to fix;
holding contributions to the same standard is the point.
