# memtrust

Standardized, reproducible benchmarks for agent-memory backends, run against the vendors, not
published by them.

```bash
pip install -e ".[dev]"
memtrust run --backends mempalace,mem0,zep,openviking --eval all
```

## Why this exists

If you've compared agent-memory backends recently, you've probably noticed each one leads with a
different accuracy number, on a different benchmark, measured a different way. MemPalace's own
community already flagged the problem in public. Issue [#27](https://github.com/mempalace/mempalace/issues/27)
on the MemPalace repository, opened April 7, 2026 and still open, documents a widely-cited 100%
LongMemEval score that came from hand-tuning on the failed test questions themselves. The held-out
score is 98.4%, not 100%. A separate 96.6% figure people cite everywhere turns out to be mostly
ChromaDB's default embeddings doing the work, not MemPalace's own architecture. A "lossless"
compression claim drops accuracy by 12.4 percentage points in practice. Two internal pull requests
attempting to fix the reporting problem (#433 and #729) were both closed without merging on April
12, 2026. As of this writing, the issue has 233 thumbs-up reactions and 39 comments.

None of that means MemPalace, or any other backend, doesn't work. It means nobody outside the
vendor had run the same test, the same way, against every option, and published the raw logs.

memtrust does that. It runs LongMemEval, LoCoMo, and two evals built specifically for this project.
The first is a contradiction-detection eval, because neither LongMemEval nor LoCoMo tests the
question that actually matters once a memory system sits underneath a production agent: what
happens when a new fact contradicts an old one? Does the backend flag the conflict? Silently
overwrite the old fact with no audit trail? Serve whichever version it happens to retrieve first?
None of the four backends this project tracks publish a number for that. The second is a
compression/round-trip-fidelity eval, built to directly test claims like the "lossless" one above:
it stores content, retrieves it, and scores literal reconstruction fidelity rather than semantic
accuracy, per operating mode a backend exposes (see `MemoryBackendAdapter.supported_modes`) -- the
mechanism that would let a contributor with live MemPalace credentials actually reproduce the
12.4-point compressed-mode accuracy drop mempalace/mempalace#27 documents, instead of just citing
it. **It has not been run against any live backend as of this writing** -- see
`docs/methodology.md`.

## What it does

Every command below was actually run against this repo, with zero vendor API keys configured, to
produce the output shown. Nothing here is simulated.

```
$ memtrust run --backends mempalace,mem0,zep,openviking --eval all
memtrust 0.1.0 -- run_id=mt_2026-07-12T015520Z
Backends: mempalace, mem0, zep, openviking   Evals: longmemeval, locomo, contradiction, compression

mempalace: SKIPPED (not configured) -- mempalace is not configured: environment variable
MEMPALACE_STORAGE_PATH is not set. Skipping this backend. See docs/methodology.md for setup
instructions.
mem0: SKIPPED (not configured) -- mem0 is not configured: environment variable MEM0_API_KEY is not
set. Skipping this backend. See docs/methodology.md for setup instructions.
zep: SKIPPED (not configured) -- zep is not configured: environment variable ZEP_API_KEY is not set.
Skipping this backend. See docs/methodology.md for setup instructions.
openviking: SKIPPED (not configured) -- openviking is not configured: environment variable
OPENVIKING_API_KEY is not set. Skipping this backend. See docs/methodology.md for setup
instructions.

Cost: $0.00 (no LLM-judged evals ran -- structural evals only, or judge not configured)

Full report: memtrust-report-2026-07-12.json
```

That's the real, reproducible behavior of a fresh clone with no credentials: every backend reports
SKIPPED, the command exits cleanly, and a valid JSON report is still written. Set the relevant
environment variable for any backend you want to actually test (`MEM0_API_KEY`, `ZEP_API_KEY`,
`OPENVIKING_API_KEY`, `MEMPALACE_STORAGE_PATH`) and that backend runs for real against its live
API instead of being skipped.

The eval logic itself is proven offline, against the bundled synthetic fixtures, by the test
suite:

```
$ pytest --cov=memtrust --cov-report=term-missing
...
Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
src/memtrust/adapters/base.py             70      0   100%
src/memtrust/evals/contradiction.py       87      0   100%
src/memtrust/evals/compression.py         86      3    97%
src/memtrust/evals/longmemeval.py         58      0   100%
src/memtrust/scoring/cost_tracker.py      40      0   100%
-----------------------------------------------------------
TOTAL                                    882     42    95%

72 passed in 0.35s
```

72 tests, 95% overall coverage, 100% on the adapter interface and the contradiction-detection eval,
97% on the new compression/round-trip-fidelity eval. Every test mocks its HTTP layer or uses an
in-memory fake backend -- none of them touch a real network.

## How this differs from trusting a vendor's own numbers

Every backend memtrust tracks publishes its own benchmark numbers. None of them publish the same
benchmark, scored the same way, with the same held-out discipline. memtrust doesn't ask you to
trust it instead: it asks you to read the raw logs. Every run's methodology, prompt templates,
dataset versions, and scoring rubric are published in `docs/methodology.md`, versioned alongside
the code that produced them. If the methodology has a flaw, it's a flaw you can point to in a
specific file and line, not something buried in a vendor's internal eval pipeline.

General-purpose LLM eval frameworks (promptfoo, DeepEval, RAGAS, and similar tools) are mature and
widely used, but none of them ship a memory-backend adapter abstraction or a contradiction-
detection eval out of the box -- they're built for RAG quality, red-teaming, and general prompt
evaluation, not for comparing how different memory systems handle a fact that changes over time.
memtrust is narrower and more specific on purpose.

## The eval that actually matters: contradiction detection

LongMemEval and LoCoMo both measure recall: can the backend remember a fact you told it earlier.
That's necessary but not sufficient. The harder question is what a backend does when two facts
conflict: you tell it your meeting is at 2pm, then later say it moved to 3pm. Does it flag the
change? Overwrite silently? Serve whichever one it retrieves first? `memtrust`'s classifier stores
a fact, stores a contradicting fact, queries for it, then checks the actual retrieved content for
both values, rather than trusting whatever conflict signal the adapter itself reports. See
`src/memtrust/evals/contradiction.py` and the scoring-logic section of `docs/methodology.md` for
exactly how that classification works.

## The eval built for the other headline overclaim: compression fidelity

mempalace/mempalace#27 documents two separate overclaims, not one: the LongMemEval score gap
described above, and a "lossless" compression claim that measured 12.4 percentage points lower in
practice under a compressed operating mode. memtrust could not previously reproduce that second
number at all -- there was no way to tell an adapter "run this under mode X vs mode Y" through the
shared interface. `MemoryBackendAdapter.store()`/`query()` now accept an optional `mode: str |
None` parameter, and `MemoryBackendAdapter.supported_modes` lets an adapter declare which mode
strings it actually understands (`MemPalaceAdapter.supported_modes` is `("raw", "AAAK")`, the two
names mempalace/mempalace#27 itself uses -- see `src/memtrust/adapters/mempalace_adapter.py` for
the exact provenance and confidence caveat on those names). Adapters with no mode variants accept
and ignore the parameter, so this is a purely additive, backward-compatible interface change.

`src/memtrust/evals/compression.py` runs the same store-then-retrieve round trip once per mode a
backend reports, and scores each round trip with a direct, deterministic character-level
similarity ratio (`fidelity_ratio()`, via `difflib.SequenceMatcher` -- not an LLM judge, since a
"lossless" claim is a literal-reconstruction claim, not a semantic one). This is what would let a
contributor with live MemPalace credentials point `memtrust run --eval compression` at it and
reproduce a "raw vs AAAK" fidelity gap directly. **As of this writing this eval has not been run
against any live backend** -- see `docs/methodology.md` for the same live-credentials caveat that
applies to every other eval in this table.

## Benchmarks

**Not yet measured against live backends.** This repo was built without API credentials for
MemPalace, Mem0, Zep, or OpenViking. Publishing a number here without actually running the harness
against a live backend would be exactly the kind of unverifiable claim this project exists to
push back on, so there isn't one.

To produce real numbers:

```bash
export MEM0_API_KEY=...          # and/or
export ZEP_API_KEY=...
export OPENVIKING_API_KEY=...
export MEMPALACE_STORAGE_PATH=...
export MEMTRUST_JUDGE_API_KEY=...   # needed for LongMemEval/LoCoMo grading; contradiction-detection doesn't need it

memtrust run --backends mempalace,mem0,zep,openviking --eval all
memtrust report memtrust-report-<date>.json
```

The command prints per-backend accuracy and contradiction-handling rates, writes a full JSON
report, and prints an estimated cost for any LLM-judged evals that ran. Two of the four adapters
(MemPalace and OpenViking) are built against best-effort interpretations of documented product
concepts rather than a confirmed API reference -- see the confidence table in
`docs/methodology.md` before treating their output as authoritative, and consider that table's
gaps a standing invitation to contribute a fix.

## GitHub Actions usage

Run the suite on a schedule and publish results to the leaderboard:

```yaml
name: memtrust-leaderboard
on:
  schedule:
    - cron: "0 9 * * 1"  # weekly
  workflow_dispatch: {}

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install memtrust
      - run: memtrust run --backends mempalace,mem0,zep,openviking --eval all --output leaderboard/data.json
        env:
          MEM0_API_KEY: ${{ secrets.MEM0_API_KEY }}
          ZEP_API_KEY: ${{ secrets.ZEP_API_KEY }}
          OPENVIKING_API_KEY: ${{ secrets.OPENVIKING_API_KEY }}
          MEMTRUST_JUDGE_API_KEY: ${{ secrets.MEMTRUST_JUDGE_API_KEY }}
      - run: git add leaderboard/data.json && git commit -m "Update leaderboard" && git push
```

This repo's own CI (`.github/workflows/ci.yml`) runs lint, type-check, test, and a dependency
security audit on every push and pull request -- no vendor credentials required, since every test
runs fully offline.

## Self-host

```bash
git clone https://github.com/RudrenduPaul/memtrust
cd memtrust
pip install -e ".[dev]"
export MEM0_API_KEY=...
memtrust run --backends mem0 --eval all
```

Point an adapter at your own backend, or run the suite against your own conversation data instead
of the bundled synthetic fixtures (see `docs/methodology.md`'s note on swapping in the real
LongMemEval/LoCoMo datasets). Nothing leaves your machine unless you choose to publish it.

## Install

### npx (agent-native)

> **Coming soon -- requires PyPI + npm publish, not live yet.** memtrust has not been published to
> PyPI or npm as of this writing (`pip install memtrust` and `npm install memtrust` both 404
> today). The command below describes the planned distribution path; see `npm/` in this repo for
> the actual wrapper source.

For CI and agent runners that have Node.js available but not necessarily a Python toolchain:

```bash
npx memtrust run --backends mempalace,mem0,zep,openviking --eval all
```

This is not a zero-dependency install: `npx memtrust` still fetches `memtrust` from PyPI on first
use. What it removes is no Python toolchain to provision by hand -- `npx memtrust` handles the
interpreter and package fetch for you via a bundled, verified copy of Astral's
[`uv`](https://github.com/astral-sh/uv). Each platform package bundles a genuine, SHA-256-verified
copy of `uv`'s own GitHub release binary (fetched at npm package-publish time, never at end-user
install time), and its `bin` shim runs `uv tool run --from memtrust memtrust <args>`, which
transparently bootstraps a Python interpreter and installs `memtrust` from PyPI, caching it after
the first run.

## What a hosted trust layer would add

The harness, adapters, and leaderboard in this repo are the entire OSS surface, and they're
sufficient on their own to compare backends. A hosted layer on top of this -- described here, not
built -- would add continuous regression monitoring that re-runs the suite automatically whenever
a tracked backend ships a new release, private scorecards that run the same methodology against a
team's own data shape instead of the public sample fixtures, and a compliance-report export for
teams whose security or legal review needs a documented third-party artifact rather than a
free-text summary. None of that exists yet. If it's ever built, it stays additive to the free
harness, never a requirement for using it.

## Backend coverage

| Backend | Adapter status | Confidence (see docs/methodology.md) |
|---|---|---|
| MemPalace | Implemented | Medium on behavior, low on exact method names -- needs verification against a live instance |
| Mem0 | Implemented | High -- documented Python SDK and REST behavior |
| Zep / Graphiti | Implemented | Medium-high -- documented contradiction-handling behavior, best-effort wire format |
| OpenViking | Implemented | Medium on architecture, low on exact memory-write/query paths |

Adding a backend adapter is the primary contribution path -- see `CONTRIBUTING.md`.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy --strict src/memtrust
pytest --cov=memtrust --cov-report=term-missing --cov-fail-under=80
pip-audit
```

`.pre-commit-config.yaml` wires ruff and mypy into `pre-commit` if you'd rather run these on every
commit than remember to run them by hand.

## License

Apache 2.0. See `LICENSE`.

## Success stories

Nine real bugs, reported by real contributors against MemPalace, mem0, Zep/Graphiti, and
OpenViking, that memtrust's own harness either couldn't have caught before this work or can now
catch directly. Each one below has been re-verified live against the current codebase, not just
cited from a changelog. Full write-ups, live-validation evidence, and outreach status live in
`[redacted internal path]`
for the project's own tracking; the summary here is for anyone deciding whether this harness would
have caught their own bug.

**MemPalace**
- [#1754](https://github.com/MemPalace/mempalace/pull/1754) (@rodboev): a checkpoint recovery fix
  for silently quarantined dim-None pickles. memtrust's contradiction eval couldn't previously tell
  "silently quarantined" apart from "no update primitive at all"; it now can
  (`ConflictSignal.EMPTY_OR_LOST`).
- [#1929](https://github.com/MemPalace/mempalace/pull/1929) (@jrzmurray): a fix for NUL bytes
  silently corrupting a ChromaDB index. memtrust's `store()` used to trust "no exception" as proof
  of a durable write; an opt-in read-after-write verification step now catches this.
- [#1450](https://github.com/MemPalace/mempalace/pull/1450) (@lealbrunocalhau): a fix for an empty
  embedding response getting scored as a wrong answer instead of flagged as infra failure. Same
  fix as #1754 above.
- [#1823](https://github.com/MemPalace/mempalace/pull/1823) / [#1543](https://github.com/MemPalace/mempalace/pull/1543)
  (@fatkobra): lock and write-integrity fixes that pointed at the same read-after-write gap #1929
  closed.

**mem0**
- [#5973](https://github.com/mem0ai/mem0/pull/5973) (@abhay-codes07, superseded by
  [#5992](https://github.com/mem0ai/mem0/pull/5992)): an empty-string entity-id filter scoping bug.
  memtrust's mem0 adapter only reached the hosted Platform API and had no delete operation at all,
  so it couldn't have caught this. A self-hosted adapter with tested delete/delete_many primitives
  now can.
- [#4297](https://github.com/mem0ai/mem0/pull/4297) (@utkarsh240799): a dimension auto-detection
  fix. The self-hosted adapter now routes to the right deployment, though no test yet reproduces
  this specific bug end to end, so this one is partial, not fully caught.

**Zep / Graphiti**
- [#1489](https://github.com/getzep/graphiti/issues/1489) (@brentkearney): a bi-temporal
  `invalid_at` correctness gap. memtrust's contradiction classifier used to discard Graphiti's own
  `invalid_at` metadata and infer everything from a fixed top-5 text match, misreading a correctly
  flagged case as a silent overwrite. It now checks the metadata first.

**OpenViking**
- [#3029](https://github.com/volcengine/OpenViking/issues/3029) (@dfwgj, still open): Feishu resync
  silently deleting user-managed files. memtrust had no way to observe this failure mode at all; a
  dedicated resource-sync-safety eval now seeds generated and user files, triggers a resync, and
  checks what survives.
- [#2850](https://github.com/volcengine/OpenViking/issues/2850) (@lg320531124, still open): BM25
  search silently returning empty results at scale. memtrust now flags an empty result as distinct
  from an ordinary miss, though it doesn't yet attribute the cause or reproduce the scale
  condition, so this one stays partial too.
