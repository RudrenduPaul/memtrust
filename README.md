# memtrust

Standardized, reproducible benchmarks for agent-memory backends, run against the vendors, not
published by them.

[![CI](https://github.com/RudrenduPaul/memtrust/actions/workflows/ci.yml/badge.svg)](https://github.com/RudrenduPaul/memtrust/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/RudrenduPaul/memtrust/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/RudrenduPaul/memtrust/blob/main/pyproject.toml)
[![PyPI](https://img.shields.io/badge/pypi-memtrust-blue.svg)](https://pypi.org/project/memtrust/)

```bash
pip install memtrust
memtrust run --backends mempalace,mem0,zep,openviking --eval all
```

(For contributing to this repo instead of just running it, see [Development](#development) --
`pip install -e ".[dev]"` from a clone.)

**Contents:** [Why this exists](#why-this-exists) · [What it does](#what-it-does) ·
[Commands](#commands) · [How this differs](#how-this-differs-from-trusting-a-vendors-own-numbers) ·
[Contradiction detection](#the-eval-that-actually-matters-contradiction-detection) ·
[Compression fidelity](#the-eval-built-for-the-other-headline-overclaim-compression-fidelity) ·
[Temporal-KG boundary](#the-eval-built-from-mempalaces-own-bug-temporal-kg-boundary-detection) ·
[The landscape](#the-landscape-verified-not-benchmarked) · [Benchmarks](#benchmarks) ·
[GitHub Actions usage](#github-actions-usage) · [Self-host](#self-host) · [Install](#install) ·
[Hosted layer](#what-a-hosted-trust-layer-would-add) · [Backend coverage](#backend-coverage) ·
[Development](#development) · [License](#license) · [Success stories](#success-stories)

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

memtrust does that. It runs LongMemEval, LoCoMo, and a growing set of evals built specifically for
this project -- 14 of them as of this writing, registered in the CLI's `--eval` flag, plus a
fifteenth (temporal-KG boundary detection) that exists as a tested module but isn't wired into the
CLI yet. The two that matter most for understanding what this project is actually for:
contradiction detection, because neither LongMemEval nor LoCoMo tests the question that actually
matters once a memory system sits underneath a production agent -- what happens when a new fact
contradicts an old one? Does the backend flag the conflict? Silently overwrite the old fact with no
audit trail? Serve whichever version it happens to retrieve first? None of the four backends this
project tracks publish a number for that. And compression/round-trip fidelity, built to directly
test claims like the "lossless" one above: it stores content, retrieves it, and scores literal
reconstruction fidelity rather than semantic accuracy, per operating mode a backend exposes (see
`MemoryBackendAdapter.supported_modes`) -- the mechanism that would let a contributor with live
MemPalace credentials actually reproduce the 12.4-point compressed-mode accuracy drop
mempalace/mempalace#27 documents, instead of just citing it. **Neither has been run against any
live backend as of this writing** -- see `docs/methodology.md`. The other evals -- ranking quality,
crash recovery, extraction quality, embedding drift, scale/volume stress, lock contention, stats
accuracy, orphan cleanup, result consistency, migration rollback, filter injection, resource-sync
safety, and temporal-KG boundary detection -- each grew out of a specific real bug report against
one of the four tracked backends; see "Success stories" below for the full list.

## What it does

Every command below was actually run against this repo, with zero vendor API keys configured, to
produce the output shown. Nothing here is simulated.

```
$ memtrust run --backends mempalace,mem0,zep,openviking --eval all
memtrust 0.1.0 -- run_id=mt_2026-07-17T155402Z
Backends: mempalace, mem0, zep, openviking   Evals: longmemeval, locomo, contradiction,
resource_sync_safety, compression, ranking_quality, scale_stress, embedding_drift, crash_recovery,
extraction_quality, migration_rollback, filter_injection, lock_contention, stats_accuracy,
orphan_cleanup, result_consistency

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

Full report: memtrust-report-2026-07-17.json
```

That's the real, reproducible behavior of a fresh clone with no credentials: every backend reports
SKIPPED, the command exits cleanly, and a valid JSON report is still written. `memtrust --version`
currently prints `0.1.0` even though the package installs as `0.2.0` on PyPI -- a stale hardcoded
string in `src/memtrust/__init__.py` that hasn't caught up with `pyproject.toml`, shown here exactly
as it actually runs rather than smoothed over. Set the relevant environment variable for any backend
you want to actually test (`MEM0_API_KEY`, `ZEP_API_KEY`, `OPENVIKING_API_KEY`,
`MEMPALACE_STORAGE_PATH`) and that backend runs for real against its live API instead of being
skipped.

The eval logic itself is proven offline, against the bundled synthetic fixtures and, for several
adapters, the real installed vendor packages with only the network boundary mocked, by the test
suite:

```
$ pytest --cov=memtrust --cov-report=term-missing
...
Name                                                          Stmts   Miss  Cover
-------------------------------------------------------------------------------------
src/memtrust/adapters/base.py                                   290      1    99%
src/memtrust/adapters/mempalace_adapter.py                      265     15    94%
src/memtrust/adapters/mem0_adapter.py                            140     12    91%
src/memtrust/adapters/mem0_direct_adapter.py                     281     34    88%
src/memtrust/adapters/openviking_adapter.py                      178     18    90%
src/memtrust/adapters/zep_graphiti_adapter.py                     63      3    95%
src/memtrust/adapters/zep_graphiti_selfhosted_adapter.py         165     24    85%
src/memtrust/evals/contradiction.py                              127      2    98%
src/memtrust/evals/compression.py                                 86      1    99%
src/memtrust/evals/temporal_kg_boundary.py                        90      4    96%
src/memtrust/receipt.py                                          118     10    92%
-------------------------------------------------------------------------------------
TOTAL                                                            4130    256    94%

580 passed, 8 skipped in 3.31s
```

580 passing tests across 33 source modules, 94% overall statement coverage, 98% on the
contradiction-detection eval, 99% on compression/round-trip fidelity, 96% on the new temporal-KG
boundary eval, 94-99% across the adapter layer. The 8 skips are live-`mempalace`-package tests that
only run with the optional `mempalace-direct` extra installed (`pip install -e
'.[dev,mempalace-direct]'`). Every test mocks its HTTP or wire
layer, or uses an in-memory fake backend -- none of them touch a real network, though a meaningful
share of the adapter tests now import and exercise *real installed vendor classes* directly
(`mem0ai==2.0.12`'s embedder and vector-store modules, and -- gated behind the optional
`mempalace-direct` extra -- the real `mempalace.mcp_server` functions), mocking only the outermost
network or wire-client boundary rather than the whole library. `graphiti-core` is not installed in
this environment, so its self-hosted adapter's tests still run against a hand-written Protocol
double built to match the real package's confirmed method signatures, not the real classes -- see
`docs/methodology.md`'s adapter confidence table for exactly which claim rests on which kind of
verification.

## Commands

```
$ memtrust --help
Usage: memtrust [OPTIONS] COMMAND [ARGS]...

  memtrust: an independent, reproducible benchmark harness for agent-memory
  backends.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  keygen  Generate a new Ed25519 keypair for signing `memtrust run`...
  report  Read a prior `memtrust run` JSON report and print a formatted...
  run     Run the eval suite against the requested backends.
  verify  Verify a signed receipt produced by `memtrust run --sign`.
```

| Command | Flags | What it does |
|---|---|---|
| `memtrust run` | `--backends TEXT` comma-separated list or `all` (default `all`) · `--eval TEXT` comma-separated from `longmemeval,locomo,contradiction,resource_sync_safety,compression,ranking_quality,scale_stress,embedding_drift,crash_recovery,extraction_quality,migration_rollback,filter_injection,lock_contention,stats_accuracy,orphan_cleanup,result_consistency`, or `all` (default `all`) · `--output FILE` (defaults to `./memtrust-report-<date>.json`) · `--locomo-dataset-path FILE` points the LoCoMo eval at a real, downloaded `locomo10.json` instead of the bundled synthetic fixture (memtrust does not bundle or auto-fetch the real dataset) · `--locomo-exclude-question-ids-file FILE` excludes known-bad-ground-truth LoCoMo question IDs from scoring · `--scale-stress-n-records INTEGER` (default `500`) sets how many synthetic records the scale-stress eval stores and re-queries · `--sign FILE` writes a signed `<output>.receipt.json` alongside the report, proving it was produced by the holder of the given Ed25519 private key | Runs the eval suite against the requested backends. A backend without its credential env var set prints `SKIPPED` and the run continues -- this command never crashes on missing credentials. |
| `memtrust report REPORT_PATH` | positional path to a prior JSON report | Reads a report written by `memtrust run` and prints a formatted summary. |
| `memtrust keygen` | -- | Generates a new Ed25519 keypair for signing reports with `run --sign`. |
| `memtrust verify RECEIPT_PATH` | -- | Verifies a signed receipt produced by `memtrust run --sign`; a tampered or mismatched receipt fails verification. |
| `memtrust --version` | -- | Prints the installed version. Currently prints `0.1.0`, one release behind the `0.2.0` this package actually installs as -- see the note above. |

The temporal-KG boundary eval (`src/memtrust/evals/temporal_kg_boundary.py`,
`run_temporal_kg_boundary_eval()`) is not yet wired into `--eval`'s known-eval list -- it exists as
a tested module and a MemPalace-specific capability, callable directly, not yet a `memtrust run
--eval temporal_kg_boundary` option. See its own section below.

Every line above came straight from running `memtrust --help`, `memtrust run --help`, and
`memtrust report --help` against this repo. Nothing here is invented.

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

## The landscape (verified, not benchmarked)

Real, publicly checkable numbers as of this writing (`gh api repos/<org>/<repo>`), not
memtrust-run scores -- accuracy and contradiction-handling comparisons stay in the "Benchmarks"
section below until a live run actually produces them:

| Backend | GitHub stars | Self-reported description |
|---|---|---|
| [MemPalace](https://github.com/MemPalace/mempalace) | 57,268 | "The best-benchmarked open-source AI memory system. And it's free." |
| [Mem0](https://github.com/mem0ai/mem0) | 60,688 | "Universal memory layer for AI Agents" |
| [Zep / Graphiti](https://github.com/getzep/graphiti) | 28,648 | "Build Real-Time Knowledge Graphs for AI Agents" |
| [OpenViking](https://github.com/volcengine/OpenViking) | 26,639 | "Self-evolving Context Database for AI Agents. Unify Agent Memory, Knowledge RAG and Skills." |

None of these numbers say anything about which backend handles a contradicted fact correctly --
that's the whole reason the harness exists. Star count measures adoption, not correctness.

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

## The eval built from MemPalace's own bug: temporal-KG boundary detection

MemPalace/mempalace#1913 (fixed by merged PR#1914, contributor ggettert) described a real,
concrete bug: `_temporal_filter_sql`'s `as_of` point-in-time query used a closed interval on both
ends, so a fact whose `valid_to` equaled the query's exact `as_of` instant still matched. Hand-roll
a fact change as `kg_invalidate(ended=T)` immediately followed by `kg_add(valid_from=T)` at the
identical boundary instant -- the exact pattern MemPalace's own pre-fix agent guidance told every
caller to do -- and an `as_of=T` query returns both the just-ended fact and its just-started
successor at once, so a single-valued fact reports two contradictory answers with no error.
`src/memtrust/evals/temporal_kg_boundary.py` reproduces that exact hand-rolled sequence against
`MemPalaceAdapter`'s `kg_add()`/`kg_invalidate()`/`kg_query()` and classifies the result with a new
`TemporalBoundarySignal` taxonomy, distinct from `ConflictSignal` and `RankingSignal` because it
concerns one narrow, structurally different failure: two facts sharing one instant, not a
contradiction across time or a ranking-order question.

Honest scope, stated the same way this project states it for every other eval: the real
`mempalace` PyPI package is not installed in this build environment, and PR#1914's fix had not
shipped in a released `mempalace` version as of this adapter's live-verified 3.5.0 build -- it
lands under the package's `[Unreleased]` changelog section. `tests/test_temporal_kg_boundary.py`
proves the *classification logic* is correct against two hand-written fake implementations that
reproduce the confirmed pre-#1914 (closed-interval) and post-#1914 (half-open-interval) SQL
comparison exactly. **This has not been run against a live MemPalace instance.** It also isn't
wired into `memtrust run --eval` yet -- call `run_temporal_kg_boundary_eval()` directly, or see the
test suite, until that CLI surface exists.

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
report, and prints an estimated cost for any LLM-judged evals that ran. `MemPalaceAdapter`'s
drawer and knowledge-graph calls are now live-verified against a real installed instance (see
"Backend coverage" below), but OpenViking's memory-write/query paths, and parts of the
self-hosted Mem0 and Zep/Graphiti adapters, are still built against best-effort interpretations of
documented or source-read product concepts rather than a live-confirmed API -- see the confidence
table in `docs/methodology.md` before treating any adapter's output as authoritative, and consider
that table's gaps a standing invitation to contribute a fix.

**Labeling requirement for any future `accuracy` figure published here.** LongMemEval and LoCoMo
`accuracy` grades the LLM judge's verdict on raw retrieved-record content directly -- there is no
answer-generation step in either eval runner. This is not the same measurement as the official
LongMemEval/LoCoMo leaderboards' generate-then-judge QA-accuracy scores. Any `accuracy` number
this project publishes for those two evals must be labeled "retrieval-graded accuracy," not bare
"accuracy," and must not be directly compared to leaderboard figures without that caveat. See
`docs/methodology.md`'s "Retrieval-graded accuracy vs. generated-answer accuracy" section.

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

> **PyPI is live; the npm wrapper is publish-ready but not yet published.** `pip install memtrust`
> works today -- version 0.2.0 is on PyPI. The `memtrust-cli` npm package in `npm/memtrust-cli/` in
> this repo is built, tested, and passes both `npm pack --dry-run` and `npm publish --dry-run`
> cleanly (re-verified this session), but has not actually been pushed to the npm registry yet --
> `npm install memtrust-cli` still 404s as of this writing. The command below is the real,
> working distribution path once that publish happens; nothing about it is speculative except the
> "it's live on npm" part.

For CI and agent runners that have Node.js available but not necessarily a Python toolchain:

```bash
npx memtrust-cli run --backends mempalace,mem0,zep,openviking --eval all
```

The npm package is named `memtrust-cli` so it is unambiguous as a CLI tool at a glance (and so it
doesn't collide with any future `memtrust` JS library package). `npx` always resolves the package
name to its matching `bin` entry automatically, so `npx memtrust-cli ...` is what reliably works
for a zero-install first run. Once installed, the package also exposes the shorter `memtrust`
command as a second `bin` alias -- matching the underlying Python CLI's own command name -- so you
are not stuck typing `memtrust-cli` for every subsequent invocation; `memtrust run ...` works too.

This is not a zero-dependency install: `npx memtrust-cli` still fetches `memtrust` from PyPI on
first use. What it removes is no Python toolchain to provision by hand -- `npx memtrust-cli`
handles the interpreter and package fetch for you via a bundled, verified copy of Astral's
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

The MemPalace row below used to say "needs verification against a live instance" -- it needed more
than that. Every prior version of `MemPalaceAdapter` called a `mempalace.Palace` class
(`Palace(storage_path=...)` exposing `.remember()`/`.recall()`/`.invalidate()`) that never existed
in the real, installed package. `python3 -c "import mempalace; hasattr(mempalace, 'Palace')"`
returns `False`; grepping every `class` definition across the installed package turns up nothing
named `Palace` anywhere. Every test that appeared to pass before this rewrite was exercising a
hand-written fake standing in for that guess, never the real thing -- `store()`/`query()`/
`update()` had never actually worked against a live MemPalace install, in this project's entire
history, until this rewrite. `src/memtrust/adapters/mempalace_adapter.py` was rewritten from
scratch against the real, plain module-level functions in `mempalace.mcp_server`
(`tool_add_drawer`, `tool_search`, `tool_update_drawer`, `tool_delete_drawer`,
`tool_kg_add`/`tool_kg_invalidate`/`tool_kg_query`) -- every return shape documented in the
adapter's module docstring was captured by calling those functions live against a real, local
chromadb-backed palace, not read off a docstring and trusted. It's the kind of mistake this whole
project exists to catch in other people's benchmarks; finding it in memtrust's own adapter and
shipping the fix in the open, rather than quietly patching it, is the more useful story.

| Backend | Adapter status | Confidence (see docs/methodology.md) |
|---|---|---|
| MemPalace | Implemented -- drawer API + knowledge-graph API | High on the real `mempalace.mcp_server` functions this adapter now calls, live-verified against an installed `mempalace` 3.5.0 instance (see above). Still best-effort on compression-mode names (`"raw"`/`"AAAK"`) and on whether `degraded_retrieval` warnings are ever populated by the installed version -- see the adapter's module docstring for both caveats stated plainly. |
| Mem0 | Implemented -- hosted Platform API, self-hosted OSS server, and a direct in-process library adapter | High on the hosted Platform API and on what the installed `mem0ai==2.0.12` library's embedder/vector-store code actually does (confirmed by reading its real source, exercised directly in tests); medium-high on the self-hosted OSS server's route shape (confirmed from source, not run against a live server). |
| Zep / Graphiti | Implemented -- hosted Zep Platform API and a self-hosted `graphiti-core` adapter | Medium-high on the hosted API's documented contradiction-handling behavior; medium on the self-hosted adapter's wire-level shape (every method signature confirmed by reading `graphiti-core`'s real source, not by running it against a live Neo4j/FalkorDB instance -- the package isn't installed in this environment). |
| OpenViking | Implemented | Medium on architecture, low on exact memory-write/query paths -- still the adapter most likely to need correction against a live instance. |

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

197 real issues/PRs filed by real contributors against MemPalace, mem0, Zep/Graphiti, and
OpenViking have been independently root-caused against this codebase: does the solution, as it
actually exists today, let you diagnose or resolve what was reported? 55 (28%) verify as a clean
PASS, 16 (8%) as PARTIAL (evidence captured, needs a human to
interpret further, or only part of the issue is covered), 42 (21%) as a genuine capability gap this
harness doesn't close yet, and 84 (43%) as not actually applicable (feature requests,
already-fixed-upstream, or genuinely out of scope). Every verdict below has been re-verified live
against the current codebase by a reviewer independent of whoever built the fix, not just cited
from a changelog. Full write-ups and validation evidence are tracked internally; the summary here
is for anyone deciding whether this harness would have caught their own bug.

**The headline story is about memtrust's own bug, not a vendor's.** Every version of
`MemPalaceAdapter` before this rewrite called a `mempalace.Palace` class -- `Palace(storage_path=
...)` exposing `.remember()`/`.recall()`/`.invalidate()` -- that never existed in the real,
installed package. `python3 -c "import mempalace; hasattr(mempalace, 'Palace')"` returns `False`;
nothing named `Palace` appears anywhere in the installed package's source. Every test that appeared
to pass was exercising a hand-written fake standing in for that guess -- `store()`/`query()`/
`update()` had never once worked against a live MemPalace install. `src/memtrust/adapters/
mempalace_adapter.py` was rewritten against the real `mempalace.mcp_server` functions, with every
documented return shape captured by calling them live against a real local instance. A project
built to catch other vendors overclaiming found the same failure mode in its own code, and the fix
shipped in the open rather than quietly. See "Backend coverage" above for the full account.

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
- [#1913](https://github.com/MemPalace/mempalace/issues/1913) / [PR#1914](https://github.com/MemPalace/mempalace/pull/1914)
  (@ggettert): a temporal-KG `as_of` boundary bug where a fact ending at exactly the query instant
  still matched alongside its successor. memtrust's new temporal-KG boundary eval reproduces the
  exact hand-rolled `kg_invalidate()`-then-`kg_add()` sequence that triggers it -- see "The eval
  built from MemPalace's own bug" above for the honest not-yet-live-verified caveat.
- [PR#1890](https://github.com/MemPalace/mempalace/pull/1890) / [#1889](https://github.com/MemPalace/mempalace/issues/1889)
  (@JosefAschauer): an `authored_at` chronology tie-break fix for `_hybrid_rank`. memtrust's ranking
  classifier now credits a top-level `authored_at` field, not just one nested under `metadata`, as
  a genuine ranking-driving signal.
- [#524](https://github.com/MemPalace/mempalace/issues/524) (@gaby): a buried-comment report that
  "no API key required" doesn't mean "no network required" -- ChromaDB's default embedder still
  needs to download a model on first use, which silently breaks airgapped setups. The adapter's
  module docstring now says so explicitly.

**mem0**
- [#5973](https://github.com/mem0ai/mem0/pull/5973) (@abhay-codes07, superseded by
  [#5992](https://github.com/mem0ai/mem0/pull/5992)): an empty-string entity-id filter scoping bug.
  memtrust's mem0 adapter only reached the hosted Platform API and had no delete operation at all,
  so it couldn't have caught this. A self-hosted adapter with tested delete/delete_many primitives
  now can.
- [#4297](https://github.com/mem0ai/mem0/pull/4297) (@utkarsh240799): a dimension auto-detection
  fix. The self-hosted adapter now routes to the right deployment, though no test yet reproduces
  this specific bug end to end, so this one is partial, not fully caught.
- [#4573](https://github.com/mem0ai/mem0/issues/4573) (@jamebobob): a 32-day audit of 10,134 real
  mem0 entries finding 97.8% junk. memtrust's new extraction-quality eval and
  `ExtractionQualitySignal` taxonomy cover the audit's own junk categories, including its
  808-duplicate feedback-loop case.
- [PR#5980](https://github.com/mem0ai/mem0/pull/5980) (@HrushiYadav): a filter-injection fix for
  the Elasticsearch vector store. A new filter-injection eval exercises the real, installed
  `mem0.vector_stores.elasticsearch.ElasticsearchDB._validate_filter()` directly and confirms it
  rejects the exact malicious filter shape (`{"user_id": {"$ne": ""}}`) this PR fixed.
- [#4956](https://github.com/mem0ai/mem0/issues/4956) (@NDNM1408): an open proposal that mem0's
  add-only pipeline surfaces stale, contradictory facts with no recency signal. memtrust's
  contradiction eval now runs the same literal add-only scenario (two `store()` calls, no explicit
  update) against this taxonomy.
- [#4884](https://github.com/mem0ai/mem0/issues/4884) (@wangjiawei-vegetable): a hardcoded
  English-only tokenizer silently degrading non-Latin-script retrieval. A new
  `LanguageDegradationSignal` and non-Latin-script fixtures now catch this shape.

**Zep / Graphiti**
- [#1489](https://github.com/getzep/graphiti/issues/1489) (@brentkearney): a bi-temporal
  `invalid_at` correctness gap. memtrust's contradiction classifier used to discard Graphiti's own
  `invalid_at` metadata and infer everything from a fixed top-5 text match, misreading a correctly
  flagged case as a silent overwrite. It now checks the metadata first.
- [#1275](https://github.com/getzep/graphiti/issues/1275) (@rafaelreis-r, still open): O(n)
  entity-resolution context growth silently dropping episodes past roughly 300 ingested. A new
  self-hosted `graphiti-core` adapter plus a scale/volume-stress eval now tracks a fixed "anchor"
  record's recall across ascending checkpoints against real `add_episode()` ingestion -- the same
  shape this issue describes.
- [#836](https://github.com/getzep/graphiti/issues/836) (@matthiaslau) / [#920](https://github.com/getzep/graphiti/issues/920)
  (@markwkiehl): two separate crashes in `update_communities()`/`resolve_edge_contradictions()` --
  a too-many-values-to-unpack error and a tz-naive/aware datetime comparison error. A new
  `CrashSignal` classification recognizes both exact shapes instead of surfacing an opaque generic
  exception.
- [PR#1222](https://github.com/getzep/graphiti/pull/1222) (@david-morales) / [PR#1183](https://github.com/getzep/graphiti/pull/1183)
  (@Milofax): FalkorDB RediSearch syntax errors from empty or unescaped fulltext queries. A new
  `CrashSignal.QUERY_SANITIZATION_ERROR` recognizes both issues' verbatim filed error text.
- [#1467](https://github.com/getzep/graphiti/issues/1467) (@elimydlarz, open, zero engagement):
  `GeminiEmbedder` silently returning the wrong vector count. A new
  `CrashSignal.EMBEDDING_BATCH_COUNT_MISMATCH` catches this once Gemini embedder support is wired
  into the self-hosted adapter.

**OpenViking**
- [#3029](https://github.com/volcengine/OpenViking/issues/3029) (@dfwgj, still open): Feishu resync
  silently deleting user-managed files. memtrust had no way to observe this failure mode at all; a
  dedicated resource-sync-safety eval now seeds generated and user files, triggers a resync, and
  checks what survives.
- [#2850](https://github.com/volcengine/OpenViking/issues/2850) (@lg320531124, still open): BM25
  search silently returning empty results at scale. A dedicated scale/volume-stress eval
  (`memtrust run --eval scale_stress`) now stores a large synthetic corpus and re-queries it at
  ascending checkpoints to reproduce the *shape* of this condition -- recall collapsing past a
  volume threshold with no exception raised.
- [#1581](https://github.com/volcengine/OpenViking/issues/1581) (@0xble, fix rejected, still live
  upstream): `v2_lock_max_retries=0` silently means unlimited retries, not zero. A new
  lock-contention eval asserts a bounded response-time budget under concurrent-write contention.
- [#1255](https://github.com/volcengine/OpenViking/issues/1255) (@SeeYangZhi): a stats endpoint
  silently returning zero despite persisted memories. A new `get_stats()`/`StatsResult` primitive
  and dedicated stats-accuracy eval now catch this.
- [#2966](https://github.com/volcengine/OpenViking/issues/2966) (@lRoccoon, unaddressed upstream):
  legacy uint16-truncated records that are permanently undeletable. A new
  `CrashSignal.LEGACY_CORRUPT_RECORD_UNDELETABLE` now surfaces this instead of a silent no-op.
- [#204](https://github.com/volcengine/OpenViking/issues/204) (@ponsde, closed): non-deterministic
  search results (Jaccard similarity 0.11 across identical queries) from a self-diagnosed dimension
  mismatch. A new result-consistency eval computes pairwise Jaccard similarity over repeated
  identical queries to catch this class directly.

**Cross-project**
- [OneNomad-LLC/przm-bench](https://github.com/OneNomad-LLC/przm-bench) (@mattstvartak): a peer
  benchmarking project shipped cryptographic receipt signing; memtrust had none. `memtrust` now has
  real Ed25519 signing/verification (the `cryptography` library, not a hand-rolled scheme) via
  `memtrust keygen` / `run --sign` / `verify` -- a tampered receipt correctly fails verification,
  a genuine one correctly passes.

Several PARTIAL and FAIL -- capability gap rows above and elsewhere in the full 197-row set remain
open, deliberately not counted as fixed: some point at real gaps this harness genuinely can't close
yet without a live vendor credential, and inflating a near-miss to PASS defeats the entire point of
an independently-verified benchmark. See the confidence caveats throughout this README and in
`docs/methodology.md` for exactly which claims rest on which kind of evidence.
