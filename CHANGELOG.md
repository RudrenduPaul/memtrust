# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.2] - 2026-07-16

### Added

- `npm/` -- an unpublished npm-distributable CLI wrapper (`npx memtrust ...`) for CI and agent
  runners that have Node.js but not necessarily a Python toolchain. Six per-platform optional
  packages (`@memtrust/darwin-arm64`, `darwin-x64`, `linux-arm64`, `linux-x64`, `win32-arm64`,
  `win32-x64`) each bundle a genuine, SHA-256-verified copy of Astral's `uv` binary
  (github.com/astral-sh/uv, dual-licensed MIT OR Apache-2.0), fetched from uv's own GitHub
  release 0.11.28 at npm package-publish time via a `prepack` script, never at end-user install
  time. The `memtrust` bin shim runs `uv tool run --from memtrust memtrust <args>`, which
  bootstraps a Python interpreter and installs `memtrust` from PyPI on first use. Not yet
  published to npm -- gated on a separate publish step. See the README's "npx (agent-native)"
  section and `npm/` for the wrapper source and third-party attribution.

### Added

- `StoreResult.extraction_signal` (`ExtractionSignal`, `adapters/base.py`) -- flags when a `store()`
  call completes without raising but the response carried no usable memory id, the exact
  mem0ai/mem0#5178 "store() succeeded but silently extracted zero facts" shape. `Mem0Adapter`,
  `Mem0SelfHostedAdapter`, and `Mem0DirectAdapter` all now set `FACTS_EXTRACTED`/`EMPTY_EXTRACTION`
  instead of silently returning a normal-looking `StoreResult` with `memory_id=""`. See
  docs/methodology.md's "ExtractionSignal and mem0ai/mem0#5178" section.

### Fixed

- PyPI `project.urls` now link each author to their GitHub profile instead of leaving the `Author` field email-less with no way to reach either maintainer

## [0.1.0] - 2026-07-11

Initial release.

### Added

- Shared `MemoryBackendAdapter` interface (`store()`/`query()`/`update()`) in
  `src/memtrust/adapters/base.py`, plus `ConflictSignal` classification used by the
  contradiction-detection eval.
- Four backend adapters: MemPalace, Mem0, Zep/Graphiti, OpenViking. Each reads its configuration
  from a single environment variable and raises `BackendNotConfiguredError` (never crashes) when
  it's missing. Confidence level per adapter documented in `docs/methodology.md`.
- Three eval runners: LongMemEval-style long-horizon recall, LoCoMo-style multi-session recall,
  and memtrust's original multi-hop contradiction-detection eval (flagged / silently-overwrote /
  served-stale / not-applicable).
- LLM-judge scoring pipeline (`scoring/llm_judge.py`), model-configurable via environment
  variables, with a no-crash `NOT_RUN` fallback when no judge API key is configured.
- Cost tracker (`scoring/cost_tracker.py`) with a dated, approximate per-model pricing table.
- `memtrust run` and `memtrust report` CLI commands.
- Static leaderboard site (`leaderboard/index.html` + `leaderboard/data.json`) with a documented
  schema, shipped as an example rather than fabricated live results.
- Full test suite: 57 tests, 95% overall coverage, 100% on `adapters/base.py`,
  `evals/contradiction.py`, `evals/longmemeval.py`, and `scoring/`. All tests run fully offline.
- CI workflow: lint (ruff), type-check (mypy --strict), test (pytest + coverage across Python
  3.11-3.13), security (pip-audit).
- `docs/methodology.md`, `CONTRIBUTING.md`, `SECURITY.md`.

### Known limitations (v0.1)

- Adapters for MemPalace and OpenViking are built against best-effort interpretations of
  documented product concepts, not a confirmed API reference -- see the confidence table in
  `docs/methodology.md`. They should be verified against a live instance before their output is
  treated as authoritative.
- LongMemEval and LoCoMo eval runners ship against small, explicitly synthetic sample fixtures
  matching each benchmark's real published schema, not the full public datasets.
- No live benchmark numbers are published in the README -- running the harness against real
  backends requires vendor API keys not available at the time of this release. See the README's
  "Benchmarks" section for exactly what was and wasn't measured.
