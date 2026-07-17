# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-07-17

### Added

- `ZepGraphitiSelfHostedAdapter` (`adapters/zep_graphiti_selfhosted_adapter.py`) -- a second,
  separately-configured Zep/Graphiti adapter (`GRAPHITI_NEO4J_URI` or `GRAPHITI_FALKORDB_URL`)
  that instantiates `graphiti_core.Graphiti` directly in-process, reaching internal
  graphiti-core bugs (getzep/graphiti#1302, #836, #1013, #1001) the existing Zep-Cloud REST
  adapter can never see. Adds `MemoryRecord.attributes` so structured per-record properties
  survive the adapter boundary, and `ConflictSignal.EDGE_INTEGRITY_VIOLATION` for edge records
  with a missing `source_node_uuid`/`target_node_uuid`.
- `Mem0DirectAdapter` (`adapters/mem0_direct_adapter.py`) -- a direct, in-process `mem0.Memory`
  handle via `Memory.from_config()`, reaching mem0's construction-time `graph_store`/`embedder`/
  `vector_store` config surface that the REST-only `Mem0Adapter`/`Mem0SelfHostedAdapter` cannot.
  Built and tested against the real installed `mem0ai==2.0.12` package: confirms mem0ai/mem0
  #5671, #4362, #4711, and #2304 are fixed in that release, and that #3558 (Kuzu) cannot be
  reproduced because the installed package has no `graph_store` field or kuzu dependency at all.
  Adds `CorruptionSignal` (`CONFIG_REJECTED`/`VECTOR_ZEROED`/`CLEAN`/`NOT_APPLICABLE`) and an
  optional `mem0-direct` dependency group. Opt-in only, registered as `mem0_direct` in
  `ADAPTER_REGISTRY`, not in `cli.ALL_BACKENDS`.
  - `custom_instructions` passthrough constructor argument, threading a caller-supplied
    fact-extraction prompt into `MemoryConfig` (mem0ai/mem0#4573's junk-retention finding;
    `custom_instructions` is the real top-level key after mem0 renamed it away from
    `custom_fact_extraction_prompt` in mem0ai/mem0#4740).
  - Qdrant support (`vector_store_provider="qdrant"`) and a `query(threshold=...)` parameter
    forwarded to `Memory.search()`, giving mem0ai/mem0#4297 (embedding-dimension mismatch,
    confirmed still reachable in the installed package) and #4453 (search-threshold inversion,
    confirmed fixed) a real construction-time surface to reach.
  - Elasticsearch vector-store support (host/api_key/embedding-dims threading) plus a new
    `filter_injection` eval and `FilterInjectionSignal` taxonomy (`FILTER_REJECTED`/
    `FILTER_ACCEPTED_SAFELY`/`INJECTION_SUCCEEDED`/`NOT_APPLICABLE`), built on
    `probe_raw_filter()`/`RawFilterProbeResult` (`adapters/base.py`), which submits an
    adversarial filter dict directly to a backend's filter-building layer. Confirms the
    installed `mem0ai==2.0.12` already carries the `_validate_filter()` fix from
    mem0ai/mem0#5980.
- Ed25519-signed receipts for `memtrust run` output (`src/memtrust/receipt.py`): canonical JSON
  encoding, Ed25519 sign/verify, PEM keypair I/O, and three new CLI commands (`memtrust keygen`,
  `memtrust run --sign <keyfile>`, `memtrust verify <receipt.json>`). Signing is opt-in and off
  by default; unsigned `memtrust run` output is unchanged.
- Crash-recovery eval (`evals/crash_recovery.py`, `crash_stress` capability flags in
  `adapters/base.py`) modeling volcengine/OpenViking#2644's silent index-rebuild skip on
  restart, with `CrashRecoverySignal` (`RECOVERED`/`INDEX_LOST_DATA_SURVIVED`/`DATA_LOST`/
  `NOT_APPLICABLE`). Built at the harness level against a fake adapter; no adapter in this repo
  has real process-lifecycle control over a live backend.
- Embedding-drift/consistency eval (`evals/embedding_drift.py`) for volcengine/OpenViking#1523's
  in-place vector overwrite during an embedder migration, with `EmbeddingDriftSignal`
  (`EMBEDDING_DRIFT`/`CLEAN`/`NOT_APPLICABLE`).
- Extraction-quality-at-scale eval (`evals/extraction_quality.py`) modeled on a real 32-day mem0
  audit (mem0ai/mem0#4573, jamebobob: 97.8% of 10,134 stored entries were junk) plus a
  documented feedback-loop case (one hallucinated memory re-extracted into 808 duplicate
  stores). Adds `ExtractionQualitySignal` (`RETAINED_JUNK`/`REJECTED_JUNK`/`RETAINED_VALID`/
  `LOST_VALID`/`FEEDBACK_LOOP_DUPLICATE`/`NO_UNEXPECTED_GROWTH`/`NOT_APPLICABLE`).
- Scale/volume stress-testing eval (`evals/scale_stress.py`, `evals/scale_fixtures.py`,
  `scale_stress`) for volcengine/OpenViking#2850 (BM25 search going silently empty at volume)
  and getzep/graphiti#1275 (O(n) entity-resolution growth silently dropping old episodes), with
  a deterministic large-scale synthetic corpus generator and a `--scale-stress-n-records` CLI
  flag.
- Migration-rollback-safety eval (`evals/migration_rollback.py`) verifying the concept behind
  MemPalace's real rename-aside swap fix (mempalace/mempalace#1028, PR#935) for an unguarded
  `shutil.rmtree()`-then-`shutil.move()` migration swap that could permanently lose data on a
  partial failure. Adds `MigrationRollbackSignal` (`RESTORED`/`DATA_LOST`/`NOT_APPLICABLE`); no
  real adapter sets the new capability flag since none has filesystem control over a live
  `migrate()` call.
- New crash-classification signals in `adapters/base.py`'s `CrashSignal`: `UNPACK_ERROR` and
  `TYPE_COMPARISON_ERROR` for graphiti-core's `store()` exceptions (getzep/graphiti#836's
  tuple-unpack `ValueError`, #920's tz-naive/tz-aware `TypeError`), and
  `QUERY_SANITIZATION_ERROR` for FalkorDB RediSearch syntax errors in `query()`
  (getzep/graphiti#1222, superseded by #1475; #1183, merged).
- MemPalace degraded-retrieval signal: `RetrievalWarning` and `QueryResult.degraded_retrieval`
  (`adapters/base.py`), surfacing MemPalace's real merged `search_memories()` fix
  (mempalace/mempalace#1005) that degrades vector-query failures into a response carrying
  warnings and partial results instead of raising -- a failure mode
  `ConflictSignal.EMPTY_OR_LOST` could not see.
- MemPalace `authored_at` ranking tie-breaker: `_RANKING_METADATA_KEYS` and query-result parsing
  now recognize `authored_at` (top-level or nested under `metadata`) as a ranking-driving field,
  matching MemPalace's real merged PR#1890/#1889.
- Retrieval-graded vs. generated-answer accuracy disclosure in `docs/methodology.md`: memtrust's
  LongMemEval/LoCoMo accuracy metric grades raw retrieved-record content directly, with no
  answer-generation step, so it is not the same measurement as the official leaderboards'
  generate-then-judge QA-accuracy scores (closes mempalace/mempalace#367).

### Fixed

- MemPalace adapter docstring no longer conflates "no API key required" with "no network
  required" -- `mempalace mine .` can still fail offline because chromadb's default embedder
  downloads its ONNX model on first use (mempalace/mempalace#524). LongMemEval now flags cases
  where `top_k` already covers the whole corpus (`top_k_exceeds_corpus`), so a small haystack
  can no longer read as artificially high recall.
- Report table width corrected for 12-column output as new evals were added.
- OpenViking adapter's `BackendAPIError` now reads the real HTTP response body
  (`exc.response.text`) instead of only `httpx`'s status line, across `store`/`query`/`update`/
  `delete`/`list_resource_paths`/`trigger_resync` -- volcengine/OpenViking#1227's server-side
  Pydantic validation detail was previously swallowed down to a useless status-line-only
  message.

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
