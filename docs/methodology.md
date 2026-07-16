# Methodology

This document is the source of truth for how memtrust scores agent-memory backends. If a scoring
decision, a prompt, or a dataset choice is not written down here, it should not be trusted and it
should not ship. Every claim in the README traces back to something on this page.

Last updated: 2026-07-11, alongside the v0.1 release.

## What requires a live vendor API key, and what runs fully offline

This matters because it changes what a number *means*.

| Component | Requires live credentials? | Notes |
|---|---|---|
| `pytest` test suite | No | Every HTTP call is mocked (pytest-httpx). No test ever reaches a real network endpoint. |
| Eval runners against the bundled synthetic fixtures | Yes, one vendor API key per backend under test | `store()`/`query()`/`update()` call the real vendor API. Without a key, the adapter raises `BackendNotConfiguredError` and the CLI reports SKIPPED. |
| LLM-judge scoring (LongMemEval, LoCoMo) | Yes, `MEMTRUST_JUDGE_API_KEY` | Without it, `judge_answer()` returns `JudgeVerdict.NOT_RUN` for every case. The eval still runs (facts get stored and queried against the real backend) but nothing gets graded, and `accuracy` is reported as `None`, not as 0%. |
| Contradiction-detection eval | Yes, one vendor API key | No LLM judge involved -- classification is done by direct substring comparison against the known fixture values (see below), which is cheaper and more auditable than an LLM judge for this specific eval. |
| Compression/round-trip-fidelity eval | Yes, one vendor API key (more if the backend declares more than one `supported_modes` entry) | No LLM judge involved -- fidelity is scored by a direct, deterministic text-similarity ratio against the literal stored content (see below). **Has not been run against any live backend as of this writing.** |
| Leaderboard site (`leaderboard/`) | No | Static HTML reading a checked-in `data.json`. No live calls of any kind. |

**No number in this repo's README or leaderboard was produced by simulating a vendor response.**
Every accuracy or conflict-rate figure comes from either (a) an actual HTTP call to the named
vendor's real API, or is explicitly labeled as not yet measured.

## Dataset versions and what is synthetic vs. real

### LongMemEval

- **Real benchmark:** Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
  Interactive Memory" (ICLR 2025). Public dataset: `xiaowu0162/longmemeval` on Hugging Face.
  Schema confirmed via the dataset card: `question_id`, `question_type` (one of
  `single-session-user`, `single-session-assistant`, `single-session-preference`,
  `temporal-reasoning`, `knowledge-update`, `multi-session`), `question`, `answer`,
  `question_date`, `haystack_session_ids`, `haystack_dates`, `haystack_sessions` (a list of
  sessions, each a list of `{role, content}` turns), `answer_session_ids`.
- **What ships in this repo:** `tests/fixtures/longmemeval_sample.json` -- 3 hand-written examples
  matching that exact schema. The conversations and facts are invented for this repo; none of the
  text is copied from the real dataset. This was a deliberate choice, not a shortcut we're hiding:
  downloading and redistributing the full public dataset was out of scope for this build pass, and
  a small, schema-accurate synthetic sample is what lets `evals/longmemeval.py` be tested fully
  offline and deterministically.
- **To run against the real dataset:** download `xiaowu0162/longmemeval` from Hugging Face,
  convert it to the same top-level `{"examples": [...]}` shape (or write a loader that reads the
  Hugging Face format directly -- `load_dataset()` in `evals/longmemeval.py` is the one function
  that would need a second code path), and pass the path via `run_longmemeval(adapter, judge,
  dataset_path=...)`. This is a documented, contribution-shaped gap -- see CONTRIBUTING.md.

### LoCoMo

- **Real benchmark:** Snap Research's LoCoMo (`snap-research/locomo` on GitHub), distributed as
  `locomo10.json`. Schema confirmed via the repository's own docs: a conversation object with
  `speaker_a`/`speaker_b`, `session_<n>`/`session_<n>_date_time` pairs (each session a list of
  `{speaker, text, dia_id}` turns), and a `qa` list of `{question, answer, category, evidence}`
  entries spanning five categories (single-hop, multi-hop, temporal, open-domain, adversarial).
- **What ships in this repo:** `tests/fixtures/locomo_sample.json` -- 1 hand-written conversation,
  2 sessions, 3 QA pairs across 3 of the 5 real categories (single-hop, temporal, multi-hop). Again,
  invented content matching the real schema, not copied data.
- **To run against the real dataset:** download `locomo10.json` from the LoCoMo repository and
  point `run_locomo(adapter, judge, dataset_path=...)` at it -- the loader expects the same
  top-level `{"conversations": [...]}` shape already.

### MemTrust Contradiction-Detection Eval (original)

- **Not derived from any published dataset.** This is memtrust's own eval, built specifically
  because neither LongMemEval nor LoCoMo tests what happens when a stored fact is contradicted by
  a later one.
- **Fixture:** `tests/fixtures/contradiction_cases.json` -- 5 hand-written cases, each with an
  `initial_fact`, a `contradicting_fact`, a `query`, and the specific `initial_value`/
  `updated_value` substrings the classifier checks for (see Scoring logic below).
- **Design constraint learned the hard way:** the `contradicting_fact` text must not restate the
  old value inside its own correction narrative (e.g. "Priya moved teams, Sam is now the lead"
  restates "Priya"). If it does, a backend that only ever returns the single latest stored string
  will still appear to satisfy both `initial_value` and `updated_value`, which the classifier
  would then score as FLAGGED even though nothing about the backend actually surfaced a conflict.
  This was caught by the test suite (`tests/test_evals.py`) before it shipped and the fixture was
  reworded. Anyone adding a new contradiction case should keep this in mind and phrase corrections
  the way a real user would ("the deadline moved to September 1st"), not by narrating the change
  ("it used to be August 15th but now it's September 1st").
- **Extending this eval:** adding more cases means adding entries to the fixture file with the
  same five fields. No code change is required. See CONTRIBUTING.md.

### MemTrust Compression/Round-Trip-Fidelity Eval (original)

- **Not derived from any published dataset.** Built specifically to test the second overclaim
  mempalace/mempalace#27 documents (see README.md's "Why this exists" section): a "lossless"
  compression claim that measured 12.4 percentage points lower in practice under a compressed
  operating mode. Neither LongMemEval, LoCoMo, nor the contradiction eval measures literal
  reconstruction fidelity -- they measure recall and conflict-handling, not "did the exact text
  survive the round trip."
- **Fixture:** `tests/fixtures/compression_cases.json` -- 5 hand-written cases covering short,
  long/multi-sentence, special-character/unicode, and structured/numeric content, each just a
  `case_id` and a `content` string (see Scoring logic below).
- **Requires the new `mode` parameter.** `MemoryBackendAdapter.store()`/`query()` (see
  `src/memtrust/adapters/base.py`) accept an optional `mode: str | None = None` parameter, and
  `MemoryBackendAdapter.supported_modes` lets an adapter declare which mode strings it actually
  understands. Adapters without mode variants (the default: `supported_modes == ()`) accept and
  ignore the parameter -- a purely additive, backward-compatible change to the shared interface.
  `MemPalaceAdapter.supported_modes` is `("raw", "AAAK")`, the two mode names
  mempalace/mempalace#27 itself uses; those names come from that community issue, not a confirmed
  API parameter in the installed `mempalace` package -- see `mempalace_adapter.py`'s module
  docstring for the full caveat, which follows the same LOW-confidence pattern already documented
  for that adapter's method names below.
- **Extending this eval:** adding more cases means adding entries to the fixture file with a
  `case_id` and `content` field. Adding a mode to an adapter means adding a string to that
  adapter's `supported_modes` tuple and threading `mode` through to the real vendor call -- no
  change to `evals/compression.py` itself is required either way. See CONTRIBUTING.md.

### MemTrust Resource-Sync-Safety Eval (original)

- **Not derived from any published dataset.** Modeled directly on volcengine/OpenViking#3029 (a
  Feishu resync mechanism silently deleting user-owned files an ingestion watcher had not itself
  generated) and, as of this change, additionally shaped to make volcengine/OpenViking#1703
  reachable (`index_resource()` in OpenViking's `embedding_utils.py` skipped every subdirectory
  during reindex, so nested-directory content was never vectorized and searches over it silently
  returned nothing -- reported by GitHub user SonicBotMan).
- **Fixture:** `tests/fixtures/resource_sync_cases.json` -- 4 hand-written cases, each seeding a
  mix of `generated`/`user`-origin files under one resource prefix. Case `mt-resync-004` is the
  one added for this change: its seed files nest 3 real directory levels deep
  (`generated/entities/people/jordan-lee.md`, `user-notes/preferences/user-482/notification-
  settings.md`, `user-notes/preferences/user-482/timezone.md`), mirroring #1703's own
  `entities/people/` and `preferences/{user_id}/` examples, rather than the single-level
  `origin-folder/file.md` shape every earlier case used.
- **Why the earlier fixture couldn't have exercised #1703 at all.** Before this change,
  `OpenVikingAdapter.store()` ignored the `resource_path` metadata key this eval already passed
  it and always wrote to a flat `memory/{session_id}/{sha256(content)[:16]}` path -- a single
  level, regardless of what nested path a seed file's `path_suffix` specified. `store()` now
  honors `resource_path` and writes to that real nested path when supplied (falling back to the
  prior flat-hash behavior when it isn't, so every other eval's calls are unaffected).
  `list_resource_paths()` now does a real recursive tree walk instead of trusting a single flat
  response, so directory entries a listing response reports are actually descended into. See
  `src/memtrust/adapters/openviking_adapter.py`'s module docstring for the full detail.
- **What this closes, precisely, and what it does not.** This change closes the *storage-layer
  precondition*: memtrust can now construct a real nested directory tree against OpenViking (or
  any adapter that honors `resource_path`), which is what makes a directory-indexing bug class
  like #1703 structurally reachable by this harness at all. It does **not** reproduce OpenViking's
  real server-side reindex bug end-to-end -- that requires `trigger_resync()` to hit a live
  OpenViking instance whose actual `index_resource()` skips subdirectories, and no test in this
  repo does that (every HTTP call in `pytest` is mocked; see the table at the top of this
  document). The new `ResourceSyncSignal.NESTED_CONTENT_UNINDEXED` classification and its
  covering unit tests (`tests/test_evals.py::test_resource_sync_detects_nested_content_unindexed_matching_issue_1703`)
  prove the *eval's own classification logic* correctly distinguishes "present on disk but never
  returned by search" from deletion (#3029's signal) or overwrite, against a fake in-memory
  adapter built to model #1703's shape -- they do not, and cannot, prove anything about a real
  OpenViking server's actual reindex code.
- **Extending this eval:** adding more cases means adding entries to the fixture file with the
  same `case_id`/`prefix`/`seed_files` shape. Adding a nested case for another adapter to be
  exercised against requires that adapter's own `store()` to honor `resource_path` metadata the
  same way `OpenVikingAdapter` now does -- no other adapter has been updated to do this as part of
  this change, since `supports_resource_sync` is currently only `True` on `OpenVikingAdapter`. See
  CONTRIBUTING.md.

## Resource-Sync-Safety scoring logic

Implemented in `src/memtrust/evals/resource_sync_safety.py`, function `classify_resource_sync_file()`,
called from `run_resource_sync_eval()`. For every seed file in every case:

1. Store the file via `adapter.store(case.prefix, seed.content, metadata={"resource_path":
   seed.path_suffix, "origin": seed.origin})`.
2. List paths under `case.prefix` before the resync (`paths_before`).
3. Call `adapter.trigger_resync(case.prefix)`.
4. List paths under `case.prefix` again (`paths_after`).
5. If the file's stored path is present in `paths_after`, issue a `query()` for the file's own
   content and check two independent things: whether *any* returned record's `memory_id` equals
   the stored path (`indexed_after_resync`), and whether any returned record's content actually
   contains the seeded content (`content_matches_after_resync`). These are deliberately kept
   separate -- a record existing under the right id with the wrong content (overwrite) is a
   different failure from no record existing under that id at all (never indexed).

Classification (first matching rule wins):

| Condition | Signal |
|---|---|
| Present before, gone after | **DELETED_USER_FILE** -- the volcengine/OpenViking#3029 shape |
| Present after, `indexed_after_resync is False` | **NESTED_CONTENT_UNINDEXED** -- the volcengine/OpenViking#1703 shape: on disk, never searchable |
| Present after, `content_matches_after_resync is False` | **OVERWRITTEN_UNCHANGED** -- a record was found under the right id, but its content had changed |
| Present after, neither of the above | **PRESERVED** |
| Never observed present before the resync | **NOT_APPLICABLE** |

`indexed_after_resync=None` (the default when a caller doesn't distinguish the two questions
above, or when `present_after` is `False` so no query was even attempted) never triggers
`NESTED_CONTENT_UNINDEXED` -- this keeps `classify_resource_sync_file()` backward compatible with
callers that only pass the first three positional arguments.

## Contradiction-detection scoring logic (the eval this project exists for)

Implemented in `src/memtrust/evals/contradiction.py`, function `classify_case()`. For every case:

1. Store `initial_fact` via `adapter.store()`.
2. Store `contradicting_fact` via `adapter.update()` against the same memory.
3. Query with the case's `query` string via `adapter.query()`.
4. Check whether the retrieved content (joined text of every returned record) contains
   `initial_value` and/or `updated_value` as a case-insensitive substring.

Classification:

| Retrieved content contains | Verdict |
|---|---|
| Both `initial_value` and `updated_value` | **FLAGGED** -- the contradiction is visible in the response, whatever the adapter itself reports |
| Only `updated_value` | **SILENT_OVERWRITE** -- the backend resolved the conflict with no trace of the prior value |
| Only `initial_value` | **SERVED_STALE** -- the backend never surfaced the update at all |
| Neither | **NOT_APPLICABLE** -- deferred to the adapter's own reported signal, or recorded as not-applicable if the adapter offered nothing meaningful |

**Why this doesn't just trust the adapter's self-reported signal.** Every adapter's `query()`
method also returns a `ConflictSignal` it derived from vendor-specific evidence (Graphiti's
`invalid_at` timestamp, MemPalace's `invalidated` marker, etc.). The classifier in step 4 above
cross-checks that signal against the literal retrieved text rather than accepting it outright. An
adapter that reports `FLAGGED` while the actual returned content contains neither value is
downgraded to `NOT_APPLICABLE`, not credited with a pass it did not earn. No backend gets a
"verified" claim it cannot support -- that rule applies at the eval-scoring level, not just the
README level.

## Compression/round-trip-fidelity scoring logic

Implemented in `src/memtrust/evals/compression.py`, function `run_compression_eval()`. For every
mode an adapter reports supporting (`adapter.supported_modes`, or a single synthetic `"default"`
mode if that tuple is empty), and for every case in the fixture:

1. Store the case's `content` via `adapter.store(session_id, content, mode=mode)` (`mode=None`
   for the synthetic `"default"` mode).
2. Query for it via `adapter.query(session_id, content, top_k=5, mode=mode)`.
3. Select the retrieved text: prefer the record whose `memory_id` matches what `store()` returned,
   falling back to the top-ranked result -- the shared adapter interface has no get-by-id, only
   `query()`, so this eval retrieves the same way any other caller would.
4. Score `fidelity_ratio(original, retrieved)`: a character-level `difflib.SequenceMatcher.ratio()`
   between the stored and retrieved text. 1.0 means byte-for-byte identical (a genuinely lossless
   round trip); measurably lower values mean measurable reconstruction loss.

**Why this doesn't use the LLM judge.** A "lossless"/compression-ratio claim is a claim about
literal reconstruction fidelity, not semantic equivalence -- an LLM judge might rate a paraphrased,
information-dropping reconstruction as "close enough," which is exactly the leniency this eval
exists to avoid. `fidelity_ratio()` is deterministic, free, and reproducible without any judge
credentials, unlike LongMemEval/LoCoMo's accuracy scores.

**What a real run would show.** `CompressionEvalResult.fidelity_drop_pp` reports the percentage-
point gap between the best- and worst-scoring mode, once at least two modes produce a scoreable
mean -- this is the number that would reproduce a "96.6% raw vs 84.2% AAAK, 12.4pp drop" style
comparison. **No such run has been performed against any live backend as of this writing**; every
number this eval could report is currently unmeasured, the same as every other eval's Benchmarks
section in the README until live credentials are configured.

## LLM-judge prompt template

Used by `src/memtrust/scoring/llm_judge.py` for LongMemEval and LoCoMo (the contradiction eval
does not use an LLM judge -- see above). Exact template, copied verbatim from the source file so
this document cannot drift from what the code actually sends:

```
You are grading whether a memory system's recalled answer is factually equivalent to the expected answer. Ignore differences in phrasing, tense, or extra detail -- grade only whether the core fact matches.

Question asked: {question}
Expected answer: {expected}
System's actual answer: {actual}

Respond with exactly one word on the first line: CORRECT, INCORRECT, or PARTIAL.
On the second line, give a one-sentence reason.
```

- **Default model:** `deepseek-chat`, called via an OpenAI-compatible `/chat/completions` endpoint
  at `https://api.deepseek.com`. Chosen for the v0.1 default because it publishes an OpenAI-
  compatible REST surface and inexpensive per-token pricing, which keeps the harness's own running
  cost low and auditable.
- **Configurable via:** `MEMTRUST_JUDGE_MODEL` (any model name the configured endpoint accepts),
  `MEMTRUST_JUDGE_BASE_URL` (any OpenAI-compatible `/chat/completions` endpoint -- this is how you
  point the judge at Gemini via an OpenAI-compatibility proxy, or at a self-hosted model).
- **No API key configured:** `judge_answer()` returns `JudgeVerdict.NOT_RUN` with an explicit
  reason. It never returns `INCORRECT` or a numeric 0 to represent "could not grade." Every eval
  result's `accuracy` property returns `None`, not `0.0`, when every case is `NOT_RUN` -- callers
  (the CLI, the README, any downstream consumer) must treat `None` as "not measured," never as a
  failing score.

## Adapter confidence levels

Every adapter in `src/memtrust/adapters/` is built against real, cited vendor documentation, but
the confidence level differs by vendor because the public documentation available during this
build differed in how precisely it specified the Python/REST method signatures. This section is
the single place that states, plainly, how much to trust each adapter's exact wire format before
relying on its output.

| Adapter | Confidence | What's confirmed | What's best-effort |
|---|---|---|---|
| `mem0_adapter.py` (`Mem0Adapter`, hosted Platform API) | **High** | `MemoryClient(api_key=...)` reading `MEM0_API_KEY`, `.add()`/`.search()`/`.update()` method names and behavior, confirmed via docs.mem0.ai and the June 2026 SDK v2.0.8 release notes. Mem0's internal ADD/UPDATE/DELETE memory-pipeline decision is documented vendor behavior. | Exact REST path strings (`/v1/memories/`, `/v1/memories/search/`) are a best-effort reconstruction of what the documented SDK wraps, not copied from an OpenAPI spec. |
| `mem0_adapter.py` (`Mem0SelfHostedAdapter`, self-hosted OSS server) | **Medium-High on route shape, Low on live end-to-end behavior** | Route shape (`POST /memories`, `GET /memories`, `PUT`/`DELETE /memories/{id}`, `DELETE /memories`, `POST /search`, `GET /memories/{id}/history`, `POST /reset` -- unprefixed, no `/v1/...`) and request models (`MemoryCreate`, `MemoryUpdate`, `SearchRequest` fields including `filters`, `top_k`, `threshold`, and deprecated top-level `user_id`/`run_id`/`agent_id`) were confirmed by fetching the actual `server/main.py` and `server/auth.py` source from `mem0ai/mem0`'s `main` branch on GitHub during this build (2026-07-11) -- not reconstructed from documentation. No auth by default (`AUTH_DISABLED`), default local port 8888, confirmed via both `server/auth.py` and mem0's own Docker self-hosting guide. | This was never run against a live self-hosted instance in this environment -- no HTTP request in this adapter's test suite reaches a real server. `main` is an unpinned, moving branch that can drift from any specific deployment's actual server version. The exact JSON shape `Memory.search()`/`Memory.add()` return (as opposed to the FastAPI request models, which were confirmed) is reused from the hosted adapter's `{"results": [...]}` parsing, not independently re-verified against this server's response handling. |
| `zep_graphiti_adapter.py` | **Medium-High** | Graphiti's `add_episode()`/`search()` behavior and its bi-temporal `invalid_at` contradiction-handling mechanism are confirmed via Graphiti's own docs and DeepWiki. This is real, documented product behavior, not a memtrust assumption. | Exact REST path strings under `api.getzep.com` are best-effort. The choice to target Zep Cloud's hosted API rather than self-hosted `graphiti-core` + Neo4j is a deliberate scope decision (see below), not an uncertainty. |
| `mempalace_adapter.py` | **Medium on behavior, Low on exact method names** | MemPalace is confirmed local-first, no API key required, SQLite + chromadb backed, and documented as shipping a temporal entity-relationship graph with add/query/invalidate/timeline operations. | The exact Python class and method names (`mempalace.Palace(storage_path=...)`, `.remember()`/`.recall()`/`.invalidate()`) were **not** confirmed against `mempalaceofficial.com/reference/python-api` -- that page was not fetchable during this build. The adapter is written against the documented *concepts*, isolated behind `_get_palace()` so a wrong guess fails with a clear `BackendAPIError` naming the exact assumption, not a confusing `AttributeError` three calls deep. `supported_modes = ("raw", "AAAK")` is the same kind of best-effort assumption: those two names come from mempalace/mempalace#27's community-documented compression-mode claim, not a confirmed `mode` keyword on the real package's `remember()`/`recall()`. **A contributor with access to the real API reference should verify and correct this adapter before treating its output as trustworthy against a live MemPalace instance.** |
| `openviking_adapter.py` | **Medium on architecture, Low on exact memory-write/query paths** | OpenViking's `viking://` virtual-filesystem paradigm, REST server on port 1933, and `OpenViking`/`SyncHTTPClient`/`AsyncHTTPClient` Python client classes are confirmed via the project's own docs. | The documentation fetched during this build covered resource/skill ingestion (`add_resource`, `add_skill`) in detail but did not surface a confirmed endpoint for writing or querying a conversational *memory* entry specifically -- OpenViking's memory layer is described as automatic session-derived extraction, not a direct "store this fact" call. This adapter's `store()`/`query()`/`update()` are written best-effort against the confirmed filesystem paradigm (write a file under a session-scoped `viking://` path, search that path, overwrite on update). `store()` now honors a `resource_path` metadata key to write to a real nested path (falling back to the prior flat content-hash path otherwise), and `list_resource_paths()` now recursively walks directory entries instead of trusting one flat response -- both are still best-effort against the same unconfirmed `/v1/fs/write`/`/v1/fs/list` paths, not newly-confirmed wire format. **This is the adapter most likely to need correction against a live instance.** |

## Read-after-write verification (opt-in, off by default)

`store()` returning without raising `BackendAPIError` has never been proof that a write is
durable or even retrievable. A vendor can return a normal success response while silently
dropping or corrupting the write server-side -- this is not hypothetical: two independently
root-caused MemPalace bug classes did exactly this (checkpoint corruption via NUL bytes in
stored content; stale/self-deadlocked locks that no-op a write). Neither raises an exception.
Left undetected, a silently dropped write just looks like weaker recall on whatever eval touches
it later, and gets misattributed to model quality instead of a backend durability bug.

`MemoryBackendAdapter.verify_store()` (in `src/memtrust/adapters/base.py`) closes that gap: it
issues a `query()` immediately after a `store()` call and checks whether the just-written content
is actually retrievable, reporting the result on `StoreResult.verified` (`True`/`False`). If an
adapter never attempts verification, `StoreResult.verified` stays `None` -- read as "not measured,"
never as an implicit pass, the same rule this document already applies to `JudgeVerdict.NOT_RUN`
and `accuracy=None` above.

**Why this is opt-in, not automatic.** Turning read-after-write verification on unconditionally
for every `store()` call would silently double the number of vendor API calls (and latency) memtrust
itself makes against every backend under test on every eval run. A benchmark harness that quietly
doubled its own cost footprint without the caller asking for it would be its own credibility
problem -- exactly the kind of unstated cost this document exists to surface, not hide. So
verification only runs when a caller explicitly passes `verify=True` to an adapter's `store()`; the
default behavior of every adapter, including MemPalace, is completely unchanged from before this
was added.

**Reference implementation:** `mempalace_adapter.py`'s `store()` accepts a keyword-only
`verify: bool = False` parameter and calls `verify_store()` when `verify=True`, since MemPalace is
the specific vendor whose silent-write bugs motivated this feature. The other three adapters
(`mem0_adapter.py`, `zep_graphiti_adapter.py`, `openviking_adapter.py`) have not been wired up yet --
`verify_store()` is available on the shared base class for any of them to adopt the same way, but
doing so is a separate, adapter-by-adapter contribution, not implied by this change.

None of the above uncertainty is hidden behind a passing test. The adapters' unit tests mock the
adapter's own HTTP layer (or inject a fake object matching the documented interface for MemPalace)
-- they confirm the adapter's *internal logic* is correct given a response shape, not that the
shape itself matches the real vendor's live API. Verifying the wire format against a live vendor
instance is explicitly listed as a first contribution path in CONTRIBUTING.md.

**Why Zep targets the hosted Cloud API, not self-hosted Graphiti.** Self-hosted `graphiti-core`
requires a running graph database (Neo4j or FalkorDB) plus its own LLM credentials for entity
extraction -- there is no single environment variable that gates "is this configured," which
breaks the harness's "one env var, or SKIPPED" contract used by every other adapter. Zep Cloud's
hosted API (`ZEP_API_KEY`) wraps Graphiti and fits that contract. If self-hosted Graphiti support
is wanted later, it should be a second adapter (e.g. `zep_graphiti_selfhosted_adapter.py`) with its
own configuration story, not a silent branch inside this one.

**Why Mem0 has two adapters, `Mem0Adapter` and `Mem0SelfHostedAdapter`, not one with a deployment
flag.** Mem0 ships two materially different deployment shapes: the hosted Platform API
(`api.mem0.ai`, `/v1/...` routes, `MEM0_API_KEY` required) and a self-hosted OSS `server/` FastAPI
wrapper (unprefixed routes, no auth by default, run by the user on their own infrastructure). A
meaningful fraction of the most rigorous, best-evidenced Mem0 bug reports memtrust's outreach
turned up -- entity-id filter scoping (mem0ai/mem0#5973), multi-entity-delete truncation
(#5936/#5970), embedding-dimension mismatch (#4297), and search-threshold inversion (#4453) -- live
entirely in the self-hosted server/`Memory` class code paths that `Mem0Adapter` never talks to and,
before this change, had no configuration surface to reach. Following the same precedent this
document already sets for Zep below ("a second adapter ... with its own configuration story, not a
silent branch inside this one"), `Mem0SelfHostedAdapter` is a separate class, gated on
`MEM0_SELFHOSTED_BASE_URL` (a base URL, not an API key -- the server has no auth by default, the
same reasoning given below for MemPalace's storage-path gate) with an optional
`MEM0_SELFHOSTED_API_KEY` for deployments that do front it with auth.

Of the four bug classes above, this change makes two directly exercisable through the adapter's own
code: `Mem0SelfHostedAdapter.query()` accepts optional `run_id`/`agent_id` parameters, including a
deliberately empty string, and always places them inside the JSON `filters` dict using an
`is not None` check rather than a truthy check -- so a caller's empty string reaches the server
intact instead of being silently dropped the way the server's own deprecated top-level-field merge
path drops falsy values (this asymmetry is the concrete, source-confirmed shape of #5973, described
in full in `mem0_adapter.py`'s module docstring). The same method accepts an optional `threshold`
parameter, passed straight through to the confirmed `SearchRequest.threshold` field, which is what
makes #4453 (threshold inversion) reachable -- `Mem0Adapter` (hosted) has no equivalent parameter.
The other two bug classes become reachable only in the weaker sense that eval traffic now has a
route to a self-hosted instance at all: #4297 (dimension mismatch) lives in self-hosted vector-store
configuration this adapter has no parameter surface to trigger directly, and #5936/#5970
(multi-entity-delete truncation) requires a `delete()` operation that does not exist on
`MemoryBackendAdapter` today -- adding one is a larger interface change than this backlog item
scopes to, and is called out here as unfinished rather than silently left out. See the confidence
table above and `mem0_adapter.py`'s module docstring for exactly what was and was not confirmed
against live source.

**Why MemPalace's "configuration" is a storage path, not an API key.** MemPalace is genuinely
local-first and documented as requiring no API key at all. Forcing it to read a fake API key
env var to match the other three adapters would misrepresent how the product actually works.
Instead, `MEMPALACE_STORAGE_PATH` (the local palace directory) is the value gated on --
`BackendNotConfiguredError` still fires if it's unset, preserving the "SKIPPED, never crashed"
contract, just with a variable name that describes what MemPalace actually needs.

## Cost-tracking pricing table

`src/memtrust/scoring/cost_tracker.py` ships an approximate, dated per-model price table
(`MODEL_PRICING_PER_MILLION_TOKENS`, last verified 2026-07-11) used only to print an estimated
cost alongside a run's output. It is explicitly not a billing guarantee -- provider pricing
changes, and the table should be treated as a cost-awareness aid, not an invoice. If you're running
memtrust against a model not in that table, the tracker falls back to a conservative default price
rather than silently reporting $0.00.

## Vendor-pushback self-check

Before publishing any run's numbers, the honest question is asked and answered here, not skipped:
*if MemPalace's, Mem0's, or Zep's own team read this methodology, could they point to a specific,
defensible flaw?*

As of this writing, the most defensible objection would be: two of the five adapters
(MemPalace, OpenViking) are built against best-effort interpretations of documented concepts
rather than a confirmed API reference, and their output should not be treated as authoritative
until someone verifies the exact wire format against a live instance. That objection is valid,
which is exactly why it's stated plainly in the confidence table above rather than left for a
vendor or a user to discover on their own.

`Mem0SelfHostedAdapter` deserves its own version of the same objection, even though its route
shape was confirmed against real source rather than documentation: it has never been run against a
live self-hosted Mem0 instance in this environment, `main` is a moving target that can drift from
any given deployment, and two of the four bug classes motivating this adapter's addition
(dimension mismatch, multi-entity-delete truncation) are not directly exercised by any code this
adapter adds -- only made reachable in principle by routing traffic at a self-hosted server at all,
and in the delete case, not reachable at all without an interface change this backlog item did not
make. Anyone relying on this adapter to reproduce a specific self-hosted bug report should verify
against a live instance first, not take the source-code read as equivalent to a live-tested
integration.

The same objection applies, in the same shape, to the `NESTED_CONTENT_UNINDEXED` signal added to
the resource-sync-safety eval for volcengine/OpenViking#1703. What this change actually verified:
`OpenVikingAdapter.store()` now constructs a real nested directory tree when given a
`resource_path`, `list_resource_paths()` now really walks it recursively, and
`classify_resource_sync_file()` correctly separates "never indexed" from "deleted" and
"overwritten" against a fake adapter purpose-built to model #1703's shape (see
`tests/test_evals.py::NestedIndexSkipFakeAdapter`). What it did not verify: that OpenViking's real,
live server actually has this bug, still has this bug as of any date after #1703 was reported, or
would produce exactly this signal if run against this eval today. Nobody should read a
`NESTED_CONTENT_UNINDEXED` result in a report generated by this repo as confirmation that a live
OpenViking instance is currently affected -- it confirms only that memtrust's storage layer and
scoring logic are now capable of detecting that failure mode *if* a live instance exhibits it.
Confirming the live bug itself requires running `resource_sync_safety` against a real, running
OpenViking server with `OPENVIKING_API_KEY` configured, which this build pass did not do.
