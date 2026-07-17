# Methodology

This document is the source of truth for how memtrust scores agent-memory backends. If a scoring
decision, a prompt, or a dataset choice is not written down here, it should not be trusted and it
should not ship. Every claim in the README traces back to something on this page.

Last updated: 2026-07-16, adding the Scale/Volume-Stress eval.

## What requires a live vendor API key, and what runs fully offline

This matters because it changes what a number *means*.

| Component | Requires live credentials? | Notes |
|---|---|---|
| `pytest` test suite | No | Every HTTP call is mocked (pytest-httpx). No test ever reaches a real network endpoint. `tests/test_mem0_direct_adapter.py` additionally requires the optional `mem0-direct` dependency group (`pip install -e ".[dev,mem0-direct]"`) to exercise the real, installed `mem0ai` package's embedder/vector-store classes -- still fully offline (only the vendor SDK/wire-client boundary is mocked), but the test *file* skips cleanly with an explained reason if that group isn't installed, rather than failing collection. CI installs it. |
| Eval runners against the bundled synthetic fixtures | Yes, one vendor API key per backend under test | `store()`/`query()`/`update()` call the real vendor API. Without a key, the adapter raises `BackendNotConfiguredError` and the CLI reports SKIPPED. |
| LLM-judge scoring (LongMemEval, LoCoMo) | Yes, `MEMTRUST_JUDGE_API_KEY` | Without it, `judge_answer()` returns `JudgeVerdict.NOT_RUN` for every case. The eval still runs (facts get stored and queried against the real backend) but nothing gets graded, and `accuracy` is reported as `None`, not as 0%. |
| Contradiction-detection eval | Yes, one vendor API key | No LLM judge involved -- classification is done by direct substring comparison against the known fixture values (see below), which is cheaper and more auditable than an LLM judge for this specific eval. |
| Compression/round-trip-fidelity eval | Yes, one vendor API key (more if the backend declares more than one `supported_modes` entry) | No LLM judge involved -- fidelity is scored by a direct, deterministic text-similarity ratio against the literal stored content (see below). **Has not been run against any live backend as of this writing.** |
| Ranking-Quality eval | Yes, one vendor API key | No LLM judge involved -- classification is a direct comparison of returned record order against per-record metadata values and known insertion order (see below). **Has not been run against any live backend as of this writing.** |
| Scale/Volume-Stress eval | Yes, one vendor API key, to reach a real backend at all -- but see the honest limitation below, this has only ever been run against fake in-memory adapters so far | No LLM judge involved -- classification is a direct comparison of re-query recall at small vs. large synthetic-corpus checkpoints (see below). **Has not been run against any live backend at real scale (10K+ records / 300+ episodes) as of this writing; `pytest`'s coverage is entirely against fake adapters engineered to model the two motivating bug shapes.** |
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
  The published dataset totals 1,986 questions: 1,540 "regular" questions across the first four
  categories, plus a 446-question adversarial category 5 -- deliberately unanswerable questions
  that test whether a backend admits it doesn't know rather than fabricating a confident answer.
- **What ships in this repo:** `tests/fixtures/locomo_sample.json` -- 1 hand-written conversation,
  2 sessions, 4 QA pairs across all 4 non-adversarial categories represented by 3 cases
  (single-hop, temporal, multi-hop) plus 1 adversarial case. Again, invented content matching the
  real schema, not copied data.
- **To run against the real dataset:** download `locomo10.json` from the LoCoMo repository and
  point `run_locomo(adapter, judge, dataset_path=...)` at it -- the loader expects the same
  top-level `{"conversations": [...]}` shape already.

#### Headline accuracy vs. non-adversarial accuracy (category-5 exclusion)

This gap was raised by an independent audit (`dial481/locomo-audit`, referenced from
mempalace/mempalace#29 and #875): a benchmark's headline accuracy number, if it silently folds
the 446-question adversarial category 5 in with the 1,540 regular questions, is not the same
measurement the LoCoMo paper itself reports, and a reader comparing numbers across vendors has no
way to tell whether adversarial questions were included unless the harness makes the distinction
explicit.

`LoCoMoResult` reports two accuracy numbers, both always computed, neither hidden behind extra
API surface a caller has to know to ask for:

- **`accuracy`** -- every graded case, all categories included (adversarial included). This is
  the property's original meaning; nothing that already reads `.accuracy` changes behavior.
- **`non_adversarial_accuracy`** -- the same computation restricted to categories other than
  `"adversarial"`, mirroring the real benchmark's own 1,540/446 split. This is the number that is
  directly comparable to a vendor's claim that excludes category 5.

Both numbers are surfaced everywhere `accuracy` already was: `memtrust run`'s console output
prints both lines (`accuracy (all categories, incl. adversarial)` and `non_adversarial_accuracy
(excludes category 5)`), the JSON report's `locomo` block carries both keys, and `memtrust
report`'s table renders both as `all / non-adversarial` in the LoCoMo column. `accuracy_by_category()`
already existed and still gives the full per-category breakdown, including `"adversarial"` on its
own -- `non_adversarial_accuracy` is the additive, headline-visible version of "exclude that one
category," not a replacement for the finer-grained breakdown.

**What this does not claim to fix.** memtrust cannot make a vendor disclose which number they
reported; it can only make sure memtrust's own numbers, and any report generated from a memtrust
run, never blend the two without saying so.

#### Known-bad ground-truth exclusion

The same audit separately catalogued 99 ground-truth labeling errors in the released dataset --
a different problem from the category-5 blending above: even the 1,540 "regular" questions
include some where the published expected answer is itself wrong, which would unfairly penalize
a backend that answered correctly against the *actual* conversation content.

`run_locomo(adapter, judge, dataset_path=..., exclude_question_ids=...)` accepts an optional set
of question IDs to exclude from scoring entirely -- the case is still recorded (so `n_cases` stays
honest and a reader can see it was excluded, not silently dropped) via
`LoCoMoCaseResult.excluded_ground_truth`, but it is never queried, never judged, and never counted
toward `accuracy`, `non_adversarial_accuracy`, or `accuracy_by_category()`.

**This repo does not ship dial481's specific 99 question IDs.** They were not independently
verified against his published audit data during this change, and hardcoding an unverified list
into memtrust's default scoring would be exactly the kind of unstated, unauditable adjustment this
document exists to prevent. The mechanism is real and pluggable; the specific list is left for
whoever runs memtrust against the real dataset to supply, once they have a verified corrected list.

To use it:

1. Build (or obtain) a verified list of known-bad question IDs. The published LoCoMo schema has
   no `question_id` field, so `run_locomo()` derives one per case as
   `f"{conversation_id}::{index_in_conversation}"` (or uses `qa["question_id"]` directly if the
   dataset provides it) -- a corrected list must use IDs in that same shape.
2. Load it with `load_exclude_question_ids(path)` (`src/memtrust/evals/locomo.py`), which accepts
   either a JSON array of ID strings or a plain-text file with one ID per line (`#`-prefixed lines
   ignored, for inline annotation of why an ID is excluded).
3. Pass the result as `exclude_question_ids=...` to `run_locomo()`, or point `memtrust run` at the
   file directly with `--locomo-exclude-question-ids-file <path>`.

### MemTrust Contradiction-Detection Eval (original)

- **Not derived from any published dataset.** This is memtrust's own eval, built specifically
  because neither LongMemEval nor LoCoMo tests what happens when a stored fact is contradicted by
  a later one.
- **Fixture:** `tests/fixtures/contradiction_cases.json` -- 7 hand-written cases, each with an
  `initial_fact`, a `contradicting_fact`, a `query`, and the specific `initial_value`/
  `updated_value` substrings the classifier checks for (see Scoring logic below). Cases 6 and 7
  were added alongside `ZepGraphitiSelfHostedAdapter` (2026-07-16): case 6's query
  (`"What is the status of order ORTAND-88?"`) deliberately contains every uppercase letter
  (O, R, N, T, A, D) that getzep/graphiti#1302's `lucene_sanitize()` mis-escapes
  character-by-character instead of only escaping the `AND`/`OR`/`NOT` boolean operators it's
  meant to target -- this cannot demonstrate the bug against any adapter in this repo today (it
  lives entirely inside self-hosted graphiti-core's internal search pipeline, which no adapter's
  own code can intercept), but gives a contributor running the self-hosted adapter against a live
  instance a ready-made query to compare BM25 ranking on. Case 7 carries an optional `metadata`
  field (`ContradictionCase.metadata`, threaded into `adapter.store()`) with non-fact structured
  key/value pairs, exercising `MemoryRecord.attributes` end-to-end -- see
  `zep_graphiti_selfhosted_adapter.py`'s module docstring for why that specific adapter accepts
  and ignores this metadata rather than surfacing it back out (graphiti-core's real
  `add_episode()` has no generic metadata parameter to receive it).
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

### MemTrust Ranking-Quality Eval (original)

- **Not derived from any published dataset.** Built specifically to close a gap that
  `ConflictSignal` (see below) structurally cannot see: whether returned *content* is correct is a
  different question from whether it came back in the *right order*, and a backend can be
  perfectly correct on the first axis while silently broken on the second. Neither LongMemEval,
  LoCoMo, the contradiction eval, nor the compression eval measures ordering at all.
- **Origin: mempalace/mempalace#1733** (GitHub user Kartalops, found while validating memtrust
  against real MemPalace usage, not a synthetic scenario invented for this repo).
  `mempalace/layers.py`'s `Layer1.generate()` sorts drawers by `importance`/`emotional_weight`/
  `weight`, but no ingest path in the real package ever writes those keys -- confirmed 0/45,969
  drawers on a real palace. `importance` silently defaults to a constant, so the documented "high
  importance, recent" `wake-up` ordering degenerates to plain insertion order (oldest moments
  first) with zero errors raised anywhere in the pipeline. Every individual returned drawer is
  itself a real, uncorrupted memory -- there is no contradiction anywhere in this bug, which is
  exactly why it was invisible to `ConflictSignal`.
- **Fixture:** `tests/fixtures/ranking_quality_cases.json` -- 4 hand-written cases, each a
  `session_id`, `query`, `ranking_field` (which metadata key this case tests, e.g. `"importance"`),
  and a list of `records` to store in order (`content` + `metadata`). Case `mt-rank-001` and
  `mt-rank-004` reproduce the #1733 shape directly (constant value, and field never written at
  all, respectively); `mt-rank-002` is the negative control (genuinely varied values); `mt-rank-003`
  models a backend with a real varying signal that still isn't used to order results.
- **RankingSignal is a distinct taxonomy from ConflictSignal, not a variant of it.** Defined in
  `src/memtrust/adapters/base.py` alongside `ConflictSignal`. `QueryResult.ranking_signal` follows
  the exact same "adapter self-reports, eval never blindly trusts it" convention `conflict_signal`
  established: `MemPalaceAdapter.query()` (the one adapter with a documented, sort-relevant
  metadata field) computes a coarse self-report via `_classify_ranking_signal()`, and
  `evals/ranking_quality.py`'s `classify_ranking_case()` derives the actual scored signal from
  ground truth (the case's known insertion order and per-record field values) rather than trusting
  that self-report outright.
- **Extending this eval:** adding more cases means adding entries to the fixture file with the
  same fields described above. Wiring a new adapter into detection means implementing its own
  `_classify_ranking_signal`-equivalent and setting `ranking_signal` on the `QueryResult` it
  returns -- adapters that don't implement this simply keep the field's default,
  `RankingSignal.NOT_APPLICABLE`, and are scored the same way `ConflictSignal.NOT_APPLICABLE`
  scores an adapter with `supports_update = False`: recorded explicitly, never silently dropped.
  See CONTRIBUTING.md.

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

### MemTrust Scale/Volume-Stress Eval (original)

- **Not derived from any published dataset.** Every other eval in this package (contradiction,
  ranking-quality, resource-sync-safety, compression) runs against a hand-written fixture of
  4-7 cases -- fine for exercising correctness logic, but structurally incapable of reaching the
  corpus size at which two real, documented, still-open vendor reports actually manifest:
  **volcengine/OpenViking#2850** (lg320531124) -- BM25 search silently returning empty results
  once a corpus grows large -- and **getzep/graphiti#1275** (rafaelreis-r) -- O(n)
  entity-resolution context growth silently dropping episodes once ingestion passes roughly 300
  episodes. Both are the same underlying gap: nothing in this repo, before this change, ever
  stored more than a handful of records, so neither bug shape had anywhere to show up.
- **Fixture generator:** `src/memtrust/evals/scale_fixtures.py`'s `generate_scale_corpus(n,
  seed=...)`, not a checked-in file. A hand-typed fixture cannot reach 10,000+ realistic-looking
  records; a deterministic generator (seeded `random.Random`) can, reproducibly, for any `n` a
  caller asks for. Every generated record embeds a unique, greppable marker token
  (`SCALEMARK{index:06d}`) in fact-shaped prose, so "is this specific record still recoverable"
  reduces to a literal substring check -- the same shape of query BM25-style lexical search
  actually serves, and the shape #2850 concerns (a literal keyword search coming back empty).
- **Eval:** `src/memtrust/evals/scale_stress.py`'s `run_scale_stress_eval()`. Stores `n_records`
  synthetic records incrementally and, at a small ascending set of checkpoints (default
  `[5, n//10, n//2, n]` for a typical `n=500` run: `[5, 50, 250, 500]`), re-queries by marker
  token to measure recall as a function of corpus size, in two independent shapes: a fixed
  **anchor** record (the very first one ever stored -- becoming unrecoverable as volume grows is
  the #1275 shape, old content silently evicted) and a **sample** spread across everything stored
  so far (recall collapsing as volume grows, independent of which record, is the #2850 shape,
  search itself degrading). `ScaleTestResult`/`ScaleSignal` (`WORKED_AT_SCALE`,
  `SILENTLY_DEGRADED_AT_SCALE`, `PARTIAL_DEGRADATION`, `ERROR`, `NOT_APPLICABLE`) follow the same
  ground-truth-driven, never-trust-"didn't raise" convention every other eval's signal enum in
  this package already follows.
- **What this closes, precisely, and what it does not.** This change closes the *volume
  precondition*: memtrust can now generate and store an arbitrarily large synthetic corpus against
  any adapter and measure recall as corpus size grows, which is what makes a scale-dependent bug
  class like #2850 or #1275 structurally reachable by this harness at all. It does **not**
  reproduce either vendor's real production behavior end-to-end -- that requires
  `run_scale_stress_eval()` to be pointed at a live, credentialed OpenViking or Graphiti backend
  with `n_records` in the thousands (and, for #1275 specifically, at least ~300 real episodes
  through Graphiti's actual `add_episode()` path, not memtrust's generic `store()`), and no test in
  this repo does that. `tests/test_scale_stress.py`'s three purpose-built fake adapters
  (`ScaleCleanFakeAdapter` as the negative control; `ScaleEmptyAtVolumeFakeAdapter` modeling
  #2850's "search goes silently empty past a threshold" shape; `ScaleEvictsOldFakeAdapter`
  modeling #1275's "oldest content silently falls out of the search window" shape) prove the
  *eval's own classification logic* correctly tells scale-invariant recall apart from both
  degradation shapes -- they do not, and cannot, prove that OpenViking's or Graphiti's real
  production systems currently exhibit either bug.
- **Extending this eval:** pointing a real run at the scale the motivating bugs need means passing
  a much larger `--scale-stress-n-records` to `memtrust run` (e.g. `10000`) against a real,
  credentialed backend -- the harness itself places no cap beyond
  `scale_fixtures.generate_scale_corpus`'s 999,999-record limit (a fixed-width marker format, not
  a scale judgment). See CONTRIBUTING.md.

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
| Any retrieved record is edge-shaped (`raw` carries both a `source_node_uuid` and `target_node_uuid` key) but at least one is missing/falsy | **EDGE_INTEGRITY_VIOLATION** -- checked before the value-level rules below; see `ConflictSignal.EDGE_INTEGRITY_VIOLATION` in `adapters/base.py` for the two real graphiti-core bugs (getzep/graphiti#1013, #1001) this is built to catch if reproduced against an affected version |
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

## Ranking-quality scoring logic

Implemented in `src/memtrust/evals/ranking_quality.py`, function `classify_ranking_case()`. For
every case:

1. Store each of the case's `records` in order via `adapter.store()`, keeping the returned
   `memory_id`s in the exact order stored -- this is the eval's ground truth for "insertion order."
2. Query once via `adapter.query(session_id, query, top_k=len(records))`.
3. Read the case's `ranking_field` (e.g. `"importance"`) off each returned record's metadata,
   parsed as a float where present.

Classification:

| Observed evidence | Verdict |
|---|---|
| `ranking_field` missing from at least one returned record, or present on all but carrying the identical value everywhere | **MISSING_ORDERING_KEY** -- no real per-record signal exists to have driven the order, whatever the backend's documentation claims |
| `ranking_field` present and genuinely varied, and the returned order is sorted by descending value | **SIGNAL_DRIVEN** -- a real signal exists and the backend appears to actually use it |
| `ranking_field` present and genuinely varied, but the returned order does NOT correlate with it | **ORDER_INCONSISTENT** -- a real signal exists and the backend isn't ordering by it, a distinct bug from MISSING_ORDERING_KEY |
| Fewer than 2 returned records | **NOT_APPLICABLE** -- nothing to compare an ordering claim against |

**Why this doesn't just trust the adapter's self-reported signal.** `QueryResult.ranking_signal`
is a coarse, adapter-derived claim (see `MemPalaceAdapter._classify_ranking_signal()` -- it can say
"this field is present and varies" but cannot itself confirm the returned order actually
correlates with it, because it has no access to any case's ground-truth insertion order).
`classify_ranking_case()` recomputes the real verdict from the case's own known insertion order and
field values, and stores the adapter's self-report separately (`adapter_reported_signal`) purely
for comparison -- exactly the same non-negotiable rule `evals/contradiction.py`'s `classify_case`
applies to `conflict_signal`.

**Honest limitation -- read this before trusting a MISSING_ORDERING_KEY number.** This eval, run
purely against a live backend's query responses, can only ever prove one thing: *no real
per-record signal was observed driving this response's order.* It cannot always distinguish two
different underlying causes that produce the identical observable symptom:

  * the backend genuinely has nothing meaningful to rank by for this particular query (every
    candidate really is equally important), versus
  * the backend forgot to populate the ranking field at all (mempalace/mempalace#1733's actual
    root cause -- 0/45,969 drawers on a real palace ever got a real `importance` value).

`MISSING_ORDERING_KEY` is named for what is actually detected (absence of a driving signal), not
for a claim about why. Kartalops's #1733 finding is the strong form of this -- direct inspection of
a live palace's write path, not a black-box query-response inference -- and this eval's
query-response-only view would, on its own, only ever justify the weaker claim above. A
`MISSING_ORDERING_KEY` result is a strong prompt to go verify the stronger claim by inspecting the
backend's actual ingest/write path (as #1733 did), not proof of it by itself.

## Scale/volume stress scoring logic

Implemented in `src/memtrust/evals/scale_stress.py`, function `classify_scale_result()`, called
from `run_scale_stress_eval()`. For a run of `n_records`:

1. Generate `n_records` deterministic synthetic records via `scale_fixtures.generate_scale_corpus()`.
2. Store them one at a time via `adapter.store()`, tracking which indices actually succeeded
   (a `BackendAPIError` on an individual `store()` call is counted, not fatal to the run).
3. At each checkpoint size (default `[5, n//10, n//2, n]`), re-query for the anchor record
   (index 0) plus a fixed-size sample (`SAMPLE_SIZE_PER_CHECKPOINT=5`) of everything stored so
   far, by each record's unique marker token, and check whether the marker is actually present
   in the joined text of the returned records -- never trusts "`query()` didn't raise" as proof a
   record came back, the same rule every other eval in this package applies.
4. Compute `recall_rate` per checkpoint, `recall_degradation_pct` (first scoreable checkpoint's
   recall minus the last, in percentage points), and `anchor_lost_at_n` (the smallest checkpoint
   at which a previously-recoverable anchor stopped being recoverable).

Classification (first matching rule wins):

| Condition | Signal |
|---|---|
| Fewer than 2 checkpoints produced a scoreable recall rate | **NOT_APPLICABLE** -- nothing to compare "small scale" against "large scale" with |
| Anchor record recoverable at an earlier checkpoint, unrecoverable at a later one | **SILENTLY_DEGRADED_AT_SCALE** -- the getzep/graphiti#1275 shape: old content silently evicted as volume grows |
| Sample recall dropped by >= `DEGRADATION_THRESHOLD_PP` (15pp) between the first and last scoreable checkpoint | **SILENTLY_DEGRADED_AT_SCALE** -- the volcengine/OpenViking#2850 shape: search itself degrades at volume |
| Final-checkpoint recall below `MIN_ACCEPTABLE_FINAL_RECALL` (0.9), but not via a scale-correlated drop | **PARTIAL_DEGRADATION** -- recall is genuinely incomplete, but not distinguishably volume-triggered (could be an ordinary indexing miss present at every scale) |
| Otherwise | **WORKED_AT_SCALE** |

Every `store()`/`query()` call that raises `BackendAPIError` is caught and counted rather than
crashing the run; if every single `store()` call fails, the run is classified `ERROR` (a distinct,
more informative outcome than the generic `NOT_APPLICABLE` "not enough data" bucket -- see
`ScaleSignal.ERROR`'s docstring).

**Honest limitation -- read this before trusting a `SILENTLY_DEGRADED_AT_SCALE` (or
`WORKED_AT_SCALE`) result.** As stated above and in `scale_stress.py`'s own module docstring, this
eval has only ever been exercised in this build against fake, in-memory adapters purpose-built to
either degrade at a hard-coded threshold or scale cleanly. It proves the *classification logic*
correctly separates those two shapes given real recall measurements. It does not, on its own, say
anything about whether a live OpenViking or Graphiti deployment currently exhibits #2850 or #1275
-- that requires an actual `memtrust run --backends openviking --eval scale_stress
--scale-stress-n-records 10000` (or the Graphiti equivalent, driven through real episode
ingestion) against a real, credentialed instance, which this build did not do.

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
| `mem0_direct_adapter.py` (`Mem0DirectAdapter`, direct in-process library, embedder/vector-store selection) | **High on what the installed `mem0ai==2.0.12` package's code actually does (confirmed by reading its real, installed source, not documentation or the GitHub issues), Low on live end-to-end behavior** | Every embedder-dims-forwarding and vector-store vector=None-guard claim below was confirmed by reading the *installed* `mem0.embeddings.{aws_bedrock,fastembed,gemini,openai}.py` and `mem0.vector_stores.{redis,valkey}.py` source directly, and `tests/test_mem0_direct_adapter.py` exercises those real classes with only the vendor SDK/wire-client boundary mocked (`boto3`, `openai`, `google.genai`, `fastembed.TextEmbedding`, `redis`/`valkey`) -- see "Mem0DirectAdapter and the retired Kuzu bug" below for the full finding, including the one surprising negative result (`graph_store`/Kuzu support does not exist in this package version at all). | Never run against a live Redis/Valkey server, a live Bedrock/Gemini/OpenAI embedding endpoint, or a live FastEmbed model download in this environment -- every test mocks the vendor boundary, same convention this document already states for `mem0_adapter.py`. `_read_raw_embedding_bytes()`'s raw-client inspection (used by `update_metadata_only()` to derive `CorruptionSignal.CLEAN`/`VECTOR_ZEROED`) reaches into each vector store's private-ish `schema`/`prefix`/`client` attributes rather than a documented public API, since `VectorStoreBase.get()` does not expose raw vector bytes at all -- confirmed by reading the base class, but still an internals-reaching workaround that could break on a future `mem0ai` release's internal refactor (it would fail closed, into `NOT_APPLICABLE`, not silently misreport). |
| `zep_graphiti_adapter.py` | **Medium-High** | Graphiti's `add_episode()`/`search()` behavior and its bi-temporal `invalid_at` contradiction-handling mechanism are confirmed via Graphiti's own docs and DeepWiki. This is real, documented product behavior, not a memtrust assumption. | Exact REST path strings under `api.getzep.com` are best-effort. The choice to target Zep Cloud's hosted API rather than self-hosted `graphiti-core` + Neo4j is a deliberate scope decision (see below), not an uncertainty. |
| `zep_graphiti_selfhosted_adapter.py` (`ZepGraphitiSelfHostedAdapter`, self-hosted `graphiti-core`) | **Medium on wire-level shape, Low on live end-to-end behavior** | Every `graphiti_core` constructor/method signature this adapter calls (`Graphiti(uri=, user=, password=)`, `Graphiti(graph_driver=)`, `FalkorDriver(host=, port=, username=, password=)`, `add_episode(name=, episode_body=, source_description=, reference_time=, group_id=, update_communities=)` returning an `AddEpisodeResults` with an `.episode.uuid`, `search(query, group_ids=, num_results=)` returning `list[EntityEdge]` directly, `remove_episode(episode_uuid)`) was confirmed by fetching the real source files from `getzep/graphiti`'s `main` branch on GitHub (`raw.githubusercontent.com/getzep/graphiti/main/...`) on 2026-07-16 and reading them directly -- not reconstructed from documentation. The four bug citations this adapter's module docstring makes (getzep/graphiti#1302, #836, #1013, #1001) were confirmed the same way: #1302's per-character `O`/`R`/`N`/`T`/`A`/`D` escape-map entries and #836's `communities, community_edges = await semaphore_gather(...)` unpack-of-a-list-of-2-tuples were read verbatim out of `helpers.py`/`graphiti.py`/`community_operations.py` on `main`; #1013's fix (`SET e = edge` replacing an enumerated field list in the Neo4j bulk edge-save query -- Neo4j is the default case in `get_entity_edge_save_bulk_query()`; FalkorDB's own branch of the same function uses the equivalent `SET r = edge`) and #1001's closure (FalkorDB's old `add_triplet()` no longer exists anywhere in the rewritten `falkordb_driver.py`) were confirmed the same way. | The real `graphiti-core` package is not installed in this build environment, and no Neo4j or FalkorDB instance was started or reached during this build -- every signature above was confirmed by reading source, never by importing and calling the real package or a live database. This adapter's own unit tests (`tests/test_adapters.py`) mock a `graphiti_core`-shaped Protocol double, the same convention `mempalace_adapter.py`'s `_PalaceProtocol` already establishes. The `update_communities` toggle is confirmed to thread through to `add_episode()` (and is unit-tested doing so), but nothing in this build can demonstrate it actually triggers #836's `ValueError` without a live instance and real LLM credentials driving entity extraction. `lucene_sanitize()` (#1302) is internal to graphiti-core's search pipeline and is not called anywhere in this adapter's own code -- fixture case `mt-contra-006` (see below) only sets up a query a contributor with a live instance could use to observe the ranking degradation directly, it does not reproduce the bug in this repo's own test suite. `EDGE_INTEGRITY_VIOLATION` (#1013/#1001) is checked at the harness level in `evals/contradiction.py`, but both underlying bugs are confirmed fixed/closed on the `graphiti-core` version this adapter was built against, so a live run today should not actually trigger it. |
| `mempalace_adapter.py` | **Medium on behavior, Low on exact method names** | MemPalace is confirmed local-first, no API key required, SQLite + chromadb backed, and documented as shipping a temporal entity-relationship graph with add/query/invalidate/timeline operations. | The exact Python class and method names (`mempalace.Palace(storage_path=...)`, `.remember()`/`.recall()`/`.invalidate()`) were **not** confirmed against `mempalaceofficial.com/reference/python-api` -- that page was not fetchable during this build. The adapter is written against the documented *concepts*, isolated behind `_get_palace()` so a wrong guess fails with a clear `BackendAPIError` naming the exact assumption, not a confusing `AttributeError` three calls deep. `supported_modes = ("raw", "AAAK")` is the same kind of best-effort assumption: those two names come from mempalace/mempalace#27's community-documented compression-mode claim, not a confirmed `mode` keyword on the real package's `remember()`/`recall()`. **A contributor with access to the real API reference should verify and correct this adapter before treating its output as trustworthy against a live MemPalace instance.** The `importance`/`emotional_weight`/`weight` metadata keys `_classify_ranking_signal()` checks (see the Ranking-Quality eval above) are the same LOW-confidence category: they come from mempalace/mempalace#1733's own root-cause report on `layers.py`, not a confirmed field-name reference for what `recall()`'s response `metadata` actually contains on a live instance. |
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
hosted API (`ZEP_API_KEY`) wraps Graphiti and fits that contract.

**Self-hosted Graphiti support (`ZepGraphitiSelfHostedAdapter`, added 2026-07-16).** Following the
precedent stated above, this is a second adapter, `zep_graphiti_selfhosted_adapter.py`, with its own
configuration story (`GRAPHITI_NEO4J_URI` or `GRAPHITI_FALKORDB_URL`, not `ZEP_API_KEY`) rather than
a branch inside `ZepGraphitiAdapter`. It exists specifically because four real, independently-verified
graphiti-core bug reports -- getzep/graphiti#1302 (`lucene_sanitize()` mis-escaping BM25 queries),
#836 (`add_episode(update_communities=True)` raising `ValueError` on non-2-node episodes), #1013
(Neo4j bulk edge-save silently omitting `attributes`/`reference_time`, now fixed upstream), and
#1001 (FalkorDB's old `add_triplet()` silently no-oping and never setting edge endpoint UUIDs, now
closed via #1013) -- all live entirely inside the self-hosted `graphiti-core` library layer that
`ZepGraphitiAdapter`'s hosted REST calls can never reach. See that new adapter's module docstring for
the full citation trail (each bug was confirmed by fetching and reading the real source from
`getzep/graphiti`'s `main` branch on GitHub during this build, not by running it against a live
database) and, plainly, what this build could NOT verify: no Neo4j or FalkorDB instance was ever
started or reached in this environment, so nothing here demonstrates any of the four bugs actually
reproducing end-to-end. `ConflictSignal.EDGE_INTEGRITY_VIOLATION` (`adapters/base.py`) and
`MemoryRecord.attributes` (also `adapters/base.py`) are the two shared-interface additions this
adapter needed: the former lets `evals/contradiction.py`'s `classify_case()` flag a structurally
broken edge (missing `source_node_uuid`/`target_node_uuid`) as its own distinct signal instead of
folding it into an ordinary text-classification miss; the latter lets a backend's structured
per-record properties (graphiti-core's `EntityEdge.attributes`) survive the adapter boundary at all.
Both are purely additive to the shared interface -- no existing adapter's behavior changes.

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

**Why `Mem0DirectAdapter` exists, and the retired Kuzu bug it could not reproduce.** Neither
`Mem0Adapter` nor `Mem0SelfHostedAdapter` can select a `graph_store`, `embedder`, or
`vector_store` *provider* -- those are construction-time Python config
(`MemoryConfig(embedder=..., vector_store=..., graph_store=...)`), not REST parameters either
adapter's HTTP surface exposes. Five real, cited mem0 bug reports trace to exactly that
unreachable configuration surface: mem0ai/mem0#3558 (Kuzu graph store raising `ValueError` on a
bad `embedding_dims`), #5671 (AWS Bedrock not forwarding `embedding_dims`), #4362 (Redis/Valkey
silently zeroing a vector on a metadata-only update), #4711 (FastEmbed defaulting to a hardcoded
1536 instead of the loaded model's real dimension), and #2304 (Gemini/OpenAI silently dropping
`embedding_dims`). `Mem0DirectAdapter` (`mem0_direct_adapter.py`) holds a direct, in-process
`mem0.Memory` handle via `Memory.from_config()` specifically to make that configuration surface
reachable, gated on `MEM0_DIRECT_EMBEDDER_PROVIDER` and not included in `cli.ALL_BACKENDS` --
same opt-in-only precedent as `Mem0SelfHostedAdapter` above, since this adapter targets a
self-assembled in-process stack rather than a single hosted vendor API.

Of the five, **four are confirmed fixed in the installed `mem0ai==2.0.12` package** (the newest
version on PyPI as of this build, 2026-07-16) by reading that package's actual source, not by
trusting the GitHub issues' "merged" status: `aws_bedrock.py` forwards `embedding_dims` into the
Bedrock Titan V2 request body (#5671); `fastembed.py`'s `FastEmbedEmbedding` reads
`self.dense_model.embedding_size` at init instead of defaulting to 1536 (#4711); `gemini.py` and
`openai.py` both forward `embedding_dims` into their respective embed calls (#2304); and
`redis.py`/`valkey.py` both guard `if vector is not None:` before overwriting the stored
`"embedding"` field (#4362). `tests/test_mem0_direct_adapter.py` exercises the real, installed
classes for all four directly (mocking only each vendor's network/model-load boundary), so a
regression that reintroduced any of them in a future `mem0ai` release would fail those tests
against the *installed* package -- this is what justifies re-validating all four as PASS against
the currently pinned `mem0ai` version, not just restating the issue tracker.

**#3558 (Kuzu) could not be reproduced, and this was the one genuinely surprising finding of this
build.** The installed `mem0ai==2.0.12`'s `MemoryConfig` (`mem0/configs/base.py`) has no
`graph_store` field at all; no `kuzu` dependency appears in any of the package's declared
extras; and no graph/kuzu module exists anywhere in the installed `mem0/` package tree (the only
"kuzu" string in the entire installed package is an illustrative example inside
`mem0/exceptions.py`'s `DependencyError` docstring, not a real code path). `kuzu_memory.py`, the
file mem0ai/mem0#3558 and its fix concern, is not present in this release. Worse, passing
`graph_store` to `MemoryConfig(**config_dict)` anyway does not raise -- it is silently ignored
(confirmed empirically during this build: `MemoryConfig(graph_store={"provider": "kuzu", ...})`
produces a config object with no trace the key was ever given, pydantic's default
`extra="ignore"` behavior on this model). Rather than reproduce that silent no-op,
`Mem0DirectAdapter.__init__` refuses any `graph_store_provider` request outright, at
construction, with a `BackendAPIError` naming this finding. In its place, the adapter reproduces
the *bug class* #3558 established -- a backend rejecting a missing/invalid embedding-dimension
config at construction time, before any write can silently corrupt state -- against a component
that still has exactly that validation shape: `ValkeyConfig.embedding_model_dims` is a required
(no-default) `int` field, so constructing this adapter with `vector_store_provider="valkey"` and
an explicit `embedding_dims=None` raises a real `pydantic.ValidationError` (a `ValueError`
subclass) from the installed package, caught and reported as `StoreResult.corruption_signal =
CorruptionSignal.CONFIG_REJECTED` rather than an unhandled crash. This is an honest substitution
for a retired code path, not a re-creation of it -- a `memtrust` user reading "CONFIG_REJECTED"
against `mem0_direct` should not conclude mem0ai/mem0#3558 itself was reproduced, only that the
*shape* of bug it represents (construction-time config rejection, not a silent graph-store
no-op) is what this adapter can demonstrate against the currently installed package. See
`mem0_direct_adapter.py`'s module docstring and `CorruptionSignal.CONFIG_REJECTED`'s docstring
in `base.py` for the same caveat stated where the code itself lives.

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

`ZepGraphitiSelfHostedAdapter` deserves the strongest version of this objection of any adapter in
this repo: it was built and unit-tested entirely against a Protocol double, because the real
`graphiti-core` package is not installed in this build environment and no Neo4j or FalkorDB
instance was ever started or reached. Of the four bugs motivating this adapter's addition, two
(getzep/graphiti#1013, #1001) are confirmed already fixed/closed upstream on the version of
graphiti-core this adapter was built against -- so a live run today should not reproduce them at
all, and `ConflictSignal.EDGE_INTEGRITY_VIOLATION` exists to catch them only if this adapter is
ever run against an older, affected version. Of the remaining two, `update_communities=True` is
confirmed to thread through to `add_episode()` correctly (unit-tested), but nothing in this build
demonstrates it actually triggers #836's `ValueError` -- that requires a live instance with real
LLM credentials driving entity extraction, which is outside this build's scope. `lucene_sanitize()`
(#1302) is never called by this adapter's own code at all; it is internal to graphiti-core's search
pipeline, and this adapter can only supply a fixture query (`mt-contra-006`) a contributor with a
live instance could use to observe the effect, not reproduce it here. If a reader wants confidence
that this adapter reproduces any of these four issues, the honest answer today is: it does not,
demonstrably, in this build -- verifying against a live Neo4j/FalkorDB deployment is the necessary
next step, not an optional nice-to-have.

The same objection, in its strongest form yet, applies to the Scale/Volume-Stress eval added for
volcengine/OpenViking#2850 and getzep/graphiti#1275. What this change actually built and verified:
a deterministic synthetic-corpus generator that can produce an arbitrarily large, realistic-shaped
record set; an eval that stores that corpus incrementally and measures recall as a function of
corpus size at a series of checkpoints; and a classifier, unit-tested against three purpose-built
fake adapters, that correctly tells "recall stayed scale-invariant" apart from both "anchor record
silently evicted as volume grew" (#1275's shape) and "search collapsed to empty past a volume
threshold" (#2850's shape). What it did not do: run against a live OpenViking or Graphiti instance
at any scale, let alone the 10K+ records or 300+ real episodes the two cited issues describe.
`DEFAULT_N_RECORDS=500` is a CI-speed default, not a claim about the scale where either bug
manifests -- reaching that scale for real requires `--scale-stress-n-records 10000` (or larger)
against a credentialed backend, which this build pass did not run. Nobody should read a
`SILENTLY_DEGRADED_AT_SCALE` or `WORKED_AT_SCALE` result in a report generated by this repo as
confirmation of either vendor's current live behavior -- it confirms only that memtrust's harness
is now structurally capable of detecting that shape of bug *if* a live backend exhibits it.
