# CLAUDE.md -- memtrust

## Project identity

- **What this is:** an independent, reproducible benchmark harness that runs standardized evals
  (LongMemEval, LoCoMo, and an original multi-hop contradiction-detection eval) against
  agent-memory backends (MemPalace, Mem0, Zep/Graphiti, OpenViking) and publishes results with
  full raw logs and a documented methodology.
- **Repo:** github.com/RudrenduPaul/memtrust
- **Package:** `memtrust` on PyPI
- **Language:** Python (`src/memtrust/` layout, PEP 621 `pyproject.toml`)
- **License:** Apache 2.0
- **Goal:** be the benchmark someone can point to instead of a vendor's own self-reported number,
  because the methodology, prompts, and raw logs are fully public and reproducible from a fresh
  clone. Trust is earned by being independently reproducible, not claimed.

## Git workflow

When asked to commit, push, or "update GitHub" -- just do it. No confirmation prompts.

- `git add` relevant files -> `git commit` -> `git push origin main` in one shot.
- Every commit message ends with:
  `Built by Rudrendu Paul and Sourav Nandy, developed with Claude Code`
- Do not add `Co-Authored-By:` trailers.
- Prefer small, checkpoint-shaped commits over one giant commit.

## Engineering standards (block all tasks until these pass)

1. **Lint:** `ruff check . && ruff format --check .`
2. **Types:** `mypy --strict src/memtrust` -- zero errors, zero unexplained `# type: ignore`.
3. **Tests:** `pytest --cov=memtrust --cov-report=term-missing --cov-fail-under=80` -- 80% minimum
   overall; 90%+ on `adapters/base.py`, `evals/contradiction.py`, and `scoring/`. Every test must
   run fully offline -- mock or stub any network/vendor call. No test may require a real API key.
4. **Security:** `pip-audit` -- no unfixed HIGH/CRITICAL CVEs in the dependency tree.
5. **Reproducibility:** if you changed a scoring prompt, an adapter, or an eval dataset version,
   re-run the affected eval and show the before/after in your response -- never state a number
   without showing the command that produced it.

Do not mark a task complete if any of these fail. Fix the root cause; do not suppress errors or
add a blanket `# type: ignore`.

## Planning rules

Enter plan mode for any task that:
- Touches more than 2 files.
- Changes the `MemoryBackendAdapter` interface or a scoring-pipeline contract.
- Adds a new eval family or a new tracked backend.
- Modifies `.github/workflows/ci.yml`.

Write the plan before touching code. If something goes wrong mid-task, stop and re-plan rather
than patching around the original plan.

## Anti-sycophancy rules

These override default behavior in every session working on this repo:

1. **No benchmark number without a fresh, reproducible run.** Before publishing or citing a score
   for any backend, run the eval and show the command output. Never state a number from memory or
   a stale prior run without re-verifying it's current.
2. **Every eval runs identically across every tracked backend.** No per-vendor prompt tuning, no
   per-vendor dataset subset. If a backend's API genuinely cannot support a given eval, document
   the gap explicitly in the results (`ConflictSignal.NOT_APPLICABLE`, or an equivalent explicit
   marker) rather than silently excluding the backend from that eval's table.
3. **No "verified"/"safe"/"best" claim about any backend.** This project publishes comparative
   scores across a defined eval set, not an endorsement or a safety certification. Report numbers;
   let the reader draw the conclusion.
4. **Every methodology decision lives in `docs/methodology.md`, versioned with the code.** Prompt
   templates, dataset versions, scoring rubrics, and adapter confidence levels all belong there. If
   a methodology choice can't be explained in that file, it does not belong in the harness.
5. **Vendor-pushback check.** Before publishing a run, ask: "if this backend's own maintainers
   read this methodology, could they point to a specific, defensible flaw?" If yes, fix the flaw
   before publishing, not after someone complains publicly.

## What Claude must never do in this repo

- Publish or cite a benchmark number without a fresh command-output run in the same session.
- Ship a new eval or backend adapter without a corresponding `docs/methodology.md` entry.
- Commit with `--no-verify`.
- Merge a change to scoring logic without re-running the affected eval and showing the delta.
- Present a best-effort adapter (see the confidence table in `docs/methodology.md`) as a confirmed
  vendor API integration.
- State or imply that this project's purpose is anything other than the eval harness and
  leaderboard described above.

## Key files

| File | Purpose |
|---|---|
| `src/memtrust/adapters/base.py` | The shared adapter interface every backend implements. Read this before touching any adapter. |
| `src/memtrust/adapters/` | One adapter per backend (MemPalace, Mem0, Zep/Graphiti, OpenViking). |
| `src/memtrust/evals/contradiction.py` | The original wedge eval -- the most important file in the repo. |
| `src/memtrust/evals/` | LongMemEval and LoCoMo runners. |
| `src/memtrust/scoring/` | LLM-judge scoring pipeline and cost tracker. |
| `src/memtrust/cli.py` | `memtrust run`, `memtrust report`. |
| `docs/methodology.md` | Full, versioned methodology -- read before publishing any number. |
| `leaderboard/` | Static leaderboard site (`index.html` + `data.json`). |
| `CONTRIBUTING.md` | Read before adding a new backend adapter -- the primary contribution path. |
| `.github/workflows/ci.yml` | lint -> type-check -> test -> security. |

## Session start checklist

1. Run `git status` and `git log --oneline -5` to understand current state.
2. Run `pytest` to confirm the baseline is green before touching anything.
3. Read `docs/methodology.md`'s relevant section before changing an eval or adapter.
4. If a score looks off, re-run the specific eval against the specific backend with verbose output
   before assuming the harness (rather than the backend) is wrong.
