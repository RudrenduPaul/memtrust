# Can an AI Agent Install Your CLI and Trust What It Reports, With No Human Checking Either Step?

### Building an npm-to-uv-to-PyPI bridge for a memory-benchmark harness, and the "expand scope, then adversarially re-verify" habit that keeps its output honest enough for an agent to act on alone

*Co-authored by [Rudrendu Paul](https://dev.to/rudrendu_paul) and [Sourav Nandy](https://dev.to/sourav-nandy).*

**Repo:** [github.com/RudrenduPaul/memtrust](https://github.com/RudrenduPaul/memtrust), an independent, reproducible benchmark harness for agent-memory backends. `pip install memtrust`.

**Quick summary:** giving an AI agent or a CI job a CLI it can install with one command is only half the job; the other half is making sure the thing it installs is telling the truth once it runs. Below is the real, honest state of an npm wrapper (`memtrust-cli`) that bootstraps Astral's `uv` to run a Python benchmark tool with zero manual provisioning, built and dry-run-verified but not yet live on the registry, plus a concrete before-and-after of how this project keeps its own eval claims from becoming exactly the kind of thing it would flag in someone else's benchmark.

---

In Part 1 of this series we found that our own MemPalace adapter had been calling an API method that did not exist. Every result it had ever reported for that backend was fiction, and closing that gap required an adversarial, from-scratch re-verification of every adapter in the project. That story is worth reading on its own. This one picks up after it, on a question Part 1 only implied: when an agent, rather than a person, is running this tool, what does "trustworthy" even mean with nobody left to double-check the output?

Here is the real command surface memtrust ships today, the one an agent or a CI script would actually parse.

![Terminal recording of `memtrust run --help`, showing the real flag list: --backends, --eval, --output, --locomo-dataset-path, and --sign](https://raw.githubusercontent.com/RudrenduPaul/memtrust/main/docs/assets/dev-to-demos/demo-3-cli-surface.gif)

*Every flag shown here is real, pulled straight from `memtrust run --help`. Nothing in this GIF is staged copy.*

## What "agent-native" actually has to mean

"Agent-native" gets used loosely enough that it is worth pinning down. It does not mean a tool has a `--json` flag bolted onto an otherwise human-facing CLI. It means two separate things have to both be true at once:

1. **Installable without a human choosing anything.** An agent or a CI job needs to go from "nothing installed" to "tool running" with one command it can construct programmatically, on whatever runtime the environment already has, not the runtime the tool's author happened to use.
2. **Legible without a human interpreting anything.** Every response needs to be something a program can parse and act on, not a paragraph a person reads and translates into a decision.

Most tools built for developers, memtrust's earlier releases included, satisfy neither by default. `pip install memtrust` assumes Python is already provisioned, a specific version, a virtual environment discipline, a place `pip` itself is trusted to write to. A CI runner or an agent sandbox that only ships Node has no path in at all. And most CLI output, memtrust's early output included, is designed to be read, not parsed: colored tables, progress bars, a summary line meant for a human's eyes.

The npm side of this project exists to close gap one. The rest of this article is about gap two, because gap one turns out to be the easier problem.

## The honest current state: built, dry-run verified, not live

Here is the plain fact first, stated without hedging: `memtrust-cli` is not installable via `npx memtrust-cli` right now. It does not exist on the npm registry. Running `npm view memtrust-cli version` against the real registry today returns a 404.

What does exist: a complete, tested npm package at `npm/memtrust-cli/` in the repo, six per-platform companion packages (`@memtrust-cli/darwin-arm64`, `@memtrust-cli/darwin-x64`, `@memtrust-cli/linux-arm64`, `@memtrust-cli/linux-x64`, `@memtrust-cli/win32-arm64`, `@memtrust-cli/win32-x64`), and two dry runs against the real npm registry that both passed cleanly: `npm pack --dry-run` produces a correct 5.8 kB tarball, and `npm publish --dry-run` walks through the exact publish flow, registry auth included, without touching production. That is as close to "ready" as a package gets before someone actually presses publish. It is not the same claim as "live," and this article is not going to blur that line.

In the meantime, the real, live path is `pip install memtrust`, version 0.3.0, confirmed against PyPI's own API as this was written. Every example below works today through that path. The npm wrapper below is the distribution mechanism this section describes; you cannot run it yet.

Here is what `bin/memtrust.js` actually does, in full, because the whole design fits in about thirty lines:

```js
// uv is a bootstrap tool here, not the final CLI: "uv tool run --from memtrust
// memtrust <args>" transparently provisions a Python interpreter (if needed)
// and installs/caches memtrust from PyPI on first use, then runs it.
const result = spawnSync(
  uvPath,
  ["tool", "run", "--from", "memtrust", "memtrust", ...process.argv.slice(2)],
  { stdio: "inherit" }
);
```

The script resolves a platform-specific `uv` binary from `require.resolve()`, then hands off every argument to it unchanged, reimplementing none of memtrust's actual logic. Reading the whole file top to bottom takes about a minute, which is the design goal: a thirty-line bridge, small enough to audit at a glance.

The part worth explaining is where the `uv` binary itself comes from, because this is the one design decision in the whole wrapper that took real back-and-forth to get right. Each platform package's `prepack` script fetches the matching `uv` release archive directly from Astral's own GitHub releases, downloads the per-archive `.sha256` checksum file `uv` publishes alongside it, and refuses to proceed if the two don't match. That verification happens once, at `npm publish` time, by whoever is running the release, not once per end user at install time. The tradeoff we chose deliberately: a checksum verified once by a maintainer who can see the whole supply chain in front of them is a stronger guarantee than the same checksum re-fetched fresh by every downstream `npm install`, trusting whatever the network happens to say months or years later. It also means an offline or network-restricted install never needs to reach Astral's servers at all. The `uv` binary is just there, already verified, sitting in `node_modules`.

`uv tool run --from memtrust memtrust <args>` is the one line doing the real work after that: it provisions a Python interpreter if none is present, installs `memtrust` from PyPI into an isolated tool environment on first use, caches it, and runs it. An agent or a CI job that has Node available, which is most of them, gets a fully working Python benchmark tool without anyone provisioning a Python toolchain by hand.

Node's own footprint makes that gap worth closing. Sonatype's 2026 State of the Software Supply Chain Report puts npm at 7.97 trillion package downloads across 2025, up 65.43% year over year, against 804.97 billion for PyPI in the same period ("2026 State of the Software Supply Chain," Sonatype). That's a scale signal, correlational rather than a direct measurement of any specific CI image's contents: Node-based tooling reaching a larger share of environments by default meaningfully raises the odds it's already present somewhere a curated Python toolchain isn't, which is the actual gap a Node wrapper closes for a Python tool.

`uv` itself is not a niche bet either. Astral's repository sits at 87,595 GitHub stars as of this writing (GitHub API, `astral-sh/uv`), and pypistats.org reports 165.67 million downloads in the trailing month. The Python Developers Survey run by the PSF and JetBrains found `uv` usage went from 0% to 11% adoption in the single year it was introduced, reported in JetBrains' "The State of Python 2025". Stack Overflow's 2025 Developer Survey puts `uv` at 9.5% adoption against pip's 40.9% and npm's 56.8%, a real and measured minority share. `uv` is a reasonable thing to build a bootstrap step on top of.

## Every run already produces structured JSON output

Every `memtrust run` invocation writes a full JSON report to disk by default, `./memtrust-report-<date>.json` unless `--output` says otherwise. There is no `--json` switch to remember, because there is no unstructured mode to opt out of: the JSON report is the only thing the command produces past its console summary. The report's `results` block breaks down per-backend, per-eval outcomes with the same signal taxonomy the underlying evals use internally (`memtrust run --eval` currently exposes sixteen registered evals, `longmemeval` through `result_consistency`), so a program reading the file gets the identical verdict a person reading the console table would.

Here is what an agent or CI job actually parses, pulled directly from a real `memtrust run --backends mempalace,mem0,zep,openviking --eval all` invocation against `memtrust` 0.3.1, zero credentials configured, nothing edited or shortened:

```json
{
  "mempalace": {
    "status": "skipped",
    "reason": "mempalace is not configured: environment variable MEMPALACE_STORAGE_PATH is not set. Skipping this backend. See docs/methodology.md for setup instructions.",
    "missing_env_var": "MEMPALACE_STORAGE_PATH"
  },
  "mem0": {
    "status": "skipped",
    "reason": "mem0 is not configured: environment variable MEM0_API_KEY is not set. Skipping this backend. See docs/methodology.md for setup instructions.",
    "missing_env_var": "MEM0_API_KEY"
  }
}
```

No prose to translate, no ambiguity about whether a backend ran: `status` and `missing_env_var` are both machine-checkable fields, which is the entire point of the nuance below.

There's a real `memtrust report <path>` subcommand too, for reading a prior run's JSON back and printing a formatted summary, useful when the agent that ran the eval and the agent (or person) reviewing it aren't the same process.

One nuance worth stating plainly, because it is exactly the kind of thing a parser can get wrong silently: a backend without a configured credential environment variable does not fail the run. It prints `SKIPPED` in that backend's section and the run continues. That is the right behavior for a CLI meant to run unattended in CI, a missing API key for one backend shouldn't crash the whole eval suite. But it means a program parsing the JSON has to check each backend's own `status` field instead of relying on the command's exit code alone, or it will silently treat "we never actually tested this backend" as "this backend passed."

There's also a `--sign` flag, backed by a real Ed25519 keypair generated with `memtrust keygen`, that writes a signed receipt alongside the JSON report, proving which key produced it and that the file hasn't been altered since. That detail exists for a reason specific to agent-to-agent consumption: structured output only helps if the agent reading it can also trust where it came from. A JSON file with no origin guarantee is exactly as spoofable as a human-readable summary, just easier to spoof automatically. Signing is optional, and most runs won't need it, but it's there for the case where a report crosses a trust boundary the way this whole article is about.

## The harder half: making the tool's own claims something an agent can act on

None of the distribution mechanics above matter if the tool being distributed can't be trusted. An agent calling `memtrust run` autonomously, parsing the JSON, and deciding what to do next has no human in the loop to notice if a result is quietly wrong. That's the actual stakes of Part 1's discovery: a fictional API call inside the MemPalace adapter meant every downstream consumer of that adapter's output, human or agent, had been acting on a claim nobody had checked. That downstream consumer includes more than a CI pipeline: the same reader Part 1 already named, a product manager comparing vendors or an investor doing technical diligence, has no way to know a benchmark's own machinery was quietly wrong unless someone goes back, checks it, and says so in public.

The discipline that came out of that re-verification runs on repeat: every time the project's own backlog says a capability is missing, someone investigates the real gap, builds the smallest genuinely closeable piece of it, and re-verifies the result independently before calling it done. Nobody inflates a partial fix into a full one just to shrink the backlog. The clearest recent example of that loop start to finish is a knowledge-graph boundary bug memtrust's own eval suite couldn't detect until a few days ago.

### A concrete before-and-after: the temporal-KG boundary bug

The bug itself lives upstream, inside MemPalace's own knowledge-graph store. `MemPalace/mempalace#1913` reported that querying a knowledge-graph fact with an `as_of` point-in-time filter, at the exact instant that fact was invalidated, could return both the old value and the new value simultaneously. The cause was a closed-interval SQL comparison (`valid_to >= as_of`) instead of a half-open one (`valid_to > as_of`): a fact that ended exactly at the query instant still matched, and if its replacement started at that identical instant, a single-valued predicate like `uses_model` reported two contradictory answers at once with no error or warning. Worse, the hand-rolled pattern that triggers it, invalidate a fact and add its replacement at the same shared timestamp, is exactly what its own pre-fix onboarding guidance told every agent to do. The maintainers fixed it in merged PR `#1914`, switching to the half-open interval and adding a new `supersede()` primitive as the preferred atomic replacement.

memtrust's own backlog had flagged the capability to detect this class of bug months earlier, bundled together with a second, unrelated capability (drawer neighbor-expansion leak scoping) into one item. That item sat OPEN because, honestly, it was too large for a single scoped fix. Forcing it through anyway, just to close the backlog line, would have meant either shipping something half-built or quietly widening what counted as "done." Neither is acceptable under this project's own rules, so the item stayed open until the scope actually shrank to something real.

The expand-scope investigation is the actual mechanism, and it's worth walking through in detail. Splitting the bundled item required reading the adapter code directly rather than re-describing the backlog entry from memory. That reading turned up something useful: the underlying wiring, `kg_query(entity, as_of=..., direction="both")`, already existed post-rewrite and already passed straight through to the real, confirmed `tool_kg_query`. Nothing was calling it with a nonzero `as_of` in an eval. The real capability gap was that nobody had built a classifier or a test exercising the path that already worked, a materially smaller job than the original bundled item implied. That distinction only surfaces when someone reads the actual code with the specific goal of shrinking the claim, rather than re-estimating the same vague item a second time.

What got built from there: a new `TemporalBoundarySignal` enum (`CLEAN`, `DOUBLE_COUNT`, `NOT_APPLICABLE`) in `adapters/base.py`, and a dedicated eval, `evals/temporal_kg_boundary.py`, that seeds a fact, invalidates it and adds its replacement at the identical boundary instant (the exact `#1913` shape), queries at that instant, and classifies the response two separate ways. First, the adapter's own self-reported signal. Second, and this is the part that matters, the eval's own independent classification, derived straight from the raw list of facts the query actually returned, never trusting the adapter's self-report as the final answer. If those two disagree, that's a bug in the adapter's own classification logic, a distinct failure from whether the backend itself double-counts, and the eval is built to catch both separately rather than conflating them.

The test suite backing this holds to the same standard. It runs against two from-scratch fake implementations that faithfully reproduce the actual pre-`#1914` closed-interval comparison and the actual post-`#1914` half-open comparison, computing a real answer from real logic each time rather than a stub tuned to whatever value makes the test pass.

![Terminal recording of `pytest tests/test_temporal_kg_boundary.py -q --no-cov` passing 15 tests](https://raw.githubusercontent.com/RudrenduPaul/memtrust/main/docs/assets/dev-to-demos/demo-4-temporal-kg-tests.gif)

*Fifteen tests, all passing, against fakes that implement both the buggy closed-interval logic and the fixed half-open logic in full.*

Now the honest limitations, stated exactly as plainly as the eval's own docstring states them. First: the real `mempalace` package is not installed in the environment this was built in, and PR `#1914` is merged to MemPalace's `develop` branch, not yet in the `3.5.0` release this adapter was built and live-verified against. That makes this a detection-capable harness: proven correct against faithful synthetic reproductions of both the buggy and fixed comparison logic. It has not yet run against a live instance. The eval would flag `#1913`'s exact failure shape the moment it's pointed at a pre-fix deployment, but nobody has pointed it at one yet, and the article isn't going to describe that as done when it isn't. Second: `temporal_kg_boundary` is not yet one of the sixteen evals `memtrust run --eval` can invoke by name. It exists today as a standalone module with its own dedicated, passing test suite. Wiring it into the CLI's eval registry is a small, separate, still-open task.

Before this change merged, someone other than whoever wrote the fix independently re-ran the full suite themselves and confirmed 580 tests passed, 8 skipped, with `ruff check`, `ruff format --check`, and `mypy --strict` all clean against the actual merge commit. That's the same discipline Part 1's re-verification introduced, applied here to a much smaller, much less dramatic change. The habit has to hold on boring changes like this one the same way it held on the dramatic one that made Part 1 a story worth telling.

And the drawer neighbor-expansion half of that original bundled backlog item is still open, untouched, deliberately. Splitting the item didn't mean quietly closing both halves. It meant closing the one that had actually shrunk to a real, verifiable piece of work, and leaving the other exactly where it was.

## Distribution and verification are the same story, told from two directions

An agent that can install your CLI with zero setup but can't trust a word it reports back has automated the squinting, without actually removing the human who has to check the result. And a benchmark tool with airtight verification discipline that only runs for a person with a curated Python environment fails to reach the audience that increasingly needs it: agents and CI pipelines making decisions with nobody watching in real time. Neither half works without the other. The actual thesis here is that a tool built to be called autonomously has to hold its own claims to the same evidentiary bar it holds the systems it benchmarks.

Where this genuinely still falls short: the npm package isn't live, so none of the distribution mechanics above are usable today through `npx`, only through the existing `pip install memtrust` path. A `Kuzu` graph-database driver another adapter could support stays unbuilt on purpose, because the upstream project that would back it has formally deprecated that path and dropping default test coverage for it, building against a dependency its own maintainers are walking away from is a bad trade regardless of how easy the fix would be. And "detection-capable" and "field-tested against a live deployment of the exact bug" are different claims, a distinction that generally gets flattened in security and eval tooling more broadly and is worth naming directly, rather than letting the softer claim ride on the stronger one's credibility.

## The evolving landscape of agent-facing tool distribution and eval verification

Four distinct categories of existing practice sit around the specific gap this project is trying to close, and it's worth being precise about what each one actually solves.

**Protocol-level standards for agent-tool-calling** have emerged as the most direct answer to "how does an agent invoke a tool at all." The Model Context Protocol, introduced in November 2024 and donated in December 2025 to the newly formed Agentic AI Foundation, a directed fund under the Linux Foundation, reported over 97 million monthly SDK downloads and more than 10,000 active public servers at the time of that announcement. What a protocol standard like this solves is the invocation contract: how a tool describes its own capabilities and how an agent calls them in a predictable shape. What it doesn't solve is whether the values a tool returns through that contract are actually true. Standardizing the envelope isn't the same as verifying the contents.

**Traditional package-manager install flows**, pip and npm's default behavior included, were built around a human reading terminal output and making judgment calls: accept a prompt, notice a warning, interpret a stack trace. There's no standard machine-parseable report built into a default `pip install` or `npm install`, because there was never an audience other than a person watching the terminal when those tools were designed. That's a description of the design target those tools were built for, from an era when a person watching the terminal was the only audience there was.

**Benchmark and leaderboard self-reporting** in the broader ML ecosystem has a well-documented gaming problem. A 2025 analysis found that one major model provider privately tested 27 model variants against a popular crowdsourced arena benchmark before release, with proprietary models receiving access to 54.3% to 70.1% of the benchmark's evaluation data compared to far less for openly available models, and that additional arena data access alone produced score gains of up to 112% relative to a baseline (Singh, Nan, Wang, et al., "The Leaderboard Illusion," NeurIPS 2025). A separate survey has documented the broader pattern of train/test contamination undermining leaderboard comparisons across the field. The general shape across this category: a self-reported number, with no independent re-run requirement, is trivially easy to optimize toward rather than earn.

**Detection-capable claims in vulnerability and bug-finding tooling more broadly** get conflated with live-reproduced claims often enough that it's now a recognized distinction in its own right. Recent work building automated exploit-verification pipelines makes the point directly: identifying that a flaw plausibly exists is a categorically easier claim than producing a working reproduction against a real running system, and treating the two as equivalent overstates what a detection tool has actually shown (a 2025 paper on automated exploit generation and validation makes this argument explicitly).

A gap runs through all four categories: none of them, on their own, forces a tool to prove its own output is true rather than merely well-formatted or plausible. A protocol standardizes the call. A package manager gets the bits onto disk. A leaderboard reports a number. A detection tool flags a candidate. None of the four requires the kind of adversarial re-verification, independent of the person or process that produced the original claim, that this project treats as non-optional. That's the specific practice an agent-native benchmark tool has to add on top of everything those four categories already do well: install cleanly, call predictably, report a number, flag a candidate, and then prove that number or that flag actually holds up when someone who didn't write the fix goes and checks.

## The forced choice

If you're building a CLI or an eval suite you want an agent to call autonomously, which comes first for you: getting installation down to one command it can run without a human, or building the re-verification discipline that makes its output worth acting on without one? We went distribution-first here because it was the more tractable half, but we'd genuinely like to hear if that's backwards for how you're actually deploying agents against tools like this.

---

### References

1. Sonatype. ["2026 State of the Software Supply Chain."](https://www.sonatype.com/state-of-the-software-supply-chain/2026/software-infrastructure-growth) 2026.
2. GitHub API, [`astral-sh/uv`](https://github.com/astral-sh/uv) repository metadata, fetched 2026-07-17.
3. [pypistats.org: `uv` package download statistics.](https://pypistats.org/packages/uv) Fetched 2026-07-17.
4. JetBrains. ["The State of Python 2025."](https://blog.jetbrains.com/pycharm/2025/08/the-state-of-python-2025/) August 2025, based on the Python Developers Survey (PSF + JetBrains).
5. Stack Overflow. [2025 Developer Survey, Technology section.](https://survey.stackoverflow.co/2025/technology) 2025.
6. MemPalace. [`mempalace#1913`, "kg query --as-of returns two values for a single-valued fact at a supersession boundary."](https://github.com/MemPalace/mempalace/issues/1913)
7. MemPalace. [`mempalace#1914`, "fix: half-open as-of interval + supersede() to end KG boundary double-count."](https://github.com/MemPalace/mempalace/pull/1914) Merged.
8. Anthropic. ["Donating the Model Context Protocol and Establishing the Agentic AI Foundation."](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) December 2025.
9. Linux Foundation. ["Linux Foundation Announces the Formation of the Agentic AI Foundation."](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation) December 2025.
10. Singh, Nan, Wang, et al. ["The Leaderboard Illusion."](https://arxiv.org/abs/2504.20879) NeurIPS 2025, arXiv:2504.20879.
11. ["Benchmark Data Contamination of LLMs: A Survey."](https://arxiv.org/html/2406.04244v1) arXiv:2406.04244, June 2024.
12. "CVE-Genie: From CVE Entries to Verifiable Exploits." [arXiv:2509.01835.](https://arxiv.org/html/2509.01835v1) 2025.

---

*Rudrendu Paul is a founder and engineer building agent-native open-source tooling, including `memtrust` and a companion suite of AI-agent infrastructure projects. Co-authored with Sourav Nandy. Find the code at [github.com/RudrenduPaul](https://github.com/RudrenduPaul).*
