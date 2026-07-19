# Would Your AI Benchmark Survive Being Audited Like the Vendors It Judges? Ours Didn't, at First

### Part 1 of 2: how we found our own AI-memory benchmark tool calling a fictional API for one of the four vendors it tests, and the two PASS verdicts the honest fix forced us to take back

*Co-authored by [Rudrendu Paul](https://dev.to/rudrendu_paul) and [Sourav Nandy](https://dev.to/sourav-nandy).*

**Repo:** [github.com/RudrenduPaul/memtrust](https://github.com/RudrenduPaul/memtrust), an independent, reproducible CLI benchmark harness for agent-memory backends. `pip install memtrust`.

**Quick summary:** memtrust exists because AI-memory vendors publish their own benchmark numbers, scored their own way, and nobody outside the vendor checks the raw logs. Midway through building it, we found that our own adapter for one of the four vendors we test, MemPalace, had been calling a Python class that has never existed in the real package, for the tool's entire history. Below: how we found it, the full rewrite against the real API, and why we then spent more effort re-checking our own prior "PASS" verdicts than we spent on the original fix, two of which turned out to be wrong.

---

![Terminal recording of memtrust run against all four backends with zero API keys configured, showing every backend reporting SKIPPED and the run exiting cleanly with a valid JSON report](https://raw.githubusercontent.com/RudrenduPaul/memtrust/main/docs/assets/dev-to-demos/demo-1-zero-credentials.gif)

*A fresh clone, zero credentials, `memtrust run --backends mempalace,mem0,zep,openviking --eval all`. Every backend reports a real SKIPPED status, and the run still exits cleanly with a real report. That honesty standard is exactly what the tool is supposed to hold itself to. Keep that in mind for what comes next.*

Stanford's 2026 AI Index Report has a line in its technical performance chapter that should worry anyone who cites a benchmark number in a sales deck: on SWE-bench Verified, model performance went from 60% to near 100% in a single year (Stanford HAI, 2026 AI Index Report). A jump like that signals a benchmark saturating faster than anyone can redesign it, a polite way of saying the number stopped measuring the thing it was built to measure. This problem has real precedent: a 2019 study that built fresh test sets for ImageNet and CIFAR-10, matching the original collection methodology as closely as possible, found accuracy dropped 11 to 14 percentage points on ImageNet and 3 to 15 points on CIFAR-10 the moment the benchmark stopped being one everybody had already been implicitly tuning against for years (Recht et al., ICML 2019).

AI-agent memory backends are having their own version of this problem right now, in full public view. One of the four vendors we benchmark has two public GitHub issues, one with 233 thumbs-up reactions and 39 comments, the other with 42 and 8 (both verified live via the GitHub API while writing this), documenting a widely cited 100% LongMemEval score that turned out to be hand-coded against three specific dev-set questions. The held-out score on the other 450 questions is 98.4%, and the maintainer confirmed it directly in the issue thread. A separate 96.6% figure that gets quoted everywhere turns out to be mostly the default embedding model doing the work, not the vendor's own architecture. A "lossless" compression claim drops accuracy by 12.4 percentage points once you actually measure it. Two pull requests attempting to fix the reporting were opened and closed, unmerged, the same day (MemPalace/mempalace#27, MemPalace/mempalace#29).

We built memtrust to stop guessing which vendor's number to trust: run LongMemEval, LoCoMo, and a growing set of evals we wrote ourselves against all four backends the same way, publish the raw logs, and let the numbers be whatever they actually are. Then, while doing exactly that work, we discovered our own tool had been making the identical kind of unverified claim about itself, in its own code, since the very first version.

## What we found when we finally checked

The adapter that talks to it is a normal piece of glue code: memtrust calls `store()`, `query()`, `update()`, and `delete()`, and the adapter translates those into whatever the vendor's real API expects. Every version of that adapter, until this rewrite, called `mempalace.Palace(storage_path=...)`, a class with `.remember()`, `.recall()`, and `.invalidate()` methods.

We ran one line to confirm it before starting the rewrite:

```
$ python3 -c "import mempalace; hasattr(mempalace, 'Palace')"
False
```

![Terminal recording of the real check that first surfaced the fictional Palace class problem: python3 check_palace.py printing False against the actually installed mempalace 3.5.0 package](https://raw.githubusercontent.com/RudrenduPaul/memtrust/main/docs/assets/dev-to-demos/demo-2-fictional-api-discovery.gif)

*The exact terminal check that surfaced this. `Palace` is not an attribute of the installed `mempalace` package, version 3.5.0, full stop. Every store/query/update call this adapter ever made against it ran through a class that does not exist.*

We grepped every `^class ` definition across the entire installed package to make sure we weren't missing a re-export somewhere. `mempalace/palace.py`, the file the class name implies it should live in, defines exactly two things: `MineAlreadyRunning` and `MineValidationError`, both exceptions. No `Palace` class anywhere in the package.

That means `store()`, `query()`, and `update()` had never worked against the real vendor package, in this project's entire history. Every test that appeared to pass was exercising a hand-written fake standing in for a guess about an API that was never confirmed, and that guess was wrong. The adapter's own pre-rewrite documentation had already flagged this honestly as a low-confidence guess, "Confidence: MEDIUM on product behavior, LOW on exact Python method names." Low confidence turned out to mean zero confidence. The class was fictional.

Here is the part that made this more than an embarrassing bug: memtrust exists specifically to catch a vendor claiming something its product doesn't actually do. We had just found our own tool doing precisely that, about its own capability, unnoticed, the entire time.

## The rewrite: calling the real code, and writing down exactly what it actually does

The fix was not swapping one guess for another. The rewrite calls the vendor's real MCP server instead, whose tool calls dispatch to plain, real, module-level functions: `tool_status`, `tool_list_wings`, `tool_list_rooms`, `tool_add_drawer`, `tool_search`, `tool_update_drawer`, `tool_delete_drawer`, `tool_kg_add`, `tool_kg_invalidate`, `tool_kg_query`. Those functions had already been in limited, correct use in one narrow corner of the same adapter for metadata calls. The rewrite made them the only way the adapter talks to it, for everything.

We didn't take the vendor's own docstrings as ground truth this time either. Every response shape documented in the rewritten adapter was captured by calling the real functions live, against a real local, chromadb-backed palace: `pip install mempalace`, point `MEMPALACE_PALACE_PATH` at a temp directory, call `mempalace.mcp_server.tool_add_drawer(...)`, and read what actually comes back. Two things that live-verification caught, that a docs read alone would have missed or gotten wrong:

- `tool_search` results carry no per-record ID field at all. There's no `id`, no `drawer_id`, nothing, only content and scoring fields. We considered recomputing an ID client-side from the content, then rejected it: the vendor's `update_drawer()` deliberately keeps a drawer's original ID stable across a content edit, live-verified by updating a drawer and confirming the pre-update ID still comes back. Recomputing from a query response's current content would silently produce the wrong ID for anything that had ever been edited. The adapter now reports `memory_id=""`, an honest "the real vendor response has nothing to put here," instead of a guess that would be wrong some of the time.
- The knowledge-graph subsystem ignores `MEMPALACE_PALACE_PATH` entirely when called as a library, not just as a CLI. We traced this to a module-level flag that's only ever set `True` when the process's own command-line arguments carried an explicit `--palace` flag, which a library caller never does. The practical effect: two `MemPalaceAdapter` instances pointed at two different storage paths in the same process both read and write the exact same physical knowledge-graph file on disk. That's a real vendor bug we found by exercising the adapter's own test setup, not something reported in an issue anywhere, and we documented it rather than quietly working around it.

One finding cut the other way. The vendor's own docstring for `tool_add_drawer` claims that content chunked across multiple physical rows can't be updated or deleted by its logical ID, "tool_get_drawer(drawer_id) and tool_delete_drawer(drawer_id) report 'not found' on the chunked path." We read the actual function that resolves those calls and it explicitly handles the chunked case correctly. Live-testing confirmed it: storing 2,000 characters of content, well over the 800-character default chunk size, then updating or deleting it by its logical ID worked exactly right, every time we tried it. An earlier draft of this rewrite had trusted the vendor's docstring text here without checking it live, which would have shipped a false "confirmed limitation" in the other direction. Both kinds of error, trusting an API that doesn't exist and trusting documentation that's stale and wrong, come from the same root cause: reading a claim instead of running the code.

## The harder step: not trusting our own fix either

A rewrite that fixes the fictional API is still just a fix, made and validated by the person who wrote it. The standard this project applies to every vendor is that a self-report doesn't count until someone else checks it against reality. We applied that same standard to ourselves.

Thirteen rows in memtrust's issue-validation log, our internal record of every real GitHub issue we've investigated against these four backends, had a verdict on this vendor's repo sitting on PASS or deferred, built and tested against the old, fictional adapter. Once the real rewrite existed, every one of those thirteen went back through independent, freshly spawned reviewers with no memory of building the original fix, checking each claim against the real, working code from scratch.

Eleven held up. Two of those eleven had real, working mechanisms that the rewrite had accidentally dropped along the way, not from dishonesty, just from the ordinary collateral damage of restructuring a file: an `authored_at` ranking fallback and a docstring section distinguishing "no API key required" from "no network access required" both got restored, with new tests proving them against the real adapter this time, not a fake. One row was confirmed correctly still failing, exactly as it had been marked before. Two rows were genuinely wrong, and got downgraded.

### #1005: the PR that never actually shipped

A prior version of memtrust claimed it could detect when the vendor's search silently degrades, falling back and reporting a partial result instead of failing outright, and that the claim was "confirmed against the real, merged PR diff" for MemPalace/mempalace#1005. Two things turned out to be false in that sentence. First, PR #1005 was never merged. GitHub shows it closed, never merged, closed by its own author as self-superseded; we confirmed this directly against the live PR. The retrieval-fallback mechanism it proposed did land eventually, through separate PRs, but the specific observability fields memtrust's code was reading, `warnings` and `available_in_scope`, were never shipped in any of them. Second, and more directly, we grepped the real, installed package's source for those exact field names and found zero matches anywhere. The code path that would populate them does not exist in the shipping product.

The parsing logic itself is fine and won't crash or misread a response if that shape ever does appear. It simply never fires, because the vendor never sends that response. Call it correctly implemented against a capability the vendor hasn't built yet, a meaningfully weaker claim than the one we'd been making. The verdict moved from PASS to PARTIAL to reflect exactly that gap.

### #1733: the fix that solved a different bug than the one reported

The second downgrade is the sharper one. A contributor named Kartalops filed a real, well-diagnosed bug: the vendor's memory "wake-up" sort was supposed to prioritize high-importance, recent memories, but nothing in the real ingest path ever wrote the `importance` field it sorted by, so every memory defaulted to the same value and the sort silently degenerated into plain insertion order. A prior memtrust fix built a ranking-quality signal and marked this PASS, on the theory that it could now detect that exact failure shape.

Once we traced the real code paths after the rewrite, the fix and the bug turned out to live in two different parts of the vendor's own codebase entirely. Kartalops's bug is in `mempalace/layers.py`, in a function called `Layer1.generate()`, the actual "wake-up" sort. memtrust's adapter has never called that function, in any version, including the fictional one. What memtrust actually calls is `tool_search`, a structurally separate code path that sorts by vector similarity, not by `importance` at all. The ranking-quality signal we built is real and does catch a real class of ranking degradation, a different one. It was checking a field that belongs to a method this adapter has never invoked.

A capability gap sits a category above a parsing bug or a missed edge case: there is currently no way for memtrust to reach, measure, or detect the specific bug Kartalops reported, through any code path this adapter has. The verdict moved from PASS to FAIL, capability gap, which is the honest label for "we cannot currently do this," not "we mostly do this but missed a detail."

## Where the numbers landed

Across all four backends, 197 real GitHub issues and PRs have been investigated this way, each one asking the same question: given the code as it actually exists today, can this tool diagnose or resolve the exact failure that was reported. After this re-verification and the two downgrades above:

| Verdict | Count | Share |
|---|---|---|
| PASS | 55 | 28% |
| PARTIAL | 16 | 8% |
| FAIL, capability gap | 42 | 21% |
| FAIL, not applicable | 84 | 43% |
| **Total investigated** | **197** | **100%** |

55 out of 197 is not a number we're proud of in the way a vendor is proud of a 96.6%. It's the number that was still standing after we went looking, on purpose, for exactly the kind of overclaiming this project exists to catch, and found it in our own code first. A PASS rate that survives someone actively trying to break it means something a PASS rate that's never been challenged doesn't.

## Where this still falls short, stated plainly on purpose

This adapter is the one we can currently stand behind at this level of confidence, because it's the one we rewrote against real, live-verified functions and then independently re-checked. The other three adapters are not there yet. memtrust's own methodology documentation grades adapter confidence on a scale from High to Low, and today that grade genuinely varies: high confidence where source code or a documented SDK was read directly and exercised against real installed packages, low confidence anywhere a method signature is still a best-effort reconstruction from documentation that wasn't fully available during the build. OpenViking's adapter is explicitly flagged as the one most likely to need correction against a live instance, because its public documentation covers resource and skill ingestion in detail but never surfaced a confirmed endpoint for writing or querying a conversational memory entry specifically.

This fix settles something narrower than every number in this project. One specific, previously false claim is now either genuinely true or honestly labeled as not yet true, and the process used to get there, independent re-verification by reviewers who didn't build the original fix, is the one we intend to keep pointing at the rest of the codebase.

The test suite behind all of this currently runs 580 passing tests, 8 skipped, entirely offline, no live vendor credentials required (`pytest -q --no-cov`, confirmed at time of writing). That number proves the eval logic is internally consistent. It does not, on its own, prove any single adapter's wire format matches a live vendor's real API, which is exactly why the re-verification step above exists as a separate, independent check rather than something the test suite alone can stand in for.

## How people currently try to answer whether an AI benchmark's number is real

Set aside memtrust specifically for a moment. If you're evaluating an AI-agent memory backend today, or any AI system whose vendor publishes its own accuracy claims, you have roughly three existing paths to figuring out whether a number is real, and each one leaves a gap.

**General-purpose LLM and RAG evaluation frameworks** give you real, usable metrics: faithfulness, answer relevancy, hallucination detection, and custom LLM-as-judge scoring, built for evaluating a single model's output against a single-turn or single-session query. We checked the documentation of the three most widely used open-source frameworks in this category directly. None of them ship a first-class way to test memory specifically: recall across separate sessions days apart, what happens when a new fact contradicts an old one, or how retrieval quality decays as the knowledge base ages. The closest built-in metric, in the framework where one exists at all, tracks retained facts "throughout a conversation" by its own documentation's wording, meaning one continuous session, not the multi-session boundary an agent-memory product is actually supposed to solve. You can build that evaluation on top of these frameworks' general primitives. Nobody ships it out of the box.

**Vendor self-reported benchmarks** are the default path, because they're the ones already sitting in the product's own documentation. Set dishonesty aside as the explanation: the structural problem is that self-grading and public grading face different incentives, and there's a documented asymmetry in how freely that grading gets checked. A 2024 industry analysis found that four of thirteen vector-database vendors it reviewed contractually prohibited customers from publishing independent benchmark results against them (BenchANT, "To Benchmark Vector Databases or to Get Sued for Breaching a DeWitt Clause?", 2024). Vendors remain free to publish comparative numbers about themselves. Whether someone else can check that number against reality is, in a documented number of real cases, a decision the vendor's own contract makes for you.

**Academic benchmark suites** are the strongest, most rigorous foundation available, and they're also the ones with the least reason to overstate a result. LongMemEval, from researchers at UCLA, Tencent AI Lab, and UC San Diego, tests five distinct long-term memory abilities across curated question sets embedded in chat histories that scale up to 500 sessions and roughly 1.5 million tokens, and its own headline finding is not flattering to the systems it tested: ChatGPT, running on GPT-4o, answered 91.8% of questions correctly when given the full conversation directly, and only 57.7% once its own memory retrieval sat between the model and the answer (Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory," ICLR 2025). LoCoMo, from UNC Chapel Hill, USC, and Snap Research, built very long synthetic conversations and found an even starker gap: human accuracy on its question-answering task measured 87.9 F1, while the best model tested, GPT-4-turbo, reached 32.4 (Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents," ACL 2024). Both are exactly the kind of rigorous, adversarial ground truth a vendor benchmark should be measured against. Neither one, on its own, tells you whether a specific vendor's product, wired up the way you'd actually deploy it, behaves the way the vendor's marketing page says it does, or catches the specific failure a real user already hit in production.

A persistent gap runs through all three. None of them make it someone else's job, on an ongoing basis, to point an adversarial standard at real, reported production failures against a specific vendor's actual shipped product, republish the raw evidence, and be willing to take back a result once it's shown to be wrong. That's the specific job an independent, continuously re-audited benchmark harness, one that treats its own prior verdicts as no more trustworthy than a vendor's until they've survived a fresh, adversarial check, is built to do.

## What we'd actually change, based on what broke

Obviously, verify vendor APIs before you ship an adapter. The narrower, less comfortable lesson underneath that one: a tool built specifically to catch overclaiming is not exempt from producing it, and the only thing that caught it here was treating our own PASS verdicts with the same suspicion we point at everyone else's, on a schedule, not just when something felt off.

That's also the open question we don't have a settled answer to yet. Right now, this re-verification was a one-time, deliberate pass, triggered by discovering one specific fictional API. It was not a standing, automated part of every release. Should re-verification against fresh reviewers be a recurring, scheduled part of this project's own release process, the way a security audit or a dependency scan is, rather than something that only happens after a bug this obvious forces it? Or does that kind of continuous self-audit only work as a targeted response to a specific, credible reason to doubt a specific claim, the way it worked here? We're genuinely unsure, and we'd rather ask the people who'll actually be reading these PASS/FAIL rows than guess.

Part 2 of this series looks at what the same standard found when it was pointed at the other three backends, and where the remaining 145 rows outside this one vendor still carry the same open confidence gaps this piece just admitted to.

**Repo:** [github.com/RudrenduPaul/memtrust](https://github.com/RudrenduPaul/memtrust). The full 197-row validation log, commit by commit, is in the repo if you want to check any of the numbers above yourself, which is the entire point of publishing them this way.

Would you want a benchmark tool like this to publish a permanent, append-only "verdicts we took back" log as a first-class part of every report, right next to the PASS count, or does burying a correction inside a changelog entry (the way this rewrite currently is) do enough?

---

### References

- Stanford Institute for Human-Centered Artificial Intelligence. [The 2026 AI Index Report.](https://hai.stanford.edu/ai-index/2026-ai-index-report) 2026.
- [MemPalace/mempalace#27: "Multiple issues between README claims and codebase."](https://github.com/MemPalace/mempalace/issues/27) Opened April 7, 2026.
- [MemPalace/mempalace#29: "Multiple issues with benchmark methodology and scoring."](https://github.com/MemPalace/mempalace/issues/29) Opened April 7, 2026.
- BenchANT. ["To Benchmark Vector Databases or to Get Sued for Breaching a DeWitt Clause?"](https://benchant.com/blog/vectordb-de-witt) 2024.
- Wu, Wang, Yu, Zhang, Chang, Yu. ["LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory."](https://arxiv.org/abs/2410.10813) ICLR 2025, arXiv:2410.10813.
- Maharana, Lee, Tulyakov, Bansal, Barbieri, Fang. ["Evaluating Very Long-Term Conversational Memory of LLM Agents."](https://arxiv.org/abs/2402.17753) ACL 2024, arXiv:2402.17753.
- Recht, Roelofs, Schmidt, Shankar. ["Do ImageNet Classifiers Generalize to ImageNet?"](https://arxiv.org/abs/1902.10811) ICML 2019, arXiv:1902.10811.

---

*Rudrendu Paul is a founder and engineer building independent, agent-native AI infrastructure tooling, including `memtrust` and a companion suite of AI-agent evaluation projects. Find the code at [github.com/RudrenduPaul](https://github.com/RudrenduPaul).*
