"""memtrust's vendor embedder-call counting and cost-attribution eval for
`Mem0DirectAdapter`.

Motivating case: spike-spiegel-21 (rank 104, mem0ai/mem0#1900, "[improvement]:
Duplicate embedding generation removed", merged). That PR's own description:
"If the new memory is not modified after `new_memories_with_actions`, we can
utilise the embedding that were already created while `search`." memtrust
had zero vendor-side embedder-call instrumentation anywhere --
`scoring/cost_tracker.py` tracked only memtrust's own judge-LLM spend, never
what a backend's own embedding API actually did per call. This eval and the
`CostTracker.record_embed_calls()`/`EmbedCallEntry` primitive it depends on
(see `mem0_direct_adapter.py`'s `_CountingEmbedder`) close that
instrumentation gap.

Honest scope -- read this before trusting a "redundant re-embed" verdict.
PR #1900 targeted an OLDER mem0 architecture: a separate LLM-driven
UPDATE/DELETE decision pass (`new_memories_with_actions`) that reused an
embedding already computed during an earlier search step. That code path is
gone in the currently installed `mem0ai==2.0.12`: confirmed by reading
`mem0/memory/main.py`'s `_add_to_vector_store()` directly, its "V3 PHASED
BATCH PIPELINE" only ever embeds the incoming query text once (Phase 1, for
similarity search against existing memories) and then unconditionally
batch-embeds every LLM-extracted fact (Phase 3) with no code path that
checks whether an extracted fact matches content already embedded in Phase
1. This eval therefore does not, and cannot, reproduce PR #1900's original
diff -- the same honest substitution shape `Mem0DirectAdapter`'s own module
docstring already establishes for mem0ai/mem0#3558 (Kuzu): it reproduces the
bug *class* (vendor embedder-call instrumentation surfacing whether a
redundant re-embed happens) against the pipeline that actually exists today,
not a literal re-run of a since-superseded code path.

What this eval actually measures: seed a record via `store()`, `query()`
it back (the "search()" half of PR #1900's own phrasing), then `store()`
the exact same, unmodified content again (the "then add()" half) --
and reports whether the second `store()` call still triggered a real
vendor embedder call. Every `store()` call in the currently installed
package's pipeline does its own independent batch-embed with no reuse
logic at all, so a REDUNDANT_REEMBED verdict against a live mem0 instance
here should be read as "this is simply how the installed package's
pipeline behaves today," not as a regression this build discovered --
this eval exists to make that behavior *observable* through memtrust,
which it previously could not do.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from memtrust.adapters.base import BackendAPIError
from memtrust.adapters.mem0_direct_adapter import Mem0DirectAdapter
from memtrust.scoring.cost_tracker import CostTracker

DEFAULT_SESSION_ID = "embedder-cost-session"
DEFAULT_CONTENT = "The user's preferred deploy window is Tuesday mornings."


class EmbedderCostSignal(StrEnum):
    """Defined locally in this eval module rather than added to `base.py`,
    following the precedent `evals/scale_stress.py`'s local `ScaleSignal`
    enum already establishes for a capability that is not part of every
    adapter's shared `MemoryBackendAdapter` interface -- this eval is
    `Mem0DirectAdapter`-specific (it depends on `_CountingEmbedder`
    wrapping a real, installed mem0 embedder instance), not something
    every adapter could meaningfully report.
    """

    REDUNDANT_REEMBED = "redundant_reembed"
    """The second `store()` call -- for the exact same, unmodified content
    a prior `store()`+`query()` sequence already embedded -- still
    triggered at least one real vendor embedder call. See module
    docstring for the honest scope of what this does and does not prove
    against the currently installed `mem0ai` package."""

    NO_REDUNDANT_REEMBED = "no_redundant_reembed"
    """The second `store()` call triggered zero vendor embedder calls --
    the backend genuinely reused an existing embedding for unmodified
    content instead of re-embedding it. The good outcome PR #1900's own
    fix (for a now-superseded code path) achieved."""

    NOT_APPLICABLE = "not_applicable"
    """The store()/query() call sequence raised BackendAPIError, or this
    adapter has no `cost_tracker`/embedder surface to observe at all --
    recorded explicitly, never silently read as either signal above, same
    convention every other signal enum's NOT_APPLICABLE member in this
    package follows."""


@dataclass
class EmbedderCostEvalResult:
    backend_name: str
    signal: EmbedderCostSignal
    first_store_embed_calls: int
    """Vendor embedder-call count `_CountingEmbedder` observed for the
    FIRST store() call (seeding the record)."""
    second_store_embed_calls: int
    """Vendor embedder-call count observed for the SECOND store() call --
    the one this eval's verdict is actually about."""
    error: str | None = None


def run_embedder_cost_eval(
    adapter: Mem0DirectAdapter,
    cost_tracker: CostTracker,
    session_id: str = DEFAULT_SESSION_ID,
    content: str = DEFAULT_CONTENT,
) -> EmbedderCostEvalResult:
    """Seed a record, query() it back (search()), then store() the exact
    same unmodified content again (add()) -- see module docstring for the
    real PR #1900 phrasing this mirrors -- and classify whether the second
    store() call triggered a redundant vendor embedder call.

    `cost_tracker` must be the same `CostTracker` instance `adapter` was
    constructed with (`Mem0DirectAdapter(..., cost_tracker=cost_tracker)`)
    -- this function reads `cost_tracker.embed_entries` after each
    store() call to observe what `_CountingEmbedder` counted, rather than
    requiring a new field on `StoreResult` (keeping this eval's scope to
    exactly `mem0_direct_adapter.py` + `cost_tracker.py` + this new eval
    module, per the backlog item's own `Where:` scope -- no `base.py`
    change).
    """
    try:
        adapter.store(session_id, content)
        first_calls = cost_tracker.embed_entries[-1].call_count if cost_tracker.embed_entries else 0

        adapter.query(session_id, content)
        adapter.store(session_id, content)
        second_calls = (
            cost_tracker.embed_entries[-1].call_count if cost_tracker.embed_entries else 0
        )
    except BackendAPIError as exc:
        return EmbedderCostEvalResult(
            backend_name=adapter.name,
            signal=EmbedderCostSignal.NOT_APPLICABLE,
            first_store_embed_calls=0,
            second_store_embed_calls=0,
            error=str(exc),
        )

    if not cost_tracker.embed_entries:
        # adapter.cost_tracker wasn't wired to the same CostTracker, or
        # this adapter's underlying memory has no embedding_model surface
        # to observe at all -- see Mem0DirectAdapter.store()'s gating.
        return EmbedderCostEvalResult(
            backend_name=adapter.name,
            signal=EmbedderCostSignal.NOT_APPLICABLE,
            first_store_embed_calls=0,
            second_store_embed_calls=0,
            error="no embed-call instrumentation observed -- adapter may not share "
            "this cost_tracker, or its memory.embedding_model is unavailable",
        )

    signal = (
        EmbedderCostSignal.REDUNDANT_REEMBED
        if second_calls > 0
        else EmbedderCostSignal.NO_REDUNDANT_REEMBED
    )
    return EmbedderCostEvalResult(
        backend_name=adapter.name,
        signal=signal,
        first_store_embed_calls=first_calls,
        second_store_embed_calls=second_calls,
    )
