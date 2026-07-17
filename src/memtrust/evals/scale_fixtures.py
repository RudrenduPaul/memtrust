"""Deterministic large-scale synthetic corpus generator for
evals/scale_stress.py.

Every other fixture in this repo (`tests/fixtures/*.json`) is a hand-written
file with 3-7 examples -- fine for exercising contradiction/ranking/
resource-sync logic, but 3-5 orders of magnitude below the ~100K-record or
~300-episode scale where several real, documented vendor bugs actually
manifest:

  * volcengine/OpenViking#2850 (lg320531124, still open): BM25 search
    silently returning empty results once a corpus grows large. A
    hand-written 5-record fixture structurally cannot reach the corpus size
    where this shows up.
  * getzep/graphiti#1275 (rafaelreis-r): O(n) entity-resolution context
    growth causing episodes to be silently dropped once ingestion passes
    roughly 300 episodes. Same problem -- no bundled fixture has 300 of
    anything.

A hand-written JSON file cannot solve this: nobody is going to hand-type
10,000 realistic-looking memory facts into a fixture file, and even if they
did, it would not be reproducible/adjustable the way a parameterized
generator is. This module is a function, not a file -- `generate_scale_corpus
(n, seed=...)` produces N synthetic records with realistic-shaped content,
deterministically, for any N a caller asks for (5 or 100,000), so
evals/scale_stress.py can be pointed at whatever scale a real run needs
without checking a giant file into git.

Design notes:
  * Determinism is via `random.Random(seed)` -- the same (n, seed) pair
    always produces byte-identical output, which is what makes a scale run
    reproducible across two separate `memtrust run` invocations.
  * Every record's content embeds a unique, greppable marker token
    (`SCALEMARK{index:06d}`) that appears nowhere else in the corpus. This
    is deliberate: it turns "is this specific record still findable" into a
    literal substring/keyword search, the same shape of query BM25-style
    lexical search actually serves, and the shape #2850 concerns (a literal
    keyword search coming back empty). A record's `marker` field is what
    evals/scale_stress.py queries for when it wants to test recall of one
    specific record.
  * The generated content is deliberately fact-shaped prose (a name, a
    topic, an action, a detail), not lorem-ipsum noise or a bare marker
    string -- so a backend that does real embedding/tokenization work has
    something realistic to index, not a degenerate one-token input that
    would never occur in a real ingestion pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

#: Format for each record's unique, greppable marker token. Six digits
#: comfortably covers every N this module is meant to be used at (up to
#: 999,999); a caller requesting more than that gets a ValueError rather
#: than silently colliding marker tokens.
_MARKER_FORMAT = "SCALEMARK{index:06d}"
_MAX_RECORDS = 999_999

_FIRST_NAMES = (
    "Priya",
    "Jordan",
    "Sam",
    "Alex",
    "Morgan",
    "Taylor",
    "Casey",
    "Jamie",
    "Riley",
    "Devon",
    "Nadia",
    "Kenji",
    "Elena",
    "Marcus",
    "Ines",
    "Tobias",
    "Ling",
    "Omar",
    "Freya",
    "Dante",
)
_TOPICS = (
    "the quarterly budget review",
    "the onboarding checklist",
    "the API rate limit increase",
    "the deployment pipeline",
    "the customer escalation queue",
    "the on-call rotation",
    "the vendor security questionnaire",
    "the data retention policy",
    "the staging environment refresh",
    "the incident postmortem",
    "the roadmap planning doc",
    "the support ticket backlog",
    "the pricing experiment",
    "the migration runbook",
    "the compliance audit",
)
_ACTIONS = (
    "moved to",
    "was updated to reference",
    "got flagged for follow-up regarding",
    "was rescheduled around",
    "now depends on",
    "was assigned to",
    "was blocked by",
    "was reprioritized after",
    "was archived following",
    "needs another look because of",
)
_DETAILS = (
    "Thursday afternoon",
    "the new staging environment",
    "ticket ORT-4471",
    "the Q3 roadmap",
    "last week's incident review",
    "the new compliance requirement",
    "a customer escalation from EMEA",
    "the vendor's revised SLA",
    "the platform team's capacity",
    "next sprint's planning session",
)


@dataclass
class ScaleFixtureRecord:
    """One synthetic record in a generated scale corpus.

    `index` is this record's 0-based position in generation order --
    evals/scale_stress.py uses it as the "insertion order" ground truth,
    the same role `RankingQualitySeedRecord`'s list position plays in
    evals/ranking_quality.py. `marker` is the unique greppable token
    embedded in `content`; querying for it (rather than for arbitrary
    natural-language text) is what turns "is this record still
    recoverable" into a deterministic, unambiguous substring check.
    """

    index: int
    content: str
    marker: str
    metadata: dict[str, str] = field(default_factory=dict)


def generate_scale_corpus(
    n: int, seed: int = 42, session_id: str = "scale-stress-session"
) -> list[ScaleFixtureRecord]:
    """Generate `n` deterministic, realistic-shaped synthetic records.

    Same (n, seed) always produces the same output -- this is what lets a
    scale run be reproduced exactly (e.g. to confirm a degradation finding
    wasn't a one-off fluke) without checking a giant fixture file into git.
    Architected to scale from a handful of records (a fast CI smoke test)
    up to 10K+ (a real stress run against a live backend); the only cost of
    a larger N is the loop below, there is no fixed-size data file to
    outgrow.

    Args:
        n: how many records to generate. Must be >= 1 and <= 999,999 (see
            `_MAX_RECORDS` -- the marker format's fixed width).
        seed: random seed. Same seed -> byte-identical corpus.
        session_id: stamped into each record's metadata for callers that
            want it; scale_stress.py's own session scoping is independent
            of this (it passes its own session_id to store()/query()).

    Raises:
        ValueError: if `n` is out of the supported range.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n > _MAX_RECORDS:
        raise ValueError(f"n must be <= {_MAX_RECORDS}, got {n}")

    rng = random.Random(seed)
    records: list[ScaleFixtureRecord] = []
    for index in range(n):
        name = rng.choice(_FIRST_NAMES)
        topic = rng.choice(_TOPICS)
        action = rng.choice(_ACTIONS)
        detail = rng.choice(_DETAILS)
        marker = _MARKER_FORMAT.format(index=index)
        content = f"On day {index}, {name} noted that {topic} {action} {detail}. (ref: {marker})"
        records.append(
            ScaleFixtureRecord(
                index=index,
                content=content,
                marker=marker,
                metadata={"scale_index": str(index), "session_id": session_id},
            )
        )
    return records
