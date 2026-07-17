"""Direct-library adapter for Mem0: constructs a real `mem0.Memory` object
via `Memory.from_config()`, in-process, instead of going over HTTP like
`Mem0Adapter`/`Mem0SelfHostedAdapter` in `mem0_adapter.py`.

## Why this adapter exists

Neither of the two REST adapters above can select a `graph_store` provider,
an `embedder` provider, or a `vector_store` provider -- those are
construction-time Python config (`MemoryConfig(embedder=..., vector_store=...,
graph_store=...)`), not REST parameters either the hosted Platform API or the
self-hosted server's `POST /memories` / `POST /search` bodies expose. Five
real, cited mem0 bug reports trace to exactly that unreachable configuration
surface:

  1. mem0ai/mem0#3558 (merged) -- `kuzu_memory.py` raised `ValueError` if
     `embedding_dims` was `None`/`<=0` before opening the Kuzu graph DB.
  2. mem0ai/mem0#5671 (merged, fixed upstream) -- `aws_bedrock.py` now
     forwards `embedding_dims` to Bedrock Titan V2 requests.
  3. mem0ai/mem0#4362 (merged, fixed upstream) -- `redis.py`/`valkey.py`
     now guard against a silent zero-vector corruption on metadata-only
     updates (`if vector is not None:` before overwriting `"embedding"`).
  4. mem0ai/mem0#4711 (merged, fixed upstream) -- `FastEmbedEmbedding` now
     reads the real dimension from the loaded model at init instead of
     defaulting to 1536.
  5. mem0ai/mem0#2304 (merged, fixed upstream) -- Gemini/OpenAI embedders
     now forward `embedding_dims`/`dimensions` into the embed call instead
     of silently dropping it.
  6. mem0ai/mem0#4297 (fixed in the TS SDK; the Python OSS SDK's Qdrant
     path was never patched and still exhibits the bug *class* today --
     see "Qdrant support" below) -- a vector-store collection created with
     a hardcoded/default 1536-dim size regardless of the actual embedder's
     output dimension, causing a dimension-mismatch `Bad Request` on
     insert for any non-1536-dim embedder.
  7. mem0ai/mem0#4453 (companion bug to #4297; confirmed fixed in the
     installed Python package -- see "Qdrant support" below) -- threshold
     filtering assuming similarity (higher=better) when a vector store
     returns raw distance (lower=better), silently inverting which results
     pass the threshold.
  8. mem0ai/mem0#5980 (merged, GitHub user HrushiYadav; confirmed fixed in
     the installed Python package -- see "Elasticsearch support" below) --
     `ElasticsearchVectorStore` embedded caller-supplied filter values
     directly into Elasticsearch `term` queries with no type validation, so
     a dict/list-valued filter value (e.g. `{"user_id": {"$ne": ""}}`)
     could inject arbitrary Elasticsearch query-DSL operators, enabling
     access-control bypass / cross-user memory enumeration. Part of a
     coordinated 5-backend injection-prevention series covering
     elasticsearch/neptune/azure/opensearch/databricks; this adapter and
     its eval only concern the elasticsearch fix.

## Confidence and what was actually confirmed against the real package

Confidence: HIGH on what the installed package's code actually does --
every claim below was confirmed by reading the *installed* `mem0ai==2.0.12`
source directly (the newest version on PyPI as of this build, 2026-07-16;
`pip index versions mem0ai` lists no newer release), not by re-reading the
GitHub issues or guessing method signatures. LOW on live end-to-end
behavior: this adapter has never been run against a live Redis/Valkey
server, a live Bedrock/Gemini/OpenAI embedding endpoint, or a live FastEmbed
model download in this environment -- see docs/methodology.md.

**mem0ai#5671, #4362, #4711, #2304 are all confirmed fixed in the installed
package**, by direct source inspection:

  * `mem0/embeddings/aws_bedrock.py::_get_embedding` forwards
    `self.config.embedding_dims` as `input_body["dimensions"]` whenever it
    is set and the model string contains `"v2"` -- exactly the #5671 fix.
  * `mem0/embeddings/fastembed.py::FastEmbedEmbedding.__init__` sets
    `self.config.embedding_dims = self.dense_model.embedding_size` when the
    caller didn't set one -- exactly the #4711 fix (no more hardcoded 1536
    default).
  * `mem0/embeddings/gemini.py::GoogleGenAIEmbedding.embed`/`embed_batch`
    build an `EmbedContentConfig(output_dimensionality=self.config.
    embedding_dims)` and pass it to every `embed_content` call -- and
    `mem0/embeddings/openai.py::OpenAIEmbedding.embed`/`embed_batch` pass
    `dimensions=self.config.embedding_dims` in the request kwargs whenever
    the caller set one -- exactly the #2304 fix for both vendors.
  * `mem0/vector_stores/redis.py::RedisDB.update` and
    `mem0/vector_stores/valkey.py::ValkeyDB.update` both guard
    `if vector is not None: data["embedding"] = np.array(vector, ...)` --
    exactly the #4362 fix. Neither store zeroes or drops the existing
    embedding when `vector` is omitted.

`tests/test_mem0_direct_adapter.py` exercises the real, installed
`AWSBedrockEmbedding`/`GoogleGenAIEmbedding`/`OpenAIEmbedding`/
`FastEmbedEmbedding`/`RedisDB`/`ValkeyDB` classes directly (mocking only
each vendor's own network/model-load boundary -- `boto3`'s client, the
`openai`/`google.genai` SDK clients, `fastembed.TextEmbedding`, and the
`redis`/`valkey` wire client), so a regression that reintroduced any of
these four bugs in a future `mem0ai` release would fail those tests against
the *installed* package, not against a copy of the fixed logic memtrust
reimplemented itself. That is what justifies treating these four as
re-validated PASS against the currently pinned `mem0ai` version, not just
"the GitHub issue says merged."

**Qdrant support (`vector_store_provider="qdrant"`) and #4297/#4453 --
mixed result, confirmed by reading the installed `mem0.vector_stores.qdrant`,
`mem0.configs.vector_stores.qdrant`, `mem0.vector_stores.base`, and
`mem0.utils.scoring` source directly, not by trusting either GitHub issue's
"merged" status.**

*#4297 (embedding-dimension mismatch): the bug class is still reachable in
the installed Python package, not fixed.* `QdrantConfig.embedding_model_dims`
(`mem0/configs/vector_stores/qdrant.py`) defaults to `1536` -- the exact
same hardcoded OpenAI dimension #4297's TS SDK fix removed for the
equivalent JS config -- and nothing in `Memory.__init__`/`Memory.from_config()`
(`mem0/memory/main.py`) reconciles that default against the embedder's real
output dimension: `EmbedderFactory.create(...)` and
`VectorStoreFactory.create(...)` are constructed independently, from
`config.embedder.config` and `config.vector_store.config` respectively,
with no cross-check between them. `tests/test_mem0_direct_adapter.py`
exercises the real, installed `mem0.vector_stores.qdrant.Qdrant` class
directly (mocking only the `QdrantClient` wire boundary) and shows it
creates a collection sized to whatever `embedding_model_dims` it is given
-- 1536 if a caller relies on `QdrantConfig`'s own default, the wrong size
for any non-1536-dim embedder (e.g. `fastembed`'s 384-dim model), and the
correct size only when a caller explicitly overrides it. This is exactly
the failure shape #4297 originally described for the TS SDK, still present
in the Python OSS package today. `Mem0DirectAdapter` itself never hits
this bug in practice, for the same reason it never hits an analogous
class of bug for Redis/Valkey: `_build_vector_store_config()` below always
threads the resolved embedder dimension into `embedding_model_dims`
explicitly, the same defensive pattern already used for
`vector_store_provider="redis"`/`"valkey"`. That is a property of this
adapter's own config-building code, not evidence that mem0ai's Qdrant path
was fixed upstream -- a caller constructing `mem0.Memory.from_config()`
directly, without this adapter, remains exposed.

*#4453 (search-threshold inversion): confirmed fixed in the installed
package, comprehensively, not just for Qdrant.* `mem0.vector_stores.base
.VectorStoreBase.search()`'s docstring states an explicit, binding
contract: "All implementations must return similarity scores where higher
values indicate greater similarity ... Implementations using distance
metrics must convert to similarity before returning." Every vector-store
implementation this build inspected in the installed package complies --
`chroma.py`, `faiss.py`, `milvus.py`, `pgvector.py`, `redis.py`,
`valkey.py`, `supabase.py`, `turbopuffer.py`, `vertex_ai_vector_search.py`,
and `s3_vectors.py` all convert a raw distance to a `[0, 1]`-ish similarity
score (`max(0, 1 - distance)` or `1 / (1 + distance)`) before returning it;
`qdrant.py`, `pinecone.py`, `weaviate.py`, `opensearch.py`,
`elasticsearch.py`, `azure_ai_search.py`, and `mongodb.py` return each
vendor's own already-higher-is-better native score directly, with no
conversion needed. `mem0.utils.scoring.score_and_rank()` -- the real
function `Memory._search_vector_store()` calls to apply `threshold` --
then filters with `if semantic_score < threshold: continue`, i.e. "drop
anything below the threshold, keep the rest," which is only correct
because every store's `score` is already similarity-oriented by the time
it gets there. `tests/test_mem0_direct_adapter.py` exercises this real
function directly with similarity-shaped scores and confirms it keeps the
closest matches and drops the farthest, plus a clearly-labeled hypothetical
test showing what would happen if a store *didn't* comply (the best match
gets dropped and the worst kept -- the literal inversion #4453 described)
to make the contract's importance concrete without claiming any real,
installed store violates it. `Mem0DirectAdapter.query()`'s new `threshold`
parameter (forwarded straight through to `Memory.search()`) is what makes
this reachable through this adapter at all -- the same role
`Mem0SelfHostedAdapter.query()`'s `threshold` parameter plays over REST.

**Elasticsearch support (`vector_store_provider="elasticsearch"`) and
#5980 -- confirmed fixed in the installed package, by reading
`mem0.vector_stores.elasticsearch` and `mem0.configs.vector_stores.elasticsearch`
source directly, not by trusting the GitHub PR's "merged" status.**
`mem0/vector_stores/elasticsearch.py` defines a module-level
`_validate_filter(key, value)` helper (compiled against
`_SAFE_FILTER_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")` for the key,
and `isinstance(value, (str, int, float, bool))` for the value) and calls
it on every `(key, value)` pair immediately before building a `{"term":
{f"metadata.{key}": value}}` clause, in all three places
`ElasticsearchDB` constructs term-query filter clauses:
`search()` (the KNN pre-filter path), `keyword_search()` (the BM25 path),
and `list()`. A dict/list-valued filter -- the exact #5980 shape, e.g.
`{"user_id": {"$ne": ""}}` -- fails the `isinstance` check and raises
`ValueError` before the query is ever built or sent to the Elasticsearch
client, closing the access-control-bypass/cross-user-enumeration path the
issue describes. This was confirmed against real, merged upstream state:
`gh pr view 5980 --repo mem0ai/mem0` shows PR #5980 ("fix(elasticsearch):
validate filter keys and values to prevent term injection", author
HrushiYadav, merged 2026-07-02, closing #5976) describes exactly this
`_validate_filter()` addition, and the installed `mem0ai==2.0.12` source
matches that description line for line -- this is not a case where the
issue says "merged" but the pinned version predates the fix.
`tests/test_mem0_direct_adapter.py` exercises the real, installed
`mem0.vector_stores.elasticsearch.ElasticsearchDB` class directly (mocking
only the `elasticsearch.Elasticsearch` wire client) with both a benign,
scalar-valued filter (accepted, forwarded to a real `client.search()`
call) and a malicious dict-valued filter (rejected with `ValueError`
before the client is ever touched) -- see `evals/filter_injection.py` for
the harness-level eval built on top of this, via
`Mem0DirectAdapter.probe_raw_filter()` and `MemoryBackendAdapter.
supports_raw_filter_probe`. `ElasticsearchConfig.embedding_model_dims`
(`mem0/configs/vector_stores/elasticsearch.py`) is a plain `int` field
defaulting to `1536`, the same field name Redis/Valkey/Qdrant configs use
-- `_build_vector_store_config()` below threads the resolved embedder
dimension into it the same defensive way it already does for the other
three providers, so this adapter never relies on that 1536 default either.
`ElasticsearchConfig` also requires either `cloud_id` or `host`, and
either `api_key` or `user`+`password` (a `model_validator(mode="before")`
raises `ValueError` otherwise) -- this adapter threads
`MEM0_DIRECT_VECTOR_STORE_URL`/`MEM0_DIRECT_ELASTICSEARCH_URL` into
`host` and an optional `MEM0_DIRECT_ELASTICSEARCH_API_KEY` into `api_key`,
but does not synthesize a fake credential when neither is set: a caller
who configures `vector_store_provider="elasticsearch"` with no API key
and no `user`/`password` override in `vector_store_config` gets a real
`pydantic.ValidationError` from the installed package at first use,
caught and reported as `CorruptionSignal.CONFIG_REJECTED` -- the same
honest "fail loudly and specifically" shape Valkey's missing-dims case
already establishes above, not a bug this adapter works around silently.
This adapter has never been run against a live Elasticsearch cluster in
this environment -- see "What this adapter is NOT" below.

**mem0ai#3558 (Kuzu) cannot be reproduced against the installed package --
the code path it lived in does not exist here.** This was the one
surprising finding of this build: `mem0ai==2.0.12`'s `MemoryConfig`
(`mem0/configs/base.py`) has no `graph_store` field at all, no `kuzu`
dependency appears in any of the package's declared extras
(`mem0ai[extras]`, `[vector-stores]`, `[llms]`, ...), and no graph/kuzu
module exists anywhere under the installed `mem0/` tree (confirmed via
`grep -rl kuzu` over the full installed package -- the only hit is an
illustrative string in `mem0/exceptions.py`'s `DependencyError` docstring,
not a real code path). `mem0ai/mem0#3558`'s `kuzu_memory.py` is not present
in this release; whatever GitHub state that issue and its merged fix
targeted either predates this PyPI release or was refactored out of the
OSS package entirely. Worse, this was not just "unsupported" -- it was
**silently unsupported**: `MemoryConfig(graph_store={"provider": "kuzu",
...})` raises nothing and produces a `MemoryConfig` with no trace the
`graph_store` key was ever given (pydantic's default `extra="ignore"`
behavior on this model, confirmed empirically during this build). That is
exactly the kind of silent no-op `MemoryBackendAdapter`'s own class
docstring in `base.py` says an adapter must never let happen ("If a
backend cannot support an operation ... report that fact ... rather than
faking a response"). So this adapter refuses a `graph_store_provider`
request outright, at construction, with a message naming this finding,
instead of passing it through to `Memory.from_config()` and pretending
graph-store selection did anything.

Because that removes the one bug this adapter was originally supposed to
make reachable, `Mem0DirectAdapter` instead reproduces the *bug class*
#3558 established -- a backend rejecting a missing/invalid
embedding-dimension config at construction time, before any store()/
query() call can silently corrupt state -- against a component that
*does* still exist and still has exactly that validation shape:
`mem0.configs.vector_stores.valkey.ValkeyConfig.embedding_model_dims` is a
required (no-default) `int` field, so constructing this adapter with
`vector_store_provider="valkey"` and no `embedding_model_dims` raises a
real `pydantic.ValidationError` (a `ValueError` subclass) from the
installed package, at the same point in the lifecycle (construction,
before any write) that the original Kuzu bug's `ValueError` fired. This is
an honest substitution for the retired code path, not a re-creation of it
-- see `CorruptionSignal.CONFIG_REJECTED` in `base.py` and
`test_config_rejected_...` in the test module for exactly what is and is
not being claimed.

## What this adapter is NOT

It does not support `graph_store` (see above -- refused outright, loudly).
It does not implement every vector-store or embedder provider mem0 ships,
only the ones the eight motivating bugs concern: embedders `openai`,
`aws_bedrock`, `gemini`, `fastembed`; vector stores `redis`, `valkey`,
`qdrant`, `elasticsearch`. It has never been run against a live backend of
any kind in this environment -- every test mocks the vendor SDK/wire-client
boundary, same convention `docs/methodology.md` already documents for
`mem0_adapter.py`. `query()`'s `threshold` parameter is forwarded to the
real `Memory.search()` unexercised against a live Qdrant server -- see the
Qdrant support section above for exactly what was and was not confirmed
about #4297/#4453 against the installed package alone. `probe_raw_filter()`
(see base.py) is exercised against the real, installed
`mem0.vector_stores.elasticsearch.ElasticsearchDB` class with the
`Elasticsearch` wire client mocked, never against a live Elasticsearch
cluster -- see the Elasticsearch support section above for exactly what
was and was not confirmed about #5980 against the installed package alone.

## Configuration

Gated on `MEM0_DIRECT_EMBEDDER_PROVIDER` (one of `openai`, `aws_bedrock`,
`gemini`, `fastembed`) as the primary "one env var, or SKIPPED" switch --
this backend is not in `cli.ALL_BACKENDS`, the same opt-in-only precedent
`Mem0SelfHostedAdapter` sets, since it targets a self-assembled in-process
stack rather than a single hosted vendor API. Depending on the chosen
embedder, a provider-specific credential is also required (missing ->
`BackendNotConfiguredError` naming that exact variable, never a silent
fallback):

  * `openai` -> `OPENAI_API_KEY`
  * `aws_bedrock` -> `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
  * `gemini` -> `GOOGLE_API_KEY`
  * `fastembed` -> none (local ONNX model, no credential)

`MEM0_DIRECT_VECTOR_STORE_PROVIDER` (`redis`, `valkey`, `qdrant`, or
`elasticsearch`, default `redis`) selects the vector store;
`MEM0_DIRECT_VECTOR_STORE_URL` (or the provider-specific
`MEM0_DIRECT_QDRANT_URL`/`MEM0_DIRECT_REDIS_URL`/`MEM0_DIRECT_VALKEY_URL`/
`MEM0_DIRECT_ELASTICSEARCH_URL`) is required for whichever one is selected
-- for `elasticsearch` this value is threaded into `ElasticsearchConfig.host`,
not a `url` field (the installed config class has no such field; see the
Elasticsearch support section above), or a caller may instead pass an
explicit `cloud_id` via `vector_store_config` for Elastic Cloud. An
optional `MEM0_DIRECT_ELASTICSEARCH_API_KEY` is threaded into
`ElasticsearchConfig.api_key`; the installed package's own validator
requires either that or a `user`+`password` override in
`vector_store_config`, and raises loudly if neither is present (see
Elasticsearch support section above). `MEM0_DIRECT_EMBEDDING_DIMS`
optionally overrides the per-provider default dimension count threaded
into both the embedder config and the vector store config (including
Qdrant's and Elasticsearch's `embedding_model_dims`, which both otherwise
default to a hardcoded `1536` in the installed package -- see the Qdrant
and Elasticsearch support sections above), so they never silently
disagree. `query()` also accepts an optional `threshold` keyword argument,
forwarded straight
through to the real `Memory.search()`.

## Custom fact-extraction prompt passthrough

`Mem0DirectAdapter.__init__` accepts an optional `custom_instructions`
constructor argument, threaded straight into the top-level
`custom_instructions` key of the config dict passed to
`mem0.Memory.from_config()`. Confirmed by reading the installed
`mem0ai==2.0.12` source directly (`mem0/configs/base.py`'s `MemoryConfig`):
there is no `custom_fact_extraction_prompt` field on the installed
`MemoryConfig` at all -- mem0 renamed that field to `custom_instructions`
(mem0ai/mem0#4740, documented in the installed package's own
`docs/changelog/sdk.mdx` and `docs/migration/oss-v2-to-v3.mdx`), and
`Memory.from_config()` (`mem0/memory/main.py`) does a plain
`MemoryConfig(**config_dict)`, so `custom_instructions` is the real,
current top-level key this adapter must set, not the older, since-renamed
name.

This closes the gap mem0ai/mem0#4573 (jamebobob's 32-day production audit
-- the founding rationale for `evals/extraction_quality.py` and its
`ExtractionQualitySignal`) and a follow-up production report from GitHub
user farrrr documented: farrrr reported that rewriting mem0's default
fact-extraction prompt measurably reduced the junk-retention rate the
audit found, but `Mem0DirectAdapter` had no constructor argument to set
one, so that specific mitigation was unreachable through this adapter.
With `custom_instructions` now threaded through, a caller can construct
two adapters -- one with the default prompt, one with a rewritten
`custom_instructions` -- and compare their junk-retention rate via
`extraction_quality.py`'s existing `ExtractionQualitySignal` classification
(building an automated A/B-comparison mode for that eval is a separate,
larger item, out of scope here).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Protocol

from memtrust.adapters.base import (
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
    CorruptionSignal,
    DeleteResult,
    ExtractionSignal,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    RawFilterProbeResult,
    StoreResult,
    UpdateResult,
)

#: The four embedder providers mem0ai#5671 (aws_bedrock), #4711
#: (fastembed), and #2304 (gemini, openai) concern. This adapter
#: deliberately wires up only these four, not mem0's full provider list --
#: see module docstring.
SUPPORTED_EMBEDDER_PROVIDERS = ("openai", "aws_bedrock", "gemini", "fastembed")

#: The vector store providers this adapter wires up: `redis`/`valkey` for
#: mem0ai#4362 (vector-zeroed-on-metadata-update), `qdrant` for
#: mem0ai#4297 (embedding-dimension mismatch) and mem0ai#4453 (search-
#: threshold inversion), and `elasticsearch` for mem0ai#5980 (filter-value
#: term-injection) -- see module docstring for what this build actually
#: confirmed about each against the installed package.
SUPPORTED_VECTOR_STORE_PROVIDERS = ("redis", "valkey", "qdrant", "elasticsearch")

#: Field name each installed `mem0.configs.vector_stores.*` config class
#: uses for its connection endpoint, confirmed by reading all four
#: installed config classes directly -- `redis`/`valkey` name it
#: `redis_url`/`valkey_url`, `qdrant` names it plain `url`, and
#: `elasticsearch` has no `url` field at all, only `host`/`port`/`cloud_id`
#: (see module docstring's Elasticsearch support section). Shared by
#: `__init__`'s configuration presence-check and `_build_vector_store_config`
#: below so the two never disagree about which key a given provider uses.
_VECTOR_STORE_URL_KEYS: dict[str, str] = {
    "redis": "redis_url",
    "valkey": "valkey_url",
    "qdrant": "url",
    "elasticsearch": "host",
}

#: Per-provider embedding dimension defaults, used only when neither
#: `embedder_config`/`vector_store_config` nor `MEM0_DIRECT_EMBEDDING_DIMS`
#: specify one -- keeps the embedder and vector-store dims in agreement by
#: construction for the common case, without silently overriding a caller
#: who set one explicitly (including a deliberate `None`, which is what the
#: CONFIG_REJECTED test case below relies on).
_DEFAULT_EMBEDDING_DIMS_BY_PROVIDER: dict[str, int] = {
    "openai": 1536,
    "aws_bedrock": 1024,
    "gemini": 768,
    "fastembed": 384,
}

_EMBEDDER_CREDENTIAL_ENV_VARS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "aws_bedrock": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    "gemini": ("GOOGLE_API_KEY",),
    "fastembed": (),
}


class _MemoryProtocol(Protocol):
    """Shape this adapter needs from `mem0.Memory` (or a test double).

    Defined as a Protocol, same convention `mempalace_adapter.py` uses for
    `_PalaceProtocol`, so tests can inject a fake implementation without
    the real, network-and-credential-dependent `mem0ai` package doing any
    real vendor calls.
    """

    vector_store: Any

    def add(
        self,
        messages: object,
        *,
        user_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
        infer: bool = ...,
    ) -> dict[str, object]: ...

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, object] | None = None,
        top_k: int = ...,
        threshold: float | None = ...,
    ) -> dict[str, object]: ...

    def update(
        self, memory_id: str, text: str | None = None, metadata: Mapping[str, object] | None = None
    ) -> dict[str, object]: ...

    def delete(self, memory_id: str) -> object: ...


class Mem0DirectAdapter(MemoryBackendAdapter):
    """Adapter that holds a direct, in-process `mem0.Memory` handle.

    See this module's docstring for the full confidence breakdown and the
    Kuzu/graph_store finding in particular before trusting this adapter's
    CONFIG_REJECTED output as a literal reproduction of mem0ai/mem0#3558.
    """

    name = "mem0_direct"
    env_var = "MEM0_DIRECT_EMBEDDER_PROVIDER"
    supports_update = True
    supports_raw_filter_probe = True
    """This adapter holds a direct, in-process handle to the real
    `mem0.Memory` object, including its constructed `vector_store` -- see
    `probe_raw_filter()` below, which calls straight into
    `vector_store.list(filters=...)`, bypassing `query()`'s hardcoded
    `{"user_id": session_id}` filter. True unconditionally (not gated to
    `vector_store_provider="elasticsearch"`): every installed vector-store
    class this adapter wires up implements the same `list(filters=...)`
    signature (`mem0.vector_stores.base.VectorStoreBase.list`), so the
    probe itself always works -- what differs per provider is whether the
    underlying store validates the filter value before using it, which is
    exactly the fact evals/filter_injection.py exists to surface."""

    def __init__(
        self,
        embedder_provider: str | None = None,
        embedder_config: dict[str, object] | None = None,
        vector_store_provider: str | None = None,
        vector_store_config: dict[str, object] | None = None,
        graph_store_provider: str | None = None,
        custom_instructions: str | None = None,
        memory: _MemoryProtocol | None = None,
    ) -> None:
        if memory is not None:
            # Test-injection path: skip all env/credential resolution and
            # real mem0.Memory construction entirely. Mirrors
            # MemPalaceAdapter.__init__(palace=...) above.
            self._memory: _MemoryProtocol | None = memory
            self._construction_error: str | None = None
            self._embedder_provider = embedder_provider or "injected"
            self._vector_store_provider = vector_store_provider or "redis"
            self._custom_instructions = custom_instructions
            return

        resolved_embedder_provider = embedder_provider or os.environ.get(self.env_var)
        if not resolved_embedder_provider:
            raise BackendNotConfiguredError(self.name, self.env_var)
        if resolved_embedder_provider not in SUPPORTED_EMBEDDER_PROVIDERS:
            raise BackendAPIError(
                self.name,
                f"unsupported embedder_provider {resolved_embedder_provider!r}; this adapter "
                f"only wires up {SUPPORTED_EMBEDDER_PROVIDERS} -- the ones mem0ai/mem0#5671, "
                "#4711, and #2304 concern. See module docstring.",
            )
        for required_var in _EMBEDDER_CREDENTIAL_ENV_VARS[resolved_embedder_provider]:
            if not os.environ.get(required_var):
                raise BackendNotConfiguredError(self.name, required_var)

        resolved_graph_store_provider = graph_store_provider or os.environ.get(
            "MEM0_DIRECT_GRAPH_STORE_PROVIDER"
        )
        if resolved_graph_store_provider:
            raise BackendAPIError(
                self.name,
                f"graph_store_provider={resolved_graph_store_provider!r} was requested, but "
                "the installed mem0ai package has no graph_store support at all: "
                "MemoryConfig has no `graph_store` field, no `kuzu` dependency appears in any "
                "declared extra, and no graph/kuzu module exists in the installed package "
                "tree. mem0ai/mem0#3558's kuzu_memory.py is not present in this release. "
                "Passing graph_store to Memory.from_config() would be silently ignored "
                "(confirmed empirically during this adapter's build), not rejected -- this "
                "adapter refuses the request loudly instead of reproducing that silent "
                "no-op. See module docstring for the full finding.",
            )

        resolved_dims = _resolve_embedding_dims(embedder_config, resolved_embedder_provider)
        built_embedder_config = _build_embedder_config(
            resolved_embedder_provider, embedder_config or {}, resolved_dims
        )

        resolved_vector_store_provider = (
            vector_store_provider or os.environ.get("MEM0_DIRECT_VECTOR_STORE_PROVIDER") or "redis"
        )
        if resolved_vector_store_provider not in SUPPORTED_VECTOR_STORE_PROVIDERS:
            raise BackendAPIError(
                self.name,
                f"unsupported vector_store_provider {resolved_vector_store_provider!r}; this "
                f"adapter only wires up {SUPPORTED_VECTOR_STORE_PROVIDERS} -- the ones "
                "mem0ai/mem0#4362, #4297, #4453, and #5980 concern. See module docstring.",
            )
        url_env_var = f"MEM0_DIRECT_{resolved_vector_store_provider.upper()}_URL"
        resolved_url = os.environ.get("MEM0_DIRECT_VECTOR_STORE_URL") or os.environ.get(url_env_var)
        url_key = _VECTOR_STORE_URL_KEYS[resolved_vector_store_provider]
        _configured_via_vector_store_config = url_key in (vector_store_config or {}) or (
            resolved_vector_store_provider == "elasticsearch"
            and "cloud_id" in (vector_store_config or {})
        )
        if not resolved_url and not _configured_via_vector_store_config:
            raise BackendNotConfiguredError(self.name, "MEM0_DIRECT_VECTOR_STORE_URL")
        built_vector_store_config = _build_vector_store_config(
            resolved_vector_store_provider,
            vector_store_config or {},
            resolved_url,
            resolved_dims,
        )

        self._embedder_provider = resolved_embedder_provider
        self._vector_store_provider = resolved_vector_store_provider
        self._custom_instructions = custom_instructions
        self._config_dict: dict[str, object] = {
            "embedder": {"provider": resolved_embedder_provider, "config": built_embedder_config},
            "vector_store": {
                "provider": resolved_vector_store_provider,
                "config": built_vector_store_config,
            },
            # `MemoryConfig.custom_instructions` (mem0/configs/base.py) is
            # the real, current top-level key -- mem0 renamed the field
            # from `custom_fact_extraction_prompt` to `custom_instructions`
            # in mem0ai/mem0#4740. See module docstring's "Custom
            # fact-extraction prompt passthrough" section. Set
            # unconditionally, same convention `_build_embedder_config`
            # uses for `embedding_dims`: `None` here is what mem0's own
            # `MemoryConfig.custom_instructions` field already defaults to,
            # so passing it explicitly changes nothing when the caller
            # didn't set one.
            "custom_instructions": custom_instructions,
        }
        self._memory = None
        self._construction_error = None

    def _get_memory(self) -> _MemoryProtocol:
        """Lazily construct the real `mem0.Memory`, same "fails loudly and
        specifically at first use" convention `MemPalaceAdapter._get_palace()`
        uses above -- and, unlike that method, deliberately *not* re-raising
        a caught construction-time `ValueError`/`pydantic.ValidationError`
        here. store() is the one place that classifies that specific
        failure as `CorruptionSignal.CONFIG_REJECTED` instead of letting it
        propagate as an unhandled crash (see module docstring's Kuzu/
        `graph_store` finding and CONFIG_REJECTED's docstring in base.py);
        query()/update()/delete() re-raise it as a normal `BackendAPIError`
        since neither `QueryResult` nor `DeleteResult` has a
        `corruption_signal` field to report it through gracefully.
        """
        if self._memory is not None:
            return self._memory
        if self._construction_error is not None:
            raise ValueError(self._construction_error)
        try:
            # `mem0ai` is declared in the optional `mem0-direct` extra (see
            # pyproject.toml), which CI's typecheck job installs -- so mypy
            # sees it as installed-but-untyped (no py.typed marker in the
            # package) rather than missing. A contributor running
            # `mypy --strict` without that extra installed would see
            # import-not-found here instead; that's expected, not a bug --
            # see module docstring for why this dependency is optional.
            import mem0  # type: ignore[import-untyped]
        except ImportError as exc:
            raise BackendAPIError(
                self.name,
                "the `mem0ai` package is not installed. Install it with `pip install mem0ai` "
                "-- this adapter needs the real library in-process, unlike Mem0Adapter/"
                "Mem0SelfHostedAdapter in mem0_adapter.py, which only need httpx.",
            ) from exc
        try:
            self._memory = mem0.Memory.from_config(self._config_dict)
        except ValueError as exc:
            # Cache the message so a second store()/query() call doesn't
            # need to re-attempt (and re-fail) real construction.
            self._construction_error = str(exc)
            raise
        return self._memory

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        # No documented operating-mode variant, same no-op convention as
        # every other adapter's store() -- see MemoryBackendAdapter.supported_modes.
        del mode
        timer = self._timed()
        try:
            memory = self._get_memory()
        except ValueError as exc:
            # This is the CONFIG_REJECTED path -- see module docstring and
            # _get_memory()'s docstring for exactly what this does and
            # does not reproduce.
            return StoreResult(
                memory_id="",
                latency_ms=timer.elapsed_ms(),
                raw={"error": str(exc)},
                corruption_signal=CorruptionSignal.CONFIG_REJECTED,
                # NOT_APPLICABLE, not EMPTY_EXTRACTION -- memory_id="" here
                # means construction was rejected before any extraction
                # could run, a different failure class than "extraction ran
                # and found nothing." See ExtractionSignal.NOT_APPLICABLE
                # in base.py.
                extraction_signal=ExtractionSignal.NOT_APPLICABLE,
            )
        try:
            data = memory.add(content, user_id=session_id, metadata=metadata or {})
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        memory_id = _extract_memory_id(data)
        # Same mem0ai/mem0#5178 gap this adapter's REST siblings in
        # mem0_adapter.py guard against -- Memory.add() can return without
        # raising and carry no usable id anywhere in its response.
        extraction_signal = (
            ExtractionSignal.EMPTY_EXTRACTION if not memory_id else ExtractionSignal.FACTS_EXTRACTED
        )
        return StoreResult(
            memory_id=memory_id,
            latency_ms=timer.elapsed_ms(),
            raw=data if isinstance(data, dict) else {"results": data},
            corruption_signal=CorruptionSignal.CLEAN,
            extraction_signal=extraction_signal,
        )

    def query(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        mode: str | None = None,
        threshold: float | None = None,
    ) -> QueryResult:
        """Retrieve memories relevant to `query` within `session_id`.

        `threshold` is forwarded straight through to the real, installed
        `mem0.Memory.search()`'s own `threshold` parameter (default `0.1`
        there if this adapter passes `None`, per the installed
        `Memory._search_vector_store()` -- confirmed by reading
        `mem0/memory/main.py` during this build). This is the parameter
        that makes mem0ai/mem0#4453 (search-threshold inversion) reachable
        through this adapter, the same way `Mem0SelfHostedAdapter.query()`'s
        `threshold` parameter makes it reachable over REST -- see module
        docstring for what this build actually confirmed about #4453
        against the installed package (short version: the bug class does
        not currently reproduce -- every installed vector store, including
        Qdrant, is confirmed to return similarity-oriented, higher-is-
        better scores before `Memory`'s own threshold filtering ever sees
        them, per `VectorStoreBase.search()`'s documented contract and
        `tests/test_mem0_direct_adapter.py`'s real-package regression
        tests against `mem0.utils.scoring.score_and_rank`).
        """
        del mode  # no-op, see store() above
        timer = self._timed()
        try:
            memory = self._get_memory()
        except ValueError as exc:
            raise BackendAPIError(self.name, f"config rejected: {exc}") from exc
        try:
            data = memory.search(
                query, filters={"user_id": session_id}, top_k=top_k, threshold=threshold
            )
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc

        raw_results_obj = data.get("results", []) if isinstance(data, dict) else []
        raw_results: list[dict[str, object]] = (
            [item for item in raw_results_obj if isinstance(item, dict)]
            if isinstance(raw_results_obj, list)
            else []
        )
        records = [
            MemoryRecord(
                memory_id=str(item.get("id", "")),
                content=str(item.get("memory", "")),
                score=item.get("score"),  # type: ignore[arg-type]
                created_at=item.get("created_at"),  # type: ignore[arg-type]
                metadata={
                    k: str(v)
                    for k, v in item.items()
                    if k not in {"id", "memory", "score", "created_at"}
                },
                raw=item,
            )
            for item in raw_results
        ]
        # Same reasoning as Mem0Adapter/Mem0SelfHostedAdapter's query() in
        # mem0_adapter.py: the real Memory.search() response carries no
        # documented per-result conflict marker either -- conflict
        # resolution happens invisibly inside mem0's own add/update
        # pipeline. NOT_APPLICABLE here, not a guess.
        return QueryResult(
            records=records,
            conflict_signal=ConflictSignal.NOT_APPLICABLE,
            latency_ms=timer.elapsed_ms(),
            raw=data if isinstance(data, dict) else {"results": data},
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        """Store a fact that may contradict a previously stored one, via a
        full-content `Memory.update(memory_id, text=content)` call.

        Because `text` is provided, the real `Memory._update_memory()`
        always computes a fresh embedding for it (see module docstring and
        `update_metadata_only()` below for the metadata-only case, which
        behaves very differently) -- there is no vector-corruption surface
        to inspect for this specific call shape, so `corruption_signal` is
        reported as CLEAN rather than guessed at NOT_APPLICABLE: this
        adapter does have the surface to observe corruption (see
        `update_metadata_only()`), it just has nothing to report for a
        call that always re-embeds.
        """
        del session_id  # Memory.update() scopes by memory_id, not session
        timer = self._timed()
        memory = self._get_memory_or_raise_backend_error()
        try:
            data = memory.update(memory_id, text=content)
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        return UpdateResult(
            memory_id=memory_id,
            acknowledged=True,
            latency_ms=timer.elapsed_ms(),
            raw=data if isinstance(data, dict) else {},
            corruption_signal=CorruptionSignal.CLEAN,
        )

    def update_metadata_only(self, memory_id: str, metadata: dict[str, str]) -> UpdateResult:
        """Update only a memory's metadata, omitting `text` entirely -- the
        call shape needed to exercise mem0ai/mem0#4362's fixed guard.

        This deliberately does NOT go through `Memory.update(memory_id,
        text=None, metadata=...)`. Reading the installed
        `Memory._update_memory()` source (see module docstring) shows that
        method falls back to the *previous* text (`data = prev_value`) and
        always recomputes a real embedding for it -- `vector` is never
        `None` by the time it reaches `vector_store.update()` through that
        path in this mem0ai version, so it can never exercise the guard
        #4362 added. Instead, this method calls
        `self._memory.vector_store.update(vector_id=memory_id, vector=None,
        payload=...)` directly -- the exact call shape (and the exact
        component) #4362's bug and fix concern, and, per this build's
        reading of `Memory._add_to_vector_store()`'s entity-linking phase,
        a call shape mem0's own code does make under other circumstances
        (`entity_store.update(..., vector=None, ...)`), just not through
        the metadata-only `Memory.update()` path this method's name
        suggests it would.

        Returns `corruption_signal=VECTOR_ZEROED` if the embedding this
        adapter can observe after the update no longer matches what was
        there before (best-effort raw-client inspection -- see
        `_read_raw_embedding_bytes()`; this is not part of mem0's own
        `VectorStoreBase.get()` contract, which does not return raw vector
        bytes at all), `CLEAN` if it is unchanged, or `NOT_APPLICABLE` if
        this adapter has no raw-inspection support for the configured
        vector store provider.
        """
        timer = self._timed()
        memory = self._get_memory_or_raise_backend_error()
        vector_store = memory.vector_store
        try:
            existing = vector_store.get(vector_id=memory_id)
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        if existing is None:
            raise BackendAPIError(self.name, f"no stored memory with id {memory_id!r} to update")
        payload = dict(getattr(existing, "payload", existing) or {})
        payload.update(metadata)

        before_bytes = _read_raw_embedding_bytes(
            vector_store, self._vector_store_provider, memory_id
        )
        try:
            vector_store.update(vector_id=memory_id, vector=None, payload=payload)
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        after_bytes = _read_raw_embedding_bytes(
            vector_store, self._vector_store_provider, memory_id
        )

        if before_bytes is None or after_bytes is None:
            corruption_signal = CorruptionSignal.NOT_APPLICABLE
        elif after_bytes == before_bytes:
            corruption_signal = CorruptionSignal.CLEAN
        else:
            corruption_signal = CorruptionSignal.VECTOR_ZEROED

        return UpdateResult(
            memory_id=memory_id,
            acknowledged=True,
            latency_ms=timer.elapsed_ms(),
            raw={
                "before_embedding_bytes": before_bytes,
                "after_embedding_bytes": after_bytes,
            },
            corruption_signal=corruption_signal,
        )

    def delete(self, memory_id: str) -> DeleteResult:
        timer = self._timed()
        memory = self._get_memory_or_raise_backend_error()
        try:
            memory.delete(memory_id)
        except Exception as exc:  # noqa: BLE001 - real vendor call, wrap uniformly
            raise BackendAPIError(self.name, str(exc)) from exc
        return DeleteResult(success=True, memory_id=memory_id, latency_ms=timer.elapsed_ms())

    def probe_raw_filter(self, filters: dict[str, object]) -> RawFilterProbeResult:
        """Submit `filters` directly to `self._memory.vector_store.list()`,
        the real, installed vector store's own filter-query-building layer
        -- bypassing `query()`'s hardcoded `{"user_id": session_id}` filter
        entirely. `list()` is used rather than `search()` because it needs
        no query embedding (`mem0.vector_stores.base.VectorStoreBase.list`
        takes only `filters`/`top_k`), so this probes the filter-building
        layer in isolation, without also depending on the embedder being
        configured/reachable. This is the primitive
        evals/filter_injection.py needs to reproduce mem0ai/mem0#5980's
        exact injection shape (a dict/list-valued filter value) against the
        real, installed `mem0.vector_stores.elasticsearch.ElasticsearchDB`
        -- see this module's docstring, "Elasticsearch support" section.

        A construction-time config rejection (e.g. Valkey's missing
        `embedding_model_dims`, or Elasticsearch's missing auth credential
        -- see module docstring) is reported as `accepted=False` here too,
        same as any other exception this call can raise -- it is not
        itself evidence about filter validation, but it is not silently
        swallowed either.
        """
        try:
            memory = self._get_memory()
        except ValueError as exc:
            # Never reached the vector store's filter-building layer at
            # all -- applicable=False, not a filter-validation verdict.
            return RawFilterProbeResult(
                accepted=False, error=f"config rejected: {exc}", applicable=False, raw={}
            )
        vector_store = memory.vector_store
        try:
            result = vector_store.list(filters=dict(filters))
        except Exception as exc:  # noqa: BLE001 - real vendor call, classify uniformly
            return RawFilterProbeResult(accepted=False, error=str(exc), raw={})
        return RawFilterProbeResult(
            accepted=True, error=None, raw={"result_repr": repr(result)[:500]}
        )

    def _get_memory_or_raise_backend_error(self) -> _MemoryProtocol:
        try:
            return self._get_memory()
        except ValueError as exc:
            raise BackendAPIError(self.name, f"config rejected: {exc}") from exc


def _resolve_embedding_dims(embedder_config: dict[str, object] | None, provider: str) -> int | None:
    """Resolve the embedding dimension count to thread through both the
    embedder config and the vector-store config, so they agree unless a
    caller deliberately breaks that (see CONFIG_REJECTED test case).

    Priority: an explicit `embedding_dims` key in `embedder_config` (even
    `None`, which a caller sets on purpose to reproduce the bad-config
    class) > `MEM0_DIRECT_EMBEDDING_DIMS` > this module's per-provider
    default.
    """
    if embedder_config is not None and "embedding_dims" in embedder_config:
        value = embedder_config["embedding_dims"]
        if value is None:
            return None
        if isinstance(value, int | float | str):
            return int(value)
        raise TypeError(f"embedding_dims must be an int, float, str, or None, got {type(value)!r}")
    env_value = os.environ.get("MEM0_DIRECT_EMBEDDING_DIMS")
    if env_value:
        return int(env_value)
    return _DEFAULT_EMBEDDING_DIMS_BY_PROVIDER.get(provider)


def _build_embedder_config(
    provider: str, embedder_config: dict[str, object], resolved_dims: int | None
) -> dict[str, object]:
    config: dict[str, object] = dict(embedder_config)
    config["embedding_dims"] = resolved_dims
    if provider == "openai" and "api_key" not in config:
        config["api_key"] = os.environ.get("OPENAI_API_KEY")
    elif provider == "aws_bedrock":
        config.setdefault("aws_access_key_id", os.environ.get("AWS_ACCESS_KEY_ID"))
        config.setdefault("aws_secret_access_key", os.environ.get("AWS_SECRET_ACCESS_KEY"))
        config.setdefault("aws_region", os.environ.get("AWS_REGION", "us-west-2"))
        config.setdefault("model", "amazon.titan-embed-text-v2:0")
    elif provider == "gemini" and "api_key" not in config:
        config["api_key"] = os.environ.get("GOOGLE_API_KEY")
    return config


def _build_vector_store_config(
    provider: str,
    vector_store_config: dict[str, object],
    resolved_url: str | None,
    resolved_dims: int | None,
) -> dict[str, object]:
    config: dict[str, object] = dict(vector_store_config)
    # `mem0.configs.vector_stores.{redis,valkey}.py` name their URL field
    # `redis_url`/`valkey_url`; `mem0.configs.vector_stores.qdrant.QdrantConfig`
    # names it plain `url`; `mem0.configs.vector_stores.elasticsearch.
    # ElasticsearchConfig` has no `url` field at all, only `host` (plus
    # `port`/`cloud_id`) -- confirmed by reading all four installed config
    # classes directly. See _VECTOR_STORE_URL_KEYS above.
    url_key = _VECTOR_STORE_URL_KEYS[provider]
    if url_key not in config and resolved_url is not None:
        config[url_key] = resolved_url
    if provider == "elasticsearch" and "api_key" not in config and "user" not in config:
        # ElasticsearchConfig.validate_auth requires api_key OR user+password
        # -- threaded only when actually set (never a fake/empty credential;
        # a caller with neither gets a real, honest pydantic.ValidationError
        # from the installed package at first use -- see module docstring's
        # Elasticsearch support section).
        env_api_key = os.environ.get("MEM0_DIRECT_ELASTICSEARCH_API_KEY")
        if env_api_key:
            config["api_key"] = env_api_key
    if "embedding_model_dims" not in config:
        config["embedding_model_dims"] = resolved_dims
    config.setdefault("collection_name", "memtrust_mem0_direct")
    return config


def _read_raw_embedding_bytes(vector_store: object, provider: str, memory_id: str) -> bytes | None:
    """Best-effort direct inspection of the raw stored embedding bytes for
    `memory_id`, reaching past `VectorStoreBase.get()` (which only exposes
    metadata, not the vector itself -- confirmed by reading
    `RedisDB.get()`/`ValkeyDB`'s inherited base `get()` during this
    build) into each store's own real, low-level wire client.

    Returns `None` (never raises) if the configured provider has no known
    raw-inspection path, or if the underlying client call itself fails --
    a raw-inspection miss is reported as NOT_APPLICABLE by callers, not
    treated as proof of corruption.
    """
    try:
        if provider == "redis":
            prefix = vector_store.schema["index"]["prefix"]  # type: ignore[attr-defined]
            key = f"{prefix}:{memory_id}"
            raw = vector_store.client.hget(key, "embedding")  # type: ignore[attr-defined]
        elif provider == "valkey":
            prefix = vector_store.prefix  # type: ignore[attr-defined]
            key = f"{prefix}:{memory_id}"
            raw = vector_store.client.hget(key, "embedding")  # type: ignore[attr-defined]
        else:
            return None
    except Exception:  # noqa: BLE001 - best-effort inspection, never fatal
        return None
    return raw if isinstance(raw, bytes) else None


def _extract_memory_id(data: object) -> str:
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return str(results[0].get("id", ""))
        if "id" in data:
            return str(data["id"])
    return ""
