"""Tests for Mem0DirectAdapter (`src/memtrust/adapters/mem0_direct_adapter.py`).

Two distinct layers are tested here, deliberately kept separate:

1. **Adapter control-flow tests** -- `Mem0DirectAdapter` exercised via its
   `memory=` constructor injection point (a hand-written `_FakeMemory`
   double matching the `_MemoryProtocol` shape), same convention
   `test_adapters.py` uses for `MemPalaceAdapter`/`FakePalace`. These prove
   the adapter's own logic (config resolution, StoreResult/UpdateResult
   shaping, corruption_signal derivation) is correct given a response
   shape -- they do NOT touch the real `mem0ai` package at all.

2. **Real-package regression tests** -- the embedder-dims-forwarding,
   redis/valkey vector=None-guard, qdrant dimension-sizing, and
   score_and_rank/VectorStoreBase threshold-contract tests import and
   exercise the actual, installed `mem0.embeddings.*`/`mem0.vector_stores.*`/
   `mem0.configs.vector_stores.*`/`mem0.utils.scoring` classes and functions
   directly (mocking only each vendor's own network/model-load boundary:
   `boto3`, the `openai`/`google.genai` SDK clients, `fastembed.TextEmbedding`,
   the `redis`/`valkey`/`qdrant_client` wire clients). These are what let
   this build honestly claim mem0ai/mem0#5671, #4362, #4711, #2304, and
   #4453 are confirmed fixed in the currently pinned `mem0ai` version (not
   just "the GitHub issue says merged"), and, just as importantly, that
   #4297's dimension-mismatch bug class is confirmed STILL REACHABLE in
   the same version -- see mem0_direct_adapter.py's module docstring for
   the full, honest breakdown of which is which.

Layer 2 requires the optional `mem0-direct` dependency group
(`pip install -e ".[dev,mem0-direct]"`) -- `pytest.importorskip` below
degrades this whole file to a clean, explained skip if it isn't installed,
rather than a collection error, matching every other adapter's "never
crash on missing dependency" contract. CI installs this group (see
.github/workflows/ci.yml) so these tests do run for real on every push.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

mem0 = pytest.importorskip(
    "mem0",
    reason=(
        "mem0_direct_adapter.py tests require the optional `mem0-direct` extra: "
        "pip install -e '.[dev,mem0-direct]'. See mem0_direct_adapter.py's module docstring."
    ),
)
boto3 = pytest.importorskip("boto3")
pytest.importorskip("google.genai")
pytest.importorskip("fastembed")
pytest.importorskip("redis")
pytest.importorskip("valkey")
pytest.importorskip("qdrant_client")
pytest.importorskip("elasticsearch")

from memtrust.adapters.base import (  # noqa: E402
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
    CorruptionSignal,
    ExtractionSignal,
)
from memtrust.adapters.mem0_direct_adapter import (  # noqa: E402
    SUPPORTED_EMBEDDER_PROVIDERS,
    SUPPORTED_VECTOR_STORE_PROVIDERS,
    Mem0DirectAdapter,
)

# ---------------------------------------------------------------------------
# Layer 1: adapter control-flow tests, via a fake in-process Memory double
# ---------------------------------------------------------------------------


class _FakeVectorStore:
    """Emulates the real, fixed redis.py/valkey.py `update()` guard
    (`if vector is not None: ...`) so adapter-level tests can assert on
    the resulting corruption_signal without a live server.
    """

    def __init__(self, initial_payload: dict[str, str], initial_embedding: bytes) -> None:
        self._payload = dict(initial_payload)
        self.schema = {"index": {"prefix": "mem0:test"}}
        self.client = MagicMock()
        self._embedding = initial_embedding
        self.client.hget.side_effect = lambda key, field: self._embedding
        self.update_calls: list[tuple[str, list[float] | None, dict[str, object]]] = []

    def get(self, vector_id: str) -> Any:
        return _FakePoint(dict(self._payload))

    def update(
        self, vector_id: str | None = None, vector: list[float] | None = None, payload=None
    ) -> None:
        self.update_calls.append((vector_id or "", vector, payload or {}))
        if vector is not None:
            self._embedding = b"\x00\x00\x00\x00" * len(vector)
        # else: leave self._embedding untouched -- this is the fix #4362
        # made real; a pre-fix implementation would have zeroed it here
        # unconditionally.


class _FakePoint:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload


class _FakeMemory:
    """Matches `_MemoryProtocol` in mem0_direct_adapter.py."""

    def __init__(self, vector_store: _FakeVectorStore | None = None) -> None:
        self.vector_store = vector_store or _FakeVectorStore({"data": "seed"}, b"\xaa\xbb\xcc\xdd")
        self.add_calls: list[dict[str, object]] = []
        self.search_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []
        self.delete_calls: list[str] = []
        self.raise_on_add: Exception | None = None
        self.raise_on_search: Exception | None = None
        self.add_returns_empty_results: bool = False
        """When True, add() returns a well-formed response with zero
        extracted memories -- the exact mem0ai/mem0#5178 shape: no
        exception, a normal-looking dict, but nothing usable for
        `_extract_memory_id()` to find. Lets tests exercise
        ExtractionSignal.EMPTY_EXTRACTION without needing a real vendor
        call that legitimately extracted nothing."""

    def add(
        self, messages: object, *, user_id=None, metadata=None, infer=True
    ) -> dict[str, object]:
        if self.raise_on_add:
            raise self.raise_on_add
        self.add_calls.append({"messages": messages, "user_id": user_id, "metadata": metadata})
        if self.add_returns_empty_results:
            return {"results": []}
        return {"results": [{"id": "mem-1", "memory": str(messages), "event": "ADD"}]}

    def search(self, query: str, *, filters=None, top_k=5, threshold=None) -> dict[str, object]:
        if self.raise_on_search:
            raise self.raise_on_search
        self.search_calls.append(
            {"query": query, "filters": filters, "top_k": top_k, "threshold": threshold}
        )
        return {
            "results": [
                {
                    "id": "mem-1",
                    "memory": "My dog is named Baxter.",
                    "score": 0.91,
                    "created_at": "2026-07-16T00:00:00Z",
                }
            ]
        }

    def update(self, memory_id: str, text=None, metadata=None) -> dict[str, object]:
        self.update_calls.append({"memory_id": memory_id, "text": text, "metadata": metadata})
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id: str) -> None:
        self.delete_calls.append(memory_id)


def test_store_returns_clean_corruption_signal_on_success() -> None:
    fake = _FakeMemory()
    adapter = Mem0DirectAdapter(memory=fake)
    result = adapter.store("session-1", "My dog is named Baxter.", metadata={"topic": "pets"})
    assert result.memory_id == "mem-1"
    assert result.corruption_signal == CorruptionSignal.CLEAN
    assert result.extraction_signal == ExtractionSignal.FACTS_EXTRACTED
    assert fake.add_calls == [
        {
            "messages": "My dog is named Baxter.",
            "user_id": "session-1",
            "metadata": {"topic": "pets"},
        }
    ]


def test_store_wraps_vendor_exception_in_backend_api_error() -> None:
    fake = _FakeMemory()
    fake.raise_on_add = RuntimeError("bedrock throttled")
    adapter = Mem0DirectAdapter(memory=fake)
    with pytest.raises(BackendAPIError):
        adapter.store("session-1", "content")


def test_store_reports_empty_extraction_signal_when_add_returns_no_results() -> None:
    """mem0ai/mem0#5178: Memory.add() can complete without raising and
    return a normal-shaped `{"results": []}` response -- no exception, no
    other signal that anything unusual happened -- when its extraction
    pipeline found nothing worth persisting. Before this fix,
    `_extract_memory_id()` silently returned "" here and the resulting
    StoreResult was indistinguishable from a genuine successful store with
    an unusual id shape. This proves the adapter now flags that gap
    instead of masking it.
    """
    fake = _FakeMemory()
    fake.add_returns_empty_results = True
    adapter = Mem0DirectAdapter(memory=fake)
    result = adapter.store("session-1", "just saying hi, nothing to remember")
    assert result.memory_id == ""
    assert result.corruption_signal == CorruptionSignal.CLEAN
    assert result.extraction_signal == ExtractionSignal.EMPTY_EXTRACTION


def test_query_parses_records_and_reports_not_applicable_conflict_signal() -> None:
    fake = _FakeMemory()
    adapter = Mem0DirectAdapter(memory=fake)
    result = adapter.query("session-1", "what is my dog's name?")
    assert len(result.records) == 1
    assert result.records[0].memory_id == "mem-1"
    assert result.records[0].content == "My dog is named Baxter."
    assert result.conflict_signal == ConflictSignal.NOT_APPLICABLE
    assert fake.search_calls == [
        {
            "query": "what is my dog's name?",
            "filters": {"user_id": "session-1"},
            "top_k": 5,
            "threshold": None,
        }
    ]


def test_query_wraps_vendor_exception_in_backend_api_error() -> None:
    fake = _FakeMemory()
    fake.raise_on_search = RuntimeError("vector store unreachable")
    adapter = Mem0DirectAdapter(memory=fake)
    with pytest.raises(BackendAPIError):
        adapter.query("session-1", "query")


def test_query_forwards_threshold_to_memory_search() -> None:
    """mem0ai/mem0#4453 reachability: this adapter's own `threshold`
    parameter must reach the real `Memory.search()` call, the same way
    `Mem0SelfHostedAdapter.query()`'s `threshold` reaches `SearchRequest`
    over REST. See module docstring for what this build confirmed about
    #4453 against the installed package (the bug class does not currently
    reproduce -- this test only proves the parameter is plumbed through).
    """
    fake = _FakeMemory()
    adapter = Mem0DirectAdapter(memory=fake)
    adapter.query("session-1", "what is my dog's name?", threshold=0.6)
    assert fake.search_calls[-1]["threshold"] == 0.6


def test_update_full_content_reports_clean_and_calls_memory_update_with_text() -> None:
    fake = _FakeMemory()
    adapter = Mem0DirectAdapter(memory=fake)
    result = adapter.update("session-1", "mem-1", "My dog is actually named Max.")
    assert result.acknowledged is True
    assert result.corruption_signal == CorruptionSignal.CLEAN
    assert fake.update_calls == [
        {"memory_id": "mem-1", "text": "My dog is actually named Max.", "metadata": None}
    ]


def test_delete_calls_memory_delete() -> None:
    fake = _FakeMemory()
    adapter = Mem0DirectAdapter(memory=fake)
    result = adapter.delete("mem-1")
    assert result.success is True
    assert fake.delete_calls == ["mem-1"]


# ---------------------------------------------------------------------------
# update_metadata_only() -- the call shape that exercises mem0ai/mem0#4362
# ---------------------------------------------------------------------------


def test_update_metadata_only_bypasses_memory_update_and_passes_vector_none() -> None:
    vector_store = _FakeVectorStore({"data": "seed", "hash": "h0"}, b"\xaa\xbb\xcc\xdd")
    fake = _FakeMemory(vector_store=vector_store)
    adapter = Mem0DirectAdapter(memory=fake)
    adapter._vector_store_provider = "redis"  # type: ignore[attr-defined]

    result = adapter.update_metadata_only("mem-1", {"pinned": "true"})

    # This must NOT go through Memory.update() -- see module docstring.
    assert fake.update_calls == []
    assert vector_store.update_calls == [
        ("mem-1", None, {"data": "seed", "hash": "h0", "pinned": "true"})
    ]
    assert result.corruption_signal == CorruptionSignal.CLEAN


def test_update_metadata_only_detects_vector_zeroed_on_a_pre_fix_style_store() -> None:
    """Same control flow as above, but against a vector store double that
    behaves the *pre-fix* way (unconditionally overwrites the embedding
    even when vector=None) -- proves this adapter's corruption_signal
    derivation actually distinguishes CLEAN from VECTOR_ZEROED, not just
    always reporting CLEAN.
    """

    class _PreFixVectorStore(_FakeVectorStore):
        def update(self, vector_id=None, vector=None, payload=None) -> None:
            self.update_calls.append((vector_id or "", vector, payload or {}))
            # Pre-fix mem0ai#4362 behavior: always overwrites, even for
            # vector=None (np.array(None, ...) in the real bug produced a
            # 4-byte garbage vector -- emulated here as an all-zero one
            # distinguishable from the real 4-float embedding below).
            self._embedding = b"\x00\x00\x00\x00"

    vector_store = _PreFixVectorStore({"data": "seed"}, b"\xaa\xbb\xcc\xdd\xee\xff\x11\x22")
    fake = _FakeMemory(vector_store=vector_store)
    adapter = Mem0DirectAdapter(memory=fake)
    adapter._vector_store_provider = "redis"  # type: ignore[attr-defined]

    result = adapter.update_metadata_only("mem-1", {"pinned": "true"})
    assert result.corruption_signal == CorruptionSignal.VECTOR_ZEROED


def test_update_metadata_only_reports_not_applicable_when_provider_unrecognized() -> None:
    vector_store = _FakeVectorStore({"data": "seed"}, b"\xaa\xbb\xcc\xdd")
    fake = _FakeMemory(vector_store=vector_store)
    adapter = Mem0DirectAdapter(memory=fake)
    adapter._vector_store_provider = "some_future_provider"  # type: ignore[attr-defined]

    result = adapter.update_metadata_only("mem-1", {"pinned": "true"})
    assert result.corruption_signal == CorruptionSignal.NOT_APPLICABLE


def test_update_metadata_only_raises_when_memory_id_not_found() -> None:
    class _EmptyVectorStore(_FakeVectorStore):
        def get(self, vector_id: str) -> Any:
            return None

    fake = _FakeMemory(vector_store=_EmptyVectorStore({}, b""))
    adapter = Mem0DirectAdapter(memory=fake)
    with pytest.raises(BackendAPIError):
        adapter.update_metadata_only("does-not-exist", {"pinned": "true"})


# ---------------------------------------------------------------------------
# Construction / configuration gating
# ---------------------------------------------------------------------------


def test_raises_when_embedder_provider_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEM0_DIRECT_EMBEDDER_PROVIDER", raising=False)
    with pytest.raises(BackendNotConfiguredError) as excinfo:
        Mem0DirectAdapter()
    assert excinfo.value.missing_env_var == "MEM0_DIRECT_EMBEDDER_PROVIDER"
    assert excinfo.value.backend_name == "mem0_direct"


@pytest.mark.parametrize(
    ("provider", "missing_var"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("aws_bedrock", "AWS_ACCESS_KEY_ID"),
        ("gemini", "GOOGLE_API_KEY"),
    ],
)
def test_raises_when_provider_credential_not_configured(
    monkeypatch: pytest.MonkeyPatch, provider: str, missing_var: str
) -> None:
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", provider)
    for var in ("OPENAI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(BackendNotConfiguredError) as excinfo:
        Mem0DirectAdapter()
    assert excinfo.value.missing_env_var == missing_var


def test_fastembed_provider_needs_no_credential_but_still_needs_vector_store_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.delenv("MEM0_DIRECT_VECTOR_STORE_URL", raising=False)
    monkeypatch.delenv("MEM0_DIRECT_REDIS_URL", raising=False)
    with pytest.raises(BackendNotConfiguredError) as excinfo:
        Mem0DirectAdapter()
    assert excinfo.value.missing_env_var == "MEM0_DIRECT_VECTOR_STORE_URL"


def test_rejects_unsupported_embedder_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "ollama")
    with pytest.raises(BackendAPIError, match="unsupported embedder_provider"):
        Mem0DirectAdapter()


def test_rejects_unsupported_vector_store_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "pinecone")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_URL", "redis://localhost:6379")
    with pytest.raises(BackendAPIError, match="unsupported vector_store_provider"):
        Mem0DirectAdapter()


def test_supported_provider_tuples_match_the_bugs_this_adapter_targets() -> None:
    assert SUPPORTED_EMBEDDER_PROVIDERS == ("openai", "aws_bedrock", "gemini", "fastembed")
    assert SUPPORTED_VECTOR_STORE_PROVIDERS == ("redis", "valkey", "qdrant", "elasticsearch")


# ---------------------------------------------------------------------------
# graph_store / Kuzu: refused outright, no mem0ai call ever made (see
# module docstring's finding that MemoryConfig silently drops this key)
# ---------------------------------------------------------------------------


def test_graph_store_provider_raises_backend_api_error_naming_the_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_URL", "redis://localhost:6379")
    with pytest.raises(BackendAPIError) as excinfo:
        Mem0DirectAdapter(graph_store_provider="kuzu")
    message = str(excinfo.value)
    assert "graph_store" in message
    assert "kuzu_memory.py" in message
    assert "silently ignored" in message


def test_memory_config_silently_drops_unknown_graph_store_key() -> None:
    """Pins the empirical finding cited in the module docstring and the
    error message above: passing `graph_store` to the installed
    mem0ai==2.0.12 `MemoryConfig` raises nothing and produces no trace of
    the key. If a future mem0ai release adds real graph_store support (or
    starts rejecting unknown keys), this test fails loudly, which is
    exactly the signal that this adapter's hand-rolled rejection in
    __init__ needs to be revisited.
    """
    from mem0.configs.base import MemoryConfig

    config = MemoryConfig(graph_store={"provider": "kuzu", "config": {}})
    assert not hasattr(config, "graph_store")
    assert "graph_store" not in config.model_dump()


# ---------------------------------------------------------------------------
# CONFIG_REJECTED: real installed pydantic validation on ValkeyConfig's
# required embedding_model_dims -- the honest substitute for the retired
# Kuzu code path (see module docstring)
# ---------------------------------------------------------------------------


def test_store_reports_config_rejected_for_missing_embedding_dims_on_valkey(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "valkey")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_URL", "redis://localhost:6379")

    # embedding_dims=None is a deliberate, explicit override (see
    # _resolve_embedding_dims's priority order) -- this is what makes the
    # real installed ValkeyConfig(embedding_model_dims: int, required)
    # reject construction with a pydantic.ValidationError (a ValueError
    # subclass), the same construction-time-before-any-write failure shape
    # mem0ai/mem0#3558 established for Kuzu.
    adapter = Mem0DirectAdapter(embedder_config={"embedding_dims": None})

    result = adapter.store("session-1", "content")
    assert result.memory_id == ""
    assert result.corruption_signal == CorruptionSignal.CONFIG_REJECTED
    # NOT_APPLICABLE, not EMPTY_EXTRACTION -- memory_id="" here means
    # construction was rejected before any extraction could run, a
    # different failure class than "extraction ran and found nothing."
    assert result.extraction_signal == ExtractionSignal.NOT_APPLICABLE
    assert "embedding_model_dims" in result.raw["error"]


def test_query_raises_backend_api_error_for_the_same_rejected_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "valkey")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_URL", "redis://localhost:6379")
    adapter = Mem0DirectAdapter(embedder_config={"embedding_dims": None})
    with pytest.raises(BackendAPIError, match="config rejected"):
        adapter.query("session-1", "query")


def test_config_rejection_is_cached_and_does_not_reattempt_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "valkey")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_URL", "redis://localhost:6379")
    adapter = Mem0DirectAdapter(embedder_config={"embedding_dims": None})
    first = adapter.store("session-1", "content")
    second = adapter.store("session-1", "content again")
    assert first.corruption_signal == CorruptionSignal.CONFIG_REJECTED
    assert second.corruption_signal == CorruptionSignal.CONFIG_REJECTED


# ---------------------------------------------------------------------------
# Real-package embedder-dims-forwarding regressions: mem0ai/mem0#5671,
# #4711, #2304 -- exercising the actual installed mem0.embeddings classes,
# not a memtrust reimplementation.
# ---------------------------------------------------------------------------


def test_real_aws_bedrock_embedder_forwards_embedding_dims_to_titan_v2() -> None:
    """mem0ai/mem0#5671: confirms the installed AWSBedrockEmbedding forwards
    `embedding_dims` into the Bedrock Titan V2 request body.
    """
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

    with patch("mem0.embeddings.aws_bedrock.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({"embedding": [0.1] * 256}).encode()
        mock_client.invoke_model.return_value = {"body": response_body}

        config = BaseEmbedderConfig(model="amazon.titan-embed-text-v2:0", embedding_dims=256)
        embedder = AWSBedrockEmbedding(config)
        embedder.embed("hello world")

        sent_body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert sent_body.get("dimensions") == 256


def test_real_aws_bedrock_embedder_omits_dimensions_for_v1_model() -> None:
    """Negative case for the same fix: V1 Titan doesn't accept a
    `dimensions` param, and the installed embedder correctly gates the
    forward on `"v2" in model`, not on `embedding_dims` alone.
    """
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

    with patch("mem0.embeddings.aws_bedrock.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({"embedding": [0.1] * 1536}).encode()
        mock_client.invoke_model.return_value = {"body": response_body}

        config = BaseEmbedderConfig(model="amazon.titan-embed-text-v1", embedding_dims=256)
        embedder = AWSBedrockEmbedding(config)
        embedder.embed("hello world")

        sent_body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert "dimensions" not in sent_body


def test_real_fastembed_embedder_reads_real_model_dims_not_hardcoded_1536() -> None:
    """mem0ai/mem0#4711: confirms the installed FastEmbedEmbedding reads
    `embedding_dims` from the loaded model's own `embedding_size` when the
    caller didn't set one, instead of defaulting to 1536.
    """
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.fastembed import FastEmbedEmbedding

    with patch("mem0.embeddings.fastembed.TextEmbedding") as mock_text_embedding_cls:
        mock_model = MagicMock()
        mock_model.embedding_size = 384
        mock_text_embedding_cls.return_value = mock_model

        config = BaseEmbedderConfig()  # embedding_dims deliberately unset
        embedder = FastEmbedEmbedding(config)

        assert embedder.config.embedding_dims == 384
        assert embedder.config.embedding_dims != 1536


def test_real_gemini_embedder_forwards_embedding_dims_as_output_dimensionality() -> None:
    """mem0ai/mem0#2304 (Gemini half): confirms the installed
    GoogleGenAIEmbedding forwards `embedding_dims` as
    `output_dimensionality` on every embed_content call.
    """
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.gemini import GoogleGenAIEmbedding

    with patch("mem0.embeddings.gemini.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1] * 768)]
        )

        config = BaseEmbedderConfig(api_key="test-key", embedding_dims=768)
        embedder = GoogleGenAIEmbedding(config)
        embedder.embed("hello world")

        sent_config = mock_client.models.embed_content.call_args.kwargs["config"]
        assert sent_config.output_dimensionality == 768


def test_real_openai_embedder_forwards_embedding_dims_as_dimensions() -> None:
    """mem0ai/mem0#2304 (OpenAI half): confirms the installed
    OpenAIEmbedding only sends `dimensions` when the caller set
    `embedding_dims`, and forwards the exact value when they did.
    """
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.openai import OpenAIEmbedding

    with patch("mem0.embeddings.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 512)]
        )

        config = BaseEmbedderConfig(api_key="sk-test", embedding_dims=512)
        embedder = OpenAIEmbedding(config)
        embedder.embed("hello world")
        sent_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert sent_kwargs.get("dimensions") == 512

        # And the negative case in the same test: no embedding_dims set ->
        # no `dimensions` kwarg sent at all (needed for non-matryoshka
        # OpenAI-compatible backends that reject the parameter outright).
        config_no_dims = BaseEmbedderConfig(api_key="sk-test")
        embedder_no_dims = OpenAIEmbedding(config_no_dims)
        embedder_no_dims.embed("hello world")
        sent_kwargs_2 = mock_client.embeddings.create.call_args.kwargs
        assert "dimensions" not in sent_kwargs_2


# ---------------------------------------------------------------------------
# Real-package vector=None guard regressions: mem0ai/mem0#4362 -- exercising
# the actual installed RedisDB.update()/ValkeyDB.update(), not a memtrust
# reimplementation.
# ---------------------------------------------------------------------------


def test_real_redis_vector_store_update_leaves_embedding_untouched_when_vector_none() -> None:
    """mem0ai/mem0#4362 (Redis half): confirms the installed RedisDB.update()
    only writes an "embedding" field when `vector` is not None.
    """
    import mem0.vector_stores.redis as redis_mod

    with (
        patch.object(redis_mod, "redis"),
        patch.object(redis_mod, "SearchIndex") as mock_search_index_cls,
    ):
        mock_index = MagicMock()
        mock_search_index_cls.from_dict.return_value = mock_index
        db = redis_mod.RedisDB(
            redis_url="redis://localhost:6379", collection_name="test", embedding_model_dims=8
        )

        db.update(
            vector_id="mem-1",
            vector=None,
            payload={"data": "x", "hash": "h", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        written = mock_index.load.call_args.kwargs["data"][0]
        assert "embedding" not in written

        db.update(
            vector_id="mem-1",
            vector=[0.1, 0.2],
            payload={"data": "x", "hash": "h", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        written_2 = mock_index.load.call_args.kwargs["data"][0]
        assert "embedding" in written_2


def test_real_valkey_vector_store_update_leaves_embedding_untouched_when_vector_none() -> None:
    """mem0ai/mem0#4362 (Valkey half): confirms the installed
    ValkeyDB.update() only writes an "embedding" field when `vector` is
    not None.
    """
    import mem0.vector_stores.valkey as valkey_mod

    with patch.object(valkey_mod, "valkey") as mock_valkey_mod:
        mock_client = MagicMock()
        mock_valkey_mod.from_url.return_value = mock_client
        mock_client.execute_command.return_value = []
        db = valkey_mod.ValkeyDB.__new__(valkey_mod.ValkeyDB)
        db.client = mock_client
        db.prefix = "mem0:test"
        db.timezone = "UTC"

        db.update(
            vector_id="mem-1",
            vector=None,
            payload={"data": "x", "hash": "h", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        written = mock_client.hset.call_args.kwargs["mapping"]
        assert "embedding" not in written

        db.update(
            vector_id="mem-1",
            vector=[0.1, 0.2],
            payload={"data": "x", "hash": "h", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        written_2 = mock_client.hset.call_args.kwargs["mapping"]
        assert "embedding" in written_2


# ---------------------------------------------------------------------------
# Qdrant support: adapter-level config-building tests for
# vector_store_provider="qdrant" -- see mem0_direct_adapter.py's module
# docstring for the #4297/#4453 findings these exist to make reachable.
# ---------------------------------------------------------------------------


def test_qdrant_vector_store_config_threads_embedder_dims_into_embedding_model_dims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defensive pattern this adapter already uses for redis/valkey
    (see _build_vector_store_config) applies identically to qdrant: the
    resolved embedder dimension is threaded into `embedding_model_dims`
    explicitly, so this adapter never relies on QdrantConfig's own 1536
    default -- see the #4297 finding in the module docstring.
    """
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "qdrant")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_URL", "http://localhost:6333")

    adapter = Mem0DirectAdapter()

    vector_store_config = adapter._config_dict["vector_store"]["config"]  # type: ignore[index]
    assert vector_store_config["embedding_model_dims"] == 384  # fastembed's default
    assert vector_store_config["url"] == "http://localhost:6333"
    assert adapter._config_dict["vector_store"]["provider"] == "qdrant"  # type: ignore[index]


def test_qdrant_vector_store_url_env_var_is_provider_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEM0_DIRECT_QDRANT_URL (the provider-specific fallback) works the
    same way MEM0_DIRECT_REDIS_URL/MEM0_DIRECT_VALKEY_URL already do.
    """
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "qdrant")
    monkeypatch.delenv("MEM0_DIRECT_VECTOR_STORE_URL", raising=False)
    monkeypatch.setenv("MEM0_DIRECT_QDRANT_URL", "http://qdrant.internal:6333")

    adapter = Mem0DirectAdapter()

    vector_store_config = adapter._config_dict["vector_store"]["config"]  # type: ignore[index]
    assert vector_store_config["url"] == "http://qdrant.internal:6333"


# ---------------------------------------------------------------------------
# Real-package regression: mem0ai/mem0#4297 (embedding-dimension mismatch).
# Exercises the actual installed mem0.vector_stores.qdrant.Qdrant and
# mem0.configs.vector_stores.qdrant.QdrantConfig classes directly (mocking
# only the QdrantClient wire boundary), not a memtrust reimplementation.
# See module docstring's "Qdrant support" section for the full finding:
# this bug class is confirmed STILL REACHABLE in the installed Python
# package -- unlike #5671/#4362/#4711/#2304 above, this is not a PASS.
# ---------------------------------------------------------------------------


def test_real_qdrant_config_defaults_embedding_model_dims_to_1536() -> None:
    """Pins the exact field default that makes the #4297 bug class possible
    in the installed mem0ai==2.0.12 Python package: QdrantConfig.
    embedding_model_dims defaults to 1536 -- OpenAI's dimension -- the same
    hardcoded value #4297's TS SDK fix removed for the equivalent JS
    config. If a future mem0ai release changes this default (or starts
    deriving it from the embedder), this test fails loudly, which is the
    signal to revisit the module docstring's #4297 finding.
    """
    from mem0.configs.vector_stores.qdrant import QdrantConfig

    config = QdrantConfig(path="/tmp/memtrust-qdrant-config-test")
    assert config.embedding_model_dims == 1536


def test_real_qdrant_store_creates_default_1536_collection_when_dims_not_overridden() -> None:
    """Reproduces the #4297 bug-class mechanism against the real, installed
    Qdrant vector store: constructing it with QdrantConfig's own 1536
    default (i.e. what happens if a caller doesn't override
    embedding_model_dims for a non-OpenAI embedder) creates a Qdrant
    collection sized 1536 -- the wrong size for, e.g., fastembed's 384-dim
    model. This is the exact "Bad Request on insert" setup #4297 describes,
    reproduced against the real, installed mem0ai package with only the
    QdrantClient wire boundary mocked.
    """
    from mem0.vector_stores.qdrant import Qdrant
    from qdrant_client.models import VectorParams

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    Qdrant(collection_name="test_default_dims", embedding_model_dims=1536, client=mock_client)

    create_kwargs = mock_client.create_collection.call_args.kwargs
    vectors_config = create_kwargs["vectors_config"]
    assert isinstance(vectors_config, VectorParams)
    assert vectors_config.size == 1536


def test_real_qdrant_vector_store_creates_correct_size_collection_when_dims_are_explicit() -> None:
    """Same real Qdrant class, but with the dimension a non-OpenAI embedder
    (fastembed, 384-dim) would actually produce, threaded through
    explicitly -- exactly what Mem0DirectAdapter._build_vector_store_config
    does for vector_store_provider="qdrant" (see the adapter-level tests
    above). Proves the collection is created at the *correct* size once a
    caller supplies the real dims, instead of silently trusting the 1536
    default -- this is what keeps Mem0DirectAdapter itself from hitting the
    bug in practice, even though the bug class remains reachable through
    the raw mem0ai package (see the test above).
    """
    from mem0.vector_stores.qdrant import Qdrant
    from qdrant_client.models import VectorParams

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    Qdrant(collection_name="test_384_dims", embedding_model_dims=384, client=mock_client)

    create_kwargs = mock_client.create_collection.call_args.kwargs
    vectors_config = create_kwargs["vectors_config"]
    assert isinstance(vectors_config, VectorParams)
    assert vectors_config.size == 384


# ---------------------------------------------------------------------------
# Real-package regression: mem0ai/mem0#4453 (search-threshold inversion).
# Exercises the actual installed mem0.vector_stores.base.VectorStoreBase
# contract and mem0.utils.scoring.score_and_rank -- the real function
# Memory._search_vector_store() calls to apply `threshold` -- not a
# memtrust reimplementation. See module docstring's "Qdrant support"
# section: this bug class is confirmed FIXED in the installed package.
# ---------------------------------------------------------------------------


def test_real_vector_store_base_documents_the_similarity_score_contract() -> None:
    """Pins the exact fix that closes mem0ai/mem0#4453 in the installed
    mem0ai==2.0.12 package: VectorStoreBase.search()'s docstring now states
    a binding contract every vector-store implementation must follow --
    always return similarity scores where higher is better, converting
    from a raw distance metric before returning. This is a documented
    interface contract, not an accident -- if a future mem0ai release
    weakens or removes this docstring, this test fails loudly.
    """
    from mem0.vector_stores.base import VectorStoreBase

    doc = VectorStoreBase.search.__doc__ or ""
    assert "higher values" in doc
    assert "indicate greater similarity" in doc


def test_real_score_and_rank_keeps_closest_matches_and_drops_farthest() -> None:
    """Exercises the real, installed mem0.utils.scoring.score_and_rank with
    already-similarity-scored candidates (the contract every vector store
    must satisfy per VectorStoreBase.search() above) and confirms it keeps
    candidates at/above threshold, drops those below, and ranks best-first
    by that same higher-is-better score -- i.e. genuine similarity
    semantics, not the inverted "lower score = closer, keep it" semantics
    #4453 originally described.
    """
    from mem0.utils.scoring import score_and_rank

    candidates = [
        {"id": "close", "score": 0.92, "payload": {"data": "close match"}},
        {"id": "mid", "score": 0.55, "payload": {"data": "mid match"}},
        {"id": "far", "score": 0.10, "payload": {"data": "far match"}},
    ]
    results = score_and_rank(
        semantic_results=candidates, bm25_scores={}, entity_boosts={}, threshold=0.5, top_k=10
    )
    kept_ids = {r["id"] for r in results}
    assert kept_ids == {"close", "mid"}
    assert "far" not in kept_ids
    assert results[0]["id"] == "close"  # best (highest score) ranked first


def test_hypothetical_raw_distance_scores_would_invert_threshold_filtering() -> None:
    """Negative/hypothetical case, deliberately NOT a claim about any real
    installed vector store: if a store violated VectorStoreBase.search()'s
    documented contract and returned raw distance (lower = closer/better)
    instead of similarity, the real, installed score_and_rank's
    `if semantic_score < threshold: continue` would silently keep only the
    *worst* (most distant) matches and drop the *best* (closest) ones --
    the exact inversion mem0ai/mem0#4453 originally described. Every real
    vector store this build inspected (see module docstring) complies with
    the contract, so this specific failure is not reachable through any
    adapter this repo ships today; this test exists only to make the
    contract's importance concrete against the real scoring function, not
    to claim the bug reproduces live anywhere.
    """
    from mem0.utils.scoring import score_and_rank

    raw_distance_candidates = [
        {"id": "actually_close", "score": 0.05, "payload": {"data": "closest (small distance)"}},
        {"id": "actually_far", "score": 0.95, "payload": {"data": "farthest (large distance)"}},
    ]
    results = score_and_rank(
        semantic_results=raw_distance_candidates,
        bm25_scores={},
        entity_boosts={},
        threshold=0.5,
        top_k=10,
    )
    kept_ids = {r["id"] for r in results}
    # If scores were misinterpreted as similarity, the CLOSEST real match
    # (raw distance 0.05) is wrongly dropped and the FARTHEST (0.95) kept.
    assert kept_ids == {"actually_far"}
    assert "actually_close" not in kept_ids


# ---------------------------------------------------------------------------
# Elasticsearch support: adapter-level config-building tests for
# vector_store_provider="elasticsearch" -- see mem0_direct_adapter.py's
# module docstring for the #5980 finding these exist to make reachable.
# ---------------------------------------------------------------------------


def test_elasticsearch_vector_store_config_threads_embedder_dims_into_embedding_model_dims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same defensive dimension-threading pattern already used for
    redis/valkey/qdrant (see _build_vector_store_config) applies identically
    to elasticsearch: the resolved embedder dimension is threaded into
    `embedding_model_dims` explicitly, so this adapter never relies on
    ElasticsearchConfig's own 1536 default.
    """
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "elasticsearch")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_URL", "https://es.internal:9200")
    monkeypatch.setenv("MEM0_DIRECT_ELASTICSEARCH_API_KEY", "fake-test-api-key")

    adapter = Mem0DirectAdapter()

    vector_store_config = adapter._config_dict["vector_store"]["config"]  # type: ignore[index]
    assert vector_store_config["embedding_model_dims"] == 384  # fastembed's default
    assert vector_store_config["host"] == "https://es.internal:9200"
    assert vector_store_config["api_key"] == "fake-test-api-key"
    assert adapter._config_dict["vector_store"]["provider"] == "elasticsearch"  # type: ignore[index]


def test_elasticsearch_vector_store_url_env_var_is_provider_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEM0_DIRECT_ELASTICSEARCH_URL (the provider-specific fallback) works
    the same way MEM0_DIRECT_REDIS_URL/MEM0_DIRECT_VALKEY_URL/
    MEM0_DIRECT_QDRANT_URL already do -- threaded into `host`, not `url`
    (ElasticsearchConfig has no `url` field; see module docstring).
    """
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "elasticsearch")
    monkeypatch.delenv("MEM0_DIRECT_VECTOR_STORE_URL", raising=False)
    monkeypatch.setenv("MEM0_DIRECT_ELASTICSEARCH_URL", "https://es.example.internal:9200")
    monkeypatch.setenv("MEM0_DIRECT_ELASTICSEARCH_API_KEY", "fake-test-api-key")

    adapter = Mem0DirectAdapter()

    vector_store_config = adapter._config_dict["vector_store"]["config"]  # type: ignore[index]
    assert vector_store_config["host"] == "https://es.example.internal:9200"
    assert "url" not in vector_store_config


def test_elasticsearch_cloud_id_override_satisfies_configuration_presence_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller who wants Elastic Cloud (`cloud_id`, no `host`) can pass it
    via `vector_store_config` without needing MEM0_DIRECT_VECTOR_STORE_URL
    set at all -- the presence check accepts `cloud_id` as an alternative
    to the usual url_key, matching ElasticsearchConfig's own
    `cloud_id`-or-`host` validator.
    """
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "elasticsearch")
    monkeypatch.delenv("MEM0_DIRECT_VECTOR_STORE_URL", raising=False)
    monkeypatch.delenv("MEM0_DIRECT_ELASTICSEARCH_URL", raising=False)

    adapter = Mem0DirectAdapter(
        vector_store_config={"cloud_id": "test:deployment-id", "api_key": "fake-test-api-key"}
    )

    vector_store_config = adapter._config_dict["vector_store"]["config"]  # type: ignore[index]
    assert vector_store_config["cloud_id"] == "test:deployment-id"
    assert "host" not in vector_store_config


def test_elasticsearch_without_url_or_cloud_id_raises_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "elasticsearch")
    monkeypatch.delenv("MEM0_DIRECT_VECTOR_STORE_URL", raising=False)
    monkeypatch.delenv("MEM0_DIRECT_ELASTICSEARCH_URL", raising=False)

    with pytest.raises(BackendNotConfiguredError):
        Mem0DirectAdapter()


# ---------------------------------------------------------------------------
# Real-package regression: mem0ai/mem0#5980 (Elasticsearch filter-value term
# injection). Exercises the actual installed
# mem0.vector_stores.elasticsearch.ElasticsearchDB and
# mem0.configs.vector_stores.elasticsearch.ElasticsearchConfig classes
# directly (mocking only the `elasticsearch.Elasticsearch` wire client), not
# a memtrust reimplementation. See module docstring's "Elasticsearch
# support" section: this bug class is confirmed FIXED in the installed
# package.
# ---------------------------------------------------------------------------


def _real_elasticsearch_db(mock_client: MagicMock) -> Any:
    """Constructs the real, installed ElasticsearchDB with the
    elasticsearch.Elasticsearch wire client mocked out -- same pattern the
    Qdrant real-package tests above use for QdrantClient.
    """
    from mem0.vector_stores.elasticsearch import ElasticsearchDB

    with patch("mem0.vector_stores.elasticsearch.Elasticsearch", return_value=mock_client):
        return ElasticsearchDB(
            collection_name="memtrust_filter_injection_test",
            host="https://es.internal",
            port=9200,
            api_key="fake-test-api-key",
            embedding_model_dims=384,
            auto_create_index=False,
        )


def test_real_elasticsearch_config_defaults_embedding_model_dims_to_1536() -> None:
    """Pins the exact field default that makes the #4297 bug *class*
    (embedding-dimension mismatch, not #5980's injection bug) possible for
    elasticsearch too, in the installed mem0ai==2.0.12 Python package:
    ElasticsearchConfig.embedding_model_dims defaults to 1536, the same
    hardcoded OpenAI dimension QdrantConfig defaults to. If a future
    mem0ai release changes this default, this test fails loudly, which is
    the signal to revisit the module docstring's Elasticsearch support
    finding.
    """
    from mem0.configs.vector_stores.elasticsearch import ElasticsearchConfig

    config = ElasticsearchConfig(host="localhost", api_key="fake-test-api-key")
    assert config.embedding_model_dims == 1536


def test_real_elasticsearch_benign_scalar_filter_is_accepted_and_forwarded() -> None:
    """Control case: an ordinary, single-user scalar filter (exactly what
    Mem0DirectAdapter.query() itself sends) is accepted by the real,
    installed ElasticsearchDB.list() and forwarded into a real
    Elasticsearch term-query filter clause -- a fix for #5980 must not
    reject this.
    """
    mock_client = MagicMock()
    mock_client.search.return_value = {"hits": {"hits": []}}
    db = _real_elasticsearch_db(mock_client)

    results = db.list(filters={"user_id": "alice"})

    assert results == [[]]
    query_body = mock_client.search.call_args.kwargs["body"]
    assert query_body["query"]["bool"]["must"] == [{"term": {"metadata.user_id": "alice"}}]


def test_real_elasticsearch_malicious_dict_filter_is_rejected_before_reaching_client() -> None:
    """Reproduces mem0ai/mem0#5980's literal example against the real,
    installed ElasticsearchDB: a dict-valued filter for user_id
    (`{"$ne": ""}`, designed to make an unvalidated term query match every
    user) raises ValueError from the installed `_validate_filter()` helper
    BEFORE the query is ever built or sent to the Elasticsearch client --
    confirming the fix, not just trusting the PR's "merged" status.
    """
    mock_client = MagicMock()
    db = _real_elasticsearch_db(mock_client)

    with pytest.raises(ValueError, match="Filter value"):
        db.list(filters={"user_id": {"$ne": ""}})

    # The rejection happens before any query is ever built/sent -- the
    # real Elasticsearch client is never touched for a malicious filter.
    mock_client.search.assert_not_called()


def test_real_elasticsearch_malicious_list_filter_is_rejected_before_reaching_client() -> None:
    """Same #5980 shape, list-valued instead of dict-valued -- confirms the
    real _validate_filter() rejects any non-scalar type, not just dict."""
    mock_client = MagicMock()
    db = _real_elasticsearch_db(mock_client)

    with pytest.raises(ValueError, match="Filter value"):
        db.list(filters={"user_id": ["victim-a", "victim-b"]})

    mock_client.search.assert_not_called()


def test_real_elasticsearch_malicious_key_is_also_rejected() -> None:
    """The installed fix validates the filter KEY too (a regex allowlist),
    not only the value -- confirms the key-injection half of #5980's fix
    (`_SAFE_FILTER_KEY`) is present, not just the value-type half."""
    mock_client = MagicMock()
    db = _real_elasticsearch_db(mock_client)

    with pytest.raises(ValueError, match="Invalid filter key"):
        db.list(filters={"user_id) OR 1=1 --": "alice"})

    mock_client.search.assert_not_called()


# ---------------------------------------------------------------------------
# Mem0DirectAdapter.probe_raw_filter() -- the primitive
# evals/filter_injection.py calls. Adapter-level control-flow tests via the
# memory= injection point (see _FakeMemory above), same layer-1 convention
# as the rest of this file's non-real-package tests.
# ---------------------------------------------------------------------------


def test_probe_raw_filter_reports_accepted_true_when_vector_store_list_succeeds() -> None:
    fake_vector_store = MagicMock()
    fake_vector_store.list.return_value = [[]]
    fake = _FakeMemory(vector_store=fake_vector_store)
    adapter = Mem0DirectAdapter(memory=fake)

    result = adapter.probe_raw_filter({"user_id": "alice"})

    assert result.accepted is True
    assert result.applicable is True
    assert result.error is None
    fake_vector_store.list.assert_called_once_with(filters={"user_id": "alice"})


def test_probe_raw_filter_reports_accepted_false_when_vector_store_list_raises() -> None:
    """Models a store whose filter-building layer rejects a malicious
    value, the same shape the real, installed ElasticsearchDB exhibits --
    see the real-package regression tests above for that exact case
    exercised against the installed package instead of this fake."""
    fake_vector_store = MagicMock()
    fake_vector_store.list.side_effect = ValueError("Filter value for 'user_id' must be scalar")
    fake = _FakeMemory(vector_store=fake_vector_store)
    adapter = Mem0DirectAdapter(memory=fake)

    result = adapter.probe_raw_filter({"user_id": {"$ne": ""}})

    assert result.accepted is False
    assert result.applicable is True
    assert result.error is not None
    assert "scalar" in result.error


def test_probe_raw_filter_reports_not_applicable_construction_failure_via_applicable_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A construction-time config rejection (Valkey's missing
    embedding_model_dims -- see the CONFIG_REJECTED tests above) never
    reaches the vector store's filter-building layer at all -- accepted is
    False, but applicable is also False, so
    evals/filter_injection.py's classify_filter_injection_case() can tell
    this apart from a genuine filter-validation rejection and report
    NOT_APPLICABLE instead of crediting/blaming the filter value.
    """
    monkeypatch.setenv("MEM0_DIRECT_EMBEDDER_PROVIDER", "fastembed")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_PROVIDER", "valkey")
    monkeypatch.setenv("MEM0_DIRECT_VECTOR_STORE_URL", "redis://localhost:6379")
    adapter = Mem0DirectAdapter(embedder_config={"embedding_dims": None})

    result = adapter.probe_raw_filter({"user_id": "alice"})

    assert result.accepted is False
    assert result.applicable is False
    assert result.error is not None
    assert "config rejected" in result.error
