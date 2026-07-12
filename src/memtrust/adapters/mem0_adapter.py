"""Adapter for Mem0 (https://mem0.ai).

Confidence: HIGH. Mem0 publishes a documented Python SDK (`mem0ai` on PyPI)
built around `MemoryClient(api_key=...)` with `.add()`, `.search()`, and
`.update()` methods, and reads `MEM0_API_KEY` from the environment when no
key is passed explicitly (docs.mem0.ai/platform/quickstart, June 2026 SDK
v2.0.8 release notes). This adapter talks to the same REST surface the
official SDK wraps, using httpx directly rather than depending on the
`mem0ai` package, to keep memtrust's own dependency tree small and to keep
every HTTP call independently mockable in tests without a vendor SDK
installed. Endpoint paths below are best-effort reconstructions of the
REST API the documented SDK calls sit on top of -- see
docs/methodology.md for what is confirmed vs. inferred.

Mem0's own memory pipeline decides, per stored fact, whether to ADD a new
memory, UPDATE an existing one, or leave it alone -- this is documented
Mem0 behavior, not a memtrust assumption, and it is exactly the kind of
vendor-side conflict resolution the contradiction-detection eval is built
to observe rather than short-circuit.
"""

from __future__ import annotations

import os

import httpx

from memtrust.adapters.base import (
    BackendAPIError,
    BackendNotConfiguredError,
    ConflictSignal,
    MemoryBackendAdapter,
    MemoryRecord,
    QueryResult,
    StoreResult,
    UpdateResult,
)

DEFAULT_BASE_URL = "https://api.mem0.ai"


class Mem0Adapter(MemoryBackendAdapter):
    name = "mem0"
    env_var = "MEM0_API_KEY"
    supports_update = True

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        api_key = os.environ.get(self.env_var)
        if not api_key:
            raise BackendNotConfiguredError(self.name, self.env_var)
        self._http = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def store(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        mode: str | None = None,
    ) -> StoreResult:
        # Mem0 has no documented operating-mode variant to select -- `mode`
        # is accepted and ignored (no-op) so callers can pass it uniformly
        # across every adapter. See MemoryBackendAdapter.supported_modes.
        del mode
        timer = self._timed()
        payload: dict[str, object] = {
            "messages": [{"role": "user", "content": content}],
            "user_id": session_id,
        }
        if metadata:
            payload["metadata"] = metadata
        try:
            resp = self._http.post("/v1/memories/", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        memory_id = _extract_memory_id(data)
        return StoreResult(memory_id=memory_id, latency_ms=timer.elapsed_ms(), raw=data)

    def query(
        self, session_id: str, query: str, top_k: int = 5, mode: str | None = None
    ) -> QueryResult:
        del mode  # no-op, see store() above
        timer = self._timed()
        payload = {"query": query, "filters": {"user_id": session_id}, "top_k": top_k}
        try:
            resp = self._http.post("/v1/memories/search/", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc

        raw_results = data.get("results", data if isinstance(data, list) else [])
        records = [
            MemoryRecord(
                memory_id=str(item.get("id", "")),
                content=str(item.get("memory", item.get("text", ""))),
                score=item.get("score"),
                created_at=item.get("created_at"),
                metadata=item.get("metadata") or {},
                raw=item,
            )
            for item in raw_results
        ]
        # Mem0's search response does not carry an explicit "this result
        # superseded a prior one" marker in the documented API surface --
        # conflict handling happens inside Mem0's own add/update pipeline,
        # invisibly to the caller. Until Mem0 documents a per-result
        # conflict marker, memtrust conservatively records SILENT_OVERWRITE
        # when exactly one record is returned for a query that the
        # contradiction eval knows should have two candidate facts, and
        # FLAGGED only if the response ever includes more than one
        # conflicting record for the same fact slot. See
        # evals/contradiction.py for how this signal is actually derived
        # per test case; this adapter only reports the raw record set.
        conflict_signal = ConflictSignal.NOT_APPLICABLE
        return QueryResult(
            records=records,
            conflict_signal=conflict_signal,
            latency_ms=timer.elapsed_ms(),
            raw=data,
        )

    def update(self, session_id: str, memory_id: str, content: str) -> UpdateResult:
        timer = self._timed()
        try:
            resp = self._http.put(f"/v1/memories/{memory_id}/", json={"text": content})
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendAPIError(self.name, str(exc)) from exc
        return UpdateResult(
            memory_id=memory_id, acknowledged=True, latency_ms=timer.elapsed_ms(), raw=data
        )

    def close(self) -> None:
        self._http.close()


def _extract_memory_id(data: object) -> str:
    if isinstance(data, dict):
        if "id" in data:
            return str(data["id"])
        results = data.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return str(results[0].get("id", ""))
    return ""
