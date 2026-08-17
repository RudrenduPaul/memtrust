"""MCP (Model Context Protocol) server wrapper for the memtrust CLI.

Exposes memtrust's `run` subcommand -- the only memtrust command that
produces a structured, machine-readable report -- as a single MCP tool
over stdio transport.

Note: memtrust has no `--json`-to-stdout mode (confirmed against the real
`memtrust run --help` output -- only `--output <file>` exists, and it
always writes JSON there regardless of flags). So this wrapper appends a
private temp `--output` path instead of a `--json` flag, reads that file
back once the process exits, and returns the parsed JSON as the tool
result, deleting the temp file afterward.

stdout is reserved for the JSON-RPC protocol the MCP client speaks over
stdio, so all logging here goes to stderr (see `_log`).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

#: The real memtrust console-script name (see `[project.scripts]` in
#: pyproject.toml: `memtrust = "memtrust.cli:main"`), resolved via PATH
#: rather than hardcoded to an absolute path so this works inside whatever
#: venv memtrust happens to be installed into.
_MEMTRUST_BIN = "memtrust"


def _log(message: str) -> None:
    """Write a debug/info line to stderr. Never print to stdout -- stdout
    is reserved for the JSON-RPC protocol stream."""
    print(f"[memtrust-mcp] {message}", file=sys.stderr)


_TOOL_DESCRIPTION = (
    "Runs memtrust's agent-memory benchmark suite (LongMemEval, LoCoMo, "
    "contradiction-detection, and a dozen other evals) against one or more "
    "configured memory backends -- MemPalace, Mem0, Zep/Graphiti, "
    "OpenViking -- and returns the resulting JSON report. Call this when an "
    "agent needs a fresh, reproducible comparison of memory-backend quality "
    "or performance rather than trusting a vendor's self-reported numbers; "
    'do not call it just to read documentation (pass `["--help"]` for '
    "that instead) or to inspect a report that already exists on disk (use "
    "the CLI's `report` subcommand directly for that). A backend is only "
    "evaluated if its credential env var is set in the server's "
    "environment (e.g. `MEM0_SELFHOSTED_API_KEY`, `GRAPHITI_GEMINI_API_KEY`); "
    "backends missing credentials are reported as skipped rather than "
    "causing an error, so no prerequisite check is required before "
    "calling.\n\n"
    "Behavior: this shells out to the real `memtrust run` CLI as a "
    "subprocess (network calls happen for whichever backends are "
    "configured; evals also read/write a private temp file for the JSON "
    "report, cleaned up automatically). It is read-only with respect to "
    "the backends themselves -- it queries and scores them, it does not "
    "modify backend state -- and each call is idempotent: running the same "
    "args twice re-runs the full suite and produces a fresh, independent "
    "report rather than mutating prior results. Runs can take several "
    "minutes (subprocess timeout is 600s) depending on which evals and "
    "backends are selected. On a non-zero CLI exit, the tool does not "
    'raise; it returns `{"error": ..., "returncode": ..., "stderr": '
    "...}` so the calling agent can inspect what failed.\n\n"
    "Parameter: `args` is a list[str] of argv tokens appended to `memtrust "
    'run` (NOT a shell string, and do not include "run" itself or an '
    "`--output` flag -- the wrapper supplies its own temp output path). "
    "Real examples pulled from `memtrust run --help`:\n"
    "  - [] -- run every eval against every configured backend (both flags "
    'default to "all")\n'
    '  - ["--backends", "mempalace", "--eval", "contradiction"] -- one '
    "backend, one eval\n"
    '  - ["--backends", "mem0,zep", "--eval", "longmemeval,locomo", '
    '"--scale-stress-n-records", "10000"] -- multiple backends and evals, '
    "plus a numeric tuning flag\n"
    '  - ["--help"] -- exits before writing a report, so it surfaces as an '
    "error dict whose `stdout` field holds the CLI's own live help text; "
    "useful for discovering flags this description doesn't enumerate, such "
    "as --sign or --locomo-dataset-path\n\n"
    "Return shape: on success, the parsed contents of memtrust's JSON "
    "report file (per-backend, per-eval results and scores -- see "
    "`memtrust report --help` for how a human would render the same file). "
    "On failure, a dict with an `error` key plus whichever of `returncode` "
    "/ `stderr` / `stdout` is available."
)

mcp = MCPServer("memtrust-mcp")


@mcp.tool(description=_TOOL_DESCRIPTION)
def run(args: list[str]) -> dict[str, Any]:
    """Shell out to the real memtrust CLI's `run` subcommand and return its
    JSON report as a dict.

    memtrust has no `--json`-to-stdout mode, so this always writes the
    report to a private temp file (`--output <tmp>`) rather than appending
    a `--json` flag, then reads that file back and parses it. The temp
    file is removed afterward regardless of outcome.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "memtrust-mcp-report.json"
        command = [_MEMTRUST_BIN, "run", *args, "--output", str(out_path)]
        _log(f"running: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {
                "error": "memtrust run failed",
                "returncode": result.returncode,
                "stderr": result.stderr.strip(),
            }
        try:
            report_text = out_path.read_text()
        except OSError as exc:
            return {
                "error": f"memtrust run exited 0 but no report file was found: {exc}",
                "stdout": result.stdout.strip(),
            }
        try:
            return json.loads(report_text)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            return {"error": f"report file was not valid JSON: {exc}"}


def main() -> None:
    """Entry point for the `memtrust-mcp` console script (see
    `[project.scripts]` in pyproject.toml). Runs the server over stdio
    transport, which needs no hosting: the MCP client spawns this as a
    local subprocess and speaks JSON-RPC over stdin/stdout."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
