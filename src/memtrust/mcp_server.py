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


def _capture_cli_help() -> str:
    """Run `memtrust --help` once at import time and return its stdout, so
    the exposed tool's description reflects the real CLI's own --help text
    instead of a hardcoded, driftable description string."""
    try:
        result = subprocess.run(
            [_MEMTRUST_BIN, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "memtrust: benchmark harness for agent-memory backends."
    except (OSError, subprocess.TimeoutExpired) as exc:
        _log(f"could not capture `memtrust --help` for the tool description: {exc}")
        return "memtrust: benchmark harness for agent-memory backends."


_TOOL_DESCRIPTION = (
    "Run the memtrust agent-memory benchmark harness and return the resulting "
    "JSON report. `args` are passed straight through to `memtrust run` (e.g. "
    '["--backends", "mempalace", "--eval", "contradiction"]). Backends without '
    "a configured credential env var are reported as skipped, not an error.\n\n"
    "Real `memtrust --help` output:\n"
    f"{_capture_cli_help()}"
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
