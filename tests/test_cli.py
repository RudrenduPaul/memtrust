"""CLI tests via click's CliRunner -- no real network calls, no real
backend credentials. Confirms the "never crash on missing credentials"
contract end to end through the actual command entry points.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from pytest_httpx import HTTPXMock

from memtrust.cli import ALL_EVALS, main


@pytest.fixture(autouse=True)
def _no_backend_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "MEM0_API_KEY",
        "ZEP_API_KEY",
        "OPENVIKING_API_KEY",
        "MEMPALACE_STORAGE_PATH",
        "MEMTRUST_JUDGE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_run_with_no_credentials_skips_everything_and_exits_zero(tmp_path: Path) -> None:
    runner = CliRunner()
    out_path = tmp_path / "report.json"
    result = runner.invoke(
        main,
        ["run", "--backends", "all", "--eval", "all", "--output", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert "SKIPPED" in result.output
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    assert set(data["results"].keys()) == {"mempalace", "mem0", "zep", "openviking"}
    for backend_result in data["results"].values():
        assert backend_result["status"] == "skipped"


def test_run_with_single_backend(tmp_path: Path) -> None:
    runner = CliRunner()
    out_path = tmp_path / "report.json"
    result = runner.invoke(
        main,
        ["run", "--backends", "mem0", "--eval", "contradiction", "--output", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    assert list(data["results"].keys()) == ["mem0"]


def test_run_surfaces_empty_or_lost_signal_in_json_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """A configured backend whose store()/update() calls succeed (200, no
    exception) but whose search always comes back empty must surface the
    new EMPTY_OR_LOST/records_empty signal in the JSON report -- not get
    silently folded into not_applicable_rate or an ordinary miss."""
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://api.mem0.ai/v1/memories/",
        json={"id": "mem-x"},
        is_reusable=True,
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.mem0.ai/v1/memories/search/",
        json={"results": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        method="PUT",
        is_reusable=True,
        json={"id": "mem-x"},
    )

    runner = CliRunner()
    out_path = tmp_path / "report.json"
    result = runner.invoke(
        main, ["run", "--backends", "mem0", "--eval", "all", "--output", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    evals = data["results"]["mem0"]["evals"]

    assert evals["contradiction"]["empty_or_lost_rate"] == 1.0
    assert evals["contradiction"]["not_applicable_rate"] == 0.0
    assert all(c["signal"] == "empty_or_lost" for c in evals["contradiction"]["cases"])

    assert evals["longmemeval"]["n_records_empty"] == len(evals["longmemeval"]["cases"])
    assert all(c["records_empty"] for c in evals["longmemeval"]["cases"])
    assert evals["locomo"]["n_records_empty"] == len(evals["locomo"]["cases"])
    assert all(c["records_empty"] for c in evals["locomo"]["cases"])


def test_run_rejects_unknown_backend() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--backends", "not-a-real-backend"])
    assert result.exit_code != 0
    assert "unknown backend" in result.output.lower()


def test_run_rejects_unknown_eval() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--eval", "not-a-real-eval"])
    assert result.exit_code != 0
    assert "unknown eval" in result.output.lower()


def test_report_command_reads_prior_run(tmp_path: Path) -> None:
    runner = CliRunner()
    report_path = tmp_path / "report.json"
    run_result = runner.invoke(
        main, ["run", "--backends", "all", "--eval", "all", "--output", str(report_path)]
    )
    assert run_result.exit_code == 0

    report_result = runner.invoke(main, ["report", str(report_path)])
    assert report_result.exit_code == 0, report_result.output
    assert "Backend results" in report_result.output
    assert "SKIPPED" in report_result.output


def test_report_command_rejects_invalid_json(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not valid json{{{")
    runner = CliRunner()
    result = runner.invoke(main, ["report", str(bad_path)])
    assert result.exit_code != 0


def test_version_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_default_output_path_when_not_specified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--backends", "mem0", "--eval", "contradiction"])
    assert result.exit_code == 0
    generated = list(tmp_path.glob("memtrust-report-*.json"))
    assert len(generated) == 1


def test_resource_sync_safety_registered_in_eval_list() -> None:
    assert "resource_sync_safety" in ALL_EVALS


def test_run_resource_sync_safety_skips_cleanly_for_unsupported_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mem0 has no resource-mirror concept (supports_resource_sync defaults
    to False), so this exercises the CLI's end-to-end skip path for the
    new eval without needing any HTTP mocking -- the eval must never call
    the unimplemented list_resource_paths()/trigger_resync() methods."""
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    runner = CliRunner()
    out_path = tmp_path / "report.json"
    result = runner.invoke(
        main,
        ["run", "--backends", "mem0", "--eval", "resource_sync_safety", "--output", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    rss = data["results"]["mem0"]["evals"]["resource_sync_safety"]
    assert rss["skipped"] is True
    assert rss["user_file_deletion_rate"] is None
    assert rss["n_files"] == 0


def test_run_against_configured_backend_full_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """Exercises the "backend is configured, evals actually run" code path
    end to end with a mocked Mem0 HTTP surface -- no real network call."""
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://api.mem0.ai/v1/memories/",
        json={"id": "mem-x"},
        is_reusable=True,
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.mem0.ai/v1/memories/search/",
        json={"results": [{"id": "mem-x", "memory": "some recalled content"}]},
        is_reusable=True,
    )
    httpx_mock.add_response(
        method="PUT",
        is_reusable=True,
        json={"id": "mem-x"},
    )

    runner = CliRunner()
    out_path = tmp_path / "report.json"
    result = runner.invoke(
        main, ["run", "--backends", "mem0", "--eval", "all", "--output", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    assert data["results"]["mem0"]["status"] == "configured"
    assert "longmemeval" in data["results"]["mem0"]["evals"]
    assert "locomo" in data["results"]["mem0"]["evals"]
    assert "contradiction" in data["results"]["mem0"]["evals"]
    assert "compression" in data["results"]["mem0"]["evals"]

    report_result = runner.invoke(main, ["report", str(out_path)])
    assert report_result.exit_code == 0
    assert "configured" in report_result.output


def test_compression_eval_is_registered_in_all_evals() -> None:
    from memtrust.cli import ALL_EVALS

    assert "compression" in ALL_EVALS


def test_run_accepts_compression_as_a_single_eval_selection(tmp_path: Path) -> None:
    runner = CliRunner()
    out_path = tmp_path / "report.json"
    result = runner.invoke(
        main,
        ["run", "--backends", "mem0", "--eval", "compression", "--output", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    # mem0 is unconfigured (autouse fixture strips credentials), so it's
    # skipped -- this just confirms "compression" resolves as a known eval
    # name and the command doesn't crash, matching the "never crash on
    # missing credentials" contract exercised elsewhere in this file.
    assert data["results"]["mem0"]["status"] == "skipped"


def test_scale_stress_eval_is_registered_in_all_evals() -> None:
    assert "scale_stress" in ALL_EVALS


def test_run_accepts_scale_stress_as_a_single_eval_selection(tmp_path: Path) -> None:
    runner = CliRunner()
    out_path = tmp_path / "report.json"
    result = runner.invoke(
        main,
        ["run", "--backends", "mem0", "--eval", "scale_stress", "--output", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    # mem0 is unconfigured (autouse fixture strips credentials), so it's
    # skipped -- this just confirms "scale_stress" resolves as a known eval
    # name and the command doesn't crash, matching the "never crash on
    # missing credentials" contract exercised elsewhere in this file.
    assert data["results"]["mem0"]["status"] == "skipped"


def test_run_scale_stress_against_configured_backend_serializes_full_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """Exercises the CLI's scale_stress wiring end to end against a mocked
    Mem0 HTTP surface: the --scale-stress-n-records flag is honored (kept
    small here so the mocked HTTP round trips stay fast), the JSON report
    carries a fully-populated ScaleTestResult, and `memtrust report` renders
    the new Scale/Volume Stress column without crashing.
    """
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://api.mem0.ai/v1/memories/",
        json={"id": "mem-x"},
        is_reusable=True,
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.mem0.ai/v1/memories/search/",
        json={"results": [{"id": "mem-x", "memory": "some recalled content"}]},
        is_reusable=True,
    )

    runner = CliRunner()
    out_path = tmp_path / "report.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--backends",
            "mem0",
            "--eval",
            "scale_stress",
            "--scale-stress-n-records",
            "20",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out_path.read_text())
    scale_stress = data["results"]["mem0"]["evals"]["scale_stress"]
    assert scale_stress["n_records_requested"] == 20
    assert scale_stress["backend"] == "mem0"
    assert scale_stress["signal"] in {
        "worked_at_scale",
        "silently_degraded_at_scale",
        "partial_degradation",
        "error",
        "not_applicable",
    }
    assert scale_stress["records_stored"] == 20
    assert len(scale_stress["checkpoints"]) >= 2
    assert "signal:" in result.output

    report_result = runner.invoke(main, ["report", str(out_path)])
    assert report_result.exit_code == 0, report_result.output
    assert "Scale/Volume Stress" in report_result.output
