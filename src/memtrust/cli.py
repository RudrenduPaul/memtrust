"""`memtrust run` and `memtrust report` -- the CLI entry points.

`memtrust run` executes the requested eval suite against whichever
requested backends actually have credentials configured. An unconfigured
backend prints SKIPPED and the run continues -- this command never raises
on missing credentials, which is what lets it run in a fresh clone or in
CI with zero vendor API keys. See adapters/base.py's
BackendNotConfiguredError contract.

`memtrust report` reads a prior run's JSON output and prints a formatted
summary -- it never re-runs anything or re-derives a score, it only
reformats what a `memtrust run` invocation already produced and wrote to
disk.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from memtrust import __version__
from memtrust.adapters import ADAPTER_REGISTRY
from memtrust.adapters.base import BackendNotConfiguredError, MemoryBackendAdapter
from memtrust.evals.compression import CompressionEvalResult, run_compression_eval
from memtrust.evals.contradiction import ContradictionEvalResult, run_contradiction_eval
from memtrust.evals.locomo import LoCoMoResult, load_exclude_question_ids, run_locomo
from memtrust.evals.longmemeval import LongMemEvalResult, run_longmemeval
from memtrust.evals.ranking_quality import RankingQualityEvalResult, run_ranking_quality_eval
from memtrust.evals.resource_sync_safety import ResourceSyncEvalResult, run_resource_sync_eval
from memtrust.scoring.cost_tracker import CostTracker
from memtrust.scoring.llm_judge import LLMJudge

#: Explicit width rather than relying on terminal auto-detection -- with 5
#: evals now registered, the `report` table has 7 columns; under a
#: non-tty runner (tests, CI logs) rich's default-width fallback wraps
#: cell text across lines, which is cosmetic in a real terminal but breaks
#: substring assertions on rendered output. A fixed wide width keeps
#: rendering deterministic in both contexts.
console = Console(width=200)

#: The 4 canonical, non-aliased backend names v0.1 tracks at full eval
#: depth. "zep" and "graphiti" both resolve to the same adapter in
#: ADAPTER_REGISTRY; "all" expands to this list, not to every registry key,
#: so a backend is never silently evaluated twice under two names.
ALL_BACKENDS = ["mempalace", "mem0", "zep", "openviking"]
ALL_EVALS = [
    "longmemeval",
    "locomo",
    "contradiction",
    "resource_sync_safety",
    "compression",
    "ranking_quality",
]


def _resolve_backend_names(backends_arg: str) -> list[str]:
    if backends_arg.strip().lower() == "all":
        return list(ALL_BACKENDS)
    names = [n.strip().lower() for n in backends_arg.split(",") if n.strip()]
    unknown = [n for n in names if n not in ADAPTER_REGISTRY]
    if unknown:
        raise click.BadParameter(
            f"unknown backend(s): {', '.join(unknown)}. "
            f"Known backends: {', '.join(sorted(set(ALL_BACKENDS)))}"
        )
    return names


def _resolve_eval_names(eval_arg: str) -> list[str]:
    if eval_arg.strip().lower() == "all":
        return list(ALL_EVALS)
    names = [n.strip().lower() for n in eval_arg.split(",") if n.strip()]
    unknown = [n for n in names if n not in ALL_EVALS]
    if unknown:
        raise click.BadParameter(
            f"unknown eval(s): {', '.join(unknown)}. Known evals: {', '.join(ALL_EVALS)}"
        )
    return names


def _serialize_eval_result(result: object) -> dict[str, Any]:
    """Best-effort dataclass -> plain dict for JSON output. Nested
    dataclasses (case results) are converted the same way via asdict.
    """
    if isinstance(result, LongMemEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "accuracy": result.accuracy,
            "judge_unavailable": result.judge_unavailable,
            "n_cases": len(result.case_results),
            "n_graded": len(result.graded_cases),
            "n_records_empty": result.n_records_empty,
            "cases": [asdict(c) for c in result.case_results],
        }
    if isinstance(result, LoCoMoResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "accuracy": result.accuracy,
            "non_adversarial_accuracy": result.non_adversarial_accuracy,
            "accuracy_by_category": result.accuracy_by_category(),
            "n_cases": len(result.case_results),
            "n_graded": len(result.graded_cases),
            "n_records_empty": result.n_records_empty,
            "n_excluded_ground_truth": result.n_excluded_ground_truth,
            "cases": [asdict(c) for c in result.case_results],
        }
    if isinstance(result, ContradictionEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "flagged_rate": result.flagged_rate,
            "silent_overwrite_rate": result.silent_overwrite_rate,
            "served_stale_rate": result.served_stale_rate,
            "not_applicable_rate": result.not_applicable_rate,
            "empty_or_lost_rate": result.empty_or_lost_rate,
            "n_cases": len(result.case_results),
            "cases": [
                {
                    "case_id": c.case.case_id,
                    "subject": c.case.subject,
                    "signal": str(c.signal),
                    "adapter_reported_signal": (
                        str(c.adapter_reported_signal) if c.adapter_reported_signal else None
                    ),
                    "error": c.error,
                }
                for c in result.case_results
            ],
        }
    if isinstance(result, ResourceSyncEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "user_file_deletion_rate": result.user_file_deletion_rate,
            "preserved_rate": result.preserved_rate,
            "overwritten_unchanged_rate": result.overwritten_unchanged_rate,
            "n_files": len(result.file_results),
            "files": [
                {
                    "case_id": f.case_id,
                    "path_suffix": f.path_suffix,
                    "origin": f.origin,
                    "signal": str(f.signal),
                    "error": f.error,
                }
                for f in result.file_results
            ],
        }
    if isinstance(result, RankingQualityEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "signal_driven_rate": result.signal_driven_rate,
            "missing_ordering_key_rate": result.missing_ordering_key_rate,
            "order_inconsistent_rate": result.order_inconsistent_rate,
            "not_applicable_rate": result.not_applicable_rate,
            "n_cases": len(result.case_results),
            "cases": [
                {
                    "case_id": c.case.case_id,
                    "ranking_field": c.case.ranking_field,
                    "signal": str(c.signal),
                    "adapter_reported_signal": (
                        str(c.adapter_reported_signal) if c.adapter_reported_signal else None
                    ),
                    "matches_insertion_order": c.matches_insertion_order,
                    "error": c.error,
                }
                for c in result.case_results
            ],
        }
    if isinstance(result, CompressionEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "modes": result.modes,
            "mean_fidelity_by_mode": result.mean_fidelity_by_mode(),
            "fidelity_drop_pp": result.fidelity_drop_pp,
            "mode_results": {
                mode: {
                    "mean_fidelity": mode_result.mean_fidelity,
                    "n_cases": len(mode_result.case_results),
                    "n_scored": len(mode_result.scored_cases),
                    "cases": [asdict(c) for c in mode_result.case_results],
                }
                for mode, mode_result in result.mode_results.items()
            },
        }
    raise TypeError(f"no serializer for {type(result)!r}")


@click.group()
@click.version_option(version=__version__, prog_name="memtrust")
def main() -> None:
    """memtrust: an independent, reproducible benchmark harness for agent-memory backends."""


@main.command()
@click.option(
    "--backends", default="all", show_default=True, help="Comma-separated backend list, or 'all'."
)
@click.option(
    "--eval",
    "eval_arg",
    default="all",
    show_default=True,
    help=(
        "Comma-separated eval list (longmemeval,locomo,contradiction,"
        "resource_sync_safety,compression,ranking_quality), or 'all'."
    ),
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to write the JSON report. Defaults to ./memtrust-report-<date>.json",
)
@click.option(
    "--locomo-exclude-question-ids-file",
    "locomo_exclude_ids_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to a file of known-bad-ground-truth LoCoMo question IDs to exclude from "
        "scoring (JSON array, or one ID per line). See evals/locomo.py's "
        "load_exclude_question_ids() and docs/methodology.md for the ID shape and how "
        "a corrected list (e.g. derived from a published ground-truth audit) plugs in. "
        "No such list ships with memtrust by default."
    ),
)
def run(
    backends: str,
    eval_arg: str,
    output_path: Path | None,
    locomo_exclude_ids_path: Path | None,
) -> None:
    """Run the eval suite against the requested backends.

    Backends without a configured credential env var print SKIPPED and
    the run continues -- this command never crashes on missing
    credentials.
    """
    backend_names = _resolve_backend_names(backends)
    eval_names = _resolve_eval_names(eval_arg)
    locomo_exclude_ids = (
        load_exclude_question_ids(locomo_exclude_ids_path)
        if locomo_exclude_ids_path is not None
        else None
    )
    cost_tracker = CostTracker()
    judge = LLMJudge(cost_tracker=cost_tracker)

    run_id = datetime.now(UTC).strftime("mt_%Y-%m-%dT%H%M%SZ")
    report: dict[str, Any] = {
        "run_id": run_id,
        "memtrust_version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "backends_requested": backend_names,
        "evals_requested": eval_names,
        "results": {},
    }

    console.print(f"[bold]memtrust {__version__}[/bold] -- run_id={run_id}")
    console.print(f"Backends: {', '.join(backend_names)}   Evals: {', '.join(eval_names)}\n")

    for backend_name in backend_names:
        adapter_cls = ADAPTER_REGISTRY[backend_name]
        try:
            adapter: MemoryBackendAdapter = adapter_cls()
        except BackendNotConfiguredError as exc:
            console.print(f"[yellow]{backend_name}: SKIPPED (not configured)[/yellow] -- {exc}")
            report["results"][backend_name] = {
                "status": "skipped",
                "reason": str(exc),
                "missing_env_var": exc.missing_env_var,
            }
            continue

        console.print(f"[green]{backend_name}: configured[/green], running evals...")
        backend_report: dict[str, Any] = {"status": "configured", "evals": {}}

        if "longmemeval" in eval_names:
            console.print(f"  Running LongMemEval against {backend_name}...")
            lme_result = run_longmemeval(adapter, judge)
            backend_report["evals"]["longmemeval"] = _serialize_eval_result(lme_result)
            acc = lme_result.accuracy
            if acc is not None:
                console.print(f"    accuracy: {acc:.1%}")
            else:
                console.print("    accuracy: N/A (judge not configured)")
            if lme_result.n_records_empty:
                console.print(
                    f"    [yellow]records_empty: {lme_result.n_records_empty}/"
                    f"{len(lme_result.case_results)}[/yellow] "
                    "(backend call succeeded but returned nothing)"
                )

        if "locomo" in eval_names:
            console.print(f"  Running LoCoMo against {backend_name}...")
            locomo_result = run_locomo(adapter, judge, exclude_question_ids=locomo_exclude_ids)
            backend_report["evals"]["locomo"] = _serialize_eval_result(locomo_result)
            acc = locomo_result.accuracy
            non_adv_acc = locomo_result.non_adversarial_accuracy
            if acc is not None:
                console.print(f"    accuracy (all categories, incl. adversarial): {acc:.1%}")
            else:
                console.print("    accuracy: N/A (judge not configured)")
            if non_adv_acc is not None:
                console.print(
                    f"    non_adversarial_accuracy (excludes category 5): {non_adv_acc:.1%}"
                )
            else:
                console.print("    non_adversarial_accuracy: N/A (judge not configured)")
            if locomo_result.n_records_empty:
                console.print(
                    f"    [yellow]records_empty: {locomo_result.n_records_empty}/"
                    f"{len(locomo_result.case_results)}[/yellow] "
                    "(backend call succeeded but returned nothing)"
                )
            if locomo_result.n_excluded_ground_truth:
                console.print(
                    f"    excluded_ground_truth: {locomo_result.n_excluded_ground_truth}/"
                    f"{len(locomo_result.case_results)} "
                    "(known-bad ground truth, excluded via --locomo-exclude-question-ids-file)"
                )

        if "contradiction" in eval_names:
            console.print(f"  Running Contradiction-Detection against {backend_name}...")
            contra_result = run_contradiction_eval(adapter)
            backend_report["evals"]["contradiction"] = _serialize_eval_result(contra_result)
            fr = contra_result.flagged_rate
            so = contra_result.silent_overwrite_rate
            ss = contra_result.served_stale_rate
            eol = contra_result.empty_or_lost_rate
            if fr is not None:
                console.print(
                    f"    flagged: {fr:.1%}  silent-overwrite: {so:.1%}  served-stale: {ss:.1%}"
                    f"  empty-or-lost: {eol:.1%}"
                )
            else:
                console.print("    N/A (no scoreable cases)")

        if "resource_sync_safety" in eval_names:
            console.print(f"  Running Resource-Sync Safety against {backend_name}...")
            rss_result = run_resource_sync_eval(adapter)
            backend_report["evals"]["resource_sync_safety"] = _serialize_eval_result(rss_result)
            if rss_result.skipped:
                console.print(f"    SKIPPED: {rss_result.skip_reason}")
            else:
                dr = rss_result.user_file_deletion_rate
                if dr is not None:
                    console.print(f"    user-file deletion rate: {dr:.1%}")
                else:
                    console.print("    N/A (no scoreable files)")

        if "compression" in eval_names:
            console.print(f"  Running Compression/Round-Trip-Fidelity against {backend_name}...")
            compression_result = run_compression_eval(adapter)
            backend_report["evals"]["compression"] = _serialize_eval_result(compression_result)
            means = compression_result.mean_fidelity_by_mode()
            if means:
                rendered = "  ".join(
                    f"{mode}: {value:.1%}" if value is not None else f"{mode}: N/A"
                    for mode, value in means.items()
                )
                console.print(f"    fidelity by mode -- {rendered}")
            else:
                console.print("    N/A (no scoreable cases)")

        if "ranking_quality" in eval_names:
            console.print(f"  Running Ranking-Quality against {backend_name}...")
            ranking_result = run_ranking_quality_eval(adapter)
            backend_report["evals"]["ranking_quality"] = _serialize_eval_result(ranking_result)
            mok = ranking_result.missing_ordering_key_rate
            sdr = ranking_result.signal_driven_rate
            oir = ranking_result.order_inconsistent_rate
            if mok is not None:
                console.print(
                    f"    missing-ordering-key: {mok:.1%}  signal-driven: {sdr:.1%}"
                    f"  order-inconsistent: {oir:.1%}"
                )
            else:
                console.print("    N/A (no scoreable cases)")

        report["results"][backend_name] = backend_report
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    judge.close()
    report["cost"] = {
        "total_usd": cost_tracker.total_cost_usd,
        "total_input_tokens": cost_tracker.total_input_tokens,
        "total_output_tokens": cost_tracker.total_output_tokens,
    }

    console.print()
    for line in cost_tracker.summary_lines():
        console.print(line)

    out_path = output_path or Path(f"memtrust-report-{datetime.now(UTC).strftime('%Y-%m-%d')}.json")
    out_path.write_text(json.dumps(report, indent=2, default=str))
    console.print(f"\nFull report: {out_path}")


@main.command()
@click.argument("report_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def report(report_path: Path) -> None:
    """Read a prior `memtrust run` JSON report and print a formatted summary."""
    try:
        data = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        console.print(f"[red]Could not parse {report_path} as JSON: {exc}[/red]")
        sys.exit(1)

    console.print(f"[bold]memtrust report[/bold] -- run_id={data.get('run_id', 'unknown')}")
    console.print(f"Generated: {data.get('timestamp', 'unknown')}\n")

    table = Table(title="Backend results")
    table.add_column("Backend")
    table.add_column("Status")
    table.add_column("LongMemEval")
    table.add_column("LoCoMo (all cats / non-adversarial)")
    table.add_column("Contradiction (flagged/overwrite/stale/empty-or-lost)")
    table.add_column("Resource-Sync (user-file deletion rate)")
    table.add_column("Compression fidelity by mode")
    table.add_column("Ranking Quality (missing-ordering-key rate)")

    for backend_name, backend_data in data.get("results", {}).items():
        if backend_data.get("status") == "skipped":
            table.add_row(backend_name, "SKIPPED", "-", "-", "-", "-", "-", "-")
            continue

        evals = backend_data.get("evals", {})
        lme = evals.get("longmemeval", {})
        locomo = evals.get("locomo", {})
        contra = evals.get("contradiction", {})
        rss = evals.get("resource_sync_safety", {})
        compression = evals.get("compression", {})
        ranking = evals.get("ranking_quality", {})

        def _fmt_pct(value: float | None) -> str:
            return f"{value:.1%}" if value is not None else "N/A"

        contra_str = (
            f"{_fmt_pct(contra.get('flagged_rate'))} / "
            f"{_fmt_pct(contra.get('silent_overwrite_rate'))} / "
            f"{_fmt_pct(contra.get('served_stale_rate'))} / "
            f"{_fmt_pct(contra.get('empty_or_lost_rate'))}"
            if contra
            else "-"
        )
        if rss.get("skipped"):
            rss_str = "SKIPPED (unsupported)"
        elif rss:
            rss_str = _fmt_pct(rss.get("user_file_deletion_rate"))
        else:
            rss_str = "-"
        compression_str = (
            "  ".join(
                f"{mode}: {_fmt_pct(value)}"
                for mode, value in compression.get("mean_fidelity_by_mode", {}).items()
            )
            if compression
            else "-"
        )
        locomo_acc_str = _fmt_pct(locomo.get("accuracy"))
        locomo_non_adv_str = _fmt_pct(locomo.get("non_adversarial_accuracy"))
        locomo_str = f"{locomo_acc_str} / {locomo_non_adv_str}" if locomo else "-"
        ranking_str = _fmt_pct(ranking.get("missing_ordering_key_rate")) if ranking else "-"
        table.add_row(
            backend_name,
            "configured",
            _fmt_pct(lme.get("accuracy")) if lme else "-",
            locomo_str,
            contra_str,
            rss_str,
            compression_str or "-",
            ranking_str,
        )

    console.print(table)

    cost = data.get("cost", {})
    if cost:
        console.print(f"\nEstimated cost: ${cost.get('total_usd', 0):.4f}")


if __name__ == "__main__":
    main()
