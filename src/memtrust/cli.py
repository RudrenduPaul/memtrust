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
from memtrust.adapters.mempalace_adapter import MemPalaceAdapter
from memtrust.evals.compression import CompressionEvalResult, run_compression_eval
from memtrust.evals.contradiction import ContradictionEvalResult, run_contradiction_eval
from memtrust.evals.crash_recovery import CrashRecoveryEvalResult, run_crash_recovery_eval
from memtrust.evals.embedding_drift import EmbeddingDriftEvalResult, run_embedding_drift_eval
from memtrust.evals.extraction_quality import (
    ExtractionQualityEvalResult,
    run_extraction_quality_eval,
)
from memtrust.evals.filter_injection import FilterInjectionEvalResult, run_filter_injection_eval
from memtrust.evals.lock_contention import LockContentionEvalResult, run_lock_contention_eval
from memtrust.evals.locomo import LoCoMoResult, load_exclude_question_ids, run_locomo
from memtrust.evals.longmemeval import LongMemEvalResult, run_longmemeval
from memtrust.evals.migration_rollback import (
    MigrationRollbackEvalResult,
    run_migration_rollback_eval,
)
from memtrust.evals.orphan_cleanup import OrphanCleanupEvalResult, run_orphan_cleanup_eval
from memtrust.evals.ranking_quality import RankingQualityEvalResult, run_ranking_quality_eval
from memtrust.evals.resource_sync_safety import ResourceSyncEvalResult, run_resource_sync_eval
from memtrust.evals.result_consistency import ConsistencyEvalResult, run_result_consistency_eval
from memtrust.evals.scale_stress import (
    DEFAULT_N_RECORDS,
    ScaleTestResult,
    run_scale_stress_eval,
)
from memtrust.evals.stats_accuracy import StatsAccuracyEvalResult, run_stats_accuracy_eval
from memtrust.evals.temporal_kg_boundary import (
    TemporalKGBoundaryEvalResult,
    run_temporal_kg_boundary_eval,
)
from memtrust.receipt import (
    PUBLIC_KEY_ENV_VAR,
    ReceiptError,
    receipt_path_for,
    sign_report_with_keyfile,
    verify_receipt_file,
    write_keypair,
)
from memtrust.scoring.cost_tracker import CostTracker
from memtrust.scoring.llm_judge import LLMJudge

#: Explicit width rather than relying on terminal auto-detection -- with 17
#: evals now registered, the `report` table has many columns; under a
#: non-tty runner (tests, CI logs) rich's default-width fallback wraps
#: cell text across lines, which is cosmetic in a real terminal but breaks
#: substring assertions on rendered output. A fixed wide width keeps
#: rendering deterministic in both contexts.
console = Console(width=400)

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
    "scale_stress",
    "embedding_drift",
    "crash_recovery",
    "extraction_quality",
    "migration_rollback",
    "filter_injection",
    "lock_contention",
    "stats_accuracy",
    "orphan_cleanup",
    "result_consistency",
    "temporal_kg_boundary",
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
            "nested_content_unindexed_rate": result.nested_content_unindexed_rate,
            "n_files": len(result.file_results),
            "files": [
                {
                    "case_id": f.case_id,
                    "path_suffix": f.path_suffix,
                    "origin": f.origin,
                    "signal": str(f.signal),
                    "indexed_after_resync": f.indexed_after_resync,
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
    if isinstance(result, CrashRecoveryEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "recovered_rate": result.recovered_rate,
            "index_lost_data_survived_rate": result.index_lost_data_survived_rate,
            "data_lost_rate": result.data_lost_rate,
            "n_cases": len(result.case_results),
            "cases": [
                {
                    "case_id": c.case.case_id,
                    "signal": str(c.signal),
                    "present_before_crash": c.present_before_crash,
                    "queryable_after_crash": c.queryable_after_crash,
                    "raw_store_contains_after_crash": c.raw_store_contains_after_crash,
                    "error": c.error,
                }
                for c in result.case_results
            ],
        }
    if isinstance(result, MigrationRollbackEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "restored_rate": result.restored_rate,
            "data_lost_rate": result.data_lost_rate,
            "n_cases": len(result.case_results),
            "cases": [
                {
                    "case_id": c.case.case_id,
                    "signal": str(c.signal),
                    "original_data_recoverable": c.original_data_recoverable,
                    "adapter_reported_recoverable": c.adapter_reported_recoverable,
                    "error": c.error,
                }
                for c in result.case_results
            ],
        }
    if isinstance(result, ExtractionQualityEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "junk_retained_rate": result.junk_retained_rate,
            "junk_rejected_rate": result.junk_rejected_rate,
            "valid_retained_rate": result.valid_retained_rate,
            "valid_lost_rate": result.valid_lost_rate,
            "feedback_loop_duplicate_rate": result.feedback_loop_duplicate_rate,
            "n_cases": len(result.case_results),
            "n_feedback_loop_cases": len(result.feedback_loop_results),
            "cases": [
                {
                    "case_id": c.case.case_id,
                    "category": c.case.category,
                    "should_be_stored": c.case.should_be_stored,
                    "signal": str(c.signal),
                    "retrieved": c.retrieved,
                    "error": c.error,
                }
                for c in result.case_results
            ],
            "feedback_loop_cases": [
                {
                    "case_id": c.case.case_id,
                    "signal": str(c.signal),
                    "records_after_first_store": c.records_after_first_store,
                    "records_after_second_store": c.records_after_second_store,
                    "error": c.error,
                }
                for c in result.feedback_loop_results
            ],
        }
    if isinstance(result, FilterInjectionEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "injection_succeeded_rate": result.injection_succeeded_rate,
            "malicious_rejected_rate": result.malicious_rejected_rate,
            "benign_accepted_rate": result.benign_accepted_rate,
            "benign_false_positive_rate": result.benign_false_positive_rate,
            "n_cases": len(result.case_results),
            "cases": [
                {
                    "case_id": c.case.case_id,
                    "malicious": c.case.malicious,
                    "filter_key": c.case.filter_key,
                    "signal": str(c.signal),
                    "probe_accepted": c.probe_accepted,
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
    if isinstance(result, ScaleTestResult):
        return {
            "backend": result.backend_name,
            "n_records_requested": result.n_records_requested,
            "seed": result.seed,
            "signal": str(result.signal),
            "records_stored": result.records_stored,
            "records_store_errors": result.records_store_errors,
            "records_checked": result.records_checked,
            "records_recoverable": result.records_recoverable,
            "recall_degradation_pct": result.recall_degradation_pct,
            "anchor_lost_at_n": result.anchor_lost_at_n,
            "latency_p99_ms": result.latency_p99_ms,
            "error": result.error,
            "checkpoints": [
                {
                    "checkpoint_n": c.checkpoint_n,
                    "records_stored_so_far": c.records_stored_so_far,
                    "recall_rate": c.recall_rate,
                    "anchor_recall": c.anchor_recall,
                    "latency_p50_ms": c.latency_p50_ms,
                    "latency_p99_ms": c.latency_p99_ms,
                    "n_needle_queries": len(c.needle_queries),
                }
                for c in result.checkpoints
            ],
        }
    if isinstance(result, EmbeddingDriftEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "drift_rate": result.drift_rate,
            "clean_rate": result.clean_rate,
            "not_applicable_rate": result.not_applicable_rate,
            "n_records": len(result.record_results),
            "records": [
                {
                    "case_id": r.case_id,
                    "content": r.content,
                    "model_a_label": r.model_a_label,
                    "model_b_label": r.model_b_label,
                    "signal": str(r.signal),
                    "error": r.error,
                }
                for r in result.record_results
            ],
        }
    if isinstance(result, LockContentionEvalResult):
        return {
            "backend": result.backend_name,
            "resource_path": result.resource_path,
            "budget_ms": result.budget_ms,
            "n_concurrent": result.n_concurrent,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "signal": str(result.signal),
            "stalled_count": result.stalled_count,
            "max_latency_ms": result.max_latency_ms,
            "requests": [
                {
                    "worker_index": r.worker_index,
                    "completed": r.completed,
                    "succeeded": r.succeeded,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in result.requests
            ],
        }
    if isinstance(result, StatsAccuracyEvalResult):
        return {
            "backend": result.backend_name,
            "n_records_requested": result.n_records_requested,
            "records_stored": result.records_stored,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "signal": str(result.signal),
            "verified_count": result.verified_count,
            "reported_count": result.reported_count,
            "undercount_gap": result.undercount_gap,
            "error": result.error,
        }
    if isinstance(result, OrphanCleanupEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "orphaned_vector_entry_rate": result.orphaned_vector_entry_rate,
            "clean_rate": result.clean_rate,
            "n_files": len(result.file_results),
            "files": [
                {
                    "case_id": f.case_id,
                    "path_suffix": f.path_suffix,
                    "signal": str(f.signal),
                    "present_after_delete": f.present_after_delete,
                    "queryable_after_delete": f.queryable_after_delete,
                    "error": f.error,
                }
                for f in result.file_results
            ],
        }
    if isinstance(result, ConsistencyEvalResult):
        return {
            "backend": result.backend_name,
            "dataset_path": result.dataset_path,
            "consistent_rate": result.consistent_rate,
            "inconsistent_rate": result.inconsistent_rate,
            "n_cases": len(result.case_results),
            "cases": [
                {
                    "case_id": c.case.case_id,
                    "signal": str(c.signal),
                    "average_jaccard": c.average_jaccard,
                    "n_successful_runs": c.n_successful_runs,
                    "error": c.error,
                }
                for c in result.case_results
            ],
        }
    if isinstance(result, TemporalKGBoundaryEvalResult):
        return {
            "backend": result.backend_name,
            "double_count_rate": result.double_count_rate,
            "clean_rate": result.clean_rate,
            "not_applicable_rate": result.not_applicable_rate,
            "self_report_agreement_rate": result.self_report_agreement_rate,
            "n_cases": len(result.case_results),
            "cases": [
                {
                    "case_id": c.case.case_id,
                    "signal": str(c.signal),
                    "adapter_reported_signal": str(c.adapter_reported_signal)
                    if c.adapter_reported_signal is not None
                    else None,
                    "objects_at_boundary": c.objects_at_boundary,
                    "self_report_agrees": c.self_report_agrees,
                    "error": c.error,
                }
                for c in result.case_results
            ],
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
    # ", ".join, not ",".join -- click's --help text wrapper only breaks
    # lines on whitespace, so a bare comma-joined list with no spaces
    # wraps mid-word (e.g. "longmemeval,locom" / "o,contradiction,...").
    # The parser itself (_resolve_eval_names) already .strip()s each
    # token, so accepting "a, b" on the command line still works too --
    # this only changes what gets displayed, not what's parsed.
    help=f"Comma-separated eval list ({', '.join(ALL_EVALS)}), or 'all'.",
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
@click.option(
    "--locomo-dataset-path",
    "locomo_dataset_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to a real locomo10.json (download it yourself from "
        "https://github.com/snap-research/locomo -- memtrust does not bundle or "
        "auto-fetch it) to run the locomo eval against, instead of the bundled "
        "synthetic tests/fixtures/locomo_sample.json. See docs/methodology.md's "
        '"LoCoMo" section for the schema and download link. Ignored if --eval does '
        "not include locomo."
    ),
)
@click.option(
    "--scale-stress-n-records",
    "scale_stress_n_records",
    default=DEFAULT_N_RECORDS,
    show_default=True,
    type=int,
    help=(
        "How many synthetic records the scale_stress eval stores and re-queries. "
        "Kept small by default so `memtrust run` stays fast in CI -- pass a much larger "
        "value (e.g. 10000) to actually reach the corpus size volcengine/OpenViking#2850 "
        "and getzep/graphiti#1275 manifest at, against a real configured backend. "
        "See evals/scale_stress.py and docs/methodology.md."
    ),
)
@click.option(
    "--sign",
    "sign_key_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to an Ed25519 private key PEM file (see `memtrust keygen`). When given, "
        "writes a signed receipt alongside the normal JSON report -- "
        "<output>.receipt.json -- proving the report was produced by the holder of "
        "this key and has not been altered since. Omit for plain, unsigned JSON "
        "output (the default, unchanged from before this flag existed)."
    ),
)
def run(
    backends: str,
    eval_arg: str,
    output_path: Path | None,
    locomo_exclude_ids_path: Path | None,
    locomo_dataset_path: Path | None,
    scale_stress_n_records: int,
    sign_key_path: Path | None,
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
            locomo_run_kwargs: dict[str, Any] = {"exclude_question_ids": locomo_exclude_ids}
            if locomo_dataset_path is not None:
                locomo_run_kwargs["dataset_path"] = locomo_dataset_path
            locomo_result = run_locomo(adapter, judge, **locomo_run_kwargs)
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
                ndr = rss_result.nested_content_unindexed_rate
                if dr is not None:
                    ndr_str = f"{ndr:.1%}" if ndr is not None else "N/A"
                    console.print(
                        f"    user-file deletion rate: {dr:.1%}"
                        f"  nested-content-unindexed rate: {ndr_str}"
                    )
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

        if "scale_stress" in eval_names:
            console.print(
                f"  Running Scale/Volume Stress ({scale_stress_n_records} records) "
                f"against {backend_name}..."
            )
            scale_result = run_scale_stress_eval(adapter, n_records=scale_stress_n_records)
            backend_report["evals"]["scale_stress"] = _serialize_eval_result(scale_result)
            console.print(f"    signal: {scale_result.signal}")
            console.print(
                f"    records_stored: {scale_result.records_stored}/"
                f"{scale_result.n_records_requested}"
                f"  records_recoverable: {scale_result.records_recoverable}/"
                f"{scale_result.records_checked}"
            )
            if scale_result.recall_degradation_pct is not None:
                console.print(
                    f"    recall_degradation: {scale_result.recall_degradation_pct:.1f}pp"
                )
            if scale_result.anchor_lost_at_n is not None:
                console.print(
                    f"    [yellow]anchor record lost at n={scale_result.anchor_lost_at_n}"
                    "[/yellow] (earliest-stored content became unrecoverable as volume grew)"
                )

        if "embedding_drift" in eval_names:
            console.print(f"  Running Embedding-Drift/Consistency against {backend_name}...")
            drift_result = run_embedding_drift_eval(adapter)
            backend_report["evals"]["embedding_drift"] = _serialize_eval_result(drift_result)
            dr = drift_result.drift_rate
            if dr is not None:
                console.print(
                    f"    drift-rate: {dr:.1%}  clean-rate: {drift_result.clean_rate:.1%}"
                )
            else:
                console.print("    N/A (no scoreable records)")

        if "crash_recovery" in eval_names:
            console.print(f"  Running Crash-Recovery against {backend_name}...")
            crash_result = run_crash_recovery_eval(adapter)
            backend_report["evals"]["crash_recovery"] = _serialize_eval_result(crash_result)
            if crash_result.skipped:
                console.print(f"    SKIPPED: {crash_result.skip_reason}")
            else:
                ilds = crash_result.index_lost_data_survived_rate
                rec = crash_result.recovered_rate
                if ilds is not None:
                    console.print(f"    index-lost-data-survived: {ilds:.1%}  recovered: {rec:.1%}")
                else:
                    console.print("    N/A (no scoreable cases)")

        if "extraction_quality" in eval_names:
            console.print(f"  Running Extraction-Quality against {backend_name}...")
            extraction_result = run_extraction_quality_eval(adapter)
            backend_report["evals"]["extraction_quality"] = _serialize_eval_result(
                extraction_result
            )
            jr = extraction_result.junk_retained_rate
            vl = extraction_result.valid_lost_rate
            fld = extraction_result.feedback_loop_duplicate_rate
            if jr is not None:
                vl_str = f"{vl:.1%}" if vl is not None else "N/A"
                console.print(f"    junk-retained: {jr:.1%}  valid-lost: {vl_str}")
            else:
                console.print("    junk-retained: N/A (no scoreable cases)")
            if fld is not None:
                console.print(f"    feedback-loop-duplicate: {fld:.1%}")
            else:
                console.print("    feedback-loop-duplicate: N/A (no scoreable feedback-loop cases)")

        if "migration_rollback" in eval_names:
            console.print(f"  Running Migration-Rollback against {backend_name}...")
            migration_result = run_migration_rollback_eval(adapter)
            backend_report["evals"]["migration_rollback"] = _serialize_eval_result(migration_result)
            if migration_result.skipped:
                console.print(f"    SKIPPED: {migration_result.skip_reason}")
            else:
                rr = migration_result.restored_rate
                dl = migration_result.data_lost_rate
                if rr is not None:
                    console.print(f"    restored: {rr:.1%}  data-lost: {dl:.1%}")
                else:
                    console.print("    N/A (no scoreable cases)")

        if "filter_injection" in eval_names:
            console.print(f"  Running Filter-Injection against {backend_name}...")
            filter_injection_result = run_filter_injection_eval(adapter)
            backend_report["evals"]["filter_injection"] = _serialize_eval_result(
                filter_injection_result
            )
            if filter_injection_result.skipped:
                console.print(f"    SKIPPED: {filter_injection_result.skip_reason}")
            else:
                isr = filter_injection_result.injection_succeeded_rate
                bfr = filter_injection_result.benign_false_positive_rate
                bfr_str = f"{bfr:.1%}" if bfr is not None else "N/A"
                if isr is not None:
                    console.print(
                        f"    injection-succeeded: {isr:.1%}  benign-false-positive: {bfr_str}"
                    )
                else:
                    console.print("    N/A (no scoreable cases)")

        if "lock_contention" in eval_names:
            console.print(f"  Running Lock-Contention/Hang-Detection against {backend_name}...")
            lock_contention_result = run_lock_contention_eval(adapter)
            backend_report["evals"]["lock_contention"] = _serialize_eval_result(
                lock_contention_result
            )
            if lock_contention_result.skipped:
                console.print(f"    SKIPPED: {lock_contention_result.skip_reason}")
            else:
                console.print(
                    f"    signal: {lock_contention_result.signal}"
                    f"  stalled: {lock_contention_result.stalled_count}/"
                    f"{lock_contention_result.n_concurrent}"
                )

        if "stats_accuracy" in eval_names:
            console.print(f"  Running Stats/Dashboard-Accuracy against {backend_name}...")
            stats_result = run_stats_accuracy_eval(adapter)
            backend_report["evals"]["stats_accuracy"] = _serialize_eval_result(stats_result)
            if stats_result.skipped:
                console.print(f"    SKIPPED: {stats_result.skip_reason}")
            elif stats_result.verified_count is not None:
                console.print(
                    f"    signal: {stats_result.signal}"
                    f"  verified: {stats_result.verified_count}"
                    f"  reported: {stats_result.reported_count}"
                )
            else:
                console.print("    N/A (no scoreable records)")

        if "orphan_cleanup" in eval_names:
            console.print(f"  Running Orphan-Cleanup against {backend_name}...")
            orphan_result = run_orphan_cleanup_eval(adapter)
            backend_report["evals"]["orphan_cleanup"] = _serialize_eval_result(orphan_result)
            if orphan_result.skipped:
                console.print(f"    SKIPPED: {orphan_result.skip_reason}")
            else:
                ovr = orphan_result.orphaned_vector_entry_rate
                if ovr is not None:
                    console.print(f"    orphaned-vector-entry rate: {ovr:.1%}")
                else:
                    console.print("    N/A (no scoreable files)")

        if "result_consistency" in eval_names:
            console.print(f"  Running Result-Consistency against {backend_name}...")
            consistency_result = run_result_consistency_eval(adapter)
            backend_report["evals"]["result_consistency"] = _serialize_eval_result(
                consistency_result
            )
            ir = consistency_result.inconsistent_rate
            if ir is not None:
                console.print(
                    f"    inconsistent rate: {ir:.1%}"
                    f"  consistent rate: {consistency_result.consistent_rate:.1%}"
                )
            else:
                console.print("    N/A (no scoreable cases)")

        if "temporal_kg_boundary" in eval_names:
            console.print(f"  Running Temporal-KG Boundary against {backend_name}...")
            if isinstance(adapter, MemPalaceAdapter):
                tkgb_result = run_temporal_kg_boundary_eval(adapter)
                backend_report["evals"]["temporal_kg_boundary"] = _serialize_eval_result(
                    tkgb_result
                )
                dcr = tkgb_result.double_count_rate
                if dcr is not None:
                    console.print(
                        f"    double-count: {dcr:.1%}  clean: {tkgb_result.clean_rate:.1%}"
                        f"  self-report-agreement: {tkgb_result.self_report_agreement_rate:.1%}"
                    )
                else:
                    console.print("    N/A (no scoreable cases)")
            else:
                # Only MemPalaceAdapter wires the kg_add/kg_invalidate/kg_query
                # primitives this eval calls -- see temporal_kg_boundary.py's
                # module docstring. Reported the same way resource_sync_safety
                # reports its own backend-specific skip, above.
                console.print(
                    f"    N/A (only applies to the mempalace backend, not {backend_name})"
                )
                backend_report["evals"]["temporal_kg_boundary"] = {
                    "status": "not_applicable",
                    "reason": f"temporal_kg_boundary only applies to the mempalace backend, "
                    f"not {backend_name}",
                }

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

    if sign_key_path is not None:
        try:
            receipt = sign_report_with_keyfile(report, sign_key_path)
        except ReceiptError as exc:
            console.print(f"[red]Could not sign report: {exc}[/red]")
            sys.exit(1)
        receipt_path = receipt_path_for(out_path)
        receipt_path.write_text(json.dumps(receipt, indent=2))
        console.print(f"Signed receipt: {receipt_path}  (public key: {receipt['public_key']})")


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
    table.add_column("Resource-Sync (user-file deletion / nested-content-unindexed)")
    table.add_column("Compression fidelity by mode")
    table.add_column("Ranking Quality (missing-ordering-key rate)")
    table.add_column("Scale/Volume Stress (signal, recall degradation)")
    table.add_column("Embedding Drift (drift rate)")
    table.add_column("Crash-Recovery (index-lost-data-survived rate)")
    table.add_column("Extraction Quality (junk-retained / valid-lost / feedback-loop-dup)")
    table.add_column("Migration-Rollback (restored / data-lost)")
    table.add_column("Filter Injection (injection-succeeded / benign-false-positive)")
    table.add_column("Lock Contention (signal, stalled/n)")
    table.add_column("Stats Accuracy (signal, verified/reported)")
    table.add_column("Orphan-Cleanup (orphaned-vector-entry rate)")
    table.add_column("Result-Consistency (inconsistent rate)")

    for backend_name, backend_data in data.get("results", {}).items():
        if backend_data.get("status") == "skipped":
            table.add_row(
                backend_name,
                "SKIPPED",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            )
            continue

        evals = backend_data.get("evals", {})
        lme = evals.get("longmemeval", {})
        locomo = evals.get("locomo", {})
        contra = evals.get("contradiction", {})
        rss = evals.get("resource_sync_safety", {})
        compression = evals.get("compression", {})
        ranking = evals.get("ranking_quality", {})
        scale_stress = evals.get("scale_stress", {})
        drift = evals.get("embedding_drift", {})
        crash_recovery = evals.get("crash_recovery", {})
        extraction = evals.get("extraction_quality", {})
        migration_rollback = evals.get("migration_rollback", {})
        filter_injection = evals.get("filter_injection", {})
        lock_contention = evals.get("lock_contention", {})
        stats_accuracy = evals.get("stats_accuracy", {})
        orphan_cleanup = evals.get("orphan_cleanup", {})
        result_consistency = evals.get("result_consistency", {})

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
            rss_str = (
                f"{_fmt_pct(rss.get('user_file_deletion_rate'))} / "
                f"{_fmt_pct(rss.get('nested_content_unindexed_rate'))}"
            )
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
        if scale_stress:
            degradation = scale_stress.get("recall_degradation_pct")
            degradation_str = f"{degradation:.1f}pp" if degradation is not None else "N/A"
            scale_stress_str = f"{scale_stress.get('signal', 'unknown')} ({degradation_str})"
        else:
            scale_stress_str = "-"
        drift_str = _fmt_pct(drift.get("drift_rate")) if drift else "-"
        if crash_recovery.get("skipped"):
            crash_recovery_str = "SKIPPED (unsupported)"
        elif crash_recovery:
            crash_recovery_str = _fmt_pct(crash_recovery.get("index_lost_data_survived_rate"))
        else:
            crash_recovery_str = "-"
        extraction_str = (
            f"{_fmt_pct(extraction.get('junk_retained_rate'))} / "
            f"{_fmt_pct(extraction.get('valid_lost_rate'))} / "
            f"{_fmt_pct(extraction.get('feedback_loop_duplicate_rate'))}"
            if extraction
            else "-"
        )
        if migration_rollback.get("skipped"):
            migration_rollback_str = "SKIPPED (unsupported)"
        elif migration_rollback:
            migration_rollback_str = (
                f"{_fmt_pct(migration_rollback.get('restored_rate'))} / "
                f"{_fmt_pct(migration_rollback.get('data_lost_rate'))}"
            )
        else:
            migration_rollback_str = "-"
        if filter_injection.get("skipped"):
            filter_injection_str = "SKIPPED (unsupported)"
        elif filter_injection:
            filter_injection_str = (
                f"{_fmt_pct(filter_injection.get('injection_succeeded_rate'))} / "
                f"{_fmt_pct(filter_injection.get('benign_false_positive_rate'))}"
            )
        else:
            filter_injection_str = "-"
        if lock_contention.get("skipped"):
            lock_contention_str = "SKIPPED (unsupported)"
        elif lock_contention:
            lock_contention_str = (
                f"{lock_contention.get('signal', 'unknown')} "
                f"({lock_contention.get('stalled_count', 0)}/"
                f"{lock_contention.get('n_concurrent', 0)})"
            )
        else:
            lock_contention_str = "-"
        if stats_accuracy.get("skipped"):
            stats_accuracy_str = "SKIPPED (unsupported)"
        elif stats_accuracy and stats_accuracy.get("verified_count") is not None:
            stats_accuracy_str = (
                f"{stats_accuracy.get('signal', 'unknown')} "
                f"({stats_accuracy.get('verified_count')}/"
                f"{stats_accuracy.get('reported_count')})"
            )
        elif stats_accuracy:
            stats_accuracy_str = "N/A"
        else:
            stats_accuracy_str = "-"
        if orphan_cleanup.get("skipped"):
            orphan_cleanup_str = "SKIPPED (unsupported)"
        elif orphan_cleanup:
            orphan_cleanup_str = _fmt_pct(orphan_cleanup.get("orphaned_vector_entry_rate"))
        else:
            orphan_cleanup_str = "-"
        result_consistency_str = (
            _fmt_pct(result_consistency.get("inconsistent_rate")) if result_consistency else "-"
        )
        table.add_row(
            backend_name,
            "configured",
            _fmt_pct(lme.get("accuracy")) if lme else "-",
            locomo_str,
            contra_str,
            rss_str,
            compression_str or "-",
            ranking_str,
            scale_stress_str,
            drift_str,
            crash_recovery_str,
            extraction_str,
            migration_rollback_str,
            filter_injection_str,
            lock_contention_str,
            stats_accuracy_str,
            orphan_cleanup_str,
            result_consistency_str,
        )

    console.print(table)

    cost = data.get("cost", {})
    if cost:
        console.print(f"\nEstimated cost: ${cost.get('total_usd', 0):.4f}")


@main.command()
@click.option(
    "--private-key-out",
    "private_key_path",
    default=Path("memtrust-key.pem"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the new Ed25519 private key (PEM, unencrypted). Keep this secret.",
)
@click.option(
    "--public-key-out",
    "public_key_path",
    default=Path("memtrust-key.pub"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the matching public key (PEM). Safe to publish/share.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite the output files if they already exist.",
)
def keygen(private_key_path: Path, public_key_path: Path, force: bool) -> None:
    """Generate a new Ed25519 keypair for signing `memtrust run` receipts.

    Equivalent to:

        openssl genpkey -algorithm ed25519 -out <private-key-out>
        openssl pkey -in <private-key-out> -pubout -out <public-key-out>

    Publish the public key file (or its contents via MEMTRUST_RECEIPT_PUBLIC_KEY)
    so others can run `memtrust verify` against your signed receipts. Never
    share the private key.
    """
    try:
        write_keypair(private_key_path, public_key_path, overwrite=force)
    except ReceiptError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    console.print(f"[green]Wrote private key:[/green] {private_key_path} (keep secret)")
    console.print(f"[green]Wrote public key:[/green]  {public_key_path} (safe to publish)")
    console.print(
        "\nSign a run with:   memtrust run --sign "
        f"{private_key_path} ...\n"
        "Verify a receipt with:   memtrust verify <receipt.json> --public-key "
        f"{public_key_path}"
    )


@main.command()
@click.argument("receipt_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--public-key",
    "public_key_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        f"Path to the trusted Ed25519 public key (PEM). Falls back to the "
        f"{PUBLIC_KEY_ENV_VAR} env var if omitted -- one of the two is required. "
        "Never taken from the receipt file itself: a receipt cannot vouch for its own key."
    ),
)
def verify(receipt_path: Path, public_key_path: Path | None) -> None:
    """Verify a signed receipt produced by `memtrust run --sign`.

    Proves exactly two things when valid: the receipt's payload has not
    been altered since it was signed, and it was signed by the holder of
    the supplied public key. It does NOT prove the benchmark numbers
    inside the payload are accurate -- see docs/methodology.md.

    Exits 0 and prints "valid: True" on success; exits 1 on any failure
    (bad signature, tampered payload, missing/wrong public key, malformed
    receipt).
    """
    try:
        result = verify_receipt_file(receipt_path, public_key_path=public_key_path)
    except ReceiptError as exc:
        console.print(f"[red]Could not verify: {exc}[/red]")
        sys.exit(1)

    color = "green" if result.valid else "red"
    console.print(f"[{color}]valid: {result.valid}[/{color}]")
    console.print(result.reason)
    if result.embedded_key_matches_trusted_key is False:
        console.print(
            "[yellow]Note: the public key embedded in the receipt does not match "
            "the trusted public key you supplied.[/yellow]"
        )
    if not result.valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
