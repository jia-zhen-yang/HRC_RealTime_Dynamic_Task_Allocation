"""
Batch runner for real-data no-rescheduling counterfactual simulation.

This script scans a folder for paired files:

    dynamic_events_<case_id>.csv
    gantt_data_<case_id>.txt

For each pair, it calls:

    real_data_nores_simulation_actualbase.py

and writes:
    batch_summary.csv
    batch_event_summary.csv
    batch_dynamic_benefit_overall.csv
    batch_dynamic_benefit_by_case.csv
    batch_dynamic_benefit_by_failure_type.csv
    batch_dynamic_benefit_per_event.csv
    batch_outputs.zip

Example:
    python batch_real_nores_runner.py ^
      --input-dir data ^
      --output-dir batch_real_nores_outputs ^
      --simulator real_data_nores_simulation_actualbase.py ^
      --failure-module failure_simulation.py ^
      --seed 43
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EVENT_RE = re.compile(r"^dynamic_events_(.+)\.csv$", re.IGNORECASE)
GANTT_RE = re.compile(r"^gantt_data_(.+)\.txt$", re.IGNORECASE)


def safe_name(text: str) -> str:
    """Make a string safe for folder/file names."""
    text = text.strip()
    text = re.sub(r"[^\w.\-()]+", "_", text)
    return text or "case"


def load_simulator(simulator_path: Path):
    """Import the single-case simulator as a Python module."""
    simulator_path = simulator_path.resolve()
    spec = importlib.util.spec_from_file_location("real_nores_simulator_batch", simulator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import simulator from {simulator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "run_real_nores_counterfactual"):
        raise RuntimeError(
            f"{simulator_path} does not define run_real_nores_counterfactual(). "
            "Please use the latest real-data no-rescheduling simulator."
        )
    return module


def find_pairs(input_dir: Path) -> Tuple[List[Tuple[str, Path, Path]], List[Path], List[Path]]:
    """Return matched event/gantt pairs plus unpaired files."""
    event_files: Dict[str, Path] = {}
    gantt_files: Dict[str, Path] = {}

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue

        m_event = EVENT_RE.match(path.name)
        if m_event:
            event_files[m_event.group(1)] = path
            continue

        m_gantt = GANTT_RE.match(path.name)
        if m_gantt:
            gantt_files[m_gantt.group(1)] = path
            continue

    keys = sorted(set(event_files) & set(gantt_files))
    pairs = [(key, event_files[key], gantt_files[key]) for key in keys]

    unpaired_events = [event_files[k] for k in sorted(set(event_files) - set(gantt_files))]
    unpaired_gantts = [gantt_files[k] for k in sorted(set(gantt_files) - set(event_files))]
    return pairs, unpaired_events, unpaired_gantts


def count_csv_rows(csv_path: Path) -> int:
    """Count data rows in a CSV file."""
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return -1


def read_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """Read CSV rows. Return empty list if file does not exist."""
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def zip_folder(folder: Path, zip_path: Path) -> None:
    """Zip the full output folder."""
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path != zip_path:
                z.write(path, arcname=path.relative_to(folder))


def coerce_result_value(result: Dict[str, Any], key: str, default: Any = "") -> Any:
    value = result.get(key, default)
    if value is None:
        return default
    return value




def to_float(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def compute_dynamic_benefit_stats(
    *,
    event_rows: List[Dict[str, Any]],
    output_dir: Path,
    tolerance: float = 0.1,
) -> Dict[str, Path]:
    """
    Compute how often dynamic rescheduling has a smaller makespan than no rescheduling.

    dynamic better means:
        dynamic_makespan_after_event < nores_makespan

    Equivalently:
        nores_minus_dynamic > 0
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_rows: List[Dict[str, Any]] = []
    for row in event_rows:
        nores = to_float(row.get("nores_makespan"))
        dynamic = to_float(row.get("dynamic_makespan_after_event"))

        if nores is None or dynamic is None:
            continue

        diff = nores - dynamic
        dynamic_better = diff > tolerance
        nores_better = diff < -tolerance
        tie = abs(diff) <= tolerance

        enriched = dict(row)
        enriched["nores_makespan"] = nores
        enriched["dynamic_makespan_after_event"] = dynamic
        enriched["nores_minus_dynamic"] = diff
        enriched["dynamic_better"] = int(dynamic_better)
        enriched["nores_better"] = int(nores_better)
        enriched["tie"] = int(tie)
        clean_rows.append(enriched)

    total = len(clean_rows)
    dynamic_better_count = sum(int(r["dynamic_better"]) for r in clean_rows)
    nores_better_count = sum(int(r["nores_better"]) for r in clean_rows)
    tie_count = sum(int(r["tie"]) for r in clean_rows)

    overall_rows = [{
        "failure_count": total,
        "dynamic_better_count": dynamic_better_count,
        "dynamic_better_rate": (dynamic_better_count / total) if total else "",
        "dynamic_better_percent": (dynamic_better_count / total * 100.0) if total else "",
        "nores_better_count": nores_better_count,
        "nores_better_rate": (nores_better_count / total) if total else "",
        "nores_better_percent": (nores_better_count / total * 100.0) if total else "",
        "tie_count": tie_count,
        "tie_rate": (tie_count / total) if total else "",
        "avg_nores_minus_dynamic": (
            sum(float(r["nores_minus_dynamic"]) for r in clean_rows) / total
            if total else ""
        ),
    }]

    overall_fields = [
        "failure_count",
        "dynamic_better_count",
        "dynamic_better_rate",
        "dynamic_better_percent",
        "nores_better_count",
        "nores_better_rate",
        "nores_better_percent",
        "tie_count",
        "tie_rate",
        "avg_nores_minus_dynamic",
    ]

    overall_csv = output_dir / "batch_dynamic_benefit_overall.csv"
    write_csv(overall_csv, overall_rows, overall_fields)

    # Per-case summary
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for row in clean_rows:
        by_case.setdefault(str(row.get("case_id", "")), []).append(row)

    case_rows: List[Dict[str, Any]] = []
    for case_id, rows in sorted(by_case.items()):
        n = len(rows)
        dyn = sum(int(r["dynamic_better"]) for r in rows)
        nor = sum(int(r["nores_better"]) for r in rows)
        tie = sum(int(r["tie"]) for r in rows)
        diffs = [float(r["nores_minus_dynamic"]) for r in rows]
        case_rows.append({
            "case_id": case_id,
            "failure_count": n,
            "dynamic_better_count": dyn,
            "dynamic_better_rate": dyn / n if n else "",
            "dynamic_better_percent": dyn / n * 100.0 if n else "",
            "nores_better_count": nor,
            "nores_better_rate": nor / n if n else "",
            "tie_count": tie,
            "avg_nores_minus_dynamic": sum(diffs) / n if n else "",
            "min_nores_minus_dynamic": min(diffs) if diffs else "",
            "max_nores_minus_dynamic": max(diffs) if diffs else "",
        })

    case_fields = [
        "case_id",
        "failure_count",
        "dynamic_better_count",
        "dynamic_better_rate",
        "dynamic_better_percent",
        "nores_better_count",
        "nores_better_rate",
        "tie_count",
        "avg_nores_minus_dynamic",
        "min_nores_minus_dynamic",
        "max_nores_minus_dynamic",
    ]
    case_csv = output_dir / "batch_dynamic_benefit_by_case.csv"
    write_csv(case_csv, case_rows, case_fields)

    # Per-failure-type summary
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in clean_rows:
        by_type.setdefault(str(row.get("failure_type", "")), []).append(row)

    type_rows: List[Dict[str, Any]] = []
    for failure_type, rows in sorted(by_type.items()):
        n = len(rows)
        dyn = sum(int(r["dynamic_better"]) for r in rows)
        nor = sum(int(r["nores_better"]) for r in rows)
        tie = sum(int(r["tie"]) for r in rows)
        diffs = [float(r["nores_minus_dynamic"]) for r in rows]
        type_rows.append({
            "failure_type": failure_type,
            "failure_count": n,
            "dynamic_better_count": dyn,
            "dynamic_better_rate": dyn / n if n else "",
            "dynamic_better_percent": dyn / n * 100.0 if n else "",
            "nores_better_count": nor,
            "nores_better_rate": nor / n if n else "",
            "tie_count": tie,
            "avg_nores_minus_dynamic": sum(diffs) / n if n else "",
        })

    type_fields = [
        "failure_type",
        "failure_count",
        "dynamic_better_count",
        "dynamic_better_rate",
        "dynamic_better_percent",
        "nores_better_count",
        "nores_better_rate",
        "tie_count",
        "avg_nores_minus_dynamic",
    ]
    type_csv = output_dir / "batch_dynamic_benefit_by_failure_type.csv"
    write_csv(type_csv, type_rows, type_fields)

    # Per-event detail with boolean columns.
    detail_fields = [
        "case_id",
        "step_idx",
        "failure_type",
        "nores_makespan",
        "dynamic_makespan_after_event",
        "nores_minus_dynamic",
        "dynamic_better",
        "nores_better",
        "tie",
        "target_task",
        "t1",
        "t2",
        "earlier_task",
        "later_task",
        "order_required_task_actual_duration",
        "order_wrong_first_task_actual_duration",
        "order_later_standard_duration",
        "order_compensation_upper",
    ]
    detail_csv = output_dir / "batch_dynamic_benefit_per_event.csv"
    write_csv(detail_csv, clean_rows, detail_fields)

    return {
        "dynamic_benefit_overall_csv": overall_csv,
        "dynamic_benefit_by_case_csv": case_csv,
        "dynamic_benefit_by_failure_type_csv": type_csv,
        "dynamic_benefit_per_event_csv": detail_csv,
    }




def to_int_or_zero(value: Any) -> int:
    try:
        if value in ("", None):
            return 0
        return int(float(value))
    except Exception:
        return 0


def compute_final_timeline_stats(
    *,
    batch_rows: List[Dict[str, Any]],
    output_dir: Path,
    tolerance: float = 0.1,
    exact_epsilon: float = 1e-9,
) -> Dict[str, Path]:
    """
    Compare only the final timeline makespan of each case.

    Updated rule:
    - If num_events == 0:
        Treat the two methods as a tie when
            abs(final_nores_makespan - final_dynamic_makespan) <= tolerance
        Default tolerance is 0.1 second.
    - If num_events != 0:
        Do not apply the 0.1-second tie tolerance. Compare the actual difference:
            final_nores_makespan > final_dynamic_makespan  -> dynamic is better
            final_nores_makespan < final_dynamic_makespan  -> no-rescheduling is better
            nearly exactly equal within exact_epsilon        -> tie
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: List[Dict[str, Any]] = []
    for row in batch_rows:
        if row.get("status") != "ok":
            continue

        nores = to_float(row.get("final_nores_makespan"))
        dynamic = to_float(row.get("final_dynamic_makespan"))
        if nores is None or dynamic is None:
            continue

        num_events = to_int_or_zero(row.get("num_events"))
        diff = nores - dynamic

        if num_events == 0:
            tie = abs(diff) <= tolerance
            dynamic_better = (diff > tolerance) if not tie else False
            nores_better = (diff < -tolerance) if not tie else False
            comparison_rule = "no_failure_tolerance"
            applied_tolerance = tolerance
        else:
            tie = abs(diff) <= exact_epsilon
            dynamic_better = diff > exact_epsilon
            nores_better = diff < -exact_epsilon
            comparison_rule = "failure_case_exact_difference"
            applied_tolerance = 0.0

        detail_rows.append({
            "case_id": row.get("case_id", ""),
            "num_events": num_events,
            "offline_makespan": row.get("offline_makespan", ""),
            "final_nores_makespan": nores,
            "final_dynamic_makespan": dynamic,
            "final_nores_minus_dynamic": diff,
            "dynamic_final_better": int(dynamic_better),
            "nores_final_better": int(nores_better),
            "tie": int(tie),
            "comparison_rule": comparison_rule,
            "tie_tolerance_seconds": applied_tolerance,
            "final_comparison_png": row.get("final_comparison_png", ""),
            "output_dir": row.get("output_dir", ""),
        })

    total = len(detail_rows)
    dynamic_better_count = sum(int(r["dynamic_final_better"]) for r in detail_rows)
    nores_better_count = sum(int(r["nores_final_better"]) for r in detail_rows)
    tie_count = sum(int(r["tie"]) for r in detail_rows)
    no_failure_case_count = sum(1 for r in detail_rows if int(r["num_events"]) == 0)
    failure_case_count = total - no_failure_case_count

    overall_rows = [{
        "case_count": total,
        "failure_case_count": failure_case_count,
        "no_failure_case_count": no_failure_case_count,
        "dynamic_final_better_count": dynamic_better_count,
        "dynamic_final_better_rate": (dynamic_better_count / total) if total else "",
        "dynamic_final_better_percent": (dynamic_better_count / total * 100.0) if total else "",
        "nores_final_better_count": nores_better_count,
        "nores_final_better_rate": (nores_better_count / total) if total else "",
        "nores_final_better_percent": (nores_better_count / total * 100.0) if total else "",
        "tie_count": tie_count,
        "tie_rate": (tie_count / total) if total else "",
        "no_failure_tie_tolerance_seconds": tolerance,
        "failure_case_tolerance_seconds": 0.0,
        "avg_final_nores_minus_dynamic": (
            sum(float(r["final_nores_minus_dynamic"]) for r in detail_rows) / total
            if total else ""
        ),
    }]

    overall_fields = [
        "case_count",
        "failure_case_count",
        "no_failure_case_count",
        "dynamic_final_better_count",
        "dynamic_final_better_rate",
        "dynamic_final_better_percent",
        "nores_final_better_count",
        "nores_final_better_rate",
        "nores_final_better_percent",
        "tie_count",
        "tie_rate",
        "no_failure_tie_tolerance_seconds",
        "failure_case_tolerance_seconds",
        "avg_final_nores_minus_dynamic",
    ]
    overall_csv = output_dir / "batch_final_timeline_overall.csv"
    write_csv(overall_csv, overall_rows, overall_fields)

    detail_fields = [
        "case_id",
        "num_events",
        "offline_makespan",
        "final_nores_makespan",
        "final_dynamic_makespan",
        "final_nores_minus_dynamic",
        "dynamic_final_better",
        "nores_final_better",
        "tie",
        "comparison_rule",
        "tie_tolerance_seconds",
        "final_comparison_png",
        "output_dir",
    ]
    by_case_csv = output_dir / "batch_final_timeline_by_case.csv"
    write_csv(by_case_csv, detail_rows, detail_fields)

    return {
        "final_timeline_overall_csv": overall_csv,
        "final_timeline_by_case_csv": by_case_csv,
    }

def run_batch(
    *,
    input_dir: Path,
    output_dir: Path,
    simulator_path: Path,
    failure_module_path: Optional[Path],
    seed: int,
    order_policy: str,
    compensation_mode: str = "fixed",
    compensation_value: float = 3.39,
    dry_run: bool = False,
    final_timeline_tolerance: float = 0.1,
) -> Dict[str, Path]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    simulator_path = simulator_path.resolve()
    failure_module_path = failure_module_path.resolve() if failure_module_path else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not simulator_path.exists():
        raise FileNotFoundError(f"Simulator file does not exist: {simulator_path}")
    if failure_module_path and not failure_module_path.exists():
        raise FileNotFoundError(f"Failure module does not exist: {failure_module_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, unpaired_events, unpaired_gantts = find_pairs(input_dir)

    if dry_run:
        print("[DRY RUN] Matched pairs:")
        for key, event_path, gantt_path in pairs:
            print(f"  {key}: {event_path.name} + {gantt_path.name}")
        if unpaired_events:
            print("[DRY RUN] Unpaired event files:")
            for path in unpaired_events:
                print(f"  {path.name}")
        if unpaired_gantts:
            print("[DRY RUN] Unpaired gantt files:")
            for path in unpaired_gantts:
                print(f"  {path.name}")
        return {}

    sim = load_simulator(simulator_path)

    batch_rows: List[Dict[str, Any]] = []
    all_event_rows: List[Dict[str, Any]] = []

    for case_id, events_csv, gantt_txt in pairs:
        case_folder = output_dir / safe_name(case_id)
        case_folder.mkdir(parents=True, exist_ok=True)

        row: Dict[str, Any] = {
            "case_id": case_id,
            "events_csv": str(events_csv),
            "gantt_txt": str(gantt_txt),
            "output_dir": str(case_folder),
            "num_events": count_csv_rows(events_csv),
            "status": "ok",
            "error": "",
        }

        print(f"[RUN] {case_id}")

        try:
            result = sim.run_real_nores_counterfactual(
                events_csv=events_csv,
                gantt_txt=gantt_txt,
                output_dir=case_folder,
                failure_module_path=str(failure_module_path) if failure_module_path else None,
                seed=seed,
                order_policy=order_policy,
                compensation_mode=compensation_mode,
                compensation_value=compensation_value,
            )

            row.update(
                {
                    "offline_makespan": coerce_result_value(result, "offline_makespan"),
                    "initial_offline_best_cost": coerce_result_value(result, "initial_offline_best_cost"),
                    "initial_real_nores_baseline_makespan": coerce_result_value(
                        result, "initial_real_nores_baseline_makespan"
                    ),
                    "final_nores_makespan": coerce_result_value(result, "final_nores_makespan"),
                    "final_dynamic_makespan": coerce_result_value(result, "final_dynamic_makespan"),
                    "final_nores_minus_dynamic": coerce_result_value(result, "final_nores_minus_dynamic"),
                    "summary_csv": coerce_result_value(result, "summary_csv"),
                    "comparison_gantt_data": coerce_result_value(result, "gantt_txt"),
                    "event_delay_report": coerce_result_value(result, "event_delay_report"),
                    "final_comparison_png": coerce_result_value(result, "final_comparison_png"),
                }
            )

            # Combine per-failure rows into one batch-level event summary.
            summary_path = Path(str(row["summary_csv"]))
            for event_row in read_csv_rows(summary_path):
                event_row = {"case_id": case_id, **event_row}
                all_event_rows.append(event_row)

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            error_txt = case_folder / "error_traceback.txt"
            error_txt.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[FAILED] {case_id}: {row['error']}")

        batch_rows.append(row)

    # Include unpaired files in the batch report.
    for path in unpaired_events:
        batch_rows.append(
            {
                "case_id": "",
                "events_csv": str(path),
                "gantt_txt": "",
                "output_dir": "",
                "num_events": count_csv_rows(path),
                "status": "unpaired_event_file",
                "error": "No matching gantt_data_<case_id>.txt found.",
            }
        )
    for path in unpaired_gantts:
        batch_rows.append(
            {
                "case_id": "",
                "events_csv": "",
                "gantt_txt": str(path),
                "output_dir": "",
                "num_events": "",
                "status": "unpaired_gantt_file",
                "error": "No matching dynamic_events_<case_id>.csv found.",
            }
        )

    batch_summary_fields = [
        "case_id",
        "status",
        "error",
        "num_events",
        "offline_makespan",
        "initial_offline_best_cost",
        "initial_real_nores_baseline_makespan",
        "final_nores_makespan",
        "final_dynamic_makespan",
        "final_nores_minus_dynamic",
        "events_csv",
        "gantt_txt",
        "output_dir",
        "summary_csv",
        "comparison_gantt_data",
        "event_delay_report",
        "final_comparison_png",
    ]
    batch_summary_csv = output_dir / "batch_summary.csv"
    write_csv(batch_summary_csv, batch_rows, batch_summary_fields)

    event_summary_fields = [
        "case_id",
        "step_idx",
        "failure_type",
        "decision_time",
        "csv_delay_time",
        "actual_incremental_delay",
        "applied_nores_delay",
        "delay_source",
        "nores_makespan",
        "dynamic_makespan_after_event",
        "nores_minus_dynamic",
        "target_task",
        "t1",
        "t2",
        "earlier_task",
        "later_task",
        "order_base_duration",
        "order_required_task_actual_duration",
        "order_wrong_first_task_actual_duration",
        "order_later_standard_duration",
        "order_compensation_upper",
        "released_previous_delay",
        "observed_order_extra_delay",
    ]
    batch_event_summary_csv = output_dir / "batch_event_summary.csv"
    write_csv(batch_event_summary_csv, all_event_rows, event_summary_fields)

    final_timeline_outputs = compute_final_timeline_stats(
        batch_rows=batch_rows,
        output_dir=output_dir,
        tolerance=final_timeline_tolerance,
    )

    zip_path = output_dir / "batch_outputs.zip"
    zip_folder(output_dir, zip_path)

    outputs = {
        "batch_summary_csv": batch_summary_csv,
        "batch_event_summary_csv": batch_event_summary_csv,
        **final_timeline_outputs,
        "batch_zip": zip_path,
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for real-data no-rescheduling simulations with final-timeline makespan statistics and event-count-aware tie handling and unified fixed/random compensation with actual base for no-delay agent/order mistakes")
    parser.add_argument("--input-dir", required=True, help="Folder containing dynamic_events_*.csv and gantt_data_*.txt")
    parser.add_argument("--output-dir", default="batch_real_nores_outputs", help="Batch output folder")
    parser.add_argument(
        "--simulator",
        default="real_data_nores_simulation_actualbase.py",
        help="Path to the single-case simulator file",
    )
    parser.add_argument(
        "--failure-module",
        default="failure_simulation.py",
        help="Path to your original failure_simulation.py",
    )
    parser.add_argument("--seed", type=int, default=43, help="Random seed for random compensation mode")
    parser.add_argument(
        "--compensation-mode",
        choices=["fixed", "random"],
        default="fixed",
        help="Unified compensation mode for both agent and order mistakes; default is fixed",
    )
    parser.add_argument(
        "--compensation-value",
        "--order-compensation",
        dest="compensation_value",
        type=float,
        default=3.39,
        help="Fixed compensation time in seconds for agent/order mistakes when compensation-mode=fixed; default is 3.39",
    )
    parser.add_argument(
        "--order-policy",
        choices=["sampled_recovery", "observed_positive", "transfer_slot", "sampled"],
        default="sampled_recovery",
        help="Default sampled_recovery uses Uniform(0, 2 * actual_duration) order compensation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list matched pairs; do not run simulations")
    parser.add_argument(
        "--final-timeline-tolerance",
        type=float,
        default=0.1,
        help=(
            "Tolerance in seconds applied only to no-failure cases. "
            "If num_events == 0 and |final_nores_makespan - final_dynamic_makespan| <= tolerance, it is counted as a tie. "
            "Cases with failures are compared by exact final makespan difference. "
            "Default: 0.1"
        ),
    )
    args = parser.parse_args()

    outputs = run_batch(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        simulator_path=Path(args.simulator),
        failure_module_path=Path(args.failure_module) if args.failure_module else None,
        seed=args.seed,
        order_policy=args.order_policy,
        compensation_mode=args.compensation_mode,
        compensation_value=args.compensation_value,
        dry_run=args.dry_run,
        final_timeline_tolerance=args.final_timeline_tolerance,
    )

    if outputs:
        print("[BATCH DONE]")
        for key, value in outputs.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
