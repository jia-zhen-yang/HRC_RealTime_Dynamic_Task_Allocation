#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze human/robot idle time ratios from existing comparison_gantt_data.txt files.

This script is independent from the no-rescheduling runner.
It DOES NOT rerun simulations. It only reads existing outputs.

Expected input structure:
    batch_real_nores_outputs/
      case_1/
        comparison_gantt_data.txt
      case_2/
        comparison_gantt_data.txt

Usage:
    python idle_time_analyzer_individual_robotwait.py --input-dir batch_real_nores_outputs

If this script is placed directly inside batch_real_nores_outputs:
    python idle_time_analyzer_individual_robotwait.py

Output:
    idle_time_ratio_summary.csv

Definitions
-----------
For each case, the script compares:
    Final no rescheduling
    Final dynamic timeline

Human idle time:
    Idle time within the full schedule horizon [0, makespan]:
      - gap from time 0 to the first human operation
      - gaps between consecutive human operations
      - gap from the last human operation to the makespan

Robot idle time:
    Robot idle time within the full schedule horizon [0, makespan]:
      - gap from time 0 to the first robot operation
      - gaps between consecutive robot operations
      - gap from the last robot operation to the makespan
      - estimated robot waiting time inside robot Gantt bars

Robot waiting time is estimated separately for each schedule as:
    robot_wait_idle_nores
    = sum(max(0, nores_robot_duration_i - offline_robot_duration_i))

    robot_wait_idle_dynamic
    = sum(max(0, dynamic_robot_duration_i - offline_robot_duration_i))

Robot total idle:
    robot_idle_nores
    = robot_external_gap_idle_nores + robot_wait_idle_nores

    robot_idle_dynamic
    = robot_external_gap_idle_dynamic + robot_wait_idle_dynamic

Output note:
    The robot_idle_* columns represent complete robot idle time:
        robot external full-span gap idle + estimated robot waiting time.

Ratio:
    idle_time_of_schedule / makespan_of_the_same_schedule

Indicator:
    *_nores_idle_ratio_higher = 1 if nores idle ratio is higher than dynamic else 0

Ratios are rounded to 2 decimal places.
Idle times are rounded to 2 decimal places.

Important:
    The *_ratio_diff_nores_minus_dynamic and *_nores_idle_ratio_higher
    columns are computed from the rounded ratios shown in the CSV.
    Therefore, if both displayed ratios are 0.25 and 0.25, the displayed
    diff is 0.00 and the higher flag is 0.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROW_RE = re.compile(
    r"^\s*(?P<task_id>\d+)\s+"
    r"(?P<agent>human|robot)\s+"
    r"(?P<start>-?\d+(?:\.\d+)?)\s+"
    r"(?P<finish>-?\d+(?:\.\d+)?)\s+"
    r"(?P<duration>-?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)

TITLE_RE = re.compile(r"^\s*title\s*:\s*(?P<title>.+?)\s*$", re.IGNORECASE)
MAKESPAN_RE = re.compile(r"makespan\s*:\s*(?P<makespan>-?\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class GanttRow:
    task_id: int
    agent: str
    start_time: float
    finish_time: float
    duration: float


@dataclass
class GanttBlock:
    title: str
    rows: List[GanttRow]
    makespan: float


def parse_comparison_gantt_data(path: Path) -> Dict[str, GanttBlock]:
    """Parse comparison_gantt_data.txt into blocks keyed by normalized title."""
    blocks: List[GanttBlock] = []
    current_title: Optional[str] = None
    current_rows: List[GanttRow] = []
    current_makespan: Optional[float] = None

    def flush_current() -> None:
        nonlocal current_title, current_rows, current_makespan
        if current_title is not None:
            blocks.append(
                GanttBlock(
                    title=current_title,
                    rows=current_rows,
                    makespan=float(current_makespan or 0.0),
                )
            )
        current_title = None
        current_rows = []
        current_makespan = None

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            title_match = TITLE_RE.match(line)
            if title_match:
                flush_current()
                current_title = title_match.group("title").strip()
                continue

            row_match = ROW_RE.match(line)
            if row_match and current_title is not None:
                current_rows.append(
                    GanttRow(
                        task_id=int(row_match.group("task_id")),
                        agent=row_match.group("agent").lower(),
                        start_time=float(row_match.group("start")),
                        finish_time=float(row_match.group("finish")),
                        duration=float(row_match.group("duration")),
                    )
                )
                continue

            makespan_match = MAKESPAN_RE.search(line)
            if makespan_match and current_title is not None:
                current_makespan = float(makespan_match.group("makespan"))

    flush_current()

    keyed: Dict[str, GanttBlock] = {}
    for block in blocks:
        keyed[normalize_title(block.title)] = block
    return keyed


def normalize_title(title: str) -> str:
    t = title.strip().lower()
    if "offline" in t:
        return "offline"
    if "no rescheduling" in t or "no-rescheduling" in t or "nores" in t:
        return "nores"
    if "final dynamic" in t or "dynamic timeline" in t:
        return "dynamic"
    return t.replace(" ", "_")


def rows_by_agent(rows: List[GanttRow], agent: str) -> List[GanttRow]:
    return sorted(
        [r for r in rows if r.agent.lower() == agent.lower()],
        key=lambda r: (r.start_time, r.finish_time, r.task_id),
    )


def sum_full_span_gap_idle(rows: List[GanttRow], agent: str, makespan: float) -> float:
    """
    Sum external idle gaps of one agent over the full schedule horizon [0, makespan].

    This includes:
    - 0 to first operation
    - gaps between consecutive operations
    - last operation to makespan
    """
    agent_rows = rows_by_agent(rows, agent)

    if not agent_rows:
        return max(0.0, makespan)

    idle = 0.0

    # Initial idle: schedule starts at 0, but this agent may start later.
    idle += max(0.0, agent_rows[0].start_time)

    # Between-task idle.
    for prev, cur in zip(agent_rows, agent_rows[1:]):
        idle += max(0.0, cur.start_time - prev.finish_time)

    # Final idle: agent finishes before the whole schedule makespan.
    idle += max(0.0, makespan - agent_rows[-1].finish_time)

    return idle


def reference_robot_duration_lookup(reference_block: GanttBlock) -> Dict[int, float]:
    """
    Return robot task_id -> reference execution duration.

    Reference source is the offline optimal schedule.
    It is used as the robot's pure execution time.
    Any robot Gantt duration longer than this reference duration is counted as robot waiting time.
    """
    return {
        row.task_id: row.duration
        for row in reference_block.rows
        if row.agent.lower() == "robot"
    }


def robot_wait_idle(
    target_block: GanttBlock,
    reference_robot_duration_by_task: Dict[int, float],
) -> float:
    """
    Estimate robot waiting time inside robot bars.

    robot_wait_idle = sum(max(0, final_robot_duration - offline_robot_duration))
    """
    wait = 0.0
    for row in target_block.rows:
        if row.agent.lower() != "robot":
            continue
        ref_duration = reference_robot_duration_by_task.get(row.task_id)
        if ref_duration is None:
            continue
        wait += max(0.0, row.duration - ref_duration)
    return wait


def safe_ratio(value: float, denominator: float) -> float:
    return value / denominator if denominator and denominator > 0 else 0.0


def compute_case_idle(case_id: str, gantt_path: Path) -> Dict[str, object]:
    blocks = parse_comparison_gantt_data(gantt_path)

    missing = [name for name in ["offline", "nores", "dynamic"] if name not in blocks]
    if missing:
        raise ValueError(f"{gantt_path} is missing block(s): {missing}")

    offline = blocks["offline"]
    nores = blocks["nores"]
    dynamic = blocks["dynamic"]

    robot_ref = reference_robot_duration_lookup(offline)

    # Human idle over [0, makespan].
    human_idle_nores = sum_full_span_gap_idle(nores.rows, "human", nores.makespan)
    human_idle_dynamic = sum_full_span_gap_idle(dynamic.rows, "human", dynamic.makespan)

    human_nores_idle_ratio_raw = safe_ratio(human_idle_nores, nores.makespan)
    human_dynamic_idle_ratio_raw = safe_ratio(human_idle_dynamic, dynamic.makespan)

    # Keep the displayed ratio, displayed difference, and indicator logically consistent.
    # Example: if both displayed ratios are 0.25, the displayed diff is 0.00
    # and human_nores_idle_ratio_higher must be 0.
    human_nores_idle_ratio = round(human_nores_idle_ratio_raw, 2)
    human_dynamic_idle_ratio = round(human_dynamic_idle_ratio_raw, 2)
    human_idle_ratio_diff = round(human_nores_idle_ratio - human_dynamic_idle_ratio, 2)
    human_nores_idle_ratio_higher = int(human_idle_ratio_diff > 0)

    # Robot idle over [0, makespan] = external full-span gaps + estimated waiting inside robot bars.
    robot_external_gap_idle_nores = sum_full_span_gap_idle(nores.rows, "robot", nores.makespan)
    robot_external_gap_idle_dynamic = sum_full_span_gap_idle(dynamic.rows, "robot", dynamic.makespan)

    robot_wait_idle_nores = robot_wait_idle(nores, robot_ref)
    robot_wait_idle_dynamic = robot_wait_idle(dynamic, robot_ref)

    robot_idle_nores = robot_external_gap_idle_nores + robot_wait_idle_nores
    robot_idle_dynamic = robot_external_gap_idle_dynamic + robot_wait_idle_dynamic

    robot_nores_idle_ratio_raw = safe_ratio(robot_idle_nores, nores.makespan)
    robot_dynamic_idle_ratio_raw = safe_ratio(robot_idle_dynamic, dynamic.makespan)

    # Keep the displayed ratio, displayed difference, and indicator logically consistent.
    # Example: if both displayed ratios are 0.25, the displayed diff is 0.00
    # and robot_nores_idle_ratio_higher must be 0.
    robot_nores_idle_ratio = round(robot_nores_idle_ratio_raw, 2)
    robot_dynamic_idle_ratio = round(robot_dynamic_idle_ratio_raw, 2)
    robot_idle_ratio_diff = round(robot_nores_idle_ratio - robot_dynamic_idle_ratio, 2)
    robot_nores_idle_ratio_higher = int(robot_idle_ratio_diff > 0)

    return {
        "case_id": case_id,

        "human_idle_nores": round(human_idle_nores, 2),
        "human_idle_nores_ratio": human_nores_idle_ratio,
        "human_idle_dynamic": round(human_idle_dynamic, 2),
        "human_idle_dynamic_ratio": human_dynamic_idle_ratio,
        "human_idle_ratio_diff_nores_minus_dynamic": human_idle_ratio_diff,
        "human_nores_idle_ratio_higher": human_nores_idle_ratio_higher,

        # Robot values include both external full-span gaps and estimated waiting inside robot bars.
        "robot_idle_nores": round(robot_idle_nores, 2),
        "robot_idle_nores_ratio": robot_nores_idle_ratio,
        "robot_idle_dynamic": round(robot_idle_dynamic, 2),
        "robot_idle_dynamic_ratio": robot_dynamic_idle_ratio,
        "robot_idle_ratio_diff_nores_minus_dynamic": robot_idle_ratio_diff,
        "robot_nores_idle_ratio_higher": robot_nores_idle_ratio_higher,
    }


def discover_case_files(input_dir: Path) -> List[Path]:
    """Find comparison_gantt_data.txt files under input_dir."""
    if input_dir.name == "comparison_gantt_data.txt" and input_dir.is_file():
        return [input_dir]
    return sorted(input_dir.rglob("comparison_gantt_data.txt"))


def infer_case_id(gantt_path: Path, input_dir: Path) -> str:
    """
    Infer case id from the parent folder name.

    Example:
      batch_outputs/20260617_162610/comparison_gantt_data.txt
      -> 20260617_162610
    """
    if gantt_path.parent == input_dir:
        return gantt_path.stem
    return gantt_path.parent.name


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "case_id",

        "human_idle_nores",
        "human_idle_nores_ratio",
        "human_idle_dynamic",
        "human_idle_dynamic_ratio",
        "human_idle_ratio_diff_nores_minus_dynamic",
        "human_nores_idle_ratio_higher",

        "robot_idle_nores",
        "robot_idle_nores_ratio",
        "robot_idle_dynamic",
        "robot_idle_dynamic_ratio",
        "robot_idle_ratio_diff_nores_minus_dynamic",
        "robot_nores_idle_ratio_higher",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_error_csv(path: Path, errors: List[Tuple[Path, str]]) -> None:
    error_path = path.with_name(path.stem + "_errors.csv")
    with error_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "error"])
        writer.writeheader()
        for p, msg in errors:
            writer.writerow({"path": str(p), "error": msg})
    print(f"[WARN] Error report: {error_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze full-span human and robot idle time ratios from comparison_gantt_data.txt files."
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Folder containing case subfolders with comparison_gantt_data.txt files. Default: current folder.",
    )
    parser.add_argument(
        "--output",
        default="idle_time_ratio_summary.csv",
        help="Output CSV filename. Default: idle_time_ratio_summary.csv",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = input_dir / output_path

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    gantt_files = discover_case_files(input_dir)
    if not gantt_files:
        raise FileNotFoundError(
            f"No comparison_gantt_data.txt files found under: {input_dir}"
        )

    rows: List[Dict[str, object]] = []
    errors: List[Tuple[Path, str]] = []

    for gantt_path in gantt_files:
        case_id = infer_case_id(gantt_path, input_dir)
        try:
            rows.append(compute_case_idle(case_id, gantt_path))
        except Exception as exc:
            errors.append((gantt_path, str(exc)))

    write_csv(output_path, rows)

    print(f"[OK] Wrote idle-time ratio summary: {output_path}")
    print(f"[OK] Cases analyzed: {len(rows)}")

    if errors:
        print(f"[WARN] Cases with errors: {len(errors)}")
        write_error_csv(output_path, errors)


if __name__ == "__main__":
    main()
