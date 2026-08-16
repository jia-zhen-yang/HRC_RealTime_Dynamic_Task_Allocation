"""
Replay the VNS search after every failure in one real experiment and export
convergence curves using the same plotting format as VNS_rescheduler.py.

This script is intended to be placed in the original scheduling project folder,
where the following project files are importable:
    read_schedule.py
    precedence_matrix.py
    VNS_dynamic_solver.py or VNS_rescheduler.py
    Standard Time Calculation/robot_task_time.csv

The supplied dynamic_events_*.csv and gantt_data_*.txt are used to rebuild the
state at each decision point. Each failure is replayed independently from the
EXPERIMENT'S previous optimized schedule, so stochastic differences in one
replay do not contaminate the next failure's starting state.

Outputs
-------
- convergence_failure_1.png ... convergence_failure_N.png
- convergence_all_failures.png
- convergence_history.csv
- convergence_summary.csv
- reconstructed_failure_states.csv

Important reproducibility note
------------------------------
The experiment did not export the random-number-generator state or the original
per-iteration history. Therefore this script performs a reproducible NEW VNS
run from the same recorded failure state. It cannot recover the exact historical
curve unless the original RNG state/seed was also saved.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import math
import os
import random
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Raw-data models and parsers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GanttRow:
    task_id: int
    agent: str
    start_time: float
    finish_time: float
    duration: float


@dataclass(frozen=True)
class GanttBlock:
    title: str
    rows: Tuple[GanttRow, ...]
    makespan: float

    @property
    def by_task(self) -> Dict[int, GanttRow]:
        return {row.task_id: row for row in self.rows}

    @property
    def task_ids(self) -> List[int]:
        return [row.task_id for row in self.rows]


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    number = _to_float(value, None)
    return default if number is None else int(number)


def parse_int_list(value: Any) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", str(value or ""))]


def parse_events_csv(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        events: List[Dict[str, Any]] = []
        for raw in reader:
            event = dict(raw)
            event["step_idx"] = int(event["step_idx"])
            event["failure_type"] = str(event["failure_type"]).strip().lower()
            event["decision_time"] = float(event.get("decision_time") or 0.0)
            event["frozen_task_ids"] = parse_int_list(event.get("frozen_task_ids"))
            event["dynamic_task_ids"] = parse_int_list(event.get("dynamic_task_ids"))
            for key in ("t", "t1", "t2"):
                event[key] = _to_int(event.get(key))
            event["new_agent"] = str(event.get("new_agent") or "").strip() or None
            event["delay_time"] = float(event.get("delay_time") or 0.0)
            events.append(event)
    return events


def parse_gantt_txt(path: str | Path) -> List[GanttBlock]:
    path = Path(path)
    row_re = re.compile(
        r"^\s*(?P<task_id>\d+)\s+"
        r"(?P<agent>human|robot)\s+"
        r"(?P<start>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<finish>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<duration>[+-]?\d+(?:\.\d+)?)\s*$",
        re.IGNORECASE,
    )

    blocks: List[GanttBlock] = []
    title: Optional[str] = None
    rows: List[GanttRow] = []
    makespan: Optional[float] = None

    def flush() -> None:
        nonlocal title, rows, makespan
        if title is not None:
            blocks.append(
                GanttBlock(
                    title=title,
                    rows=tuple(rows),
                    makespan=float(makespan or 0.0),
                )
            )
        title = None
        rows = []
        makespan = None

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("title:"):
            flush()
            title = line.split(":", 1)[1].strip()
            continue
        if title is None:
            continue
        if line.strip().startswith("makespan:"):
            makespan = float(line.split()[-1])
            continue
        match = row_re.match(line)
        if match:
            rows.append(
                GanttRow(
                    task_id=int(match.group("task_id")),
                    agent=match.group("agent").lower(),
                    start_time=float(match.group("start")),
                    finish_time=float(match.group("finish")),
                    duration=float(match.group("duration")),
                )
            )

    flush()
    if not blocks:
        raise ValueError(f"No Gantt blocks parsed from: {path}")
    return blocks


def block_for_step(blocks: Sequence[GanttBlock], step_idx: int) -> GanttBlock:
    # blocks[0] = offline optimal; blocks[1] = dynamic scheduling 1; ...
    if step_idx < 0 or step_idx >= len(blocks):
        raise IndexError(
            f"step_idx={step_idx} has no corresponding Gantt block; "
            f"parsed {len(blocks)} blocks"
        )
    return blocks[step_idx]


# ---------------------------------------------------------------------------
# Project-module loading
# ---------------------------------------------------------------------------


def configure_project_search_paths(project_dir: Optional[str | Path] = None) -> List[Path]:
    """Add likely project folders (including parents) to sys.path.

    This lets the script live inside a downloaded subfolder such as
    task_monitor/vns_convergence_replay_bundle while the original scheduling
    modules remain in task_monitor.
    """
    seeds: List[Path] = []
    if project_dir is not None:
        seeds.append(Path(project_dir).expanduser().resolve())
    seeds.extend([Path.cwd().resolve(), Path(__file__).resolve().parent])

    candidates: List[Path] = []
    seen: Set[Path] = set()
    for seed in seeds:
        for candidate in (seed, *seed.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    # Insert in reverse so explicit/current nearby paths retain highest priority.
    for candidate in reversed(candidates):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    return candidates


def _import_first_available(names: Sequence[str]):
    errors: Dict[str, str] = {}
    for name in names:
        try:
            return importlib.import_module(name), name
        except ImportError as exc:
            errors[name] = str(exc)
    return None, errors


def import_project_modules(
    solver_module_name: str,
    loader_module_name: str,
    precedence_module_name: str,
    project_dir: Optional[str | Path] = None,
):
    searched = configure_project_search_paths(project_dir)

    requested = str(solver_module_name).strip()
    if requested.lower() == "auto":
        solver_names = ["VNS_dynamic_solver", "VNS_rescheduler"]
    else:
        solver_names = [requested]

    solver_module, solver_info = _import_first_available(solver_names)
    if solver_module is None:
        searched_text = "\n  - ".join(str(path) for path in searched[:10])
        details = "; ".join(f"{name}: {msg}" for name, msg in solver_info.items())
        raise ImportError(
            "Cannot import a VNS solver module. Expected VNS_dynamic_solver.py "
            "or VNS_rescheduler.py in the original project folder.\n"
            f"Searched paths:\n  - {searched_text}\n"
            f"Import details: {details}\n"
            "Use --project-dir .. when the original project is one folder above, "
            "or --solver-module VNS_rescheduler to select it explicitly."
        )
    actual_solver_name = str(solver_info)

    try:
        loader_module = importlib.import_module(loader_module_name)
        precedence_module = importlib.import_module(precedence_module_name)
    except ImportError as exc:
        searched_text = "\n  - ".join(str(path) for path in searched[:10])
        raise ImportError(
            f"Cannot import {loader_module_name}.py and/or "
            f"{precedence_module_name}.py.\nSearched paths:\n  - {searched_text}\n"
            "These files must come from the original scheduling project. "
            "Use --project-dir .. if that project is in the parent folder."
        ) from exc

    if not hasattr(solver_module, "solver"):
        raise AttributeError(f"{actual_solver_name} has no solver() function")
    if not hasattr(loader_module, "load_schedule_csv"):
        raise AttributeError(f"{loader_module_name} has no load_schedule_csv()")
    if not hasattr(precedence_module, "build_precedence_matrix"):
        raise AttributeError(
            f"{precedence_module_name} has no build_precedence_matrix()"
        )

    print(f"[module] solver: {actual_solver_name}")
    print(f"[module] loader: {Path(loader_module.__file__).resolve()}")
    print(f"[module] precedence: {Path(precedence_module.__file__).resolve()}")
    return solver_module, loader_module, precedence_module, actual_solver_name


def call_supported(func, /, *args, **kwargs):
    """Call a project function while passing only kwargs accepted by its signature."""
    signature = inspect.signature(func)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    supported = kwargs if accepts_var_kw else {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return func(*args, **supported)


def load_robot_time_table(solver_module, robot_time_csv: Optional[str | Path]):
    loader = getattr(solver_module, "load_robot_task_time_table", None)
    if loader is None:
        return None

    if robot_time_csv is None:
        robot_time_csv = getattr(solver_module, "ROBOT_TIME_DIR", None)
    if robot_time_csv is None:
        return None

    path = Path(robot_time_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Robot time table not found: {path}. Supply --robot-time-csv or run "
            "the script from the original project folder."
        )
    return loader(path)


# ---------------------------------------------------------------------------
# Reconstruct one failure-state schedule
# ---------------------------------------------------------------------------


def reorder_schedule_from_block(
    metadata_schedule: Sequence[Mapping[str, Any]],
    reference_block: GanttBlock,
) -> List[Dict[str, Any]]:
    """Use recorded pre-failure order/agents while retaining full task metadata."""
    meta_by_id = {int(task["ID"]): deepcopy(dict(task)) for task in metadata_schedule}
    ordered_ids: List[int] = []

    # Gantt export omits zero-duration task 1; preserve any omitted metadata task
    # before the first exported task in original metadata order.
    exported_ids = set(reference_block.task_ids)
    for task in metadata_schedule:
        tid = int(task["ID"])
        if tid not in exported_ids and tid < min(reference_block.task_ids):
            ordered_ids.append(tid)
    ordered_ids.extend(reference_block.task_ids)

    # Keep any other omitted tasks at the end as a defensive fallback.
    ordered_ids.extend(
        int(task["ID"])
        for task in metadata_schedule
        if int(task["ID"]) not in set(ordered_ids)
    )

    agent_by_id = {row.task_id: row.agent for row in reference_block.rows}
    result: List[Dict[str, Any]] = []
    for tid in ordered_ids:
        if tid not in meta_by_id:
            raise KeyError(f"Task ID {tid} from Gantt is missing in schedule metadata")
        task = deepcopy(meta_by_id[tid])
        if tid in agent_by_id:
            task["agent"] = agent_by_id[tid]
        result.append(task)
    return result


def infer_status(
    *,
    task_id: int,
    agent: str,
    event: Mapping[str, Any],
    row: Optional[GanttRow],
    tolerance: float = 0.03,
) -> str:
    frozen = set(event["frozen_task_ids"])
    if task_id not in frozen:
        return "pending"

    # Task 1 is omitted from exported Gantt because the shown timeline begins
    # after the first kit-box replacement.
    if row is None:
        return "completed"

    decision = float(event["decision_time"])
    if row.finish_time <= decision + tolerance:
        return "completed"

    return "dispatched" if agent == "robot" else "executing"


def apply_failure_state(
    schedule: List[Dict[str, Any]],
    event: Mapping[str, Any],
    state_block: GanttBlock,
    cumulative_delay_by_task: Dict[int, float],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, float],
    Dict[str, float],
    Dict[int, float],
]:
    """Populate status, actual/fixed timing, and delay duration overrides."""
    result = deepcopy(schedule)
    rows = state_block.by_task
    frozen: Set[int] = set(int(x) for x in event["frozen_task_ids"])
    decision = float(event["decision_time"])

    fixed_start_by_id: Dict[int, float] = {}
    agent_ready_times = {"human": decision, "robot": decision}

    # Update cumulative delay before task reconstruction.
    if event["failure_type"] == "delay" and event.get("t") is not None:
        tid = int(event["t"])
        cumulative_delay_by_task[tid] = (
            cumulative_delay_by_task.get(tid, 0.0)
            + float(event.get("delay_time") or 0.0)
        )

    for task in result:
        tid = int(task["ID"])
        row = rows.get(tid)
        agent = str(task.get("agent", "human"))

        # Remove stale run-state fields from metadata or previous reconstruction.
        for key in (
            "_duration_override",
            "_expected_finish_time",
            "_delay_time",
            "_delay_failure",
            "_place_wait_compensation",
        ):
            task.pop(key, None)

        status = infer_status(
            task_id=tid,
            agent=agent,
            event=event,
            row=row,
        )
        task["status"] = status
        task["ur_dispatched"] = status == "dispatched"

        if row is not None:
            task["planned_start_time"] = float(row.start_time)
            task["planned_finish_time"] = float(row.finish_time)

        if tid in frozen and row is not None:
            task["actual_start_time"] = float(row.start_time)
            fixed_start_by_id[tid] = float(row.start_time)

            if status == "completed":
                task["actual_finish_time"] = float(row.finish_time)
                task["actual_time"] = float(row.duration)
                # Dynamic solvers commonly prioritize actual_time for frozen work;
                # this override also keeps fallback solvers numerically consistent.
                task["_duration_override"] = float(row.duration)
            else:
                task["actual_finish_time"] = None
                task["actual_time"] = None
                task["_duration_override"] = float(row.duration)
                agent_ready_times[agent] = max(
                    agent_ready_times[agent], float(row.finish_time)
                )
        else:
            task["actual_start_time"] = None
            task["actual_finish_time"] = None
            task["actual_time"] = None

    # For an order mistake, t2 is the opportunistically completed task. The
    # rounded Gantt finish may exceed decision_time by only a few milliseconds,
    # so force its recorded semantic state to completed.
    if event["failure_type"] == "order" and event.get("t2") is not None:
        t2 = int(event["t2"])
        for task in result:
            if int(task["ID"]) == t2:
                row = rows.get(t2)
                task["status"] = "completed"
                task["ur_dispatched"] = False
                if row is not None:
                    task["actual_start_time"] = float(row.start_time)
                    task["actual_finish_time"] = float(row.finish_time)
                    task["actual_time"] = float(row.duration)
                    task["_duration_override"] = float(row.duration)
                    fixed_start_by_id[t2] = float(row.start_time)
                break

    # Delay logic matches HRC_schedule_perception_main.py: each exported
    # delay_time is an increment, so repeated delays accumulate on top of the
    # task's human standard duration.
    if event["failure_type"] == "delay" and event.get("t") is not None:
        target_id = int(event["t"])
        for task in result:
            if int(task["ID"]) != target_id:
                continue
            standard = float(task.get("human_standard_time") or 0.0)
            estimated_duration = standard + cumulative_delay_by_task[target_id]
            start_t = task.get("actual_start_time")
            if start_t is None:
                row = rows.get(target_id)
                start_t = row.start_time if row is not None else max(
                    0.0, decision - standard
                )
            task["_duration_override"] = float(estimated_duration)
            task["_expected_finish_time"] = float(start_t) + estimated_duration
            task["_delay_time"] = float(event.get("delay_time") or 0.0)
            task["_delay_failure"] = True
            task["planned_start_time"] = float(start_t)
            task["planned_finish_time"] = float(start_t) + estimated_duration
            agent_ready_times[str(task.get("agent", "human"))] = max(
                agent_ready_times[str(task.get("agent", "human"))],
                float(start_t) + estimated_duration,
            )
            break

    return result, fixed_start_by_id, agent_ready_times, cumulative_delay_by_task


def compute_robot_pick_counts(schedule: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for task in schedule:
        if task.get("status") != "completed":
            continue
        if task.get("agent") != "robot":
            continue
        if task.get("command") != "pick and place":
            continue
        obj = str(task.get("object") or "").split("_")[0]
        if obj:
            counts[obj] = counts.get(obj, 0) + 1
    return counts


def infer_start_kit_id(solver_module, schedule, frozen_task_ids: Set[int]) -> int:
    extract_kits = getattr(solver_module, "extract_kits", None)
    if extract_kits is None:
        return 1
    kits = extract_kits(schedule)
    for kit_id in sorted(kits):
        if any(int(schedule[i]["ID"]) not in frozen_task_ids for i in kits[kit_id]):
            return int(kit_id)
    return max((int(k) for k in kits), default=1)


def patch_fallback_duration_logic(solver_module) -> None:
    """Make simple VNS_rescheduler honor reconstructed actual/delay durations."""
    module_name = getattr(solver_module, "__name__", "")
    if module_name != "VNS_rescheduler":
        return

    original = solver_module.get_task_duration

    def replay_get_task_duration(task):
        override = task.get("_duration_override")
        if override is not None:
            return float(override)
        if task.get("status") == "completed" and task.get("actual_time") is not None:
            return float(task["actual_time"])
        return float(original(task))

    solver_module.get_task_duration = replay_get_task_duration


# ---------------------------------------------------------------------------
# Solver result handling and outputs
# ---------------------------------------------------------------------------


def is_numeric_history(value: Any, max_iter: int) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if len(value) < min(2, max_iter):
        return False
    return all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in value)


def unpack_solver_result(result: Any, max_iter: int):
    if isinstance(result, Mapping):
        history = result.get("history") or result.get("convergence_history")
        if not is_numeric_history(history, max_iter):
            raise ValueError("Solver dict result does not contain numeric history")
        return (
            result.get("best") or result.get("best_schedule"),
            float(result.get("baseline_cost", history[0])),
            float(result.get("best_cost", history[-1])),
            float(result.get("search_time", 0.0)),
            list(history),
        )

    if not isinstance(result, (tuple, list)):
        raise TypeError(f"Unsupported solver result type: {type(result).__name__}")

    values = list(result)
    best = values[0] if values else None
    baseline = float(values[1]) if len(values) > 1 else float("nan")
    best_cost = float(values[2]) if len(values) > 2 else float("nan")
    search_time = float(values[3]) if len(values) > 3 else 0.0

    history = None
    if len(values) > 5 and is_numeric_history(values[5], max_iter):
        history = list(values[5])
    else:
        for value in reversed(values):
            if is_numeric_history(value, max_iter):
                history = list(value)
                break
    if history is None:
        raise ValueError(
            "Could not find convergence history in solver return value. "
            "The solver must return the per-iteration best makespan list."
        )

    return best, baseline, best_cost, search_time, history


def save_original_format_curve(history: Sequence[float], output_path: Path) -> None:
    """Intentionally mirrors VNS_rescheduler.py's convergence plotting block."""
    plt.figure(figsize=(10, 3))
    plt.plot(history)
    plt.title("Convergence (best makespan over iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Best makespan")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_combined_curve(histories: Mapping[int, Sequence[float]], output_path: Path) -> None:
    plt.figure(figsize=(10, 3))
    for step_idx, history in histories.items():
        plt.plot(history, label=f"Failure {step_idx}")
    plt.title("Convergence (best makespan over iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Best makespan")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------


def run_replay(
    *,
    events_path: Path,
    gantt_path: Path,
    output_dir: Path,
    max_iter: int,
    seed: Optional[int],
    shaking_threshold: int,
    solver_module_name: str,
    loader_module_name: str,
    precedence_module_name: str,
    robot_time_csv: Optional[Path],
    project_dir: Optional[Path],
) -> Dict[str, Any]:
    events_path = events_path.expanduser().resolve()
    gantt_path = gantt_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if robot_time_csv is not None:
        robot_time_csv = robot_time_csv.expanduser().resolve()

    events = parse_events_csv(events_path)
    blocks = parse_gantt_txt(gantt_path)

    if len(events) + 1 > len(blocks):
        raise ValueError(
            f"Found {len(events)} events but only {len(blocks)} Gantt blocks. "
            "Expected offline + one dynamic block per event."
        )

    solver_module, loader_module, precedence_module, actual_solver_name = import_project_modules(
        solver_module_name,
        loader_module_name,
        precedence_module_name,
        project_dir=project_dir,
    )
    patch_fallback_duration_logic(solver_module)

    project_root = Path(loader_module.__file__).resolve().parent
    previous_cwd = Path.cwd()
    try:
        os.chdir(project_root)
        metadata_schedule = loader_module.load_schedule_csv()
        robot_time_table = load_robot_time_table(solver_module, robot_time_csv)
    finally:
        os.chdir(previous_cwd)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed once so the four sequential calls form one reproducible replay, rather
    # than restarting the same random sequence at every failure.
    if seed is not None:
        random.seed(seed)

    cumulative_delay_by_task: Dict[int, float] = {}
    histories: Dict[int, List[float]] = {}
    summary_rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []

    for event in events:
        step_idx = int(event["step_idx"])
        reference_block = block_for_step(blocks, step_idx - 1)
        state_block = block_for_step(blocks, step_idx)

        schedule = reorder_schedule_from_block(metadata_schedule, reference_block)
        schedule, fixed_start_by_id, agent_ready_times, cumulative_delay_by_task = (
            apply_failure_state(
                schedule,
                event,
                state_block,
                cumulative_delay_by_task,
            )
        )

        frozen_ids = set(int(x) for x in event["frozen_task_ids"])
        robot_pick_counts = compute_robot_pick_counts(schedule)
        start_kit_id = infer_start_kit_id(solver_module, schedule, frozen_ids)

        P = precedence_module.build_precedence_matrix(schedule)

        solver_kwargs = {
            "start_kit_id": start_kit_id,
            "max_iter": max_iter,
            # Do not pass a per-call seed; random.seed() above preserves a single
            # deterministic sequence across the four real failure events.
            "shaking_threshold": shaking_threshold,
            "frozen_task_ids": frozen_ids,
            "robot_time_table": robot_time_table,
            "robot_pick_counts": robot_pick_counts,
            "decision_time": float(event["decision_time"]),
            "now_from_start": float(event["decision_time"]),
            "fixed_start_by_id": fixed_start_by_id,
            "agent_ready_times": agent_ready_times,
        }

        result = call_supported(
            solver_module.solver,
            schedule,
            P,
            **solver_kwargs,
        )
        best_schedule, baseline, best_cost, search_time, history = unpack_solver_result(
            result, max_iter
        )

        if len(history) != max_iter:
            raise ValueError(
                f"Failure {step_idx}: solver returned {len(history)} history points; "
                f"expected exactly {max_iter}. Check where history.append(best_cost) "
                "is placed in the solver."
            )

        histories[step_idx] = [float(x) for x in history]
        save_original_format_curve(
            histories[step_idx],
            output_dir / f"convergence_failure_{step_idx}.png",
        )

        recorded_cost = float(state_block.makespan)
        summary_rows.append(
            {
                "step_idx": step_idx,
                "failure_type": event["failure_type"],
                "decision_time": event["decision_time"],
                "target_task": event.get("t") or "",
                "t1": event.get("t1") or "",
                "t2": event.get("t2") or "",
                "delay_increment": event.get("delay_time") or 0.0,
                "start_kit_id": start_kit_id,
                "frozen_task_count": len(frozen_ids),
                "baseline_makespan": baseline,
                "rerun_best_makespan": best_cost,
                "recorded_dynamic_makespan": recorded_cost,
                "rerun_minus_recorded": best_cost - recorded_cost,
                "search_time": search_time,
                "iterations": len(history),
                "seed": "" if seed is None else seed,
                "solver_module": actual_solver_name,
            }
        )

        by_id = {int(task["ID"]): task for task in schedule}
        for tid in [int(task["ID"]) for task in schedule]:
            task = by_id[tid]
            state_rows.append(
                {
                    "step_idx": step_idx,
                    "task_id": tid,
                    "agent": task.get("agent", ""),
                    "status": task.get("status", ""),
                    "frozen": int(tid in frozen_ids),
                    "actual_start_time": task.get("actual_start_time", ""),
                    "actual_finish_time": task.get("actual_finish_time", ""),
                    "actual_time": task.get("actual_time", ""),
                    "duration_override": task.get("_duration_override", ""),
                    "expected_finish_time": task.get("_expected_finish_time", ""),
                }
            )

    save_combined_curve(histories, output_dir / "convergence_all_failures.png")

    history_rows = [
        {
            "step_idx": step_idx,
            "iteration": iteration,
            "best_makespan": cost,
        }
        for step_idx, history in histories.items()
        for iteration, cost in enumerate(history)
    ]
    write_csv(
        output_dir / "convergence_history.csv",
        ["step_idx", "iteration", "best_makespan"],
        history_rows,
    )
    write_csv(
        output_dir / "convergence_summary.csv",
        list(summary_rows[0].keys()) if summary_rows else ["step_idx"],
        summary_rows,
    )
    write_csv(
        output_dir / "reconstructed_failure_states.csv",
        list(state_rows[0].keys()) if state_rows else ["step_idx"],
        state_rows,
    )

    return {
        "output_dir": str(output_dir),
        "events": len(events),
        "iterations_per_event": max_iter,
        "summary": summary_rows,
    }


def validate_raw_data(events_path: Path, gantt_path: Path) -> None:
    events = parse_events_csv(events_path)
    blocks = parse_gantt_txt(gantt_path)
    print(f"events: {len(events)}")
    print(f"gantt blocks: {len(blocks)}")
    for event in events:
        step = int(event["step_idx"])
        block = block_for_step(blocks, step)
        print(
            f"failure {step}: type={event['failure_type']}, "
            f"decision={event['decision_time']:.4f}, "
            f"frozen={len(event['frozen_task_ids'])}, "
            f"recorded_makespan={block.makespan:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay VNS convergence curves for one real experiment"
    )
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--gantt", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("vns_convergence_output"))
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--shaking-threshold", type=int, default=50)
    parser.add_argument(
        "--solver-module",
        default="auto",
        help=(
            "Solver module name. Default 'auto' tries VNS_dynamic_solver first "
            "and falls back to VNS_rescheduler."
        ),
    )
    parser.add_argument("--loader-module", default="read_schedule")
    parser.add_argument("--precedence-module", default="precedence_matrix")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help=(
            "Original scheduling project folder. Usually use '..' when this "
            "script is inside vns_convergence_replay_bundle."
        ),
    )
    parser.add_argument("--robot-time-csv", type=Path, default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only parse and report the two raw-data files; do not import project modules",
    )
    args = parser.parse_args()

    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")

    if args.validate_only:
        validate_raw_data(args.events, args.gantt)
        return

    result = run_replay(
        events_path=args.events,
        gantt_path=args.gantt,
        output_dir=args.output_dir,
        max_iter=args.max_iter,
        seed=args.seed,
        shaking_threshold=args.shaking_threshold,
        solver_module_name=args.solver_module,
        loader_module_name=args.loader_module,
        precedence_module_name=args.precedence_module,
        robot_time_csv=args.robot_time_csv,
        project_dir=args.project_dir,
    )

    print("[VNS CONVERGENCE REPLAY DONE]")
    print(f"output_dir: {result['output_dir']}")
    print(f"events: {result['events']}")
    print(f"iterations_per_event: {result['iterations_per_event']}")
    for row in result["summary"]:
        print(
            f"failure {row['step_idx']}: "
            f"baseline={row['baseline_makespan']:.4f}, "
            f"best={row['rerun_best_makespan']:.4f}, "
            f"recorded={row['recorded_dynamic_makespan']:.4f}"
        )


if __name__ == "__main__":
    main()
