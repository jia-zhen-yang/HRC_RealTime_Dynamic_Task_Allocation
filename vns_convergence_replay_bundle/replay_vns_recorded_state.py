"""
Replay VNS convergence for one recorded experiment.

Core rule
---------
For every failure event:
1. Rebuild the schedule order/agent assignment that existed immediately before
   that failure from the exported Gantt blocks.
2. Apply the recorded failure (order / agent / delay).
3. Frozen tasks that had completed by decision_time use their recorded real
   operation duration.
4. Frozen tasks that were still running keep the recorded start, expected
   duration, and expected finish from that failure's exported Gantt block; this
   determines when the corresponding agent becomes available.
5. Tasks that had not started use the experiment-time standard duration inferred
   from the exported Gantt data. The current schedule.csv is only a fallback for
   task-agent combinations never observed in the experiment.
6. VNS may modify only non-frozen tasks.
7. Record one best makespan value after each outer VNS iteration, matching the
   convergence-history convention in VNS_rescheduler.py.

This script intentionally does NOT calculate, print, or plot a baseline.
The initial failed schedule is only the internal incumbent required to start VNS.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import math
import os
import random
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Recorded-data models and parsers
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


def parse_int_list(value: Any) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", str(value or ""))]


def optional_int(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    return int(float(text))


def parse_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            event = dict(raw)
            event["step_idx"] = int(event["step_idx"])
            event["failure_type"] = str(event["failure_type"]).strip().lower()
            event["decision_time"] = float(event.get("decision_time") or 0.0)
            event["frozen_task_ids"] = parse_int_list(event.get("frozen_task_ids"))
            event["dynamic_task_ids"] = parse_int_list(event.get("dynamic_task_ids"))
            event["t"] = optional_int(event.get("t"))
            event["t1"] = optional_int(event.get("t1"))
            event["t2"] = optional_int(event.get("t2"))
            event["new_agent"] = str(event.get("new_agent") or "").strip() or None
            event["delay_time"] = float(event.get("delay_time") or 0.0)
            events.append(event)
    events.sort(key=lambda item: item["step_idx"])
    return events


def parse_gantt(path: Path) -> List[GanttBlock]:
    row_re = re.compile(
        r"^\s*(?P<task>\d+)\s+"
        r"(?P<agent>human|robot)\s+"
        r"(?P<start>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<finish>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<duration>[+-]?\d+(?:\.\d+)?)\s*$",
        re.IGNORECASE,
    )

    blocks: List[GanttBlock] = []
    title: Optional[str] = None
    rows: List[GanttRow] = []
    makespan = 0.0

    def flush() -> None:
        nonlocal title, rows, makespan
        if title is not None:
            blocks.append(GanttBlock(title, tuple(rows), float(makespan)))
        title = None
        rows = []
        makespan = 0.0

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
                    task_id=int(match.group("task")),
                    agent=match.group("agent").lower(),
                    start_time=float(match.group("start")),
                    finish_time=float(match.group("finish")),
                    duration=float(match.group("duration")),
                )
            )
    flush()
    if not blocks:
        raise ValueError(f"No Gantt blocks parsed from {path}")
    return blocks


def block_for_step(blocks: Sequence[GanttBlock], step_idx: int) -> GanttBlock:
    # blocks[0] = offline; blocks[1] = dynamic scheduling 1; ...
    if step_idx < 0 or step_idx >= len(blocks):
        raise IndexError(
            f"Cannot map step {step_idx} to Gantt block; parsed {len(blocks)} blocks"
        )
    return blocks[step_idx]


# ---------------------------------------------------------------------------
# Project module loading
# ---------------------------------------------------------------------------


def import_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_project_modules(project_dir: Path, solver_file: Optional[Path], solver_module: str):
    project_dir = project_dir.resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    sys.path.insert(0, str(project_dir))

    if solver_file is not None:
        solver_path = solver_file
        if not solver_path.is_absolute():
            solver_path = (Path.cwd() / solver_path).resolve()
        if not solver_path.exists():
            raise FileNotFoundError(f"Solver file not found: {solver_path}")
        solver = import_module_from_file("vns_recorded_dynamic_solver", solver_path)
        solver_label = str(solver_path)
    else:
        try:
            solver = importlib.import_module(solver_module)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"Cannot import solver module '{solver_module}'. Place the solver in "
                f"{project_dir}, or pass --solver-file with its exact path."
            ) from exc
        solver_label = solver_module

    loader = importlib.import_module("read_schedule")
    precedence = importlib.import_module("precedence_matrix")

    print(f"[module] solver: {solver_label}")
    print(f"[module] loader: {Path(loader.__file__).resolve()}")
    print(f"[module] precedence: {Path(precedence.__file__).resolve()}")
    return solver, loader, precedence


# ---------------------------------------------------------------------------
# Experiment-time standard-duration reconstruction
# ---------------------------------------------------------------------------


def build_experiment_standard_lookup(
    events: Sequence[Mapping[str, Any]],
    blocks: Sequence[GanttBlock],
    *,
    tolerance: float = 0.03,
) -> Dict[Tuple[int, str], float]:
    """Infer the standard duration used by the recorded experiment.

    The offline optimal block is a standard-time schedule, so every row in it
    supplies one task-agent standard duration.  Later dynamic blocks may expose
    a task under the other agent after reassignment.  A row is accepted as a
    standard-time observation only when that task was *not frozen* at the
    corresponding decision, because non-frozen tasks had not started and were
    still evaluated with standard time.

    This prevents a newer/different schedule.csv from silently changing the
    replay's task-time parameters.
    """
    lookup: Dict[Tuple[int, str], float] = {}

    # Offline schedule: all displayed durations are planning standard times.
    for row in blocks[0].rows:
        lookup[(int(row.task_id), str(row.agent))] = float(row.duration)

    # Dynamic blocks: collect standard durations for task-agent combinations
    # that were pending at the decision time (not in frozen_task_ids).
    for event in events:
        step = int(event["step_idx"])
        if step >= len(blocks):
            continue
        frozen = {int(x) for x in event.get("frozen_task_ids", [])}
        decision = float(event.get("decision_time", 0.0))
        for row in blocks[step].rows:
            tid = int(row.task_id)
            if tid in frozen:
                continue
            # A pending row should not start before decision_time.  Keep a small
            # tolerance for the exported two-decimal timestamps.
            if float(row.start_time) + tolerance < decision:
                continue
            lookup.setdefault((tid, str(row.agent)), float(row.duration))

    return lookup


def apply_experiment_standard_times(
    schedule: Sequence[Mapping[str, Any]],
    standard_lookup: Mapping[Tuple[int, str], float],
) -> List[Dict[str, Any]]:
    """Write experiment-time standard durations into task metadata.

    Only the field corresponding to the task's current agent is overwritten.
    If VNS later reassigns the task, the other agent's observed experiment-time
    value is used when available; otherwise the project metadata remains the
    fallback.
    """
    result = deepcopy(list(schedule))
    for task in result:
        tid = int(task["ID"])
        human_key = (tid, "human")
        robot_key = (tid, "robot")
        if human_key in standard_lookup:
            task["human_standard_time"] = float(standard_lookup[human_key])
        if robot_key in standard_lookup:
            task["robot_standard_time"] = float(standard_lookup[robot_key])
    return result


# ---------------------------------------------------------------------------
# Schedule reconstruction following failure_simulation.py
# ---------------------------------------------------------------------------


def rebuild_schedule_from_block(
    metadata_schedule: Sequence[Mapping[str, Any]],
    reference_block: GanttBlock,
) -> List[Dict[str, Any]]:
    """Use a recorded pre-failure block for order/agents without leaking the
    post-failure solution. Task 1 remains structurally present at the front.
    """
    meta = {int(task["ID"]): deepcopy(dict(task)) for task in metadata_schedule}
    agents = {row.task_id: row.agent for row in reference_block.rows}

    result: List[Dict[str, Any]] = []
    if 1 in meta and 1 not in agents:
        task1 = deepcopy(meta[1])
        task1["agent"] = "human"
        result.append(task1)

    for tid in reference_block.task_ids:
        if tid not in meta:
            raise KeyError(f"Task {tid} from Gantt is absent from schedule.csv")
        task = deepcopy(meta[tid])
        task["agent"] = agents[tid]
        result.append(task)

    expected = {int(task["ID"]) for task in metadata_schedule}
    present = {int(task["ID"]) for task in result}
    missing = expected - present
    if missing:
        # This normally only catches zero-duration tasks omitted by the exporter.
        for task in metadata_schedule:
            if int(task["ID"]) in missing:
                result.append(deepcopy(dict(task)))
    return result


def get_id_to_index(schedule: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    return {int(task["ID"]): index for index, task in enumerate(schedule)}


def apply_failure(
    schedule: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
    cumulative_delay: Dict[int, float],
) -> Tuple[List[Dict[str, Any]], Dict[int, float]]:
    """Equivalent to failure_simulation.simulate_failure, using recorded inputs."""
    result = deepcopy(list(schedule))
    id_to_idx = get_id_to_index(result)
    kind = event["failure_type"]

    if kind == "order":
        t1, t2 = int(event["t1"]), int(event["t2"])
        if t1 not in id_to_idx or t2 not in id_to_idx:
            raise KeyError(f"Order event tasks not found: {t1}, {t2}")
        i, j = id_to_idx[t1], id_to_idx[t2]
        result[i], result[j] = result[j], result[i]

    elif kind == "agent":
        tid = int(event["t"])
        if tid not in id_to_idx:
            raise KeyError(f"Agent event task not found: {tid}")
        new_agent = event.get("new_agent")
        if not new_agent:
            old = str(result[id_to_idx[tid]].get("agent"))
            new_agent = "robot" if old == "human" else "human"
        result[id_to_idx[tid]]["agent"] = new_agent

    elif kind == "delay":
        tid = int(event["t"])
        if tid not in id_to_idx:
            raise KeyError(f"Delay event task not found: {tid}")
        cumulative_delay[tid] = cumulative_delay.get(tid, 0.0) + float(
            event.get("delay_time") or 0.0
        )

    else:
        raise ValueError(f"Unsupported failure type: {kind}")

    return result, cumulative_delay


def reorder_schedule_with_frozen(solver, schedule, frozen_task_ids: Set[int]):
    """Same ordering rule as failure_simulation.reorder_schedule_with_frozen."""
    kits = solver.extract_kits(schedule)
    reordered: List[Dict[str, Any]] = []

    for kit_id in sorted(kits):
        kit_tasks = [schedule[index] for index in kits[kit_id]]
        replace = next(
            (task for task in kit_tasks if task.get("command") == "replace"),
            None,
        )
        if replace is not None:
            reordered.append(replace)
            kit_tasks = [task for task in kit_tasks if int(task["ID"]) != int(replace["ID"])]

        frozen = [task for task in kit_tasks if int(task["ID"]) in frozen_task_ids]
        pending = [task for task in kit_tasks if int(task["ID"]) not in frozen_task_ids]
        reordered.extend(frozen)
        reordered.extend(pending)
    return reordered


def task_standard_duration(task: Mapping[str, Any]) -> float:
    if str(task.get("agent")) == "robot":
        return float(task.get("robot_standard_time") or 0.0)
    return float(task.get("human_standard_time") or 0.0)


def apply_recorded_execution_state(
    schedule: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
    state_block: GanttBlock,
    cumulative_delay: Mapping[int, float],
    tolerance: float = 0.03,
) -> Tuple[List[Dict[str, Any]], Dict[int, float], Dict[str, float], List[Dict[str, Any]]]:
    """Apply only completed real durations; pending work remains standard-time.

    Frozen-but-running tasks are not treated as completed. They are fixed at the
    recorded start and only establish the agent's next available time.
    """
    result = deepcopy(list(schedule))
    rows = state_block.by_task
    frozen = {int(x) for x in event["frozen_task_ids"]}
    decision = float(event["decision_time"])
    order_completed_id = (
        int(event["t2"])
        if event["failure_type"] == "order" and event.get("t2") is not None
        else None
    )
    delay_target = (
        int(event["t"])
        if event["failure_type"] == "delay" and event.get("t") is not None
        else None
    )

    fixed_start_by_id: Dict[int, float] = {}
    agent_ready_times = {"human": decision, "robot": decision}
    audit: List[Dict[str, Any]] = []

    for task in result:
        tid = int(task["ID"])
        row = rows.get(tid)

        # Clear all previous simulation-state fields. Pending work must use
        # standard time solely because actual_time is None.
        task["actual_time"] = None
        for key in (
            "_duration_override",
            "_expected_finish_time",
            "_delay_time",
            "_delay_failure",
            "actual_start_time",
            "actual_finish_time",
            "status",
            "ur_dispatched",
        ):
            task.pop(key, None)

        # Task 1 occurred before the exported timeline's t=0. Keep it for the
        # full task/kit structure, but make it a zero-time completed task.
        if tid == 1 and tid in frozen and row is None:
            task["status"] = "completed"
            task["actual_time"] = 0.0
            task["_duration_override"] = 0.0
            task["actual_start_time"] = 0.0
            task["actual_finish_time"] = 0.0
            fixed_start_by_id[tid] = 0.0
            audit.append({
                "task_id": tid,
                "agent": task.get("agent", ""),
                "state": "completed_before_time_origin",
                "duration_source": "zero_time_origin",
                "duration_used": 0.0,
                "fixed_start": 0.0,
                "recorded_finish": 0.0,
                "expected_finish_used": 0.0,
            })
            continue

        if tid not in frozen:
            task["status"] = "pending"
            audit.append({
                "task_id": tid,
                "agent": task.get("agent", ""),
                "state": "pending",
                "duration_source": "standard_time",
                "duration_used": task_standard_duration(task),
                "fixed_start": "",
                "recorded_finish": "",
                "expected_finish_used": "",
            })
            continue

        if row is None:
            raise KeyError(
                f"Frozen task {tid} is missing from Gantt block '{state_block.title}'"
            )

        is_completed = row.finish_time <= decision + tolerance or tid == order_completed_id
        fixed_start_by_id[tid] = float(row.start_time)
        task["actual_start_time"] = float(row.start_time)

        if is_completed:
            # This is the only normal path that imports a real operation time.
            task["status"] = "completed"
            task["actual_finish_time"] = float(row.finish_time)
            task["actual_time"] = float(row.duration)
            task["_duration_override"] = float(row.duration)
            audit.append({
                "task_id": tid,
                "agent": task.get("agent", ""),
                "state": "completed",
                "duration_source": "recorded_real_duration",
                "duration_used": float(row.duration),
                "fixed_start": float(row.start_time),
                "recorded_finish": float(row.finish_time),
                "expected_finish_used": float(row.finish_time),
            })
            continue

        # Frozen but still executing/dispatched at decision time.
        task["status"] = "dispatched" if task.get("agent") == "robot" else "executing"
        task["ur_dispatched"] = task["status"] == "dispatched"

        # The exported post-failure Gantt is the planning state that was sent
        # to the real dynamic solver at this decision.  For a task already in
        # progress, its row duration/finish is therefore the correct expected
        # execution boundary.  Do not rebuild it from the current schedule.csv.
        duration = float(row.duration)
        expected_finish = float(row.finish_time)
        source = "recorded_in_progress_expected_duration"
        task["_duration_override"] = duration
        task["_expected_finish_time"] = expected_finish
        if tid == delay_target:
            task["_delay_failure"] = True
            task["_delay_time"] = float(event.get("delay_time") or 0.0)

        agent = str(task.get("agent"))
        agent_ready_times[agent] = max(agent_ready_times.get(agent, decision), expected_finish)

        audit.append({
            "task_id": tid,
            "agent": agent,
            "state": task["status"],
            "duration_source": source,
            "duration_used": duration,
            "fixed_start": float(row.start_time),
            "recorded_finish": float(row.finish_time),
            "expected_finish_used": expected_finish,
        })

    return result, fixed_start_by_id, agent_ready_times, audit


def infer_start_kit(solver, schedule, event: Mapping[str, Any]) -> int:
    """Same principle as failure_simulation.infer_start_kit."""
    kits = solver.extract_kits(schedule)
    task_to_kit: Dict[int, int] = {}
    for kit_id, indices in kits.items():
        for index in indices:
            task_to_kit[int(schedule[index]["ID"])] = int(kit_id)

    if event["failure_type"] == "order":
        ids = [event.get("t1"), event.get("t2")]
    else:
        ids = [event.get("t")]
    candidates = [task_to_kit[int(tid)] for tid in ids if tid is not None and int(tid) in task_to_kit]
    return min(candidates) if candidates else 1


# ---------------------------------------------------------------------------
# VNS loop: dynamic makespan model + VNS_rescheduler history convention
# ---------------------------------------------------------------------------


def run_vns(
    solver,
    schedule: List[Dict[str, Any]],
    P,
    *,
    start_kit_id: int,
    max_iter: int,
    seed: int,
    shaking_threshold: int,
    frozen_task_ids: Set[int],
    decision_time: float,
    fixed_start_by_id: Mapping[int, float],
    agent_ready_times: Mapping[str, float],
    robot_time_table,
) -> Tuple[List[Dict[str, Any]], float, float, List[float], List[int]]:
    if not hasattr(solver, "N_SEARCH"):
        raise AttributeError("Solver module has no N_SEARCH neighborhood list")

    random.seed(seed)
    t0 = time.time()

    # Internal incumbent only. It is deliberately not returned as a baseline.
    best = deepcopy(schedule)
    best_cost = float(
        solver.get_makespan(
            best,
            P,
            frozen_task_ids=frozen_task_ids,
            decision_time=decision_time,
            fixed_start_by_id=dict(fixed_start_by_id),
            agent_ready_times=dict(agent_ready_times),
            robot_time_table=robot_time_table,
            robot_pick_counts=None,  # full schedule is supplied; avoid double counting
        )
    )

    kits = solver.extract_kits(best)
    history: List[float] = []
    shake_iterations: List[int] = []
    no_improvement_count = 0

    for iteration in range(max_iter):
        for kit_id in sorted(kits):
            if int(kit_id) < int(start_kit_id):
                continue

            operation = random.choice(solver.N_SEARCH)
            candidate = operation(best, kits, kit_id, frozen_task_ids)
            if candidate is None:
                continue

            cost = float(
                solver.get_makespan(
                    candidate,
                    P,
                    frozen_task_ids=frozen_task_ids,
                    decision_time=decision_time,
                    fixed_start_by_id=dict(fixed_start_by_id),
                    agent_ready_times=dict(agent_ready_times),
                    robot_time_table=robot_time_table,
                    robot_pick_counts=None,
                )
            )

            if cost < best_cost:
                best = candidate
                best_cost = cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= shaking_threshold:
                shaken = solver.shaking(best, kits, kit_id, frozen_task_ids)
                if shaken is not None:
                    shaken_cost = float(
                        solver.get_makespan(
                            shaken,
                            P,
                            frozen_task_ids=frozen_task_ids,
                            decision_time=decision_time,
                            fixed_start_by_id=dict(fixed_start_by_id),
                            agent_ready_times=dict(agent_ready_times),
                            robot_time_table=robot_time_table,
                            robot_pick_counts=None,
                        )
                    )
                    if shaken_cost < best_cost:
                        best = shaken
                        best_cost = shaken_cost
                no_improvement_count = 0
                shake_iterations.append(iteration)

        # Exactly the VNS_rescheduler convention: one point per outer iteration.
        history.append(best_cost)

    return best, best_cost, time.time() - t0, history, shake_iterations


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def save_curve(history: Sequence[float], output_path: Path) -> None:
    # Intentionally identical to VNS_rescheduler.py's convergence format.
    plt.figure(figsize=(10, 3))
    plt.plot(history)
    plt.title("Convergence (best makespan over iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Best makespan")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_combined(histories: Mapping[int, Sequence[float]], output_path: Path) -> None:
    plt.figure(figsize=(10, 3))
    for step, history in histories.items():
        plt.plot(history, label=f"Failure {step}")
    plt.title("Convergence (best makespan over iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Best makespan")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------


def run_replay(args) -> Dict[str, Any]:
    events_path = Path(args.events).resolve()
    gantt_path = Path(args.gantt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events = parse_events(events_path)
    blocks = parse_gantt(gantt_path)
    standard_lookup = build_experiment_standard_lookup(
        events,
        blocks,
        tolerance=float(args.completion_tolerance),
    )
    solver, loader, precedence = load_project_modules(
        Path(args.project_dir),
        Path(args.solver_file) if args.solver_file else None,
        args.solver_module,
    )

    # Some project loaders use relative paths. Run the load from project_dir.
    previous_cwd = Path.cwd()
    try:
        os.chdir(Path(args.project_dir).resolve())
        metadata_schedule = loader.load_schedule_csv()
    finally:
        os.chdir(previous_cwd)

    # Match failure_simulation.py: build P once from the original loaded schedule.
    P = precedence.build_precedence_matrix(metadata_schedule)

    robot_csv = Path(args.robot_time_csv) if args.robot_time_csv else None
    if robot_csv is None:
        robot_csv = Path(getattr(solver, "ROBOT_TIME_DIR"))
    elif not robot_csv.is_absolute():
        robot_csv = (Path(args.project_dir).resolve() / robot_csv).resolve()
    robot_time_table = solver.load_robot_task_time_table(robot_csv)

    cumulative_delay: Dict[int, float] = {}
    histories: Dict[int, List[float]] = {}
    summary_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for event in events:
        step = int(event["step_idx"])
        pre_failure_block = block_for_step(blocks, step - 1)
        post_failure_block = block_for_step(blocks, step)

        # Only the pre-failure block defines the VNS initial order/assignments.
        schedule = rebuild_schedule_from_block(metadata_schedule, pre_failure_block)
        schedule = apply_experiment_standard_times(schedule, standard_lookup)
        schedule, cumulative_delay = apply_failure(schedule, event, cumulative_delay)

        frozen = {int(x) for x in event["frozen_task_ids"]}
        schedule = reorder_schedule_with_frozen(solver, schedule, frozen)
        schedule, fixed_starts, agent_ready, audit = apply_recorded_execution_state(
            schedule,
            event,
            post_failure_block,
            cumulative_delay,
            tolerance=float(args.completion_tolerance),
        )

        start_kit_id = infer_start_kit(solver, schedule, event)
        event_seed = int(args.seed) + step - 1

        best, best_cost, search_time, history, shake = run_vns(
            solver,
            schedule,
            P,
            start_kit_id=start_kit_id,
            max_iter=int(args.max_iter),
            seed=event_seed,
            shaking_threshold=int(args.shaking_threshold),
            frozen_task_ids=frozen,
            decision_time=float(event["decision_time"]),
            fixed_start_by_id=fixed_starts,
            agent_ready_times=agent_ready,
            robot_time_table=robot_time_table,
        )

        if len(history) != int(args.max_iter):
            raise RuntimeError(
                f"Failure {step}: got {len(history)} history points, "
                f"expected {args.max_iter}"
            )

        histories[step] = history
        save_curve(history, output_dir / f"convergence_failure_{step}.png")

        for iteration, value in enumerate(history):
            history_rows.append({
                "step_idx": step,
                "iteration": iteration,
                "best_makespan": value,
            })

        completed = [row for row in audit if row["state"] == "completed"]
        time_origin = [row for row in audit if row["state"] == "completed_before_time_origin"]
        running = [row for row in audit if row["state"] in {"executing", "dispatched"}]
        pending = [row for row in audit if row["state"] == "pending"]

        summary_rows.append({
            "step_idx": step,
            "failure_type": event["failure_type"],
            "decision_time": event["decision_time"],
            "start_kit_id": start_kit_id,
            "event_seed": event_seed,
            "completed_real_count": len(completed),
            "completed_before_time_origin_count": len(time_origin),
            "running_count": len(running),
            "pending_standard_count": len(pending),
            "human_ready_time": agent_ready.get("human", event["decision_time"]),
            "robot_ready_time": agent_ready.get("robot", event["decision_time"]),
            "best_makespan": best_cost,
            "recorded_dynamic_makespan": post_failure_block.makespan,
            "search_time": search_time,
            "history_points": len(history),
            "shake_count": len(shake),
        })

        for row in audit:
            audit_rows.append({
                "step_idx": step,
                "decision_time": event["decision_time"],
                **row,
            })

        print(
            f"failure {step}: best={best_cost:.4f}, "
            f"recorded={post_failure_block.makespan:.4f}, "
            f"completed(real)={len(completed)}, before_origin={len(time_origin)}, "
            f"running={len(running)}, "
            f"pending(standard)={len(pending)}, "
            f"ready(human/robot)={agent_ready['human']:.4f}/{agent_ready['robot']:.4f}"
        )

    save_combined(histories, output_dir / "convergence_all_failures.png")
    write_csv(
        output_dir / "convergence_history.csv",
        history_rows,
        ["step_idx", "iteration", "best_makespan"],
    )
    write_csv(
        output_dir / "convergence_summary.csv",
        summary_rows,
        [
            "step_idx", "failure_type", "decision_time", "start_kit_id",
            "event_seed", "completed_real_count", "completed_before_time_origin_count", "running_count",
            "pending_standard_count", "human_ready_time", "robot_ready_time",
            "best_makespan", "recorded_dynamic_makespan", "search_time",
            "history_points", "shake_count",
        ],
    )
    write_csv(
        output_dir / "task_time_sources.csv",
        audit_rows,
        [
            "step_idx", "decision_time", "task_id", "agent", "state",
            "duration_source", "duration_used", "fixed_start", "recorded_finish",
            "expected_finish_used",
        ],
    )
    standard_rows = [
        {"task_id": tid, "agent": agent, "standard_duration": duration}
        for (tid, agent), duration in sorted(standard_lookup.items())
    ]
    write_csv(
        output_dir / "experiment_standard_times.csv",
        standard_rows,
        ["task_id", "agent", "standard_duration"],
    )

    return {
        "output_dir": str(output_dir),
        "events": len(events),
        "iterations_per_event": int(args.max_iter),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay VNS curves using recorded real completed times and experiment standard times"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--gantt", required=True)
    parser.add_argument("--output-dir", default="convergence_recorded_state_v5")
    parser.add_argument("--project-dir", default="..")
    parser.add_argument(
        "--solver-module",
        default="VNS_dynamic_solver",
        help="Import name when --solver-file is not supplied",
    )
    parser.add_argument(
        "--solver-file",
        default=None,
        help=(
            "Exact solver .py path. Useful for a filename such as "
            "'..\\VNS_dynamic_solver(1).py'."
        ),
    )
    parser.add_argument("--robot-time-csv", default=None)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--shaking-threshold", type=int, default=50)
    parser.add_argument(
        "--completion-tolerance",
        type=float,
        default=0.03,
        help="Tolerance for rounded Gantt finish time versus decision time",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_replay(args)
    print(f"output_dir: {result['output_dir']}")
    print(f"events: {result['events']}")
    print(f"iterations_per_event: {result['iterations_per_event']}")


if __name__ == "__main__":
    main()
