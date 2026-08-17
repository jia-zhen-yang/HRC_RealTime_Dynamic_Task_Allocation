"""
Real-data no-rescheduling counterfactual simulator.

Purpose
-------
This script reads:
  1) dynamic_events_*.csv: failure events exported during the adaptive run
  2) gantt_data_*.txt: raw Gantt blocks from the real/adaptive run

It then simulates the counterfactual case: the same failures occur, but the
system does NOT dynamically reschedule. The counterfactual preserves the
exported offline Gantt sequence in gantt_data_*.txt. It reuses the no-rescheduling execution
logic from failure_simulation.py, but replaces the old manual/std-time failure
parameters with delays inferred from the real Gantt data.

Recommended placement
---------------------
Put this file in the same folder as failure_simulation.py and the other project
modules: read_schedule.py, precedence_matrix.py, VNS_failure_simulation_solver.py,
VNS_verification.py.

Example
-------
python real_data_nores_simulation.py \
  --events dynamic_events_20260618_155540.csv \
  --gantt gantt_data_20260618_155540.txt \
  --failure-module failure_simulation.py \
  --output-dir real_nores_output \
  --seed 43

Main outputs
------------
- comparison_gantt.png: final three-way Gantt comparison figure
- comparison_gantt_data.txt: raw Gantt data for the three panels in comparison_gantt.png
- real_nores_summary.csv: per-failure summary table
- real_event_delay_report.csv: event-duration diagnostic table

Compensation options
--------------------
--compensation-mode fixed  uses --compensation-value, default 3.39 seconds.
--compensation-mode random uses:
  - actual operation time when the agent/order mistake has no delay
  - standard operation time when that task/event is delay-affected.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

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

    @property
    def by_task(self) -> Dict[int, GanttRow]:
        return {row.task_id: row for row in self.rows}

    @property
    def by_task_agent(self) -> Dict[Tuple[int, str], GanttRow]:
        return {(row.task_id, row.agent): row for row in self.rows}


# ---------------------------------------------------------------------------
# Import original simulator module
# ---------------------------------------------------------------------------

def load_failure_module(module_path: Optional[str] = None):
    """Import failure_simulation.py without modifying the original file."""
    if module_path:
        path = Path(module_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"failure module not found: {path}")
        spec = importlib.util.spec_from_file_location("failure_simulation_original", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    # Normal case: this script sits beside failure_simulation.py.
    return importlib.import_module("failure_simulation")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    number = _to_float(value, None)
    if number is None:
        return default
    return int(number)


def parse_int_list(text: Any) -> List[int]:
    if text is None:
        return []
    return [int(x) for x in re.findall(r"-?\d+", str(text))]


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
            event["new_agent"] = (event.get("new_agent") or "").strip() or None
            event["csv_delay_time"] = _to_float(event.get("delay_time"), 0.0) or 0.0
            events.append(event)
    return events


def parse_gantt_txt(path: str | Path) -> List[GanttBlock]:
    """Parse the raw Gantt text exported by failure_simulation.py."""
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

    def flush():
        nonlocal title, rows, makespan
        if title is not None:
            blocks.append(GanttBlock(title=title, rows=rows, makespan=float(makespan or 0.0)))
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
            parts = line.split()
            if parts:
                makespan = float(parts[-1])
            continue

        m = row_re.match(line)
        if m:
            rows.append(
                GanttRow(
                    task_id=int(m.group("task_id")),
                    agent=m.group("agent").lower(),
                    start_time=float(m.group("start")),
                    finish_time=float(m.group("finish")),
                    duration=float(m.group("duration")),
                )
            )

    flush()
    if not blocks:
        raise ValueError(f"No Gantt blocks were parsed from {path}")
    return blocks


def block_for_step(blocks: List[GanttBlock], step_idx: int) -> GanttBlock:
    """
    Convention used by the current exporter:
      blocks[0] = offline optimal schedule
      blocks[1] = dynamic scheduling 1
      blocks[2] = dynamic scheduling 2
      ...
    """
    if step_idx < 0 or step_idx >= len(blocks):
        raise IndexError(f"Cannot find Gantt block for step {step_idx}; parsed {len(blocks)} blocks")
    return blocks[step_idx]


def find_final_dynamic_makespan(blocks: List[GanttBlock]) -> float:
    for block in reversed(blocks):
        if "final" in block.title.lower():
            return block.makespan
    # Fallback: use the last dynamic scheduling block if no final timeline exists.
    return blocks[-1].makespan


# ---------------------------------------------------------------------------
# Actual-duration and actual-delay inference
# ---------------------------------------------------------------------------

def standard_duration_from_task(fs, task: Dict[str, Any]) -> float:
    agent = task.get("agent")
    if agent == "human":
        return float(task.get("human_standard_time", 0.0) or 0.0)
    if agent == "robot":
        return float(task.get("robot_standard_time", 0.0) or 0.0)
    # Last fallback: use original helper if available.
    return float(fs.get_task_duration(task))


def schedule_id_to_index(schedule: List[Dict[str, Any]]) -> Dict[int, int]:
    return {int(task["ID"]): idx for idx, task in enumerate(schedule)}


def first_event_step_by_task(events: List[Dict[str, Any]]) -> Dict[int, int]:
    first: Dict[int, int] = {}
    for event in events:
        involved: List[int] = []
        if event["failure_type"] == "delay" and event.get("t") is not None:
            involved.append(event["t"])
        elif event["failure_type"] in {"order", "agent"}:
            for key in ("t", "t1", "t2"):
                if event.get(key) is not None:
                    involved.append(event[key])
        for tid in involved:
            first[tid] = min(first.get(tid, event["step_idx"]), event["step_idx"])
    return first


def get_final_timeline_block_or_last(blocks: List[GanttBlock]) -> GanttBlock:
    for block in reversed(blocks):
        if "final" in block.title.lower():
            return block
    return blocks[-1]


def build_nominal_actual_duration_lookup(
    blocks: List[GanttBlock],
    events: List[Dict[str, Any]],
    *,
    include_offline_as_fallback: bool = True,
) -> Dict[Tuple[int, str], float]:
    """
    Build task-agent operation-time parameters from real data.

    Rules:
    1. Use the real observed duration for the same task and same agent whenever
       it is available.
    2. If the task is itself involved in a failure, use the latest observation
       strictly before that task's first failure as its normal operation time.
       Otherwise the failure delay would be counted twice when replaying the
       failure event in the counterfactual simulation.
    3. If the same task-agent pair is not observed in the real run, fall back to
       the offline/standard duration.
    """
    first_step = first_event_step_by_task(events)
    final_block = get_final_timeline_block_or_last(blocks)

    final_values: Dict[Tuple[int, str], float] = {
        (row.task_id, row.agent): row.duration for row in final_block.rows
    }
    before_failure_values: Dict[Tuple[int, str], List[Tuple[int, float]]] = {}
    other_real_values: Dict[Tuple[int, str], List[float]] = {}
    offline_values: Dict[Tuple[int, str], float] = {}

    for block_idx, block in enumerate(blocks):
        is_offline = block_idx == 0
        for row in block.rows:
            key = (row.task_id, row.agent)
            if is_offline:
                offline_values.setdefault(key, row.duration)
                continue

            task_first_event_step = first_step.get(row.task_id)
            if task_first_event_step is not None and block_idx < task_first_event_step:
                before_failure_values.setdefault(key, []).append((block_idx, row.duration))
            else:
                other_real_values.setdefault(key, []).append(row.duration)

    keys = set(offline_values) | set(final_values) | set(before_failure_values) | set(other_real_values)
    result: Dict[Tuple[int, str], float] = {}

    for key in keys:
        task_id, _agent = key

        if task_id in first_step:
            if key in before_failure_values:
                # Use the latest pre-failure real duration.
                result[key] = sorted(before_failure_values[key], key=lambda x: x[0])[-1][1]
                continue
            if key in offline_values:
                result[key] = offline_values[key]
                continue

        if key in final_values:
            result[key] = final_values[key]
        elif key in other_real_values:
            result[key] = other_real_values[key][-1]
        elif key in offline_values:
            result[key] = offline_values[key]

    return result


def infer_real_event_context(
    event: Dict[str, Any],
    blocks: List[GanttBlock],
    *,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Infer actual event delay from the adjacent real Gantt blocks."""
    step = int(event["step_idx"])
    before = block_for_step(blocks, step - 1)
    after = block_for_step(blocks, step)
    before_by_task = before.by_task
    after_by_task = after.by_task
    offline_by_task = blocks[0].by_task

    ctx: Dict[str, Any] = {
        "step_idx": step,
        "failure_type": event["failure_type"],
        "dynamic_makespan_after_event": after.makespan,
        "csv_delay_time": event.get("csv_delay_time", 0.0),
    }

    if event["failure_type"] == "delay":
        t = event.get("t")
        if t is None:
            raise ValueError(f"step {step}: delay event has no task t")
        before_d = before_by_task[t].duration
        after_d = after_by_task[t].duration
        standard_d = offline_by_task[t].duration if t in offline_by_task else before_d
        incremental = after_d - before_d
        if incremental < eps:
            # Fallback for cases where the adjacent block does not expose the increment.
            incremental = after_d - standard_d
        if incremental < eps:
            incremental = float(event.get("csv_delay_time", 0.0) or 0.0)
        ctx.update(
            {
                "target_task": t,
                "duration_before": before_d,
                "duration_after": after_d,
                "standard_duration": standard_d,
                "actual_incremental_delay": max(0.0, incremental),
                "actual_delay_vs_standard_after": max(0.0, after_d - standard_d),
            }
        )
        return ctx

    if event["failure_type"] == "order":
        t1, t2 = event.get("t1"), event.get("t2")
        if t1 is None or t2 is None:
            raise ValueError(f"step {step}: order event must have t1 and t2")
        deltas = {}
        positive_delta_sum = 0.0
        released_delay_sum = 0.0
        for tid in (t1, t2):
            bd = before_by_task[tid].duration
            ad = after_by_task[tid].duration
            delta = ad - bd
            deltas[tid] = {"before": bd, "after": ad, "delta": delta}
            if delta > eps:
                positive_delta_sum += delta
            elif delta < -eps:
                released_delay_sum += -delta
        ctx.update(
            {
                "t1": t1,
                "t2": t2,
                "order_duration_deltas": deltas,
                # This is the extra observed event cost, not the entire delayed slot.
                "observed_order_extra_delay": positive_delta_sum,
                # Useful diagnostic for delayed-then-order cases: a previously delayed
                # task may reset while the wrong/early task carries part of the slot time.
                "released_previous_delay": released_delay_sum,
                "actual_incremental_delay": positive_delta_sum,
            }
        )
        return ctx

    if event["failure_type"] == "agent":
        t = event.get("t")
        if t is None:
            raise ValueError(f"step {step}: agent event has no task t")
        before_d = before_by_task[t].duration if t in before_by_task else None
        after_d = after_by_task[t].duration if t in after_by_task else None
        ctx.update(
            {
                "target_task": t,
                "new_agent": event.get("new_agent"),
                "duration_before": before_d,
                "duration_after": after_d,
                "actual_incremental_delay": max(0.0, (after_d or 0.0) - (before_d or 0.0)),
            }
        )
        return ctx

    raise ValueError(f"Unsupported failure_type: {event['failure_type']}")


# ---------------------------------------------------------------------------
# No-rescheduling counterfactual mechanics
# ---------------------------------------------------------------------------

def ensure_actual_times(fs, schedule: List[Dict[str, Any]]) -> None:
    for task in schedule:
        if task.get("actual_time") is None:
            task["actual_time"] = standard_duration_from_task(fs, task)


def add_delay_to_task(task: Dict[str, Any], extra_delay: float) -> None:
    base = task.get("actual_time")
    if base is None:
        base = 0.0
    task["actual_time"] = float(base) + float(extra_delay)


def apply_nominal_actual_times(
    fs,
    schedule: List[Dict[str, Any]],
    nominal_lookup: Dict[Tuple[int, str], float],
) -> List[Dict[str, Any]]:
    """Set each task actual_time to its estimated normal real operation time."""
    result = deepcopy(schedule)
    for task in result:
        tid = int(task["ID"])
        agent = str(task.get("agent"))
        task["actual_time"] = float(nominal_lookup.get((tid, agent), standard_duration_from_task(fs, task)))
    return result


def apply_observed_durations_from_block(
    fs,
    schedule: List[Dict[str, Any]],
    block: GanttBlock,
    *,
    protected_task_ids: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    """Set operation duration from a real/adaptive Gantt block.

    For every task in the no-rescheduling schedule:
      - If the same task+agent is observed in the given real/adaptive block,
        use that duration. This makes delay-task execution time consistent
        between no-rescheduling and dynamic/adaptive timelines.
      - If it is not observed, fall back to the standard duration of the
        no-rescheduling agent.
      - Tasks in protected_task_ids keep their current actual_time. This is used
        for no-rescheduling-only recovery compensation, such as order/agent
        recovery, which should persist after it is sampled.
    """
    protected_task_ids = protected_task_ids or set()
    by_task_agent = block.by_task_agent

    result = deepcopy(schedule)
    for task in result:
        tid = int(task["ID"])
        if tid in protected_task_ids:
            continue
        agent = str(task.get("agent"))
        observed = by_task_agent.get((tid, agent))
        if observed is not None:
            task["actual_time"] = float(observed.duration)
        else:
            task["actual_time"] = standard_duration_from_task(fs, task)
    return result


def find_human_task_before_fail(
    fs,
    base_schedule: List[Dict[str, Any]],
    fail_task_id: int,
    P,
) -> Optional[Dict[str, Any]]:
    """Same intent as the nested helper in the original simulator."""
    id_to_idx = schedule_id_to_index(base_schedule)
    if fail_task_id not in id_to_idx:
        raise ValueError(f"Task ID {fail_task_id} not found")

    start_times = fs.compute_start_times(base_schedule, P, case=3)
    fail_start = start_times[id_to_idx[fail_task_id]]

    best_task = None
    best_start = None
    for idx, task in enumerate(base_schedule):
        if task.get("agent") != "human":
            continue
        s = start_times[idx]
        if s < fail_start and (best_start is None or s > best_start):
            best_task = task
            best_start = s
    return best_task


def sample_recovery_time_from_actual(
    fs,
    task: Dict[str, Any],
    nominal_lookup: Dict[Tuple[int, str], float],
    rng: random.Random,
    *,
    multiplier: float = 2.0,
    minimum: float = 1.0,
) -> float:
    tid = int(task["ID"])
    agent = str(task.get("agent"))
    observed_base = float(nominal_lookup.get((tid, agent), task.get("actual_time") or standard_duration_from_task(fs, task)))
    upper = max(minimum, multiplier * observed_base)
    return round(rng.uniform(minimum, upper), 0)


def get_unified_compensation(
    fs,
    task: Dict[str, Any],
    rng: random.Random,
    *,
    compensation_mode: str = "fixed",
    compensation_value: float = 3.39,
    random_base_duration: Optional[float] = None,
    random_base_source: str = "standard",
) -> Tuple[float, float, str, float, str]:
    """
    Unified compensation rule for both agent and order mistakes.

    fixed:
        compensation = compensation_value

    random:
        compensation ~ Uniform(0, 2 * base_duration)

    base_duration is selected by the failure logic:
      - no delay involved: use the relevant task's real actual duration
      - delay involved: use the task's standard duration to avoid using a
        delay-contaminated duration as the random parameter
    """
    mode = str(compensation_mode).strip().lower()
    if mode == "fixed":
        value = float(compensation_value)
        return value, value, f"fixed_{value:g}", value, "fixed_value"

    if mode == "random":
        if random_base_duration is None or float(random_base_duration) <= 0:
            base = float(standard_duration_from_task(fs, task))
            base_source = "standard_fallback"
        else:
            base = float(random_base_duration)
            base_source = str(random_base_source)
        upper = 2.0 * max(0.0, base)
        value = rng.uniform(0.0, upper)
        return value, upper, f"random_uniform_0_to_2x_{base_source}_{base:g}", base, base_source

    raise ValueError(f"Unknown compensation_mode: {compensation_mode}")


def apply_real_failure_no_rescheduling(
    fs,
    base_schedule: List[Dict[str, Any]],
    event: Dict[str, Any],
    event_ctx: Dict[str, Any],
    P,
    rng: random.Random,
    nominal_lookup: Dict[Tuple[int, str], float],
    *,
    order_policy: str = "sampled_recovery",
    compensation_mode: str = "fixed",
    compensation_value: float = 3.39,
    order_compensation: Optional[float] = None,
    delay_affected_task_ids: Optional[Set[int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply one real-data failure to the no-rescheduling state.

    Updated real-data interpretation:
      - delay: no additional simulated delay is added here. The duration has
        already been synchronized to the real/adaptive Gantt block by
        apply_observed_durations_from_block(). This guarantees that a delayed
        task has the same operation duration in no-rescheduling and dynamic
        timelines.
      - order: preserve the original task order. If the real run did the later
        task too early, the no-rescheduling earlier-task slot absorbs the actual
        duration of that wrong-first task plus the unified compensation:
            duration(earlier/required task)
            = actual_duration(later/wrong-first task) + compensation
        The later task then uses its standard duration.
      - agent: uses the same unified compensation setting as order mistakes.
        fixed mode uses compensation_value; random mode samples
        Uniform(0, 2 * target duration). If no delay is involved, target
        duration is the real actual operation time; if a delay is involved,
        target duration falls back to the standard duration.
    """
    schedule = deepcopy(base_schedule)
    id_to_idx = schedule_id_to_index(schedule)
    ensure_actual_times(fs, schedule)

    failure_type = event["failure_type"]
    if order_compensation is not None:
        # Backward-compatible alias for older commands.
        compensation_value = float(order_compensation)

    delay_affected_task_ids = delay_affected_task_ids or set()

    applied: Dict[str, Any] = {
        "failure_type": failure_type,
        "compensation_mode": compensation_mode,
        "compensation_value": compensation_value,
    }

    if failure_type == "delay":
        t = int(event["t"])
        observed_duration = float(event_ctx.get("duration_after", schedule[id_to_idx[t]].get("actual_time") or 0.0))
        # Do not add delay. The observed block duration is already the real
        # operation time after this delay failure.
        schedule[id_to_idx[t]]["actual_time"] = observed_duration
        applied.update(
            {
                "target_task": t,
                "applied_delay": 0.0,
                "delay_source": "real_gantt_duration_synchronized",
                "observed_duration_after_event": observed_duration,
            }
        )
        return schedule, applied

    if failure_type == "agent":
        t = int(event["t"])
        human_task = find_human_task_before_fail(fs, base_schedule, t, P)
        if human_task is None:
            raise ValueError(f"step {event['step_idx']}: cannot find human task before agent failure task {t}")
        target_id = int(human_task["ID"])
        agent_has_delay = (
            float(event_ctx.get("actual_incremental_delay", 0.0) or 0.0) > 1e-6
            or target_id in delay_affected_task_ids
        )
        if agent_has_delay:
            compensation_base_duration = standard_duration_from_task(fs, human_task)
            compensation_base_source = "standard_due_to_delay"
        else:
            compensation_base_duration = float(human_task.get("actual_time") or standard_duration_from_task(fs, human_task))
            compensation_base_source = "actual_no_delay"

        recovery, compensation_upper, compensation_source, compensation_base_duration, compensation_base_source = get_unified_compensation(
            fs,
            human_task,
            rng,
            compensation_mode=compensation_mode,
            compensation_value=compensation_value,
            random_base_duration=compensation_base_duration,
            random_base_source=compensation_base_source,
        )
        add_delay_to_task(schedule[id_to_idx[target_id]], recovery)
        applied.update(
            {
                "target_task": target_id,
                "failure_task": t,
                "applied_delay": recovery,
                "delay_source": f"agent_compensation_{compensation_source}",
                "compensation_upper": compensation_upper,
                "compensation_base_duration": compensation_base_duration,
                "compensation_base_source": compensation_base_source,
                "agent_has_delay": int(agent_has_delay),
                "protected_task_id": target_id,
            }
        )
        return schedule, applied

    if failure_type == "order":
        t1, t2 = int(event["t1"]), int(event["t2"])
        idx1, idx2 = id_to_idx[t1], id_to_idx[t2]
        earlier_id = t1 if idx1 < idx2 else t2
        later_id = t2 if idx1 < idx2 else t1

        earlier_task = schedule[id_to_idx[earlier_id]]
        later_task = schedule[id_to_idx[later_id]]

        order_deltas = event_ctx.get("order_duration_deltas", {}) or {}

        # actual duration of the required/earlier task after the real order event
        required_actual_duration = None
        if earlier_id in order_deltas:
            required_actual_duration = float(order_deltas[earlier_id].get("after", 0.0) or 0.0)
        if required_actual_duration is None or required_actual_duration <= 0:
            required_actual_duration = float(
                earlier_task.get("actual_time")
                or nominal_lookup.get((earlier_id, earlier_task.get("agent")), 0.0)
                or standard_duration_from_task(fs, earlier_task)
            )

        # actual duration of the task that was wrongly done first in the real run
        wrong_first_actual_duration = None
        if later_id in order_deltas:
            wrong_first_actual_duration = float(order_deltas[later_id].get("after", 0.0) or 0.0)
        if wrong_first_actual_duration is None or wrong_first_actual_duration <= 0:
            wrong_first_actual_duration = float(
                later_task.get("actual_time")
                or nominal_lookup.get((later_id, later_task.get("agent")), 0.0)
                or standard_duration_from_task(fs, later_task)
            )

        if order_policy == "sampled_recovery":
            # Unified final rule:
            #   no-res duration(earlier/required task)
            #     = actual duration(later/wrong-first task) + compensation
            #   no-res duration(later task)
            #     = standard duration(later task)
            #
            # compensation is shared with agent mistakes:
            #   fixed  -> compensation_value, default 3.39
            #   random -> Uniform(0, 2 * selected base duration)
            #
            # Base-duration rule:
            #   no delay involved -> use the actual duration of the required
            #                        earlier task as the random parameter
            #   delay involved    -> use the standard duration as the random
            #                        parameter to avoid delay contamination
            order_has_delay = (
                float(event_ctx.get("actual_incremental_delay", 0.0) or 0.0) > 1e-6
                or earlier_id in delay_affected_task_ids
                or later_id in delay_affected_task_ids
            )
            if order_has_delay:
                compensation_base_duration = standard_duration_from_task(fs, earlier_task)
                compensation_base_source = "standard_due_to_delay"
                later_duration = standard_duration_from_task(fs, later_task)
                later_duration_source = "standard_due_to_delay"
            else:
                compensation_base_duration = required_actual_duration
                compensation_base_source = "actual_no_delay"
                later_duration = wrong_first_actual_duration
                later_duration_source = "actual_no_delay"

            recovery, compensation_upper, compensation_source, compensation_base_duration, compensation_base_source = get_unified_compensation(
                fs,
                earlier_task,
                rng,
                compensation_mode=compensation_mode,
                compensation_value=compensation_value,
                random_base_duration=compensation_base_duration,
                random_base_source=compensation_base_source,
            )

            base_duration = wrong_first_actual_duration
            earlier_task["actual_time"] = wrong_first_actual_duration + recovery
            later_task["actual_time"] = later_duration
            source = f"order_wrong_first_actual_plus_{compensation_source}_on_earlier_task"

        elif order_policy == "sampled":
            # Legacy sensitivity option, but mapped to the corrected earlier-task slot.
            recovery = sample_recovery_time_from_actual(fs, earlier_task, nominal_lookup, rng)
            base_duration = wrong_first_actual_duration
            earlier_task["actual_time"] = wrong_first_actual_duration + recovery
            later_task["actual_time"] = standard_duration_from_task(fs, later_task)
            compensation_upper = ""
            source = "legacy_sampled_mapped_to_earlier_task"

        elif order_policy in {"observed_positive", "transfer_slot"}:
            # Legacy sensitivity options, but mapped to the corrected earlier-task slot.
            positive = float(event_ctx.get("observed_order_extra_delay", 0.0) or 0.0)
            released = float(event_ctx.get("released_previous_delay", 0.0) or 0.0)
            recovery = max(positive, released) if order_policy == "transfer_slot" else positive

            base_duration = wrong_first_actual_duration
            earlier_task["actual_time"] = wrong_first_actual_duration + recovery
            later_task["actual_time"] = standard_duration_from_task(fs, later_task)
            compensation_upper = ""
            source = f"legacy_{order_policy}_mapped_to_earlier_task"

        else:
            raise ValueError(f"Unknown order_policy: {order_policy}")

        applied.update(
            {
                "t1": t1,
                "t2": t2,
                "earlier_task": earlier_id,
                "later_task": later_id,
                "target_task": earlier_id,
                "applied_delay": recovery,
                "delay_source": source,
                "order_base_duration": base_duration,
                "order_required_task_actual_duration": required_actual_duration,
                "order_wrong_first_task_actual_duration": wrong_first_actual_duration,
                "order_later_standard_duration": standard_duration_from_task(fs, later_task),
                "order_later_simulated_duration": later_task.get("actual_time"),
                "order_later_duration_source": locals().get("later_duration_source", ""),
                "order_has_delay": int(locals().get("order_has_delay", 0)),
                "compensation_base_duration": locals().get("compensation_base_duration", ""),
                "compensation_base_source": locals().get("compensation_base_source", ""),
                "order_compensation_upper": compensation_upper,
                "compensation_upper": compensation_upper,
                "protected_task_id": earlier_id,
                "protected_second_task_id": later_id,
                "released_previous_delay": event_ctx.get("released_previous_delay", 0.0),
                "observed_order_extra_delay": event_ctx.get("observed_order_extra_delay", 0.0),
            }
        )
        return schedule, applied

    raise ValueError(f"Unsupported failure_type: {failure_type}")


def gantt_rows_from_execution(
    adjusted_schedule: List[Dict[str, Any]],
    start_times: Iterable[float],
    finish_times: Iterable[float],
) -> List[Dict[str, Any]]:
    rows = []
    for task, start, finish in zip(adjusted_schedule, start_times, finish_times):
        start = float(start)
        finish = float(finish)
        rows.append(
            {
                "task_id": int(task["ID"]),
                "agent": task.get("agent", ""),
                "start_time": start,
                "finish_time": finish,
                "duration": finish - start,
            }
        )
    return rows


def get_offline_start_lookup(
    blocks: List[GanttBlock],
    schedule: List[Dict[str, Any]],
    fs,
    P,
) -> Dict[int, float]:
    """Return the offline-optimized planned start time by task ID.

    Robot planned starts in the no-rescheduling counterfactual are anchored to
    this lookup. Prefer the exported offline optimal Gantt block so the robot
    starts match the experiment's offline plan exactly. Tasks omitted from the
    txt because their duration is 0 are filled from compute_start_times().
    """
    lookup: Dict[int, float] = {}
    if blocks:
        for row in blocks[0].rows:
            lookup[row.task_id] = row.start_time

    try:
        starts = fs.compute_start_times(schedule, P, case=3)
        for task, start in zip(schedule, starts):
            lookup.setdefault(int(task["ID"]), float(start))
    except Exception:
        for task in schedule:
            lookup.setdefault(int(task["ID"]), 0.0)

    return lookup


def get_observed_human_start_lookup(block: GanttBlock) -> Dict[int, float]:
    """Return task_id -> observed human start time from a real/adaptive Gantt block.

    The counterfactual represents the same real operation without dynamic
    replanning, so human task starts are anchored to the observed experiment
    starts whenever the same task was actually performed by the human.
    """
    return {
        row.task_id: row.start_time
        for row in block.rows
        if row.agent.lower() == "human"
    }


def get_observed_human_start_lookup_for_step(
    blocks: List[GanttBlock],
    step_idx: int,
) -> Dict[int, float]:
    """Observed human starts after a specific failure step."""
    return get_observed_human_start_lookup(block_for_step(blocks, step_idx))


def get_final_observed_human_start_lookup(blocks: List[GanttBlock]) -> Dict[int, float]:
    """Observed human starts in the final real/adaptive timeline."""
    return get_observed_human_start_lookup(get_final_timeline_block_or_last(blocks))


def simulate_real_nores_execution(
    fs,
    base_schedule: List[Dict[str, Any]],
    nores_schedule: List[Dict[str, Any]],
    P,
    offline_start_by_id: Dict[int, float],
    *,
    observed_human_start_by_id: Optional[Dict[int, float]] = None,
    robot_place: Optional[float] = None,
    enforce_human_feasibility: bool = True,
) -> Tuple[List[Dict[str, Any]], List[float], List[float]]:
    """
    Real-data no-rescheduling execution simulator.

    Correct no-rescheduling interpretation:
    - Human start times are NOT fixed to the experiment timestamps.
      In the no-rescheduling counterfactual, the human simply keeps executing
      the original no-rescheduling order.
    - Human operation durations still come from real observed durations when
      available, with standard-time fallback.
    - Robot starts are anchored to the offline optimized schedule.
    - Robot starts may be pushed only by robot resource conflict.
    - Robot finishes may be pushed because the last ``robot_place`` seconds
      cannot start before the corresponding kit-box replace is complete.
    - Human movement to later kit tasks is already constrained by P; because P
      contains kit-level precedence, the human naturally waits for robot tasks in
      the same/previous kit before moving on.
    - Robot place-wait is never written back to task["actual_time"]; it is
      recalculated every pass to avoid double counting.
    """
    if robot_place is None:
        robot_place = float(getattr(fs, "ROBOT_PLACE_TIME", 2.0))

    adjusted_schedule = deepcopy(nores_schedule)
    n = len(adjusted_schedule)

    kits = fs.extract_kits(adjusted_schedule)
    task_to_kit: Dict[int, int] = {}
    replace_idx_of_kit: Dict[int, int] = {}
    for kit_id, idxs in kits.items():
        for idx in idxs:
            task_to_kit[idx] = kit_id
            if adjusted_schedule[idx].get("command") == "replace":
                replace_idx_of_kit[kit_id] = idx

    start_times = [0.0] * n
    finish_times = [0.0] * n
    agent_free = {"human": 0.0, "robot": 0.0}

    for j, task in enumerate(adjusted_schedule):
        tid = int(task["ID"])
        agent = task["agent"]
        operation_duration = float(fs.get_task_duration(task))

        preds = [i for i in range(n) if P[i][j] == 1]
        pred_finish = max((finish_times[i] for i in preds), default=0.0)

        if agent == "human":
            # Human keeps working continuously in the no-rescheduling order.
            # No experimental start timestamp is imposed here.
            start = max(agent_free["human"], pred_finish)
            finish = start + operation_duration

        elif agent == "robot":
            # Robot follows the offline plan. It is shifted only if the previous
            # robot task still occupies the robot.
            planned_start = float(offline_start_by_id.get(tid, 0.0))
            start = max(planned_start, agent_free["robot"])
            extra_wait = 0.0

            if task.get("command") == "pick and place":
                kit_id = task_to_kit[j]
                replace_idx = replace_idx_of_kit.get(kit_id)
                if replace_idx is not None:
                    replace_finish = finish_times[replace_idx]
                    place_start_without_wait = start + max(0.0, operation_duration - robot_place)
                    extra_wait = max(0.0, replace_finish - place_start_without_wait)

            finish = start + operation_duration + extra_wait

        else:
            raise ValueError(f"Unknown agent for task {tid}: {agent}")

        start_times[j] = start
        finish_times[j] = finish
        agent_free[agent] = finish

    return adjusted_schedule, start_times, finish_times




def build_task_command_lookup(*schedules: Optional[List[Dict[str, Any]]]) -> Dict[int, str]:
    """Build task_id -> command lookup for drawing txt Gantt rows."""
    lookup: Dict[int, str] = {}
    for schedule in schedules:
        if not schedule:
            continue
        for task in schedule:
            try:
                lookup[int(task["ID"])] = str(task.get("command", ""))
            except Exception:
                continue
    return lookup


def gantt_rows_from_block(block: GanttBlock) -> List[Dict[str, Any]]:
    return [
        {
            "task_id": row.task_id,
            "agent": row.agent,
            "start_time": row.start_time,
            "finish_time": row.finish_time,
            "duration": row.duration,
        }
        for row in block.rows
    ]


def save_three_way_comparison_gantt(
    *,
    offline_rows: List[Dict[str, Any]],
    offline_makespan: float,
    nores_rows: List[Dict[str, Any]],
    nores_makespan: float,
    final_dynamic_rows: List[Dict[str, Any]],
    final_dynamic_makespan: float,
    task_command_lookup: Dict[int, str],
    output_path: str | Path,
) -> str:
    """Save offline vs final no-rescheduling vs final dynamic timeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {"replace": "tab:blue", "pick and place": "tab:orange"}
    panels = [
        ("Offline optimal schedule", offline_rows, offline_makespan),
        ("Final no rescheduling", nores_rows, nores_makespan),
        ("Final dynamic timeline", final_dynamic_rows, final_dynamic_makespan),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=False)
    x_limit = max(1.0, max(offline_makespan, nores_makespan, final_dynamic_makespan) * 1.05)

    for ax, (title, rows, makespan) in zip(axes, panels):
        for row in rows:
            task_id = int(row.get("task_id", 0))
            agent = str(row.get("agent", ""))
            start = float(row.get("start_time", 0.0))
            duration = float(row.get("duration", 0.0))
            if duration <= 0:
                continue

            command = str(row.get("command") or task_command_lookup.get(task_id, ""))
            ax.barh(
                agent,
                duration,
                left=start,
                color=colors.get(command, "gray"),
                edgecolor="black",
            )
            ax.text(
                start + duration / 2.0,
                agent,
                str(task_id),
                va="center",
                ha="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(f"{title} (makespan = {makespan:.2f})")
        ax.set_xlabel("Time")
        ax.set_xlim(0, x_limit)

    axes[0].tick_params(labelbottom=True)
    axes[1].tick_params(labelbottom=True)

    fig.suptitle(
        (
            "Final comparison  |  "
            f"Offline = {offline_makespan:.2f}   |   "
            f"No-rescheduling = {nores_makespan:.2f}   |   "
            f"Dynamic = {final_dynamic_makespan:.2f}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)



def write_gantt_raw_block(f, title: str, rows: List[Dict[str, Any]], makespan: float) -> None:
    f.write(f"title: {title}\n")
    f.write(f"{'task ID':>8}{'agent':>12}{'start_time':>15}{'finish_time':>15}{'duration':>15}\n")
    for row in rows:
        f.write(
            f"{str(row['task_id']):>8}"
            f"{str(row['agent']):>12}"
            f"{float(row['start_time']):>15.2f}"
            f"{float(row['finish_time']):>15.2f}"
            f"{float(row['duration']):>15.2f}\n"
        )
    f.write(f"{'makespan:':>50}{float(makespan):>15.2f}\n\n")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_real_nores_counterfactual(
    *,
    events_csv: str | Path,
    gantt_txt: str | Path,
    output_dir: str | Path,
    failure_module_path: Optional[str] = None,
    seed: int = 43,
    order_policy: str = "sampled_recovery",
    compensation_mode: str = "fixed",
    compensation_value: float = 3.39,
    order_compensation: Optional[float] = None,
) -> Dict[str, Any]:
    fs = load_failure_module(failure_module_path)
    events = parse_events_csv(events_csv)
    blocks = parse_gantt_txt(gantt_txt)
    nominal_lookup = build_nominal_actual_duration_lookup(blocks, events)
    event_contexts = [infer_real_event_context(event, blocks) for event in events]

    # Build the no-rescheduling base schedule from the exported offline Gantt block.
    #
    # Important:
    # The counterfactual "no rescheduling" must preserve the actual offline
    # sequence shown in gantt_data_*.txt. Do NOT call optimal_solution() again
    # here, because ties or solver nondeterminism can return another equivalent
    # optimal order, e.g. 22 -> 23 instead of the exported 23 -> 22.
    schedule = fs.load_schedule_csv()
    meta_by_id = {int(task["ID"]): task for task in schedule}

    offline_ids = [row.task_id for row in blocks[0].rows]
    offline_agent_by_id = {row.task_id: row.agent for row in blocks[0].rows}

    ordered_ids = []
    if 1 in meta_by_id and 1 not in offline_ids:
        ordered_ids.append(1)
    ordered_ids.extend([tid for tid in offline_ids if tid in meta_by_id])

    best_schedule = []
    for tid in ordered_ids:
        task = deepcopy(meta_by_id[tid])
        if tid in offline_agent_by_id:
            task["agent"] = offline_agent_by_id[tid]
        task["actual_time"] = None
        best_schedule.append(task)

    P = fs.build_precedence_matrix(best_schedule)
    best_cost = blocks[0].makespan

    # Current no-rescheduling state keeps the exported offline order/agent assignment.
    # Operation times are synchronized from the real Gantt block at each step.
    protected_task_ids: set[int] = set()
    delay_affected_task_ids: set[int] = set()
    nores_current = apply_observed_durations_from_block(
        fs,
        best_schedule,
        blocks[0],
        protected_task_ids=protected_task_ids,
    )
    offline_start_by_id = get_offline_start_lookup(blocks, best_schedule, fs, P)
    if order_compensation is not None:
        # Backward-compatible alias for older commands.
        compensation_value = float(order_compensation)

    rng = random.Random(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    gantt_history: List[Tuple[str, List[Dict[str, Any]], float]] = []

    # Initial counterfactual schedule with real-data operation durations.
    # Robot waiting caused by replace completion is calculated in the execution
    # layer and is not written back to actual_time.
    initial_adjusted, initial_start, initial_finish = simulate_real_nores_execution(
        fs,
        base_schedule=nores_current,
        nores_schedule=nores_current,
        P=P,
        offline_start_by_id=offline_start_by_id,
        observed_human_start_by_id=get_observed_human_start_lookup(blocks[0]),
    )
    initial_rows = gantt_rows_from_execution(initial_adjusted, initial_start, initial_finish)
    initial_makespan = max(initial_finish) if initial_finish else 0.0
    gantt_history.append(("Real-data no rescheduling baseline", initial_rows, initial_makespan))

    for event, ctx in zip(events, event_contexts):
        step = int(event["step_idx"])

        # Synchronize operation times to the real/adaptive state after this
        # failure step. This makes normal/delay execution durations match the
        # experiment; only protected no-rescheduling recovery compensations are
        # kept from previous steps.
        nores_base = apply_observed_durations_from_block(
            fs,
            nores_current,
            block_for_step(blocks, step),
            protected_task_ids=protected_task_ids,
        )

        failed_nores, applied = apply_real_failure_no_rescheduling(
            fs,
            base_schedule=nores_base,
            event=event,
            event_ctx=ctx,
            P=P,
            rng=rng,
            nominal_lookup=nominal_lookup,
            order_policy=order_policy,
            compensation_mode=compensation_mode,
            compensation_value=compensation_value,
            delay_affected_task_ids=delay_affected_task_ids,
        )

        adjusted_schedule, start_times, finish_times = simulate_real_nores_execution(
            fs,
            base_schedule=nores_base,
            nores_schedule=failed_nores,
            P=P,
            offline_start_by_id=offline_start_by_id,
            observed_human_start_by_id=get_observed_human_start_lookup_for_step(blocks, step),
        )
        makespan = max(finish_times) if finish_times else 0.0
        rows = gantt_rows_from_execution(adjusted_schedule, start_times, finish_times)
        gantt_history.append((f"Real-data no rescheduling after Failure {step}", rows, makespan))

        protected_id = applied.get("protected_task_id")
        if protected_id not in (None, ""):
            protected_task_ids.add(int(protected_id))
        protected_second_id = applied.get("protected_second_task_id")
        if protected_second_id not in (None, ""):
            protected_task_ids.add(int(protected_second_id))

        # Remember tasks with explicit delay failures. Later order/agent mistakes
        # involving these tasks use standard duration as the random-compensation
        # parameter to avoid delay-contaminated actual times.
        if event["failure_type"] == "delay" and event.get("t") not in (None, ""):
            delay_affected_task_ids.add(int(event["t"]))

        # Carry operation-time state and sampled recovery compensation forward.
        # Robot place-wait time is still not carried; it is recomputed in the
        # execution layer each pass.
        nores_current = deepcopy(failed_nores)

        dynamic_makespan = float(ctx.get("dynamic_makespan_after_event", 0.0) or 0.0)
        summary = {
            "step_idx": step,
            "failure_type": event["failure_type"],
            "decision_time": event.get("decision_time", 0.0),
            "csv_delay_time": event.get("csv_delay_time", 0.0),
            "actual_incremental_delay": ctx.get("actual_incremental_delay", ""),
            "applied_nores_delay": applied.get("applied_delay", ""),
            "delay_source": applied.get("delay_source", ""),
            "nores_makespan": makespan,
            "dynamic_makespan_after_event": dynamic_makespan,
            "nores_minus_dynamic": makespan - dynamic_makespan,
            "target_task": applied.get("target_task", ctx.get("target_task", "")),
            "t1": event.get("t1") or "",
            "t2": event.get("t2") or "",
            "earlier_task": applied.get("earlier_task", ""),
            "later_task": applied.get("later_task", ""),
            "order_base_duration": applied.get("order_base_duration", ""),
            "order_required_task_actual_duration": applied.get("order_required_task_actual_duration", ""),
            "order_wrong_first_task_actual_duration": applied.get("order_wrong_first_task_actual_duration", ""),
            "order_later_standard_duration": applied.get("order_later_standard_duration", ""),
            "order_later_simulated_duration": applied.get("order_later_simulated_duration", ""),
            "order_later_duration_source": applied.get("order_later_duration_source", ""),
            "order_has_delay": applied.get("order_has_delay", ""),
            "compensation_mode": applied.get("compensation_mode", ""),
            "compensation_value": applied.get("compensation_value", ""),
            "applied_compensation": applied.get("applied_delay", "") if event["failure_type"] in {"order", "agent"} else "",
            "compensation_base_duration": applied.get("compensation_base_duration", ""),
            "compensation_base_source": applied.get("compensation_base_source", ""),
            "compensation_upper": applied.get("compensation_upper", ""),
            "order_fixed_compensation": applied.get("applied_delay", "") if event["failure_type"] == "order" else "",
            "order_compensation_upper": applied.get("order_compensation_upper", ""),
            "released_previous_delay": applied.get("released_previous_delay", ""),
            "observed_order_extra_delay": applied.get("observed_order_extra_delay", ""),
        }
        summary_rows.append(summary)

    # Also compare against the final real/adaptive timeline. The final
    # no-rescheduling timeline uses continuous human execution, while robot
    # starts still follow the offline optimized plan.
    final_dynamic_block = get_final_timeline_block_or_last(blocks)
    final_dynamic_makespan = final_dynamic_block.makespan
    final_dynamic_rows = gantt_rows_from_block(final_dynamic_block)

    final_nores_current = apply_observed_durations_from_block(
        fs,
        nores_current,
        final_dynamic_block,
        protected_task_ids=protected_task_ids,
    )
    final_adjusted_schedule, final_start_times, final_finish_times = simulate_real_nores_execution(
        fs,
        base_schedule=final_nores_current,
        nores_schedule=final_nores_current,
        P=P,
        offline_start_by_id=offline_start_by_id,
        observed_human_start_by_id=get_final_observed_human_start_lookup(blocks),
    )
    final_nores_rows = gantt_rows_from_execution(final_adjusted_schedule, final_start_times, final_finish_times)
    final_nores_makespan = max(final_finish_times) if final_finish_times else initial_makespan
    gantt_history.append(("Final real-data no rescheduling timeline", final_nores_rows, final_nores_makespan))

    offline_rows = gantt_rows_from_block(blocks[0])
    offline_makespan = blocks[0].makespan
    task_command_lookup = build_task_command_lookup(schedule, best_schedule, final_adjusted_schedule)
    final_comparison_png = save_three_way_comparison_gantt(
        offline_rows=offline_rows,
        offline_makespan=offline_makespan,
        nores_rows=final_nores_rows,
        nores_makespan=final_nores_makespan,
        final_dynamic_rows=final_dynamic_rows,
        final_dynamic_makespan=final_dynamic_makespan,
        task_command_lookup=task_command_lookup,
        output_path=output_dir / "comparison_gantt.png",
    )

    summary_csv = output_dir / "real_nores_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["nores_makespan"]
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    comparison_gantt_data = output_dir / "comparison_gantt_data.txt"
    with comparison_gantt_data.open("w", encoding="utf-8-sig") as f:
        write_gantt_raw_block(f, "Offline optimal schedule", offline_rows, offline_makespan)
        write_gantt_raw_block(f, "Final no rescheduling", final_nores_rows, final_nores_makespan)
        write_gantt_raw_block(f, "Final dynamic timeline", final_dynamic_rows, final_dynamic_makespan)

    event_delay_report = output_dir / "real_event_delay_report.csv"
    report_fields = [
        "step_idx",
        "failure_type",
        "csv_delay_time",
        "actual_incremental_delay",
        "dynamic_makespan_after_event",
        "target_task",
        "duration_before",
        "duration_after",
        "standard_duration",
        "actual_delay_vs_standard_after",
        "t1",
        "t2",
        "observed_order_extra_delay",
        "released_previous_delay",
    ]
    with event_delay_report.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report_fields)
        writer.writeheader()
        for ctx in event_contexts:
            writer.writerow({key: ctx.get(key, "") for key in report_fields})

    return {
        "summary_csv": str(summary_csv),
        "comparison_gantt_data": str(comparison_gantt_data),
        "gantt_txt": str(comparison_gantt_data),
        "event_delay_report": str(event_delay_report),
        "comparison_gantt_png": str(final_comparison_png),
        "final_comparison_png": str(final_comparison_png),
        "offline_makespan": offline_makespan,
        "initial_offline_best_cost": best_cost,
        "initial_real_nores_baseline_makespan": initial_makespan,
        "final_nores_makespan": final_nores_makespan,
        "final_dynamic_makespan": final_dynamic_makespan,
        "final_nores_minus_dynamic": final_nores_makespan - final_dynamic_makespan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data no-rescheduling counterfactual simulator")
    parser.add_argument("--events", required=True, help="dynamic_events_*.csv path")
    parser.add_argument("--gantt", required=True, help="gantt_data_*.txt path")
    parser.add_argument("--output-dir", default="real_nores_output", help="output directory")
    parser.add_argument("--failure-module", default=None, help="path to failure_simulation.py; optional if importable")
    parser.add_argument("--seed", type=int, default=43, help="random seed for random compensation mode")
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
        help=(
            "How to convert order mistakes into no-rescheduling recovery time. "
            "sampled_recovery maps order mistakes to earlier task slot: wrong-first actual duration + unified compensation, while later task uses standard duration; "
            "observed_positive/transfer_slot/sample are legacy sensitivity options."
        ),
    )
    args = parser.parse_args()

    result = run_real_nores_counterfactual(
        events_csv=args.events,
        gantt_txt=args.gantt,
        output_dir=args.output_dir,
        failure_module_path=args.failure_module,
        seed=args.seed,
        order_policy=args.order_policy,
        compensation_mode=args.compensation_mode,
        compensation_value=args.compensation_value,
    )

    print("[REAL NO-RES COUNTERFACTUAL DONE]")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
