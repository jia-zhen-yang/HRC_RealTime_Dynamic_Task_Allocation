"""
HRC 作業監控用的 VNS 動態排程橋接模組

本模組負責：
1. 將監控端目前狀態轉成 VNS solver 限制條件
2. 每次動態排程後存下 3 張甘特圖：
   - 本輪重排前的最佳化排程
   - 失效發生當下的目前狀態
   - 本輪動態排程結果
3. 提供最後總覽圖：Planned timeline + 每次 Dynamic result + final timeline，可用 Tkinter 捲動查看 
"""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime #產生圖片檔名的時間戳
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import VNS_dynamic_solver as vns
from VNS_dynamic_solver import (
    compute_start_times,
    extract_kits,
    get_task_duration,
    solver,
    load_robot_task_time_table,
)

ROBOT_PLACE_TIME = vns.ROBOT_PLACE_TIME

AGENT_Y = {"human": 0, "robot": 1}
COLORS = {"replace": "tab:blue", "pick and place": "tab:orange"}


def get_id_to_index(schedule: Sequence[dict]) -> Dict[int, int]:
    # 建立task ID 對應 schedule index的查表
    return {task["ID"]: i for i, task in enumerate(schedule)}


def task_to_kit_map(schedule: Sequence[dict]) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """
    建立兩個查表工具:
    1. 某個 task ID 屬於哪個 kit
    2. 某個 kit 裡有哪些 task
    """
    kits = extract_kits(schedule)
    task_to_kit: Dict[int, int] = {}
    kit_to_task_ids: Dict[int, List[int]] = {}

    for kit_id, idxs in kits.items():
        ids: List[int] = []
        for idx in idxs:
            tid = schedule[idx].get("ID")
            if tid is None:
                continue
            task_to_kit[tid] = kit_id
            ids.append(tid)
        kit_to_task_ids[kit_id] = ids
    return task_to_kit, kit_to_task_ids


def infer_start_kit(schedule: Sequence[dict], task_ids: Iterable[int]) -> int:
    """動態重排從受失效影響最早的 kit 開始"""
    task_to_kit, _ = task_to_kit_map(schedule)
    kit_ids = [task_to_kit[tid] for tid in task_ids if tid in task_to_kit]
    return min(kit_ids) if kit_ids else 1


def _standard_duration(task: dict) -> float:
    # 根據 task 的 agent 回傳標準作業時間
    if task.get("agent") == "robot":
        return float(task.get("robot_standard_time", 0.0) or 0.0)
    return float(task.get("human_standard_time", 0.0) or 0.0)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_zero(value) -> bool:
    try:
        return value is not None and float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _fallback_start(task: dict, default: float = 0.0) -> float:
    """沒有actual_start 時，退回 planned_start"""
    if task.get("actual_start_time") is not None:
        return _safe_float(task.get("actual_start_time"), default)
    if task.get("planned_start_time") is not None:
        return _safe_float(task.get("planned_start_time"), default)
    return float(default)


def _find_replace_idx_for_task(schedule: Sequence[dict], task_idx: int) -> Optional[int]:
    """
    找出某個 pick&place task 所屬 kit 的 replace index。
    規則：往前找最近的 replace。
    若找不到，代表第一個 kit，回傳 None。
    """
    for i in range(task_idx - 1, -1, -1):
        if schedule[i].get("command") == "replace":
            return i
    return None


def prepare_runtime_constraints(
    schedule: Sequence[dict],
    now_from_start: float,
) -> Tuple[List[dict], Set[int], Dict[int, float], Dict[str, float]]:
    """
    動態排程前的資料整理與安全檢查函式
    判斷並記錄 frozen task
    計算 human / robot 下一次可用時間


    規則：
    - status == completed 或 actual_time 已存在：視為已完成，加入 frozen
    - status == dispatched：視為正在執行，加入 frozen，用 _duration_override 表示預期完成時間
    - pending 且無 actual_time：可重新分配與排序，清除暫存執行資訊

    回傳：
    - prepared_schedule: 整理後可以丟進 VNS solver 的 schedule 副本
    - frozen_task_ids: 已完成與正在執行中的任務，不允許重新分配或排序
    - fixed_start_by_id: frozen 任務固定開始時間
    - agent_ready_times: human / robot 下一次可用時間
    """
    prepared_schedule = deepcopy(schedule)
    frozen_task_ids: Set[int] = set()
    fixed_start_by_id: Dict[int, float] = {}

    human_ready = float(now_from_start)
    robot_ready = float(now_from_start)

    for task_idx, task in enumerate(prepared_schedule):
        tid = task.get("ID")
        if tid is None:
            continue

        status = task.get("status", "pending")
        agent = task.get("agent")
        actual_time = task.get("actual_time")
        actual_start = task.get("actual_start_time")
        actual_finish = task.get("actual_finish_time")

        # 正在執行中的任務資訊
        existing_duration_override = task.get("_duration_override")
        existing_expected_finish = task.get("_expected_finish_time")

        # 已完成任務就不再進入 VNS 
        # dispatched 和 executing不能因為曾有估計 duration 而被誤判為完成，所以先排除
        if status == "completed" or (
            actual_time is not None and status not in {"dispatched", "executing"}
        ):
            task["status"] = "completed"
            frozen_task_ids.add(tid)

            task.pop("_duration_override", None)
            task.pop("_expected_finish_time", None)
            task.pop("_place_wait_compensation", None)

            start_t = _fallback_start(task, default=0.0)
            fixed_start_by_id[tid] = start_t

            if actual_time is None:
                if actual_start is not None and actual_finish is not None:
                    task["actual_time"] = max(0.0, _safe_float(actual_finish) - _safe_float(actual_start))
                else:
                    task["actual_time"] = _standard_duration(task)

            if task.get("actual_start_time") is None:
                task["actual_start_time"] = start_t
            if task.get("actual_finish_time") is None and task.get("actual_time") is not None:
                task["actual_finish_time"] = start_t + _safe_float(task.get("actual_time"), 0.0)

            finish_t = _safe_float(task.get("actual_finish_time"), start_t + _safe_float(task.get("actual_time"), 0.0))
            if agent == "human":
                human_ready = max(human_ready, finish_t)
            elif agent == "robot":
                robot_ready = max(robot_ready, finish_t)
            continue

        if status in {"dispatched", "executing"}:
            frozen_task_ids.add(tid)
            start_t = _fallback_start(task, default=now_from_start)
            fixed_start_by_id[tid] = start_t

            # 優先使用主程式已寫入的 _duration_override
            if existing_duration_override is not None:
                duration = _safe_float(existing_duration_override, _standard_duration(task))
                expected_finish = start_t + duration
                expected_finish = max(float(now_from_start), expected_finish)
                duration = max(0.0, expected_finish - start_t)

            # 若沒有 duration override，但有 expected finish，就反推 duration
            elif existing_expected_finish is not None:
                expected_finish = max(
                    float(now_from_start),
                    _safe_float(existing_expected_finish, float(now_from_start)),
                )
                duration = max(0.0, expected_finish - start_t)

            # 一般 dispatched robot 或 executing task，使用標準時間估計
            else:
                base_duration = _standard_duration(task)
                expected_finish = max(float(now_from_start), start_t + base_duration)
                duration = max(0.0, expected_finish - start_t)

            task["_duration_override"] = duration
            task["_expected_finish_time"] = expected_finish

            # robot place wait compensation 只對 robot dispatched 有意義
            # human delay task 保留為 0，不影響排程
            if status == "dispatched" and agent == "robot":
                task["_place_wait_compensation"] = task.get("_place_wait_compensation", 0.0)
            else:
                task.pop("_place_wait_compensation", None)

            # 正在執行但尚未完成，不能寫 actual_time
            task["actual_time"] = None
            task["actual_start_time"] = start_t
            task["actual_finish_time"] = None

            # 更新 agent 下一次可用時間
            # human delay task 會讓 human_ready 延到 expected_finish。
            if agent == "human":
                human_ready = max(human_ready, expected_finish)
            elif agent == "robot":
                robot_ready = max(robot_ready, expected_finish)

            continue

        # 尚未開始：可交給 VNS 重新排程 
        task["status"] = "pending"
        task["actual_start_time"] = None
        task["actual_finish_time"] = None
        task["actual_time"] = None
        task.pop("_duration_override", None)
        task.pop("_expected_finish_time", None)
        task.pop("_place_wait_compensation", None)

    agent_ready_times = {"human": human_ready, "robot": robot_ready}
    return prepared_schedule, frozen_task_ids, fixed_start_by_id, agent_ready_times


def apply_dispatched_task_compensation(schedule: List[dict]) -> List[dict]:
    """
    在 dynamic_schedule 已經有 planned_start_time / planned_finish_time 後，
    重新修正正在執行中的 robot pick&place 任務。

    若 robot 的 place_start 早於對應 kit replace 的完成時間，
    則增加 _duration_override。
    """
    compensation_log = []

    for task_idx, task in enumerate(schedule):
        if task.get("status") != "dispatched":
            continue

        if task.get("agent") != "robot":
            continue

        if task.get("command") != "pick and place":
            continue

        replace_idx = _find_replace_idx_for_task(schedule, task_idx)
        if replace_idx is None:
            continue

        replace_task = schedule[replace_idx]

        replace_finish = replace_task.get("actual_finish_time")
        if replace_finish is None:
            replace_finish = replace_task.get("planned_finish_time")

        if replace_finish is None:
            continue

        start_t = task.get("actual_start_time")
        if start_t is None:
            start_t = task.get("planned_start_time")

        if start_t is None:
            continue

        start_t = _safe_float(start_t)
        replace_finish = _safe_float(replace_finish)

        base_duration = _standard_duration(task)

        place_start = start_t + max(0.0, base_duration - ROBOT_PLACE_TIME)
        extra_wait = max(0.0, replace_finish - place_start)

        adjusted_duration = base_duration + extra_wait
        expected_finish = start_t + adjusted_duration

        if extra_wait <= 0:
            continue

        task["_duration_override"] = adjusted_duration
        task["_expected_finish_time"] = expected_finish
        task["_place_wait_compensation"] = extra_wait

        compensation_log.append({
            "task_id": task.get("ID"),
            "replace_id": replace_task.get("ID"),
            "start": start_t,
            "base_duration": base_duration,
            "replace_finish": replace_finish,
            "place_start": place_start,
            "extra_wait": extra_wait,
            "expected_finish": expected_finish,
        })

    # if compensation_log:
    #     print("[ROBOT PLACE COMPENSATION]")
    #     for item in compensation_log:
    #         print(item)

    return compensation_log

def apply_planned_times(
    schedule: List[dict],
    P,
    *, # 後面的參數，呼叫時必須用「參數名稱」指定
    frozen_task_ids: Optional[Set[int]] = None,
    decision_time: float = 0.0,
    fixed_start_by_id: Optional[Dict[int, float]] = None,
    agent_ready_times: Optional[Dict[str, float]] = None,
) -> None:
    """把 solver 計算完的 planned_start_time / planned_finish_time 回寫到 schedule """
    start_times = compute_start_times(
        schedule,
        P,
        case=3,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
        fixed_start_by_id=fixed_start_by_id,
        agent_ready_times=agent_ready_times,
    )

    for i, task in enumerate(schedule):
        dur = get_task_duration(task)
        task["planned_start_time"] = float(start_times[i])
        task["planned_finish_time"] = float(start_times[i] + dur)
        task.setdefault("actual_start_time", None)
        task.setdefault("actual_finish_time", None)
        task.setdefault("actual_time", None)
        task.setdefault("ur_dispatched", False)
        task.setdefault("status", "pending")


def format_failure_title(failure_info: Optional[dict]) -> str:
    """將 failure_info 轉成圖表 title 使用的失效名稱"""
    if not failure_info:
        return "Failure"
    mapping = {
        "order": "Order mistake",
        "order_mistake": "Order mistake",
        "agent": "Agent mistake",
        "agent_mistake": "Agent mistake",
        "delay": "Delay",
        "wrong_zone": "Wrong zone",
        "opp_complete": "Opportunistic completion",
    }
    ftype = failure_info.get("type", "failure")
    return mapping.get(ftype, str(ftype).replace("_", " ").title())


def build_failure_moment(schedule: Sequence[dict], now_from_start: float) -> List[dict]:
    """
    建立失效當下的甘特圖

    保留：
    - 已完成任務
    - 已有 actual_time 的任務
    - 正在執行 / dispatched 的任務
    不保留後面尚未完成的 pending 任務
    """
    snapshot: List[dict] = []

    # for task in schedule:
    #     status = task.get("status")
    #     if status == "completed" or task.get("actual_time") is not None or status == "dispatched":
    #         snapshot.append(deepcopy(task))
    # return snapshot

    for task in schedule:
        status = task.get("status")

        if status == "completed" or task.get("actual_time") is not None:
            snapshot.append(deepcopy(task))

        elif status in {"dispatched", "executing"}:
            t = deepcopy(task)
            start_t = _fallback_start(t, default=now_from_start)

            t["_draw_until_now"] = True
            t["_draw_now_time"] = float(now_from_start)
            t["_duration_override"] = max(0.0, float(now_from_start) - start_t)

            snapshot.append(t)

    return snapshot


def _bar_values(task: dict, actual_only: bool = False):
    """回傳繪圖用 (start, duration)，依 task 狀態選擇 actual / expected / planned"""
    if _is_zero(task.get("actual_time")):
        return None

    status = task.get("status", "pending")

    if task.get("_draw_until_now"):
        start = _fallback_start(task, default=0.0)
        now_t = _safe_float(task.get("_draw_now_time"), start)
        dur = max(0.0, now_t - start)
        return start, dur

    if status == "completed" or task.get("actual_time") is not None:
        start = _fallback_start(task, default=0.0)
        if task.get("actual_time") is not None:
            dur = _safe_float(task.get("actual_time"), 0.0)
        elif task.get("actual_start_time") is not None and task.get("actual_finish_time") is not None:
            dur = max(0.0, _safe_float(task.get("actual_finish_time")) - _safe_float(task.get("actual_start_time")))
        else:
            dur = _standard_duration(task)
        return start, dur

    if status in {"dispatched", "executing"} or task.get("_duration_override") is not None:
        start = _fallback_start(task, default=0.0)
        if task.get("_duration_override") is not None:
            dur = _safe_float(task.get("_duration_override"), _standard_duration(task))
        elif task.get("_expected_finish_time") is not None:
            dur = max(0.0, _safe_float(task.get("_expected_finish_time")) - start)
        else:
            dur = _standard_duration(task)
        return start, dur

    # pending：使用目前最佳化排程的 planned time
    if actual_only:
        return None
    if task.get("planned_start_time") is not None and task.get("planned_finish_time") is not None:
        start = _safe_float(task.get("planned_start_time"), 0.0)
        dur = max(0.0, _safe_float(task.get("planned_finish_time")) - start)
        return start, dur

    return None


def export_monitor_timeline_rows(
    schedule: Sequence[dict],
    *,
    actual_only: bool = False,
) -> Tuple[List[dict], float]:
    """
    將監控甘特圖資料轉成 raw data rows
    """
    rows: List[dict] = []
    makespan = 0.0

    for task_order_index, task in enumerate(schedule):
        values = _bar_values(task, actual_only=actual_only)

        if values is None:
            continue

        start, duration = values

        if duration <= 0:
            continue

        finish = start + duration
        makespan = max(makespan, finish)

        rows.append({
            "task_id": task.get("ID", ""),
            "agent": task.get("agent", ""),
            "start_time": start,
            "finish_time": finish,
            "duration": duration,
        })

    return rows, makespan


def max_finish_time_by_fields(schedule: Sequence[dict]) -> float:
    """找出某一份 schedule 最晚結束時間"""
    max_finish = 0.0
    for task in schedule:
        values = _bar_values(task)
        if values is None:
            continue
        start, dur = values
        if dur <= 0:
            continue
        max_finish = max(max_finish, start + dur)
    return max_finish


def compute_time_limit(*candidate_values, margin_ratio: float = 0.05):
    """把多份 schedule 的最晚結束時間拿來比較，決定共同 x 軸上限"""
    numeric_candidates = [float(v) for v in candidate_values if v is not None]
    if not numeric_candidates:
        return None
    base = max(numeric_candidates)
    return base + max(1.0, base * margin_ratio)


def apply_time_scale(ax, time_limit):
    """把這個共同 x 軸上限套到 matplotlib 的圖上"""
    if time_limit is not None:
        ax.set_xlim(0, max(1.0, float(time_limit)))


def draw_monitor_timeline(schedule: Sequence[dict], title: str, ax=None, show: bool = True, time_limit=None, actual_only=False):
    """
    監控用甘特圖：
    - completed：actual_start_time + actual_time
    - dispatched：actual_start_time + 預期 duration
    - executing：actual_start_time + 預期 duration
    - pending：planned_start_time + planned duration
    - actual_time == 0：不畫
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 2.5))
        created_fig = True

    for task in schedule:
        agent = task.get("agent")
        y = AGENT_Y.get(agent)
        if y is None:
            continue

        values = _bar_values(task, actual_only=actual_only)
        if values is None:
            continue
        start, dur = values
        if dur <= 0:
            continue

        ax.barh(
            y,
            dur,
            left=start,
            height=0.55,
            color="green" if task.get("_draw_until_now") else COLORS.get(task.get("command"), "gray"),
            edgecolor="black",
        )
        ax.text(
            start + dur / 2,
            y,
            str(task.get("ID", "")),
            va="center",
            ha="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["human", "robot"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    apply_time_scale(ax, time_limit)

    if created_fig:
        plt.tight_layout()
        if show:
            plt.show()


def save_reschedule_gantt(
    *,
    optimized_schedule: Sequence[dict],
    failure_snapshot_schedule: Sequence[dict],
    dynamic_schedule: Sequence[dict],
    output_dir: str,
    step_idx: int,
    failure_info: dict,
) -> str:
    """
    每次動態排程存三張圖：
    1. 本輪最佳化排程：第一次是離線最佳，之後是前一次動態排程結果
    2. 重新排程前的目前情況：只畫 completed / executing
    3. 本輪動態排程結果
    """
    os.makedirs(output_dir, exist_ok=True)

    failure_title = format_failure_title(failure_info)
    time_limit = compute_time_limit(
        max_finish_time_by_fields(optimized_schedule),
        max_finish_time_by_fields(failure_snapshot_schedule),
        max_finish_time_by_fields(dynamic_schedule),
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 7.5), sharex=False)

    draw_monitor_timeline(
        optimized_schedule,
        title=f"Optimal schedule before Failure {step_idx}",
        ax=axes[0],
        show=False,
        time_limit=time_limit,
    )
    draw_monitor_timeline(
        failure_snapshot_schedule,
        title=f"Failure {step_idx} - {failure_title}",
        ax=axes[1],
        show=False,
        time_limit=time_limit,
    )
    draw_monitor_timeline(
        dynamic_schedule,
        title=f"Failure {step_idx} - Dynamic rescheduling result",
        ax=axes[2],
        show=False,
        time_limit=time_limit,
    )

    axes[0].tick_params(labelbottom=True)
    axes[1].tick_params(labelbottom=True)
    axes[2].set_xlabel("Time")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in failure_title.lower().replace(" ", "_"))
    path = os.path.join(output_dir, f"failure_{step_idx:03d}_{timestamp}_{safe_label}.png")

    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

def reorder_schedule_with_frozen(schedule, frozen_task_ids):
    """
    將 frozen 任務移到前面，其餘任務保持相對順序
    在同 kit 內把 frozen 任務排到該 kit 的前面
    """
    kits = extract_kits(schedule)
    reordered = []

    for kit_id in sorted(kits.keys()):
        kit_tasks = [schedule[idx] for idx in kits[kit_id]]
        
        replace_task = None
        for task in kit_tasks:
            if task["command"] == "replace":
                replace_task = task
                break
        
        if replace_task is not None:
            reordered.append(replace_task)
            kit_tasks = [t for t in kit_tasks if t["ID"] != replace_task["ID"]]
        frozen_tasks = [task for task in kit_tasks if task["ID"] in frozen_task_ids]
        remaining_tasks = [task for task in kit_tasks if task["ID"] not in frozen_task_ids]

        reordered.extend(frozen_tasks + remaining_tasks)

    return reordered


def recompute_agent_ready_times(schedule: Sequence[dict], now_from_start: float) -> Dict[str, float]:
    """
    根據目前 schedule 重新估計 human / robot 下一次可用時間

    - completed：使用 actual_finish_time，或 actual_start_time + actual_time
    - dispatched / executing：使用 _expected_finish_time，
      若沒有則用 actual_start_time + _duration_override / standard duration
    - pending：不影響 ready time
    """
    ready = {
        "human": float(now_from_start),
        "robot": float(now_from_start),
    }

    for task in schedule:
        agent = task.get("agent")
        if agent not in ready:
            continue

        status = task.get("status", "pending")
        start_t = _fallback_start(task, default=now_from_start)

        if status == "completed" or task.get("actual_time") is not None:
            if task.get("actual_finish_time") is not None:
                finish_t = _safe_float(task.get("actual_finish_time"), start_t)
            elif task.get("actual_time") is not None:
                finish_t = start_t + _safe_float(task.get("actual_time"), 0.0)
            else:
                finish_t = start_t + _standard_duration(task)

            ready[agent] = max(ready[agent], finish_t)

        elif status in {"dispatched", "executing"}:
            if task.get("_expected_finish_time") is not None:
                finish_t = _safe_float(task.get("_expected_finish_time"), now_from_start)
            elif task.get("_duration_override") is not None:
                finish_t = start_t + _safe_float(task.get("_duration_override"), _standard_duration(task))
            else:
                finish_t = start_t + _standard_duration(task)

            finish_t = max(float(now_from_start), finish_t)
            ready[agent] = max(ready[agent], finish_t)

    return ready


def run_monitor_rescheduling(
    *,
    schedule: Sequence[dict],
    P,
    now_from_start: float,
    failure_info: dict,
    output_dir: str = "dynamic_gantt_outputs",
    step_idx: int = 1,
    max_iter: int = 200,
    seed: Optional[int] = None,
    optimized_reference_schedule: Optional[Sequence[dict]] = None,
    offline_best_schedule: Optional[Sequence[dict]] = None,
    robot_time_table=None,
    robot_time_csv=None,
    robot_pick_counts=None,
    save_gantt_immediately: bool = True,
) -> Tuple[List[dict], dict]:
    """
    監控端呼叫的主函式：
        監控端發現失效
            ↓
    run_monitor_rescheduling()
            ↓
    整理目前狀態
            ↓
    決定 frozen 任務
            ↓
    決定從哪個 kit 開始重排
            ↓
    呼叫 VNS solver
            ↓
    回寫新的 planned time
            ↓
    存甘特圖
            ↓
    回傳新的 dynamic_schedule 和 info

    optimized_reference_schedule　（最當前的最佳化排程）：
    - 第一次傳離線最佳排程
    - 之後傳前一次 dynamic_schedule
    - 若未提供 optimized_reference_schedule，會退回 offline_best_schedule
    """
    decision_time = float(now_from_start)

    if robot_time_table is None:
        if robot_time_csv is None:
            robot_time_csv = vns.ROBOT_TIME_DIR
        robot_time_table = load_robot_task_time_table(robot_time_csv)

    prepared_schedule, frozen_task_ids, fixed_start_by_id, agent_ready_times = prepare_runtime_constraints(
        schedule,
        now_from_start=decision_time,
    )

    failure_snapshot_schedule = build_failure_moment(prepared_schedule, now_from_start=decision_time)

    affected_task_ids = set(failure_info.get("affected_task_ids", []))
    affected_task_ids.update(failure_info.get("completed_task_ids", []))
    current_task_id = failure_info.get("current_task_id")
    if current_task_id is not None:
        affected_task_ids.add(current_task_id)

    start_kit_id = infer_start_kit(prepared_schedule, affected_task_ids)

    
    prepared_schedule = reorder_schedule_with_frozen(prepared_schedule, frozen_task_ids)
    dynamic_input = deepcopy(prepared_schedule)
    dynamic_schedule, baseline_cost, dynamic_cost, search_time, solution_space, _ , _ = solver(
        dynamic_input,
        P,
        start_kit_id=start_kit_id,
        max_iter=50,
        seed=seed,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
        fixed_start_by_id=fixed_start_by_id,
        agent_ready_times=agent_ready_times,
        robot_time_table=robot_time_table,
        robot_pick_counts=robot_pick_counts,
    )

    apply_planned_times(
        dynamic_schedule,
        P,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
        fixed_start_by_id=fixed_start_by_id,
        agent_ready_times=agent_ready_times,
    )

    # 根據最終 planned_finish_time 修正 dispatched robot task 的 place 等待補償
    first_compensation_log = apply_dispatched_task_compensation(dynamic_schedule)
    agent_ready_times = recompute_agent_ready_times(
        dynamic_schedule,
        decision_time,
    )

    if first_compensation_log:
        second_input = deepcopy(dynamic_schedule)

        dynamic_schedule, _ , dynamic_cost, search_time_2, solution_space_2, _ , _  = solver(
            second_input,
            P,
            start_kit_id=start_kit_id,
            max_iter=max_iter,
            seed=seed,
            frozen_task_ids=frozen_task_ids,
            decision_time=decision_time,
            fixed_start_by_id=fixed_start_by_id,
            agent_ready_times=agent_ready_times,
            robot_time_table=robot_time_table,
            robot_pick_counts=robot_pick_counts,
        )

        search_time += search_time_2

        apply_planned_times(
            dynamic_schedule,
            P,
            frozen_task_ids=frozen_task_ids,
            decision_time=decision_time,
            fixed_start_by_id=fixed_start_by_id,
            agent_ready_times=agent_ready_times,
        )

        final_compensation_log = apply_dispatched_task_compensation(dynamic_schedule)
        agent_ready_times = recompute_agent_ready_times(
            dynamic_schedule,
            decision_time,
        )

    apply_planned_times(
        dynamic_schedule,
        P,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
        fixed_start_by_id=fixed_start_by_id,
        agent_ready_times=agent_ready_times,
    )

    dynamic_cost = max_finish_time_by_fields(dynamic_schedule)

    optimized_schedule = (
        deepcopy(optimized_reference_schedule)
        if optimized_reference_schedule is not None
        else deepcopy(offline_best_schedule)
        if offline_best_schedule is not None
        else deepcopy(schedule)
    )

    deferred_gantt_item = {
        "optimized_schedule": deepcopy(optimized_schedule),
        "failure_snapshot_schedule": deepcopy(failure_snapshot_schedule),
        "dynamic_schedule": deepcopy(dynamic_schedule),
        "output_dir": output_dir,
        "step_idx": step_idx,
        "failure_info": deepcopy(failure_info),
    }

    if save_gantt_immediately:
        gantt_path = save_reschedule_gantt(**deferred_gantt_item)
    else:
        gantt_path = None

    failure_title = format_failure_title(failure_info)
    info = {
        "failure_info": deepcopy(failure_info),
        "failure_title": failure_title,
        "decision_time": decision_time,
        "agent_ready_times": agent_ready_times,
        "frozen_task_ids": sorted(frozen_task_ids),
        "fixed_start_by_id": dict(fixed_start_by_id),
        "start_kit_id": start_kit_id,
        "baseline_cost": baseline_cost,
        "dynamic_cost": dynamic_cost,
        "search_time": search_time,
        "gantt_path": gantt_path,
        "deferred_gantt_item": deferred_gantt_item,
        "summary_item": {
            "title": f"Dynamic rescheduling {step_idx} - {failure_title}",
            "schedule": deepcopy(dynamic_schedule),
        },
    }
    return dynamic_schedule, info


def overlay_radar_warning_intervals_on_axis(
    ax,
    intervals,
    alpha=0.30, # 不透明度
):
    """
    將倒車雷達警示區間以透明紅色色塊覆蓋在甘特圖上
    """
    if not intervals:
        return

    for idx, interval in enumerate(intervals):
        try:
            start = float(interval["start"])
            end = float(interval["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if end <= start:
            continue

        ax.axvspan(
            start,
            end,
            ymin=0,
            ymax=1,
            color="red",
            alpha=alpha,
            zorder=20,
            label="Radar Warning" if idx == 0 else None,
        )

def make_monitor_summary_figure(rows: Sequence[dict], time_limit=None):
    """
    建立最終總表：Planned timeline + 每次 dynamic result + Actual timeline 
    """
    n = len(rows)
    fig_height = max(4.0, 2.5 * n)
    fig, axes = plt.subplots(n, 1, figsize=(16, fig_height), sharex=False)

    if n == 1:
        axes = [axes]

    if time_limit is None:
        time_limit = compute_time_limit(*(max_finish_time_by_fields(item["schedule"]) for item in rows))

    for i, (ax, item) in enumerate(zip(axes, rows)):
        title = item.get("title", f"Timeline {i + 1}")
        actual_only = "Actual" in title or "Final" in title

        draw_monitor_timeline(
            item["schedule"],
            title=item.get("title", f"Timeline {i + 1}"),
            ax=ax,
            show=False,
            time_limit=time_limit,
            actual_only=actual_only,
        )
        radar_intervals = item.get("radar_warning_intervals")
        if radar_intervals:
            overlay_radar_warning_intervals_on_axis(
                ax,
                radar_intervals,
                alpha=0.20,
            )
            
        if i < n - 1:
            ax.tick_params(labelbottom=True)

    plt.tight_layout()
    return fig


def save_monitor_summary_figure(rows: Sequence[dict], output_dir: str, filename: Optional[str] = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"timeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(output_dir, filename)
    fig = make_monitor_summary_figure(rows)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def show_scrollable_figure(fig, window_title: str = "Schedule Summary"):
    """用 Tkinter 將 matplotlib figure 包成可垂直捲動的視窗"""
    root = tk.Tk()
    root.title(window_title)
    root.geometry("1500x900")

    outer_frame = tk.Frame(root)
    outer_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(outer_frame)
    scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner_frame = tk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    fig_canvas = FigureCanvasTkAgg(fig, master=inner_frame)
    fig_widget = fig_canvas.get_tk_widget()
    fig_widget.pack(fill="both", expand=True)
    fig_canvas.draw()

    toolbar = NavigationToolbar2Tk(fig_canvas, inner_frame)
    toolbar.update()
    toolbar.pack(fill="x")

    def _on_frame_configure(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)

    def _on_mousewheel(event):
        if getattr(event, "delta", 0):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", None) == 4:
            canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(3, "units")

    inner_frame.bind("<Configure>", _on_frame_configure)
    canvas.bind("<Configure>", _on_canvas_configure)
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)

    def _on_close():
        try:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        except Exception:
            pass
        plt.close(fig)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
