# 失效模擬下的動態排程 (連續失效模擬)

from copy import deepcopy
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import random
import os
from datetime import datetime

from read_schedule import load_schedule_csv
from precedence_matrix import build_precedence_matrix
import VNS_failure_simulation_solver
from VNS_failure_simulation_solver import (
    extract_kits,
    get_makespan,
    compute_start_times,
    get_finish_times,
    get_task_duration,
    solver,
    draw_gantt,
    load_robot_task_time_table,
    print_robot_time_check,
)
from VNS_verification import optimal_solution

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAILURE_TYPES = {
    "1": "order",
    "2": "agent",
    "3": "delay",
}

ROBOT_PLACE_TIME = VNS_failure_simulation_solver.ROBOT_PLACE_TIME

# -----------------------------
# helper
# -----------------------------
def task_to_kit_map(schedule):
    """
    建立兩個查表工具:
    1. 某個 task ID 屬於哪個 kit
    2. 某個 kit 裡有哪些 task
    """
    kits = extract_kits(schedule)
    task_to_kit = {}
    kit_to_task_ids = {}
    for kit_id, idxs in kits.items():
        ids = []
        for idx in idxs:
            tid = schedule[idx]["ID"]
            task_to_kit[tid] = kit_id
            ids.append(tid)
        kit_to_task_ids[kit_id] = ids
    return task_to_kit, kit_to_task_ids

def get_id_to_index(schedule):
    return {task["ID"]: i for i, task in enumerate(schedule)}

def get_task_by_id(schedule, task_id):
    id_to_index = get_id_to_index(schedule)
    if task_id not in id_to_index:
        raise ValueError(f"Task ID {task_id} 不存在於目前排程中")
    return schedule[id_to_index[task_id]]

def infer_start_kit(schedule, failure_info):
    task_to_kit,_ = task_to_kit_map(schedule)

    if failure_info["type"] == "order":
        candidate_ids = [failure_info["t1"], failure_info["t2"]]
    else:
        candidate_ids = [failure_info["t"]]

    valid_kit_ids = [task_to_kit[t] for t in candidate_ids if t in task_to_kit]
    if not valid_kit_ids:
        return 1
    return min(valid_kit_ids)

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

def parse_int_set(text):
    text = text.strip()
    if not text:
        return set()
    return {int(x.strip()) for x in text.split(",") if x.strip()}

def build_frozen_task_ids(schedule, freeze_until_id, extra_ids):
    existing_ids = {task["ID"] for task in schedule}
    frozen = {tid for tid in existing_ids if tid <= freeze_until_id}
    frozen |= {tid for tid in extra_ids if tid in existing_ids}
    return frozen

def format_failure_info(failure_info):
    ftype = failure_info["type"]
    if ftype == "delay":
        return f"task {failure_info['t']} delay {failure_info['delay_time']} units"
    if ftype == "agent":
        if "new_agent" in failure_info:
            return f"task {failure_info['t']} agent exchange -> {failure_info['new_agent']}"
        return f"task {failure_info['t']} agent exchange"
    if ftype == "order":
        return f"task {failure_info['t1']} & {failure_info['t2']} order exchange"
    return str(failure_info)

def apply_time_scale(ax, time_limit):
    """固定甘特圖的時間尺度。"""
    if time_limit is None:
        return
    ax.set_xlim(0, max(1.0, float(time_limit)))

def compute_time_limit(*candidate_values, margin_ratio=0.05):
    """
    依據排程 makespan 計算甘特圖 x 軸上限
    """
    numeric_candidates = [float(v) for v in candidate_values if v is not None]
    if not numeric_candidates:
        return None

    base = max(numeric_candidates)
    margin = max(1.0, base * margin_ratio) #margin為右邊要多留多少空白
    return base + margin

def add_delay_to_task(task, extra_delay):
    '''
    更新實際執行時間
    '''
    base = task.get("actual_time")
    if base is not None:
        task["actual_time"] = base + extra_delay

def reset_task_to_standard_time(task):
    """
    讓 task 回到標準時間計算
    (因為 get_task_duration() 會優先使用 actual_time，
    所以若要讓任務重新依照 agent 使用 standard_time，
    必須將 actual_time 清回 None)
    """
    task["actual_time"] = None


def handle_order_mistake(schedule_fail, idx1, idx2):
    """
    order failure 的時間轉移規則:
    - 較早位置的任務 = 原本應該被執行的任務
    - 較晚位置的任務 = 實際被提前完成的錯誤任務
    - 錯誤任務完成時，耗掉的是「較早位置原任務」的實際時間區塊
    """

    if idx1 == idx2:
        return

    earlier_idx, later_idx = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)

    earlier_task = schedule_fail[earlier_idx]

    if earlier_task["agent"] == "human":
        standard_time = earlier_task["human_standard_time"]
    elif earlier_task["agent"] == "robot":
        standard_time = earlier_task["robot_standard_time"]
    else:
        raise ValueError(f"未知 agent: {earlier_task['agent']}")

    elapsed_time_of_earlier_slot = get_task_duration(earlier_task)

    need_time_transfer = (
        abs(float(elapsed_time_of_earlier_slot) - float(standard_time)) > 1e-6
    )

    schedule_fail[idx1], schedule_fail[idx2] = schedule_fail[idx2], schedule_fail[idx1]

    if not need_time_transfer:
        return

    executed_wrong_task = schedule_fail[earlier_idx]
    postponed_original_task = schedule_fail[later_idx]

    executed_wrong_task["actual_time"] = elapsed_time_of_earlier_slot
    reset_task_to_standard_time(postponed_original_task)

    executed_wrong_task["actual_time"] = elapsed_time_of_earlier_slot
    reset_task_to_standard_time(postponed_original_task)

# -----------------------------
# 失效模擬
# -----------------------------
def simulate_failure(schedule, failure_info):
    """
    根據 failure_info 在「當前排程」上產生失效後的新排程
    """
    schedule_fail = deepcopy(schedule)
    id_to_index = get_id_to_index(schedule_fail)
    failure_type = failure_info["type"]

    if failure_type == "order":
        t1, t2 = failure_info["t1"], failure_info["t2"]
        if t1 not in id_to_index or t2 not in id_to_index:
            raise ValueError("order failure 指定的 task ID 不存在")
        idx1, idx2 = id_to_index[t1], id_to_index[t2]
        handle_order_mistake(schedule_fail, idx1, idx2)

    elif failure_type == "agent":
        t = failure_info["t"]
        if t not in id_to_index:
            raise ValueError("agent failure 指定的 task ID 不存在")

        idx = id_to_index[t]
        target_agent = failure_info.get("new_agent", "human")
        schedule_fail[idx]["agent"] = target_agent

    elif failure_type == "delay":
        t = failure_info["t"]
        delay_time = failure_info["delay_time"]
        if t not in id_to_index:
            raise ValueError("delay failure 指定的 task ID 不存在")

        idx = id_to_index[t]
        task = schedule_fail[idx]
        if task["actual_time"] is None:
            task["actual_time"] = task["human_standard_time"] if task["agent"] == "human" else task["robot_standard_time"]
        add_delay_to_task(task,delay_time)

    else:
        raise ValueError(f"未知失效類型: {failure_type}")

    return schedule_fail

def update_actual_time(base_schedule, failed_schedule, frozen_task_id = None ,inplace=True):
    """
    模擬時將排程中每個 task 的 actual_time 更新為其對應 agent 的 standard_time
    """
    target_schedule = failed_schedule if inplace else deepcopy(failed_schedule)
    base_by_id = {task["ID"]: task for task in base_schedule}

    if frozen_task_id is None:
        frozen_task_id =set()

    for task in target_schedule:
        tid = task["ID"]
        update_time = False
        if tid in frozen_task_id:

            original_task = base_by_id[tid]
            original_agent = original_task["agent"]
            current_agent = task["agent"]
            original_time = original_task["actual_time"]
            current_time = task["human_standard_time"] if task["agent"]=="human" else task["robot_standard_time"]

            # 若 agent 改變，重新指定 actual_time
            if original_agent != current_agent:
                update_time = True

            # 若 agent 沒改變，但 actual_time 尚未建立，才初始化
            elif task["actual_time"] is None:
                update_time = True
            
            if update_time:
                if current_agent == "human":
                    task["actual_time"] = task.get("human_standard_time", 0.0)
                elif current_agent == "robot":
                    task["actual_time"] = task.get("robot_standard_time", 0.0)
                else:
                    raise ValueError(
                        f"Task ID {tid} 的 agent 無法辨識: {current_agent}"
                    )

    return target_schedule

def apply_parallel_delay_compensation(
    reference_schedule,
    current_schedule,
    P,
    failure_info,
    decision_time,
    frozen_task_ids=None,
):
    """
    檢查目前失效是否符合以下兩種條件，若符合則更新 current_schedule 中
    對應 pick&place 任務的 actual_time

    條件 1:
    - delay 的 task 是某 kit 的 replace
    - 若同 kit 中存在平行於 replace 且在 decision_time 前已開始執行的 pick&place

    條件 2:
    - delay 的 task 是前一個 kit 的最後一個任務
    - 若下一個 kit 中存在平行於其 replace 且在 decision_time 前已開始執行的 pick&place
    
    以上pick&place 額外延遲，延遲量 = max(0, replace_start_time - place_start_time)

    參數
    reference_schedule : 用來判斷「哪個 task 在 decision_time 已開始」的參考排程。
    current_schedule : 要被回寫 actual_time 的排程。
        建議使用：
        動態排程後的 schedule。

    回傳
    updated_schedule, compensation_log
    compensation_log記錄哪些 task 被補償、原因、補償量
    """
    compensation_log = []

    if failure_info.get("type") != "delay":
        return current_schedule, compensation_log

    if frozen_task_ids is None:
        frozen_task_ids = set()

    target_schedule = current_schedule 

    # ---------- helper ---------
    
    def get_place_start_time(task, task_start_time, robot_place = ROBOT_PLACE_TIME):
        """
        pick&place 的 place 開始時刻
        """
        duration = get_task_duration(task)
        return task_start_time + (duration - robot_place)

    def get_replace_task(schedule, kit_id, kit_to_task_ids):
        '''
        找出某個kit裡的replace task id是哪一個
        '''
        for tid in kit_to_task_ids.get(kit_id, []):
            task = schedule[get_id_to_index(schedule)[tid]]
            if task["command"] == "replace":
                return tid
        return None

    def get_last_task_of_kit(kit_id, kit_to_task_ids, finish_by_id):
        '''
        找某個 kit 中，哪個 task 是最後完成的
        '''
        candidate_ids = kit_to_task_ids.get(kit_id, [])
        if not candidate_ids:
            return None
        return max(candidate_ids, key=lambda tid: finish_by_id[tid])

    def find_started_parallel_pickplace_ids(
        schedule,
        kit_id,
        decision_time,
        start_by_id,
        finish_by_id,
        frozen_task_ids=None,
    ):
        """
        找出某個 kit 裡，平行於 replace 的 pick&place 任務(執行中)
        """
        if frozen_task_ids is None:
            frozen_task_ids = set()

        _, kit_to_task_ids = task_to_kit_map(schedule)

        result = [] # 受影響的 task ID 清單

        replace_id = get_replace_task(schedule, kit_id, kit_to_task_ids)
        # 如果找不到 replace，就代表這個 kit 無法判斷平行關係，直接回傳空結果
        if replace_id is None:
            return result

        replace_finish = finish_by_id[replace_id]
        id_to_idx = get_id_to_index(schedule)

        for tid in kit_to_task_ids.get(kit_id, []):
            task = schedule[id_to_idx[tid]]

            if task["command"] != "pick and place":
                continue

            if frozen_task_ids and tid not in frozen_task_ids:
                continue

            start_t = start_by_id[tid]
            finish_t = finish_by_id[tid]

            started_before_decision = start_t < decision_time # 任務開始時間早於decision_time
            still_relevant = finish_t > start_t # finish time 大於 start time (正常應該都成立)
            parallel_to_replace = start_t < replace_finish # 任務開始時間比 replace 完成時間早 (平行)

            if started_before_decision and still_relevant and parallel_to_replace:
                result.append(tid)

        return result

    # 用 reference_schedule 判斷
    ref_start_times = compute_start_times(
        reference_schedule,
        P,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
    )
    ref_finish_times = get_finish_times(reference_schedule, ref_start_times)
    ref_id_to_idx = get_id_to_index(reference_schedule)

    # 任務與其開始/完成時間對照
    start_by_id = {
        task["ID"]: ref_start_times[i]
        for i, task in enumerate(reference_schedule)
    }
    finish_by_id = {
        task["ID"]: ref_finish_times[i]
        for i, task in enumerate(reference_schedule)
    }

    task_to_kit, kit_to_task_ids = task_to_kit_map(reference_schedule)

    delayed_task_id = failure_info["t"]
    delay_time = float(failure_info["delay_time"])

    if delayed_task_id not in task_to_kit:
        return target_schedule, compensation_log

    delayed_kit = task_to_kit[delayed_task_id]
    compensation_targets = []

    # 判斷 delayed task 是否為該 kit 的 replace
    replace_tid_of_delayed_kit = get_replace_task(
        reference_schedule, delayed_kit, kit_to_task_ids
    )

    # 條件 1：replace 延遲，則同 kit 已開始的平行 pick&place 一起延遲
    if delayed_task_id == replace_tid_of_delayed_kit: # delay 的 task 本身就是這個 kit 的 replace
        replace_finish_time = finish_by_id[replace_tid_of_delayed_kit]

        affected_ids = find_started_parallel_pickplace_ids(
            reference_schedule,
            delayed_kit,
            decision_time,
            start_by_id,
            finish_by_id,
            frozen_task_ids=frozen_task_ids,
        )
        for tid in affected_ids:
            if tid != delayed_task_id:

                task = reference_schedule[ref_id_to_idx[tid]]
                place_start_time = get_place_start_time(task, start_by_id[tid])
                extra_delay = max(0.0, replace_finish_time+delay_time - place_start_time)
                if extra_delay > 0:
                    compensation_targets.append(
                        {
                            "task_id": tid, #哪個 task 要補償
                            "extra_delay": extra_delay, #補償多少
                            "reason": f"kit {delayed_kit} replace delayed", #補償原因
                            "trigger_task_id": delayed_task_id, # 是哪個 delayed task 造成的
                        }
                    )

    # 條件 2：前一 kit 最後任務延遲，則下一 kit 已開始的平行 pick&place 一起延遲
    if delayed_kit + 1 in kit_to_task_ids:
        last_tid_of_delayed_kit = get_last_task_of_kit(
            delayed_kit,
            kit_to_task_ids,
            finish_by_id,
        )

        if delayed_task_id == last_tid_of_delayed_kit: # 前一個 kit 的最後任務延遲
            # 找下一個 kit 中已開始且平行於 replace 的 pick&place
            next_kit = delayed_kit + 1
            next_replace_tid = get_replace_task(reference_schedule, next_kit, kit_to_task_ids)
            
            if next_replace_tid is not None:
                next_replace_finish_time = finish_by_id[next_replace_tid]
                affected_ids = find_started_parallel_pickplace_ids(
                    reference_schedule,
                    next_kit,
                    decision_time,
                    start_by_id,
                    finish_by_id,
                    frozen_task_ids=frozen_task_ids,
                )
                for tid in affected_ids:
                    task = reference_schedule[ref_id_to_idx[tid]]
                    place_start_time = get_place_start_time(task, start_by_id[tid])
                    extra_delay = max(0.0, next_replace_finish_time+delay_time - place_start_time)

                    if extra_delay > 0:
                        compensation_targets.append(
                            {
                                "task_id": tid,
                                "extra_delay": extra_delay,
                                "reason": f"last task of kit {delayed_kit} delayed, affects started pick&place in kit {next_kit}",
                                "trigger_task_id": delayed_task_id,
                            }
                        )

    # 同一 task 若同時符合兩條件，只加一次最大延遲 (防呆)
    merged = {}
    for item in compensation_targets:
        tid = item["task_id"]
        if tid not in merged or item["extra_delay"] > merged[tid]["extra_delay"]:
            merged[tid] = item

    # 回寫 current_schedule
    cur_id_to_idx = get_id_to_index(target_schedule)

    for item in merged.values():
        tid = item["task_id"]
        if tid not in cur_id_to_idx:
            continue

        task = target_schedule[cur_id_to_idx[tid]]
        add_delay_to_task(task, item["extra_delay"])

        compensation_log.append(
            {
                "task_id": tid,
                "extra_delay": item["extra_delay"],
                "updated_actual_time": task["actual_time"],
                "reason": item["reason"],
                "trigger_task_id": item["trigger_task_id"],
            }
        )

    return target_schedule, compensation_log

# No rescheduling helpers
def init_actual_time_from_agent(task):
    """
    若 actual_time 尚未設定，依照目前 agent 初始化
    """
    if task["actual_time"] is None:
        if task["agent"] == "human":
            task["actual_time"] = task.get("human_standard_time", 0.0)
        elif task["agent"] == "robot":
            task["actual_time"] = task.get("robot_standard_time", 0.0)
        else:
            raise ValueError(f"未知 agent: {task['agent']}")

def simulate_nores_execution(base_schedule, base_start_times, nores_schedule, P, robot_place=ROBOT_PLACE_TIME):
    """
    no rescheduling 的執行模擬

    規則：
    1. robot 的開始時間理論上固定為 base schedule 的開始時間
    2. 只有當前一個 robot 任務尚未完成時，robot 下一任務才順延
    3. 若 replace / precedence 延遲使 robot 無法在原節奏完成，則不改 start，
       改為把等待時間加到該 robot task 的 actual_time
    4. human 任務被動延遲傳播
    5. 保留 CASE 3 的 kit 節奏限制
    """
    adjusted_schedule = deepcopy(nores_schedule)
    n = len(adjusted_schedule)

    # base 的理論開始時間
    base_start_by_id = {task["ID"]: s for task, s in zip(base_schedule, base_start_times)}

    # kit 對照資訊 (task to kit / kit to replace task)
    kits = extract_kits(adjusted_schedule)
    task_to_kit = {}
    replace_idx_of_kit = {}
    for kit_id, idxs in kits.items():
        for idx in idxs:
            task_to_kit[idx] = kit_id
            if adjusted_schedule[idx]["command"] == "replace":
                replace_idx_of_kit[kit_id] = idx

    start_times = [0.0] * n
    finish_times = [0.0] * n
    computed = [False] * n

    # 同 agent 前一個任務完成時間
    agent_free = {"human": 0.0, "robot": 0.0}

    for j in range(n):
        task = adjusted_schedule[j]
        tid = task["ID"]
        agent = task["agent"]

        dur = get_task_duration(task)

        preds = [i for i in range(n) if P[i][j] == 1]
        pred_finish = max((finish_times[i] for i in preds), default=0.0)

        # Human: 正常被動延遲傳播
        if agent == "human":
            planned_start = base_start_by_id[tid]
            start = max(planned_start, agent_free["human"], pred_finish)
            finish = start + dur

        # Robot: start 盡量固定，只在前一 robot 任務未完時順延
        else:
            planned_start = base_start_by_id[tid]

            # robot 開始時間只受「前一個 robot 任務尚未完成」影響
            start = max(planned_start, agent_free["robot"])

            extra_wait = 0.0 # 等待量(補回 actual_time)

            if task["command"] == "pick and place":
                kit_id = task_to_kit[j]
                replace_idx = replace_idx_of_kit[kit_id]
                replace_finish = finish_times[replace_idx]

                # 原本 place 開始時刻
                place_start = start + (dur - robot_place)

                # replace 還沒完成
                extra_wait = max(extra_wait, replace_finish - place_start)

                # CASE 3：前一個 kit 的節奏限制
                if kit_id > 1:
                    prev_kit_idxs = [idx for idx in kits[kit_id - 1] if computed[idx]]
                    if prev_kit_idxs:
                        prev_kit_last_idx = max(prev_kit_idxs, key=lambda idx: start_times[idx])
                        prev_kit_last_start = start_times[prev_kit_last_idx]
                        extra_wait = max(extra_wait, prev_kit_last_start - start)

            if extra_wait > 0:
                if task["actual_time"] is None:
                    task["actual_time"] = task.get("robot_standard_time", 0.0)
                task["actual_time"] += extra_wait
                dur = get_task_duration(task)

            finish = start + dur

        start_times[j] = start
        finish_times[j] = finish
        agent_free[agent] = finish
        computed[j] = True

    return adjusted_schedule, start_times, finish_times

def simulate_failure_no_rescheduling(base_schedule, failure_info, rng=None):
    """
    - 不改 task 順序
    - 不改 agent assignment
    - 失效只轉成該 task 的 actual_time 增加
    """

    def find_human_task_before_fail(base_schedule, fail_task_id, P, case=3):
        """
        找出 fail_task 之前、最接近 fail_task 開始時間的 human task

        規則：
        1. 先用 base_schedule 計算每個 task 的 start time
        2. 取得 fail_task 的 start time = fail_start
        3. 從候選 task 中找：
        - task["agent"] == "human"
        - task 的 start time < fail_start
        - 且 start time 與 fail_start 的差距最小

        回傳:
        - human_task (dict)；若找不到則回傳 None
        """
        id_to_index = get_id_to_index(base_schedule)

        if fail_task_id not in id_to_index:
            raise ValueError(f"fail_task_id {fail_task_id} 不存在於 schedule 中")

        start_times = compute_start_times(base_schedule, P, case=case)
        fail_idx = id_to_index[fail_task_id]
        fail_start = start_times[fail_idx]

        best_task = None
        best_start = None

        for idx, task in enumerate(base_schedule):

            # 只找 human task
            if task["agent"] != "human":
                continue

            task_start = start_times[idx]

            # 必須開始時間落在 fail_task 之前
            if task_start < fail_start:
                if best_task is None or task_start > best_start:
                    best_task = task
                    best_start = task_start

        return best_task

    if rng is None:
        rng = random.Random()

    schedule_fail = deepcopy(base_schedule)
    id_to_index = get_id_to_index(schedule_fail)
    failure_type = failure_info["type"]

    for task in schedule_fail:
        init_actual_time_from_agent(task)

    if failure_type == "delay":
        t = failure_info["t"]
        delay_time = float(failure_info["delay_time"])

        if t not in id_to_index:
            raise ValueError("delay failure 指定的 task ID 不存在")

        fail_task = schedule_fail[id_to_index[t]]
        add_delay_to_task(fail_task, delay_time)

    elif failure_type == "agent":
        t = failure_info["t"]

        if t not in id_to_index:
            raise ValueError("agent failure 指定的 task ID 不存在")

        fail_task = schedule_fail[id_to_index[t]]

        upper = float(fail_task.get("human_standard_time", 0.0))
        upper = max(1.0, 2*upper)
        recovery_time = round(rng.uniform(1.0, upper),0)

        # recovery 加在原先人員執行的任務
        # 找到人員正在執行的task
        P = build_precedence_matrix(base_schedule)
        human_task_base = find_human_task_before_fail(
        base_schedule=base_schedule,
        fail_task_id=t,
        P=P,
        case=3
        )
        if human_task_base is None:
            raise ValueError("agent failure 發生時，找不到 fail_task 之前最接近的 human task")

        task_id = human_task_base["ID"]
        task = schedule_fail[id_to_index[task_id]]

        add_delay_to_task(task, recovery_time)

    elif failure_type == "order":
        t1, t2 = failure_info["t1"], failure_info["t2"]

        if t1 not in id_to_index or t2 not in id_to_index:
            raise ValueError("order failure 指定的 task ID 不存在")

        # 加在 base schedule 中原本就要執行的 task
        idx1 = id_to_index[t1]
        idx2 = id_to_index[t2]

        if idx1 < idx2:
            earlier_id, later_id = t1, t2
        else:
            earlier_id, later_id = t2, t1

        earlier_task = schedule_fail[id_to_index[earlier_id]]
        later_task = schedule_fail[id_to_index[later_id]]

        upper = float(later_task.get("human_standard_time", 0.0))
        upper = max(1.0, 2*upper)
        recovery_time = round(rng.uniform(1.0, upper),0)

        add_delay_to_task(earlier_task, recovery_time)

    else:
        raise ValueError(f"未知失效類型: {failure_type}")

    return schedule_fail

def get_nores_info(base_schedule, base_start_times, nores_schedule, P):
    adjusted_schedule, start_times, finish_times = simulate_nores_execution(
        base_schedule, base_start_times, nores_schedule, P
    )
    return adjusted_schedule, start_times, finish_times

def get_no_rescheduling_makespan(base_schedule, base_start_times, nores_schedule, P):
    _, _, finish_times = get_nores_info(base_schedule, base_start_times, nores_schedule, P)
    return max(finish_times)

# -----------------------------
# 輸入互動
# -----------------------------
def prompt_failure_info(current_schedule):
    print("\n請選擇失效類型：")
    print("  1. order  (任務順序錯誤)")
    print("  2. agent  (錯誤代理人執行)")
    print("  3. delay  (任務延遲)")
    print("  q. 離開")

    choice = input("輸入選項: ").strip().lower()
    if choice == "q":
        return None
    if choice not in FAILURE_TYPES:
        print("無效選項，請重新輸入。")
        return "retry"

    failure_type = FAILURE_TYPES[choice]

    try:
        if failure_type == "order":
            t1 = int(input("請輸入 task ID 1: ").strip())
            t2 = int(input("請輸入 task ID 2: ").strip())
            get_task_by_id(current_schedule, t1) 
            get_task_by_id(current_schedule, t2) 
            return {"type": "order", "t1": t1, "t2": t2}

        if failure_type == "agent":
            t = int(input("請輸入發生錯誤代理人的 task ID: ").strip())
            old_task = get_task_by_id(current_schedule, t)
            print(f"目前 task {t} 的 agent = {old_task['agent']}")
            new_agent = input("要改成哪個 agent? (human/robot，直接 Enter 則自動切換): ").strip().lower()
            if not new_agent:
                new_agent = "robot" if old_task["agent"] == "human" else "human"
            if new_agent not in {"human", "robot"}:
                raise ValueError("agent 只能輸入 human 或 robot")
            return {"type": "agent", "t": t, "new_agent": new_agent}

        if failure_type == "delay":
            t = int(input("請輸入 delay 的 task ID: ").strip())
            get_task_by_id(current_schedule, t) 
            delay_time = float(input("請輸入增加的延遲時間: ").strip())
            return {"type": "delay", "t": t, "delay_time": delay_time}

    except Exception as e:
        print(f"輸入有誤：{e}")
        return "retry"

def prompt_reschedule_info(failure_schedule, failure_info):
    default_start_kit = infer_start_kit(failure_schedule, failure_info)

    print("\n請輸入動態排程參數：")
    freeze_until_text = input(
        "凍結到多少task ID 為止（直接 Enter 代表不使用此前綴凍結）: "
    ).strip()
    extra_frozen_text = input(
        "其他要凍結的 task IDs（逗號分隔；直接 Enter 代表無）: "
    ).strip()
    decision_time_text = input(
        "decision_time（直接 Enter 預設 0）: "
    ).strip()

    freeze_until_id = int(freeze_until_text) if freeze_until_text else -1
    extra_ids = parse_int_set(extra_frozen_text)
    frozen_task_ids = build_frozen_task_ids(failure_schedule, freeze_until_id, extra_ids)
    decision_time = float(decision_time_text) if decision_time_text else 0.0
    start_kit_id = default_start_kit

    # print(f"Frozen task IDs: {sorted(frozen_task_ids)}")

    return {
        "freeze_until_id": freeze_until_id,
        "extra_frozen_ids": extra_ids,
        "frozen_task_ids": frozen_task_ids,
        "decision_time": decision_time,
        "start_kit_id": start_kit_id,
    }


# -----------------------------
# 視覺化與紀錄
# -----------------------------
def write_gantt_raw_block(f, title, rows, makespan):
    """
    將一張甘特圖的 raw data 寫成區塊格式
    """
    f.write(f"title: {title}\n")

    f.write(
        f"{'task ID':>8}"
        f"{'agent':>12}"
        f"{'start_time':>15}"
        f"{'finish_time':>15}"
        f"{'duration':>15}\n"
    )

    for row in rows:
        task_id = row.get("task_id", "")
        agent = row.get("agent", "")
        start_time = float(row.get("start_time", 0.0))
        finish_time = float(row.get("finish_time", 0.0))
        duration = float(row.get("duration", 0.0))

        f.write(
            f"{str(task_id):>8}"
            f"{str(agent):>12}"
            f"{start_time:>15.2f}"
            f"{finish_time:>15.2f}"
            f"{duration:>15.2f}\n"
        )

    f.write(
        f"{'makespan:':>50}"
        f"{float(makespan):>15.2f}\n"
    )

    # 每張甘特圖中間空白一行
    f.write("\n")


def show_initial_schedule(schedule, P, time_limit, title="Offline Optimal Schedule", fig=None):
    """
    顯示離線最佳排程
    """
    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass

    new_fig, ax = plt.subplots(figsize=(14, 2.5))
    draw_gantt(
        schedule,
        P,
        title=title,
        ax=ax,
        show=False,
    )

    apply_time_scale(ax, time_limit)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    return new_fig

def build_nores_gantt_rows(base_schedule, base_start_time, nores_schedule, P):
    """
    建立 no rescheduling 甘特圖 raw data rows
    """
    adjusted_schedule, start_times, finish_times = get_nores_info(
        base_schedule,
        base_start_time,
        nores_schedule,
        P,
    )

    rows = []
    makespan = 0.0

    for task, start, finish in zip(adjusted_schedule, start_times, finish_times):
        duration = finish - start

        if duration <= 0:
            continue

        actual_time = task.get("actual_time")
        try:
            if actual_time is not None and float(actual_time) == 0.0:
                continue
        except (TypeError, ValueError):
            pass

        rows.append({
            "task_id": task.get("ID", ""),
            "agent": task.get("agent", ""),
            "start_time": start,
            "finish_time": finish,
            "duration": duration,
        })

        makespan = max(makespan, finish)

    return rows, makespan

def draw_no_rescheduling_gantt(base_schedule, base_start_time, nores_schedule, P, title="No rescheduling", ax=None, show=True):
    adjusted_schedule, start_times, finish_times = get_nores_info(
        base_schedule, base_start_time, nores_schedule, P
    )
    colors = {"replace": "tab:blue", "pick and place": "tab:orange"}

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        created_fig = True

    for task, start, finish in zip(adjusted_schedule, start_times, finish_times):
        dur = finish - start
        agent = task["agent"]
        tid = task.get("ID", "")

        actual_time = task.get("actual_time")
        try:
            if actual_time is not None and float(actual_time) == 0.0:
                continue
        except (TypeError, ValueError):
            pass

        ax.barh(
            agent,
            dur,
            left=start,
            color=colors.get(task["command"], "gray"),
            edgecolor="black"
        )

        ax.text(
            start + dur / 2,
            agent,
            str(tid),
            va="center",
            ha="center",
            color="white",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_title(title)
    ax.set_xlabel("Time")

    if created_fig:
        plt.tight_layout()
        if show:
            plt.show()

def show_step_comparison(
    base_schedule,
    dynamic_schedule,
    P,
    step_idx,
    frozen_task_ids,
    decision_time,
    time_limit,
    fig=None,
):
    """
    顯示本輪三張圖：
    1) failure 前的原始排程
    2) 失效後但不做動態排程
    3) 動態排程
    """
    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass

    new_fig, axes = plt.subplots(2, 1, figsize=(14, 5.0), sharex=False)

    draw_gantt(
        base_schedule,
        P,
        title=f"Failure {step_idx} - Original Optimal schedule",
        ax=axes[0],
        show=False,
    )

    # draw_no_rescheduling_gantt(
    #     base_schedule,
    #     failed_schedule,
    #     P,
    #     title=f"Failure {step_idx} - Failure occurs (no rescheduling)",
    #     ax=axes[1],
    #     show=False,
    # )

    draw_gantt(
        dynamic_schedule,
        P,
        title=f"Failure {step_idx} - Dynamic rescheduling",
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
        ax=axes[1],
        show=False,
    )

    for ax in axes:
        apply_time_scale(ax, time_limit)
    
    # 強制把前兩張圖的 x 軸數字也打開 (預設只顯示最下面那張圖的 x 軸刻度數字)
    axes[0].tick_params(labelbottom=True)
    # axes[1].tick_params(labelbottom=True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    return new_fig

def build_dynamic_gantt_rows(schedule, P, frozen_task_ids=None, decision_time=0.0):
    """
    建立動態排程甘特圖 raw data rows
    """
    start_times = compute_start_times(
        schedule,
        P,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
    )

    finish_times = get_finish_times(schedule, start_times)

    rows = []
    makespan = 0.0

    for task, start, finish in zip(schedule, start_times, finish_times):
        duration = finish - start

        if duration <= 0:
            continue

        rows.append({
            "task_id": task.get("ID", ""),
            "agent": task.get("agent", ""),
            "start_time": start,
            "finish_time": finish,
            "duration": duration,
        })

        makespan = max(makespan, finish)

    return rows, makespan

def plot_summary(history, P, time_limit, nores = False):
    """建立 summary figure，但不直接 show，方便後續用 scrollable 視窗承載。"""
    n = len(history)
    fig_height = max(3, 2.5 * n)
    fig, axes = plt.subplots(n, 1, figsize=(16, fig_height), sharex=False)

    if n == 1:
        axes = [axes]

    for i, (ax, item) in enumerate(zip(axes, history), start=1):
        if not nores :
            draw_gantt(
                item["schedule"],
                P,
                title=item["title"],
                frozen_task_ids=item.get("frozen_task_ids"),
                decision_time=item.get("decision_time", 0.0),
                ax=ax,
                show=False,
            )
        else:
            draw_no_rescheduling_gantt(
            item["base_schedule"],
            item["base_start_time"],
            item["schedule"],
            P,
            title=item["title"],
            ax=ax,
            show=False,
            )

        apply_time_scale(ax, time_limit)
        if i < n:
            ax.tick_params(labelbottom=True)

    plt.tight_layout()
    return fig

def show_scrollable_figure(fig, window_title="Schedule Summary"):
    """
    用 Tkinter 將 matplotlib figure 包成可垂直捲動的視窗
    """
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

    fig_canvas = FigureCanvasTkAgg(fig, master=inner_frame) #matplotlib 和 Tkinter 之間的橋樑
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
        if event.delta:
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
            canvas.bind_all("<MouseWheel>", "")
            canvas.bind_all("<Button-4>", "")
            canvas.bind_all("<Button-5>", "")
        except Exception:
            pass
        plt.close(fig)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()

def export_sim_dynamic_gantt_raw_data(
    *,
    output_dir,
    dyn_summary_history,
    P,
    timestamp=None,
):
    """
    匯出動態排程甘特圖 raw data
    """
    os.makedirs(output_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = os.path.join(
        output_dir,
        f"gantt_data_{timestamp}.txt"
    )

    with open(output_path, "w", encoding="utf-8-sig") as f:
        for item in dyn_summary_history:
            title = item.get("title", "Dynamic schedule")
            schedule = item["schedule"]
            frozen_task_ids = item.get("frozen_task_ids")
            decision_time = item.get("decision_time", 0.0)

            rows, makespan = build_dynamic_gantt_rows(
                schedule,
                P,
                frozen_task_ids=frozen_task_ids,
                decision_time=decision_time,
            )

            write_gantt_raw_block(
                f,
                title=title,
                rows=rows,
                makespan=makespan,
            )

    return output_path

def export_sim_nores_gantt_raw_data(
    *,
    output_dir,
    nores_summary_history,
    P,
    timestamp=None,
):
    """
    匯出失效後不排程甘特圖 raw data
    """
    os.makedirs(output_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = os.path.join(
        output_dir,
        f"gantt_data_{timestamp}.txt"
    )

    with open(output_path, "w", encoding="utf-8-sig") as f:
        for item in nores_summary_history:
            title = item.get("title", "No rescheduling schedule")

            rows, makespan = build_nores_gantt_rows(
                base_schedule=item["base_schedule"],
                base_start_time=item["base_start_time"],
                nores_schedule=item["schedule"],
                P=P,
            )

            write_gantt_raw_block(
                f,
                title=title,
                rows=rows,
                makespan=makespan,
            )

    return output_path

def print_schedule_debug(title, schedule, P, frozen_task_ids, decision_time):
    starts = compute_start_times(
        schedule,
        P,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
    )
    print("\n", title)
    print("frozen:", sorted(frozen_task_ids))
    for task, s in zip(schedule, starts):
        print(
            f"ID={task['ID']:>2}, "
            f"agent={task['agent']:<5}, "
            f"cmd={task['command']:<14}, "
            f"frozen={task['ID'] in frozen_task_ids}, "
            f"start={s:.2f}, "
            f"dur={get_task_duration(task):.2f}"
        )

# -----------------------------
# 主流程
# -----------------------------
def main():
    plt.ion()

    schedule = load_schedule_csv()
    P = build_precedence_matrix(schedule)
    robot_time_table = load_robot_task_time_table(
        VNS_failure_simulation_solver.ROBOT_TIME_DIR
    )
    def debug_robot_time(schedule, title):
        print_robot_time_check(
            deepcopy(schedule),
            robot_time_table,
            title=title
        )

    # 初始最佳排程
    best_schedule, best_cost = optimal_solution(schedule, P, case_3=True)
    # debug_robot_time(
    #     best_schedule,
    #     "Initial best_schedule robot time check"
    # )

    current_schedule = deepcopy(best_schedule)
    failed_schedule_nores = deepcopy(best_schedule)
    fail_base_start = compute_start_times(best_schedule, P, case=3) 

    print("=================================")
    print("Initial Offline Optimal Solution")
    print(f"Makespan: {best_cost:.2f}")
    print("=================================")

    initial_fig = show_initial_schedule(
        best_schedule,
        P,
        time_limit=None,
        title="Offline Optimal Schedule",
        fig=None,
    )
    compare_fig = None

    dyn_summary_history = [
        {
            "title": "Offline Optimal Schedule",
            "schedule": best_schedule,
            "frozen_task_ids": None,
            "decision_time": 0.0,
        }
    ]
    nores_summary_history = [
        {
            "title": "Offline Optimal Schedule",
            "base_schedule": failed_schedule_nores,
            "base_start_time":fail_base_start,
            "schedule": failed_schedule_nores,
        }
    ]
    failure_logs = []
    
    step_idx = 1
    while True:
        result = prompt_failure_info(current_schedule)
        if result is None:
            break
        if result == "retry":
            continue

        failure_info = result
        dyn_base_schedule = deepcopy(current_schedule)
        nores_base_schedule = deepcopy(failed_schedule_nores)

        try:
            # ==== 失效模擬 =====
            failed_schedule = simulate_failure(current_schedule, failure_info)
            # debug_robot_time(
            #     failed_schedule,
            #     f"Failure {step_idx} - after simulate_failure"
            # )
            reschedule_info = prompt_reschedule_info(failed_schedule, failure_info)
            frozen_task_ids = reschedule_info["frozen_task_ids"]
            decision_time = reschedule_info["decision_time"]
            start_kit_id = reschedule_info["start_kit_id"]
            
            # ==== no rescheduling ====
            nores_rng = random.Random(43+step_idx)
            failed_schedule_nores = simulate_failure_no_rescheduling(
                nores_base_schedule,
                failure_info,
                rng=nores_rng,
            )
            nores_summary_history.append(
                {
                    "title": f"No rescheduling after Failure {step_idx}",
                    "base_schedule":deepcopy(nores_base_schedule),
                    "base_start_time":fail_base_start,
                    "schedule": deepcopy(failed_schedule_nores),
                }
            )

            base_cost = get_makespan(
                dyn_base_schedule,
                P,
                robot_time_table=robot_time_table,
            )
            fail_cost = get_no_rescheduling_makespan(nores_base_schedule, fail_base_start, failed_schedule_nores, P)
            _, fail_base_start, _ =  get_nores_info(nores_base_schedule, fail_base_start, failed_schedule_nores, P)

            print("\n---------------------------------")
            print(f"Failure {step_idx} - Failure info: {failure_info}")
            print(f"Original schedule makespan: {base_cost:.2f}")
            print(f"No rescheduling makespan: {fail_cost:.2f}")

            # ===== dynamic rescheduling =====
            failed_schedule = reorder_schedule_with_frozen(failed_schedule, frozen_task_ids)  #先reorder_schedule_with_frozen完才是失效排程
            # debug_robot_time(
            #     failed_schedule,
            #     f"Failure {step_idx} - after reorder before update_actual_time"
            # )
            # print_schedule_debug("after reorder_schedule_with_frozen", failed_schedule, P, frozen_task_ids, decision_time)
            failed_schedule = update_actual_time(dyn_base_schedule,failed_schedule,frozen_task_ids) # simulation自動更新actual time
            failed_schedule, compensation = apply_parallel_delay_compensation(    # simulation自動補償與replace平行的任務延遲
                dyn_base_schedule,failed_schedule,P,failure_info,decision_time,frozen_task_ids)

            dynamic_schedule, _, dynamic_cost, _, _, history, shake = solver(
                failed_schedule,
                P,
                start_kit_id=start_kit_id,
                frozen_task_ids=frozen_task_ids,
                decision_time=decision_time,
                robot_time_table=robot_time_table,
            )
            # print_schedule_debug("after solver", failed_schedule, P, frozen_task_ids, decision_time)
            # debug_robot_time(
            #     dynamic_schedule,
            #     f"Failure {step_idx} - dynamic_schedule after solver"
            # )

            print(f"Dynamic rescheduling makespan: {dynamic_cost:.2f}")
            print("---------------------------------")
            # if compensation:
            #     print(compensation)

            step_time_limit = compute_time_limit(
                base_cost,
                dynamic_cost,
            )

            if initial_fig is not None:
                try:
                    plt.close(initial_fig)
                except Exception:
                    pass
                initial_fig = None

            # 顯示本輪比較圖（原始  / 動態重排）
            compare_fig = show_step_comparison(
                base_schedule=dyn_base_schedule,
                dynamic_schedule=dynamic_schedule,
                P=P,
                step_idx=step_idx,
                frozen_task_ids=frozen_task_ids,
                decision_time=decision_time,
                time_limit=step_time_limit,
                fig=compare_fig,
            )

            dyn_summary_history.append(
                {
                    "title": f"Dynamic rescheduling after Failure {step_idx}",
                    "schedule": deepcopy(dynamic_schedule),
                    "frozen_task_ids": frozen_task_ids,
                    "decision_time": decision_time,
                }
            )

            failure_logs.append(
                {
                    "step_idx": step_idx,
                    "failure_info": deepcopy(failure_info),
                    "description": format_failure_info(failure_info),
                }
            )

            # 下一輪以本輪動態重排後的排程為基礎繼續模擬
            current_schedule = deepcopy(dynamic_schedule)
            step_idx += 1

        except Exception as e:
            print(f"發生錯誤：{e}")
            print("這次失效模擬不會更新當前排程，請重新輸入。")

    if initial_fig is not None:
        try:
            plt.close(initial_fig)
        except Exception:
            pass
    if compare_fig is not None:
        try:
            plt.close(compare_fig)
        except Exception:
            pass

    plt.ioff()
    print("\n=================================")
    print("結束連續失效模擬，顯示所有動態排程結果")
    print("=================================")

    if failure_logs:
        print("Failure Summary:")
        for item in failure_logs:
            print(f"Failure {item['step_idx']}: {item['description']}")
        
        # 匯出失效模擬 raw data
        raw_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        nores_raw_path = export_sim_nores_gantt_raw_data(
            output_dir=os.path.join(BASE_DIR, "sim_nores_raw_data"),
            nores_summary_history=nores_summary_history,
            P=P,
            timestamp=raw_timestamp,
        )

        dynamic_raw_path = export_sim_dynamic_gantt_raw_data(
            output_dir=os.path.join(BASE_DIR, "sim_dynamic_raw_data"),
            dyn_summary_history=dyn_summary_history,
            P=P,
            timestamp=raw_timestamp,
        )

        print(f"[SIM NORES RAW DATA SAVED] {nores_raw_path}")
        print(f"[SIM DYNAMIC RAW DATA SAVED] {dynamic_raw_path}")

    else:
        print("Failure Summary: 無失效事件")


    # 失效後不重排overview
    nores_summary_costs = [get_makespan(item['schedule'], P) for item in nores_summary_history]
    nores_summary_time_limit = compute_time_limit(*nores_summary_costs)
    nores_summary_fig = plot_summary(nores_summary_history, P, time_limit=nores_summary_time_limit, nores = True)
    show_scrollable_figure(nores_summary_fig, window_title="No Rescheduling Results")

    # 動態排程overview
    dyn_summary_costs = [get_makespan(item['schedule'], P) for item in dyn_summary_history]
    dyn_summary_time_limit = compute_time_limit(*dyn_summary_costs)
    dyn_summary_fig = plot_summary(dyn_summary_history, P, time_limit=dyn_summary_time_limit)
    show_scrollable_figure(dyn_summary_fig, window_title="All Dynamic Rescheduling Results")


if __name__ == "__main__":
    main()
