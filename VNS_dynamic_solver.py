"""
動態排程solver (VNS) 鄰域依序搜尋
計算開始時間、VNS 搜尋、任務重新排序與重新分配
"""

import time
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path

ROBOT_PLACE_TIME = 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_TIME_DIR = Path(BASE_DIR) / "Standard Time Calculation" / "robot_task_time.csv"

def extract_kits(schedule):
    """回傳每個kit對應內容物的schedule index"""
    kits = {}
    current_kit = 0

    for i, t in enumerate(schedule):
        if t["command"] == "replace":
            current_kit += 1

        if  current_kit not in kits:
            kits[current_kit] = []
        kits[current_kit].append(i)

    return kits

def load_robot_task_time_table(csv_path):
    """
    建立查表用 robot_task_time dictionary
    """
    table = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            task_name = row["task name"].strip()
            layout = row["layout"].strip()
            stack = int(row["stack"])
            time_sec = float(row["time(sec)"])

            obj = task_name.split()[-1]

            table[(obj, layout, stack)] = time_sec

    return table


def get_object_type(task):
    """
    從 schedule task 的 object 欄位取得物件種類
    例如：
    A_pink  -> A
    """
    return str(task["object"]).split("_")[0]


def get_layout_code(task):
    """
    將 schedule.csv 的 layout 轉成 robot_task_time.csv 的 layout 格式

    schedule.csv:
    A, B, C

    robot_task_time.csv:
    LA, LB, LC
    """
    layout = str(task["layout"]).strip()

    if layout.startswith("L"):
        return layout

    return "L" + layout


def apply_robot_height_times(
    schedule,
    robot_time_table,
    robot_pick_counts=None,
):
    """
    根據目前 schedule 順序，計算 robot 抓取每種物件的高度 stack，
    並用 robot_task_time.csv 的時間覆蓋 task["robot_standard_time"]
    """
    if robot_time_table is None:
        return schedule

    counts = dict(robot_pick_counts or {})

    # 清掉舊暫存資訊
    for task in schedule:
        task.pop("robot_stack", None)
        task.pop("robot_time_key", None)

    for task in schedule:
        if task.get("command") != "pick and place":
            continue

        if task.get("agent") != "robot":
            continue

        obj = get_object_type(task)
        layout = get_layout_code(task)

        stack = counts.get(obj, 0) + 1
        counts[obj] = stack

        key = (obj, layout, stack)

        if key not in robot_time_table:
            raise KeyError(
                f"robot_task_time.csv 找不到對應時間："
                f"object={obj}, layout={layout}, stack={stack}"
            )

        # 規劃層：直接覆蓋 robot_standard_time
        task["robot_standard_time"] = robot_time_table[key]

        # 給檢查與控制層參考
        task["robot_stack"] = stack
        task["robot_time_key"] = key

    return schedule

def compute_start_times(
    schedule,
    P,
    case=3,
    robot_place=ROBOT_PLACE_TIME,
    frozen_task_ids=None,
    decision_time=0.0,
    fixed_start_by_id=None,
    agent_ready_times=None,
):
    """
    計算每個任務的開始時間 (搭配 precedence matrix)

    追加給 HRC 動態監控用的兩個參數：
    - fixed_start_by_id:
        已完成或正在執行中的 frozen 任務會被固定在實際開始時間，
        不再被重新排程移動 
    - agent_ready_times:
        表示失效決策後兩個代理人下一次可用時間 
        例如 human 在 opp complete 後立即可用，但 robot 可能要等目前任務完成 

    CASE 1:
    - replace 完成後，後續任務才可開始

    CASE 2:
    - pick and place 可在 replace 期間先做前段搬運
    - 但最後 robot_place 時間的 place 動作要與 replace 完成同步

    CASE 3:
    - 在 CASE 2 基礎上，再額外考慮前一個 kit 的節奏限制
    """
    if frozen_task_ids is None:
        frozen_task_ids = set()
    else:
        frozen_task_ids = set(frozen_task_ids)

    fixed_start_by_id = fixed_start_by_id or {}
    agent_ready_times = agent_ready_times or {}

    n = len(schedule)
    agent_free = {
        "human": float(agent_ready_times.get("human", 0.0)),
        "robot": float(agent_ready_times.get("robot", 0.0)),
    }

    start_times = [0.0] * n
    finish_times = [0.0] * n

    def apply_fixed(j):
        """若 task 有固定開始時間，直接寫入並回傳 True"""
        task = schedule[j]
        tid = task.get("ID")
        if tid not in fixed_start_by_id:
            return False

        agent = task["agent"]
        dur = get_task_duration(task)
        start = float(fixed_start_by_id[tid])
        finish = start + dur
        start_times[j] = start
        finish_times[j] = finish
        agent_free[agent] = max(agent_free.get(agent, 0.0), finish)
        return True

    if case == 1:
        for j in range(n):
            if apply_fixed(j):
                continue

            task = schedule[j]
            agent = task["agent"]
            dur = get_task_duration(task)

            preds = [i for i in range(n) if P[i][j] == 1]
            pred_finish = max((finish_times[i] for i in preds), default=0.0)

            start = max(agent_free[agent], pred_finish)
            if task.get("ID") not in frozen_task_ids:
                start = max(start, decision_time)

            finish = start + dur
            start_times[j] = start
            finish_times[j] = finish
            agent_free[agent] = finish

    else:
        kits = extract_kits(schedule)
        task_to_kit = {}
        replace_idx_of_kit = {}

        for kit_id, idxs in kits.items():
            for idx in idxs:
                task_to_kit[idx] = kit_id
                if schedule[idx]["command"] == "replace":
                    replace_idx_of_kit[kit_id] = idx

        computed = [False] * n
        for j in range(n):
            if apply_fixed(j):
                computed[j] = True
                continue

            task = schedule[j]
            agent = task["agent"]
            dur = get_task_duration(task)

            preds = [i for i in range(n) if P[i][j] == 1]
            pred_finish = max((finish_times[i] for i in preds), default=0.0)

            earliest_start = max(agent_free[agent], pred_finish)

            if task["command"] == "pick and place":
                kit_id = task_to_kit[j]
                replace_idx = replace_idx_of_kit[kit_id]
                replace_finish = finish_times[replace_idx]
                sync_start = replace_finish - (dur - robot_place)

                if case == 2:
                    start = max(earliest_start, sync_start)
                elif case == 3:
                    if kit_id > 1:
                        prev_kit_idxs = [idx for idx in kits[kit_id - 1] if computed[idx]]
                        if prev_kit_idxs:
                            prev_kit_last_idx = max(prev_kit_idxs, key=lambda idx: start_times[idx])
                            prev_kit_last_start = start_times[prev_kit_last_idx]
                        else:
                            prev_kit_last_start = 0.0
                        start = max(agent_free[agent], prev_kit_last_start, sync_start)
                    else:
                        start = max(earliest_start, sync_start)
                else:
                    start = earliest_start
            else:
                start = earliest_start

            if task.get("ID") not in frozen_task_ids:
                start = max(start, decision_time)

            finish = start + dur
            start_times[j] = start
            finish_times[j] = finish
            agent_free[agent] = finish
            computed[j] = True

    return start_times

def get_task_duration(task):
    """
    根據任務狀態回傳繪圖/排程用 duration

    優先順序：
    1. _duration_override：正在執行但尚未完成任務的預期剩餘/完成時間 (dispatched)
    2. actual_time：真正已完成任務的實際執行時間 (completed)
    3. standard time：尚未執行任務的預期標準時間 (pending)
    """
    if task.get("_duration_override") is not None:
        return float(task.get("_duration_override") or 0.0)

    if task.get("actual_time") is not None:
        return float(task.get("actual_time") or 0.0)

    if task.get("agent") == "robot":
        return float(task.get("robot_standard_time", 0.0) or 0.0)
    return float(task.get("human_standard_time", 0.0) or 0.0)


def get_finish_times(schedule, start_times):
    return [
        start + get_task_duration(task)
        for task, start in zip(schedule, start_times)
    ]


def get_makespan(
    schedule,
    P,
    frozen_task_ids=None,
    decision_time=0.0,
    fixed_start_by_id=None,
    agent_ready_times=None,
    robot_time_table=None,
    robot_pick_counts=None,
):
    
    apply_robot_height_times(
        schedule,
        robot_time_table,
        robot_pick_counts=robot_pick_counts,
    )

    start_times = compute_start_times(
        schedule,
        P,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
        fixed_start_by_id=fixed_start_by_id,
        agent_ready_times=agent_ready_times,
    )
    finish_times = get_finish_times(schedule, start_times)
    return max(finish_times) if finish_times else 0.0


def insert(schedule, kits, kit_id, frozen_task_ids=None):
    if frozen_task_ids is None:
        frozen_task_ids = set()

    idx = [
        i for i in kits[kit_id]
        if schedule[i]["command"] == "pick and place"
        and schedule[i]["ID"] not in frozen_task_ids
    ]

    if not idx:
        return None

    i = random.choice(idx)
    new_schedule = deepcopy(schedule)

    insert_task = new_schedule.pop(i)
    insert_position = random.choice(idx)

    new_schedule.insert(insert_position, insert_task)

    return new_schedule


def swap(schedule, kits, kit_id, frozen_task_ids=None):
    """
    在指定 kit 內，隨機交換兩個同代理人的 pick & place 任務
    """
    if frozen_task_ids is None:
        frozen_task_ids = set()

    by_agent = {"human": [], "robot": []}
    for i in kits[kit_id]:
        if schedule[i]["ID"] in frozen_task_ids:
            continue
        if schedule[i]["command"] == "pick and place":
            by_agent[schedule[i]["agent"]].append(i)

    candidates = [a for a, idxs in by_agent.items() if len(idxs) >= 2]
    if not candidates:
        return None

    agent = random.choice(candidates)
    i, j = random.sample(by_agent[agent], 2)

    new_schedule = deepcopy(schedule)
    new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

    return new_schedule


def reassign_agent(schedule, kits, kit_id, frozen_task_ids=None):
    if frozen_task_ids is None:
        frozen_task_ids = set()

    idx = [
        i for i in kits[kit_id]
        if schedule[i]["command"] == "pick and place"
        and schedule[i]["ID"] not in frozen_task_ids
    ]

    if not idx:
        return None

    i = random.choice(idx)
    new_schedule = deepcopy(schedule)

    if new_schedule[i]["agent"] == "human":
        new_schedule[i]["agent"] = "robot"
    else:
        new_schedule[i]["agent"] = "human"

    return new_schedule


def shaking(
    schedule,
    kits,
    kit_id,
    frozen_task_ids,
    n_search=2,
    neighborhood_pool=None,
):
    """
    搜尋停滯時的複合擾動。

    由多個基本鄰域操作組合成較大幅度的變動。基本操作仍採隨機選取，
    但若某一操作在指定 kit 無合法候選，會保留上一個有效排程，避免
    None 被傳入下一個操作。
    """
    pool = list(neighborhood_pool or N_SEARCH)
    if not pool or n_search <= 0:
        return None

    new_schedule = deepcopy(schedule)
    changed = False

    for neighborhood_processing in random.choices(pool, k=n_search):
        candidate = neighborhood_processing(
            new_schedule,
            extract_kits(new_schedule),
            kit_id,
            frozen_task_ids,
        )
        if candidate is not None:
            new_schedule = candidate
            changed = True

    return new_schedule if changed else None


def _schedule_signature(schedule):
    """以任務順序與代理人分配建立可雜湊的排程識別值。"""
    return tuple((task.get("ID"), task.get("agent")) for task in schedule)


N_SEARCH = [swap, insert, reassign_agent]


def solver(
    schedule,
    P,
    start_kit_id=1,
    max_iter=200,
    seed=None,
    shaking_threshold=50,
    frozen_task_ids=None,
    decision_time=0.0,
    fixed_start_by_id=None,
    agent_ready_times=None,
    robot_time_csv=ROBOT_TIME_DIR,
    robot_time_table=None,
    robot_pick_counts=None,
    search_strategy="systematic",
    samples_per_neighborhood=1,
    shaking_strength=2,
):
    """
    動態排程求解器。

    search_strategy:
    - "systematic"（建議）：採 Reduced-VNS-style 的鄰域切換。
      依序搜尋 swap -> insert -> reassign；改善後回到第一個鄰域，
      未改善則切換下一個鄰域。每個鄰域內仍隨機抽樣候選解，
      以維持即時性與搜尋多樣性。
    - "random"：保留舊版行為，每次從所有鄰域中均勻隨機選擇一種。

    max_iter:
    - 為維持與舊版相近的計算規模，最大候選評估次數設定為
      max_iter × 可搜尋 kit 數量。

    samples_per_neighborhood:
    - 在同一個鄰域與 kit 中最多隨機抽樣的候選數。
      1 代表每次只評估一個隨機鄰居。

    shaking_threshold / shaking_strength:
    - 連續多次評估未改善時，組合 shaking_strength 次基本鄰域操作，
      產生距離目前最佳解較遠的候選排程。

    回傳格式與舊版相同：
    best, baseline_cost, best_cost, search_time,
    solution_space, history, shake
    """
    if search_strategy not in {"systematic", "random"}:
        raise ValueError(
            "search_strategy 必須是 'systematic' 或 'random'，"
            f"目前收到：{search_strategy!r}"
        )
    if max_iter < 0:
        raise ValueError("max_iter 不可小於 0")
    if samples_per_neighborhood < 1:
        raise ValueError("samples_per_neighborhood 必須至少為 1")
    if shaking_strength < 1:
        raise ValueError("shaking_strength 必須至少為 1")

    if robot_time_table is None:
        robot_time_table = load_robot_task_time_table(robot_time_csv)

    if seed is not None:
        random.seed(seed)

    frozen_task_ids = set(frozen_task_ids or set())
    fixed_start_by_id = fixed_start_by_id or {}
    agent_ready_times = agent_ready_times or {}

    t0 = time.time()

    baseline_cost = get_makespan(
        schedule,
        P,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
        fixed_start_by_id=fixed_start_by_id,
        agent_ready_times=agent_ready_times,
        robot_time_table=robot_time_table,
        robot_pick_counts=robot_pick_counts,
    )

    best = deepcopy(schedule)
    best_cost = baseline_cost

    initial_kits = extract_kits(best)
    eligible_kit_ids = [
        kit_id for kit_id in sorted(initial_kits)
        if kit_id >= start_kit_id
    ]

    history = []
    shake = []
    solution_space = []
    seen_signatures = {_schedule_signature(best)}

    if not eligible_kit_ids or max_iter == 0:
        search_time = time.time() - t0
        return (
            best,
            baseline_cost,
            best_cost,
            search_time,
            solution_space,
            history,
            shake,
        )

    max_evaluations = max_iter * len(eligible_kit_ids)
    evaluations = 0
    attempts = 0
    # 防止大量 None 或重複候選造成無限迴圈。
    max_attempts = max(100, max_evaluations * 20)
    no_improvement_count = 0

    def evaluate_candidate(candidate):
        """
        評估尚未出現過的候選排程。

        回傳 (cost, is_new)。candidate 無效或重複時回傳 (None, False)。
        僅真正執行 makespan 計算時才增加 evaluations。
        """
        nonlocal evaluations

        if candidate is None:
            return None, False

        signature = _schedule_signature(candidate)
        if signature in seen_signatures:
            return None, False

        seen_signatures.add(signature)
        solution_space.append(signature)

        cost = get_makespan(
            candidate,
            P,
            frozen_task_ids=frozen_task_ids,
            decision_time=decision_time,
            fixed_start_by_id=fixed_start_by_id,
            agent_ready_times=agent_ready_times,
            robot_time_table=robot_time_table,
            robot_pick_counts=robot_pick_counts,
        )
        evaluations += 1
        return cost, True

    neighborhood_index = 0
    kit_cursor = 0

    while evaluations < max_evaluations and attempts < max_attempts:
        attempts += 1

        if search_strategy == "random":
            neighborhood_index = random.randrange(len(N_SEARCH))
            kit_id = random.choice(eligible_kit_ids)
        else:
            kit_id = eligible_kit_ids[kit_cursor]

        operator = N_SEARCH[neighborhood_index]
        improved = False
        produced_new_candidate = False

        # 在同一鄰域內保留隨機抽樣；不展開完整鄰域。
        for _ in range(samples_per_neighborhood):
            current_kits = extract_kits(best)
            if kit_id not in current_kits:
                break

            candidate = operator(
                best,
                current_kits,
                kit_id,
                frozen_task_ids,
            )
            candidate_cost, is_new = evaluate_candidate(candidate)

            if not is_new:
                continue

            produced_new_candidate = True

            if candidate_cost < best_cost:
                best = candidate
                best_cost = candidate_cost
                no_improvement_count = 0
                improved = True
                break

            no_improvement_count += 1

        history.append(best_cost)

        if search_strategy == "systematic":
            if improved:
                # VNS neighborhood-change rule：改善後回到第一個鄰域。
                neighborhood_index = 0
                kit_cursor = 0
            else:
                # 先讓同一鄰域作用於所有可調整 kit，再切換下一鄰域。
                kit_cursor += 1
                if kit_cursor >= len(eligible_kit_ids):
                    kit_cursor = 0
                    neighborhood_index += 1
                    if neighborhood_index >= len(N_SEARCH):
                        neighborhood_index = 0
        else:
            # 舊版 random 模式不保留鄰域索引狀態。
            if improved:
                neighborhood_index = 0

        # 無合法候選或一直抽到重複解時，也視為一次未取得進展的嘗試；
        # 但不增加函數評估次數，避免扭曲 max_evaluations 的意義。
        if not produced_new_candidate and not improved:
            no_improvement_count += 1

        if no_improvement_count >= shaking_threshold:
            current_kits = extract_kits(best)
            valid_shake_kits = [
                kit_id for kit_id in eligible_kit_ids
                if kit_id in current_kits
            ]

            if valid_shake_kits and evaluations < max_evaluations:
                shake_kit_id = random.choice(valid_shake_kits)
                shaking_schedule = shaking(
                    best,
                    current_kits,
                    shake_kit_id,
                    frozen_task_ids,
                    n_search=shaking_strength,
                    neighborhood_pool=N_SEARCH,
                )
                shaking_cost, is_new = evaluate_candidate(shaking_schedule)

                if is_new and shaking_cost < best_cost:
                    best = shaking_schedule
                    best_cost = shaking_cost
                    neighborhood_index = 0
                    kit_cursor = 0

                history.append(best_cost)
                shake.append(len(history) - 1)

            no_improvement_count = 0

    search_time = time.time() - t0
    return (
        best,
        baseline_cost,
        best_cost,
        search_time,
        solution_space,
        history,
        shake,
    )


def draw_gantt(
    schedule,
    P,
    title="Schedule",
    frozen_task_ids=None,
    decision_time=0.0,
    fixed_start_by_id=None,
    agent_ready_times=None,
    ax=None,
    show=True,
    skip_actual_time_zero=True,
):

    start_times = compute_start_times(
        schedule,
        P,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
        fixed_start_by_id=fixed_start_by_id,
        agent_ready_times=agent_ready_times,
    )
    colors = {"replace": "tab:blue", "pick and place": "tab:orange"}
    agent_y = {"human": 0, "robot": 1}

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        created_fig = True

    for task, start in zip(schedule, start_times):
        actual_time = task.get("actual_time")
        try:
            if skip_actual_time_zero and actual_time is not None and float(actual_time) == 0.0:
                continue
        except (TypeError, ValueError):
            pass

        dur = get_task_duration(task)
        if dur is None or dur <= 0:
            continue

        agent = task["agent"]
        y = agent_y.get(agent)
        if y is None:
            continue

        tid = task.get("ID", "")

        ax.barh(
            y,
            dur,
            left=start,
            height=0.55,
            color=colors.get(task["command"], "gray"),
            edgecolor="black"
        )

        ax.text(
            start + dur / 2,
            y,
            str(tid),
            va="center",
            ha="center",
            color="white",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["human", "robot"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    if created_fig:
        plt.tight_layout()
        if show:
            plt.show()

def record_solution_space(schedule, solution_space):
    """
    記錄每個排程解的搜尋空間 
    - 如果兩個排程解的 ID 排序及代理人分配相同，則視為相同解 
    - 使用元組表示排程解，並將其加入 solution_space 中以避免重複 

    參數說明：
    - schedule          : 當前排程解 
    - solution_space    : 記錄解的集合，避免重複解
    
    返回：
    - solution_space   : 更新過的解的集合
    """
    
    # 以tuple表示排程解（ID 排序 + 代理人分配）
    schedule_signature = tuple((task["ID"], task["agent"]) for task in schedule)
    
    if schedule_signature not in solution_space:
        solution_space.append(schedule_signature)
    
    return solution_space

if __name__ == "__main__":
    from read_schedule import load_schedule_csv
    from precedence_matrix import build_precedence_matrix

    schedule = load_schedule_csv()
    P = build_precedence_matrix(schedule)

    robot_time_table = load_robot_task_time_table(ROBOT_TIME_DIR)

    # ===== 靜態排程 =====
    static_cost = get_makespan(schedule,P, robot_time_table=robot_time_table)
    
    # ===== 動態排程=====
    new_schedule, base_cost, new_cost, search_time, solution_space, history, shake = solver(
        schedule,
        P,
        start_kit_id=1,
        max_iter=200,
        #seed=200
        robot_time_table=robot_time_table,
    )
    
    # Gantt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    draw_gantt(schedule, P, title="Initial schedule",ax=axes[0], show=False)
    axes[0].tick_params(labelbottom=True)
    draw_gantt(new_schedule, P, title="Result of VSN", ax=axes[1], show=False)

    print("Baseline makespan :", base_cost)
    print("Dynamic makespan  :", new_cost)
    print(f"VSN search time   :{search_time:.2f}")
    print(f"Number of unique solutions: {len(solution_space)}")

    plt.tight_layout()
    plt.show()

    # Convergence curve
    plt.figure(figsize=(10, 3))
    plt.plot(history)
    if shake:
        plt.scatter(shake, [history[i] for i in shake], s=5)

    plt.title("Convergence (best makespan over iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Best makespan")
    plt.tight_layout()
    plt.show()
