# 動態排程solver (VNS)

import time
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path

ROBOT_PLACE_TIME = 2.0
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

def compute_start_times(schedule, P, case = 3 ,robot_place = ROBOT_PLACE_TIME,frozen_task_ids=None, decision_time=0.0):
    """
    計算每個任務的開始時間 (搭配precedence matrix)
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

    n = len(schedule)
    agent_free = {"human": 0.0, "robot": 0.0}

    start_times = [0.0] * n
    finish_times = [0.0] * n

    if case == 1:
        for j in range(n):
            task = schedule[j]
            agent = task["agent"]
            dur = get_task_duration(task)

            # 找所有必須在 j 之前完成的任務 i
            preds = [i for i in range(n) if P[i][j] == 1]

            # 計算前置任務中最晚完成的時間
            pred_finish = 0.0
            if preds:
                pred_finish = max(finish_times[i] for i in preds)

            # 任務 j 可以開始的時間，需同時滿足
            # 1. 該 agent 已經空閒
            # 2. 所有前置任務都已完成
            start = max(agent_free[agent], pred_finish)

            start_times[j] = start
            finish_times[j] = start + dur
            agent_free[agent] = finish_times[j]
    
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
                        start = max(agent_free[agent],prev_kit_last_start, sync_start)
                    else:
                        start = max(earliest_start,sync_start)
            else:
                start = earliest_start

            if task["ID"] not in frozen_task_ids:
                start = max(start, decision_time)

            finish = start + dur

            start_times[j] = start
            finish_times[j] = finish
            agent_free[agent] = finish
            computed[j] = True

    return start_times

def get_task_duration(task):
    """
    根據任務指派的agent，回傳標準/實際執行時間
    """
    if task["actual_time"] is not None:
        return task["actual_time"]

    if task["agent"] == "robot":
        return task["robot_standard_time"]
    else:
        return task["human_standard_time"]

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
    robot_time_table=None,
    robot_pick_counts=None,
):
    apply_robot_height_times(
        schedule,
        robot_time_table,
        robot_pick_counts=robot_pick_counts,
    )

    start_times = compute_start_times(schedule, P,frozen_task_ids=frozen_task_ids,decision_time=decision_time)
    finish_times = get_finish_times(schedule, start_times)
    return max(finish_times)


def insert(schedule, kits, kit_id, frozen_task_ids=None):
    if frozen_task_ids is None:
        frozen_task_ids = set()

    # 允許被reaasign的任務
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

    # 允許被交換位置的任務
    by_agent = {"human": [], "robot": []}
    for i in kits[kit_id]:
        if schedule[i]["ID"] in frozen_task_ids:
            continue
        if schedule[i]["command"] == "pick and place":
            by_agent[schedule[i]["agent"]].append(i)

    # 檢查是否有足夠任務可交換
    candidates = [a for a, idxs in by_agent.items() if len(idxs) >= 2]
    if not candidates:
        return None

    agent = random.choice(candidates)
    i, j = random.sample(by_agent[agent], 2)
    new_schedule = deepcopy(schedule) # new_schedule = schedule的話會指向同一塊記憶體
    new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
    
    return new_schedule

def reassign_agent(schedule, kits, kit_id, frozen_task_ids=None):
    if frozen_task_ids is None:
        frozen_task_ids = set()

    # 允許被reaasign的任務
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

def shaking(schedule, kits, kit_id, frozen_task_ids, n_search = 2):
    '''
    當當搜尋停滯時，Shaking 強制打破局部最佳解，引入更多隨機性
    定義shaking為多種鄰域操作的組合
    '''

    # 定義所有鄰域操作
    global N_SEARCH

    # 隨機選擇N個不同的鄰域操作
    new_schedule = deepcopy(schedule)
    shaking_processing = list(random.choices(N_SEARCH, k=n_search))
    for neighborhood_processing in shaking_processing:
        if new_schedule is None:
            new_schedule = deepcopy(schedule)
        new_schedule = neighborhood_processing(new_schedule, kits, kit_id, frozen_task_ids)  # 第一次操作

    return new_schedule

# solver
N_SEARCH = [insert, swap, reassign_agent]
def solver(
    schedule,
    P,
    start_kit_id=1, #先指定動態排程起始kit
    max_iter=200,
    seed=None,
    shaking_threshold = 50,
    frozen_task_ids=None,
    decision_time=0.0,
    robot_time_csv=ROBOT_TIME_DIR,
    robot_time_table=None,
    robot_pick_counts=None,
):
    """
    動態排程求解器（簡化版 VSN / local search）

    參數說明：
    - schedule      : 原始靜態排程
    - start_kit_id  : 從第幾個 kit 開始允許動態調整
                      在此之前的 kit 視為已完成，不可變動
    - max_iter      : 每個 kit 進行鄰域搜尋的次數
    - seed          : 隨機種子（方便重現結果）
    - refresh_kits  : 避免 kit index 漂移 (目前不用因為kit順序固定)
    - shaking_threshold : 搜索停滯時啟動shaking
    - frozen_task_ids : 不進行動態調整的任務(實務上意義為已完成)
    - decision_time   : 動態重排生效時間；未凍結任務不得早於此時刻開始

    回傳：
    - best          : 動態調整後的最佳排程
    - baseline_cost : 原始靜態排程的 makespan
    - best_cost     : 動態排程後的 makespan
    - search_time   : 動態排程花費的時間
    """

    global N_SEARCH

    if robot_time_table is None:
        robot_time_table = load_robot_task_time_table(robot_time_csv)

    solution_space = list()

    if seed is not None:
        random.seed(seed)

    t0 = time.time()

    best = deepcopy(schedule)
    baseline_cost = get_makespan(
        best,
        P,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
        robot_time_table=robot_time_table,
        robot_pick_counts=robot_pick_counts,
    )
    best_cost = baseline_cost

    kits = extract_kits(best)

    history = []
    shake = []
    no_improvement_count = 0  # 記錄無改善的次數

    for _ in range(max_iter):
        for kit_id in sorted(kits.keys()):
            if kit_id < start_kit_id:
                continue

            neighborhood_processing = random.choice(N_SEARCH)
            cand = neighborhood_processing(best, kits, kit_id, frozen_task_ids)

            history.append(best_cost)
            if cand is None:
                continue

            solution_space = record_solution_space(cand, solution_space)
            c = get_makespan(
                cand,
                P,
                frozen_task_ids=frozen_task_ids,
                decision_time=decision_time,
                robot_time_table=robot_time_table,
                robot_pick_counts=robot_pick_counts,
            )
            if c < best_cost:
                best, best_cost = cand, c
                #print(neighborhood_processing.__name__)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        
            # 如果多次無改善，進行 shaking 操作 (強制擾動)
            if no_improvement_count >= shaking_threshold:
                # shaking 不能找到更差的解
                shaking_schedule  = shaking(best, kits, kit_id, frozen_task_ids)

                if shaking_schedule is not None:
                    shaking_cost = get_makespan(
                        shaking_schedule,
                        P,
                        frozen_task_ids=frozen_task_ids,
                        decision_time=decision_time,
                        robot_time_table=robot_time_table,
                        robot_pick_counts=robot_pick_counts,
                    )
                    if shaking_cost < best_cost:
                        best = shaking_schedule
                        best_cost = shaking_cost

                # shaking不一定要找到更佳的解
                '''
                best  = shaking(best, kits, kit_id, frozen_task_ids)
                if best is None:
                    continue
                best_cost  = get_makespan(best, P,frozen_task_ids=frozen_task_ids,decision_time=decision_time)
                '''
                
                no_improvement_count = 0
                shake.append(len(history) - 1)

    search_time = time.time() - t0
    return best, baseline_cost, best_cost, search_time, solution_space, history, shake

def draw_gantt(schedule, P, title="Schedule",frozen_task_ids=None, decision_time=0.0, ax=None, show=True):
    '''排程結果'''

    start_times = compute_start_times(schedule,P,frozen_task_ids=frozen_task_ids, decision_time=decision_time)
    colors = {"replace": "tab:blue", "pick and place": "tab:orange"}

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        created_fig = True

    for task, start in zip(schedule, start_times):
        dur = get_task_duration(task)
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

        # 在 bar 中間標示任務 ID
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

def record_solution_space(schedule, solution_space):
    """
    記錄每個排程解的搜尋空間。
    - 如果兩個排程解的 ID 排序及代理人分配相同，則視為相同解。
    - 使用元組表示排程解，並將其加入 solution_space 中以避免重複。

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

def print_robot_time_check(schedule, robot_time_table, title="Robot time check"):
    """
    檢查目前 schedule 中，每個 robot pick and place 任務對應到的：
    ID、object、layout、stack、查表 key、robot_standard_time、actual_time、duration
    """

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for task in schedule:
        if task.get("command") != "pick and place":
            continue

        if task.get("agent") != "robot":
            continue

        print(
            "before apply_robot_height_times\n"
            f"ID={task.get('ID')}, "
            f"object={task.get('object')}, "
            f"layout={task.get('layout')}, "
            f"robot_stack={task.get('robot_stack')}, "
            f"robot_time_key={task.get('robot_time_key')}, "
            f"robot_standard_time={task.get('robot_standard_time')}, "
            f"actual_time={task.get('actual_time')}, "
            f"duration_used={get_task_duration(task)}"
        )

    apply_robot_height_times(schedule, robot_time_table)

    for task in schedule:
        if task.get("command") != "pick and place":
            continue

        if task.get("agent") != "robot":
            continue

        print(
            f"ID={task.get('ID')}, "
            f"object={task.get('object')}, "
            f"layout={task.get('layout')}, "
            f"robot_stack={task.get('robot_stack')}, "
            f"robot_time_key={task.get('robot_time_key')}, "
            f"robot_standard_time={task.get('robot_standard_time')}, "
            f"actual_time={task.get('actual_time')}, "
            f"duration_used={get_task_duration(task)}"
        )

if __name__ == "__main__":
    from read_schedule import load_schedule_csv
    from precedence_matrix import build_precedence_matrix

    robot_time_table = load_robot_task_time_table(ROBOT_TIME_DIR)

    schedule = load_schedule_csv()
    P = build_precedence_matrix(schedule)

    # ===== 靜態排程 =====
    static_cost = get_makespan(schedule,P, robot_time_table=robot_time_table,)
    
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
    draw_gantt(schedule, P, title="Initial schedule", ax=axes[0], show=False)
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
