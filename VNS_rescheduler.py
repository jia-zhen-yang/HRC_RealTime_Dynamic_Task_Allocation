# VNS scheduler
# 鄰域依序搜尋

import time
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path

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

    robot_pick_counts:
    - 表示目前 robot 已經抓過幾個物件
    - 例如 {"A": 1, "B": 2}
    - 若從完整排程一開始算，傳 None 即可

    """
    if robot_time_table is None:
        return schedule

    counts = dict(robot_pick_counts or {})

    # 先清掉舊的暫存資訊
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

def compute_start_times(schedule, P, case = 3 ,robot_place = 2.0):
    """
    計算每個任務的開始時間 (搭配precedence matrix)
    CASE 1:
    replace後才可開始所有pick and place
    CASE 2:
    pick and place 可在 replace 期間先做前段搬運
    最後1秒(假設)的 place 動作一定要等該 kit 的 replace 完成後才能執行
    """
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

            finish = start + dur

            start_times[j] = start
            finish_times[j] = finish
            agent_free[agent] = finish
            computed[j] = True

    return start_times

# 先假設機械手臂執行任務時間為亂數
def get_task_duration(task):
    """
    根據任務指派的agent，回傳實際執行時間
    """
    if task["agent"] == "robot":
        return task["robot_standard_time"]
    else:
        return task["human_standard_time"]

def get_makespan(
    schedule,
    P,
    robot_time_table=None,
    robot_pick_counts=None,
):
    apply_robot_height_times(
        schedule,
        robot_time_table,
        robot_pick_counts=robot_pick_counts,
    )

    start_times = compute_start_times(schedule, P)
    finish_times = [
        start + get_task_duration(task)
        for task, start in zip(schedule, start_times)
    ]
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


# 由變動程度較小至較大的鄰域順序：
# 1. swap：僅改變同一代理人的局部執行順序
# 2. insert：改變一項任務與多項任務的相對位置
# 3. reassign_agent：改變代理人負荷與任務工時結構
N_SEARCH = [swap, insert, reassign_agent]


def solver(
    schedule,
    P,
    start_kit_id=1,
    max_iter=10,
    seed=None,
    shaking_threshold=5,
    frozen_task_ids=None,
    robot_time_csv=ROBOT_TIME_DIR,
    robot_time_table=None,
    robot_pick_counts=None,
    search_strategy="systematic",
    samples_per_neighborhood=1,
    shaking_strength=2,
):
    """
    排程求解器，採系統性鄰域切換與鄰域內隨機抽樣。

    search_strategy:
    - "systematic"（預設）：依序搜尋 swap -> insert -> reassign。
      候選解改善後回到第一個鄰域；未改善時，先讓同一鄰域作用於
      所有可調整 kit，再切換至下一個鄰域。
    - "random"：保留比較用途，每次隨機選擇一種鄰域與一個 kit。

    max_iter:
    - 最大候選評估次數設定為 max_iter × 可搜尋 kit 數量，
      以維持與舊版相近的搜尋規模。

    samples_per_neighborhood:
    - 同一鄰域與 kit 中最多抽樣的候選數。鄰域種類依序切換，
      但鄰域內的任務與位置仍採隨機選擇。

    shaking_threshold / shaking_strength:
    - 連續多次未改善時，組合 shaking_strength 次基本鄰域操作，
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

    t0 = time.time()

    baseline_cost = get_makespan(
        schedule,
        P,
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

        # 同一鄰域內保留隨機抽樣，不完整展開所有鄰近解。
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
                # 改善後回到第一個鄰域與第一個可搜尋 kit。
                neighborhood_index = 0
                kit_cursor = 0
            else:
                # 同一鄰域先作用於全部 kit，再切換下一個鄰域。
                kit_cursor += 1
                if kit_cursor >= len(eligible_kit_ids):
                    kit_cursor = 0
                    neighborhood_index += 1
                    if neighborhood_index >= len(N_SEARCH):
                        neighborhood_index = 0
        else:
            # random 模式不保留鄰域索引狀態。
            if improved:
                neighborhood_index = 0

        # 沒有合法候選或只抽到重複解，也視為一次未取得進展。
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

def draw_gantt(schedule, P, title="Schedule", ax=None, show=True, skip_dur_zero=True,):
    '''排程結果'''

    start_times = compute_start_times(schedule,P)
    colors = {"replace": "tab:blue", "pick and place": "tab:orange"}

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
        created_fig = True

    for task, start in zip(schedule, start_times):
        dur = get_task_duration(task)
        agent = task["agent"]
        tid = task.get("ID", "")

        try:
            if skip_dur_zero and dur is not None and float(dur) == 0.0:
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
    驗證目前 schedule 中，每個 robot pick and place 任務對應到的：
    ID、object、layout、stack、查表 key、robot_standard_time
    """  

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

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
            f"robot_standard_time={task.get('robot_standard_time')}"
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
            f"robot_standard_time={task.get('robot_standard_time')}"
        )

if __name__ == "__main__":
    from read_schedule import load_schedule_csv
    from precedence_matrix import build_precedence_matrix
    robot_time_table = load_robot_task_time_table(ROBOT_TIME_DIR)

    schedule = load_schedule_csv()
    P = build_precedence_matrix(schedule)

    # ===== 靜態排程 =====
    static_cost = get_makespan(
        schedule,
        P,
        robot_time_table=robot_time_table,
    )
    # print_robot_time_check(
    #     schedule,
    #     robot_time_table,
    #     title="Initial schedule robot time check"
    # )
    
    # ===== 動態排程=====
    new_schedule, base_cost, new_cost, search_time, solution_space, history, shake = solver(
        schedule,
        P,
        start_kit_id=1,
        max_iter=15,
        #seed=200
        shaking_threshold = 10,
        robot_time_table=robot_time_table,
    )
    # print_robot_time_check(
    #     new_schedule,
    #     robot_time_table,
    #     title="New schedule robot time check"
    # )
    
    # Gantt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    draw_gantt(schedule, P, title="Initial schedule",ax=axes[0], show=False)
    axes[0].tick_params(labelbottom=True)
    draw_gantt(new_schedule, P, title="Result of VSN",ax=axes[1], show=False)

    print("Baseline makespan :", base_cost)
    print("Dynamic makespan  :", new_cost)
    print(f"VSN search time   :{search_time:.2f}")
    print(f"Number of unique solutions: {len(solution_space)}")

    plt.tight_layout()
    plt.show()

    # Convergence curve
    plt.figure(figsize=(10, 3))
    plt.plot(history)
    # if shake:
    #     plt.scatter(shake, [history[i] for i in shake], s=5)

    plt.title("Convergence (best makespan over iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Best makespan")
    plt.tight_layout()
    plt.show()
