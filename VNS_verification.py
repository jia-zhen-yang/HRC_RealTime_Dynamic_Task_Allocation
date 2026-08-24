# 離線模擬最佳解
# CASE 1 kit之間獨立
# CASE 2 機械手臂可提早做pick&place(pick&place不須在replace kit box後才開始)
# 以上CASE每個kit最佳解組合為全域最佳解
# CASE 3 串接kit 機械手臂可提早pick&place至前一kit最後一個任務開始後 (局部最佳非全域最佳)
# 尚未加入各類物件分配限制且kit順序固定
# 調整precedence matrix限制與compute_start_time的參數case 

import itertools
from copy import deepcopy
from tqdm import tqdm
import time
import sys
from tqdm.auto import tqdm

from read_schedule import load_schedule_csv
from precedence_matrix import build_precedence_matrix
from VNS_rescheduler import (
    solver,
    extract_kits,
    get_makespan,
    draw_gantt,
    load_robot_task_time_table,
    apply_robot_height_times,
    ROBOT_TIME_DIR,
)

n_sub_task = 3
n_agent = 2

def kit_variants(kit_block):
    """
    kit_block: 一個 kit 的 tasks list，包含 replace + 3個 pick&place
    回傳所有候選 block
    """
    replace = kit_block[0]  # 假設 replace 固定在第一個
    picks = [t for t in kit_block if t["command"] == "pick and place"]
    if len(picks) != 3:
        raise ValueError(f"Expected 3 pick&place, got {len(picks)}")

    # permutations
    global n_agent, n_sub_task
    for perm in itertools.permutations(picks, n_sub_task):
        # agent assignments
        for mask in range(n_agent**n_sub_task):
            new_picks = []
            for bit, t in enumerate(perm):
                nt = deepcopy(t)
                # >> 把二進位往右移動 bit 格
                # 取得 mask 的第 bit 位是 0 還是 1 (各任務分配的代理人組合)
                nt["agent"] = "robot" if ((mask >> bit) & 1) else "human"
                new_picks.append(nt)
                #print(nt['agent'])
            #print("-------")

            yield [deepcopy(replace)] + new_picks

_EQ_UID_KEY = "__eq_uid"

def _get_readable_task_id(task, fallback):
    """
    取得任務識別用的 ID。
    優先使用任務本身的 ID 欄位；若沒有，則用 fallback。
    這只用於判斷等價排程，不會影響原本任務資料。
    """
    for key in ("task_id", "Task ID", "Task_ID", "ID", "id", "task", "Task"):
        if key in task:
            return str(task[key])

    return str(fallback)


def unique_kit_variants_by_agent_sequence(
    kit_block,
    skip_all_robot=True,
):
    """
    產生單一 block 的去重候選解。

    等價判斷邏輯：
    若兩個 variant 的 human 任務內部順序相同，
    且 robot 任務內部順序相同，則視為等價，
    只保留第一個代表 variant。

    例如：
    1=robot, 2=human, 3=human

    [1, 2, 3]
    [2, 1, 3]
    [2, 3, 1]

    皆對應到：
    human_sequence = (2, 3)
    robot_sequence = (1)

    因此只保留一個。

    skip_all_robot=True 時：
    若該 block 內所有 pick&place 都分配給 robot，則直接排除。
    """

    work_block = deepcopy(kit_block)

    # 為每個 pick&place 任務加上暫時的唯一識別碼
    pick_counter = 0
    for task in work_block:
        if task.get("command") == "pick and place":
            readable_id = _get_readable_task_id(task, fallback=f"pick_{pick_counter}")
            task[_EQ_UID_KEY] = f"{pick_counter}_{readable_id}"
            pick_counter += 1

    seen_signatures = set()
    unique_variants = []

    raw_count = 0
    duplicate_count = 0
    all_robot_filtered_count = 0

    for variant in kit_variants(work_block):
        raw_count += 1

        picks = [
            task for task in variant
            if task.get("command") == "pick and place"
        ]

        human_sequence = tuple(
            task[_EQ_UID_KEY] for task in picks
            if task.get("agent") == "human"
        )

        robot_sequence = tuple(
            task[_EQ_UID_KEY] for task in picks
            if task.get("agent") == "robot"
        )

        # 排除全部 pick&place 都給 robot 的 block
        if skip_all_robot and len(human_sequence) == 0:
            all_robot_filtered_count += 1
            continue

        signature = (human_sequence, robot_sequence)

        if signature in seen_signatures:
            duplicate_count += 1
            continue

        seen_signatures.add(signature)

        cleaned_variant = deepcopy(variant)

        # 移除暫時加上的內部識別欄位，避免影響後續程式
        for task in cleaned_variant:
            task.pop(_EQ_UID_KEY, None)

        unique_variants.append(cleaned_variant)

    stats = {
        "raw_count": raw_count,
        "unique_count": len(unique_variants),
        "duplicate_count": duplicate_count,
        "all_robot_filtered_count": all_robot_filtered_count,
        "skip_all_robot": skip_all_robot,
    }

    return unique_variants, stats

def compute_kit_makespan(schedule, k_kits=6, robot_time_table=None, robot_time_csv=ROBOT_TIME_DIR, robot_pick_counts=None,):
    """
    計算目前每個 kit 的 makespan
    """
    if robot_time_table is None:
        robot_time_table = load_robot_task_time_table(robot_time_csv)
    
    apply_robot_height_times(schedule, robot_time_table, robot_pick_counts)

    kits = extract_kits(schedule)
    kit_ids = sorted(kits.keys())
    
    kit_makespans = {}  # 用來存放每個 kit 的 makespan
    
    for kit_id in kit_ids[:k_kits]:  # 計算前 k 個 kit
        kit_schedule = [schedule[i] for i in kits[kit_id]]  # 取得該 kit 的排程
        kit_P = build_precedence_matrix(kit_schedule)  # 計算該 kit 的 precedence matrix
        
        # 計算該 kit 的 makespan
        kit_makespans[kit_id] = get_makespan(
            kit_schedule,
            kit_P,
        )
    
    return kit_makespans

def exhaustive_global_solution_by_kit_product(
    schedule,
    P,
    k_kits=4,
    robot_time_table=None,
    robot_time_csv=ROBOT_TIME_DIR,
    robot_pick_counts=None,
    rebuild_P_each_candidate=False,
    case_1=False,
    max_eval_candidates=None,
    print_best_update=True,
):
    """
    全域窮舉版本：
    - 每個 kit 仍固定以 replace 作為該 kit 的起始任務。
    - 每個 kit 內的 3 個 pick&place 會窮舉所有順序與 agent 分配。
    - 不再逐 kit 貪婪保留最佳結果，而是將所有 kit variant 做 Cartesian product。

    參數說明：
    - k_kits: 要窮舉前幾個 kit
    - rebuild_P_each_candidate:
        False：沿用外部 P，速度較快，與你目前 optimal_solution() 寫法較接近。
        True ：每個候選排程都重建 precedence matrix，較嚴謹但更慢。
    - max_eval_candidates:
        若只想先測速，可設定例如 100000。
        若要完整跑完前 4 個 kit，設為 None。
    """

    if robot_time_table is None:
        robot_time_table = load_robot_task_time_table(robot_time_csv)

    kits = extract_kits(schedule)
    kit_ids = sorted(kits.keys())
    kit_ids = kit_ids[:k_kits]

    kit_blocks = []
    for kid in kit_ids:
        idxs = kits[kid]
        block = [deepcopy(schedule[i]) for i in idxs]

        if block[0]["command"] != "replace":
            raise ValueError(f"Kit {kid} does not start with replace. Check schedule format.")

        kit_blocks.append(block)

    # 先產生每個 kit 的所有 variant
    variant_lists = []
    total_candidates = 1

    for block in kit_blocks:
        variants = list(kit_variants(block))
        variant_lists.append(variants)
        total_candidates *= len(variants)

    if max_eval_candidates is not None:
        progress_total = min(total_candidates, max_eval_candidates)
    else:
        progress_total = total_candidates

    print("===================================")
    print("Global exhaustive search by kit product")
    print(f"Number of kits        : {len(kit_blocks)}")
    print(f"Kit IDs               : {kit_ids}")
    print(f"Variants per kit      : {[len(v) for v in variant_lists]}")
    print(f"Total candidates      : {total_candidates:,}")
    print(f"Evaluation limit      : {progress_total:,}")
    print(f"Rebuild P each cand.  : {rebuild_P_each_candidate}")
    print("===================================")

    best_schedule = None
    best_cost = float("inf")
    best_candidate_index = None

    start_time = time.time()
    evaluated_count = 0

    iterator = itertools.product(*variant_lists)

    for candidate_index, combo in enumerate(
        tqdm(
            iterator,
            total=progress_total,
            desc="Equivalence-pruned exhaustive search",
            unit="candidate",
            dynamic_ncols=True,
            leave=True,
            disable=False,
            file=sys.stdout,
            mininterval=0.5,
            miniters=1,
        ),
        start=1,
    ):
        if max_eval_candidates is not None and evaluated_count >= max_eval_candidates:
            break

        cand = []
        for block_variant in combo:
            cand.extend(deepcopy(block_variant))

        if rebuild_P_each_candidate:
            cand_P = build_precedence_matrix(cand, case_1=case_1)
        else:
            cand_P = P

        c = get_makespan(
            cand,
            cand_P,
            robot_time_table=robot_time_table,
            robot_pick_counts=robot_pick_counts,
        )

        evaluated_count += 1

        if c < best_cost:
            best_cost = c
            best_schedule = deepcopy(cand)
            best_candidate_index = candidate_index

            if print_best_update:
                tqdm.write(
                    f"[BEST UPDATE] candidate={candidate_index:,}, makespan={best_cost:.3f}"
                )

    elapsed = time.time() - start_time
    speed = evaluated_count / elapsed if elapsed > 0 else 0

    if speed > 0:
        estimated_full_time = total_candidates / speed
    else:
        estimated_full_time = None

    print("===================================")
    print(f"Evaluated candidates : {evaluated_count:,}")
    print(f"Best candidate index : {best_candidate_index:,}")
    print(f"Best makespan        : {best_cost:.3f}")
    print(f"Elapsed time         : {elapsed:.2f} sec")
    print(f"Speed                : {speed:,.2f} candidates/sec")

    if estimated_full_time is not None:
        print(f"Estimated full time  : {estimated_full_time / 3600:.2f} hours")

    print("===================================")

    return best_schedule, best_cost

def exhaustive_global_solution_by_unique_agent_sequences(
    schedule,
    P,
    k_kits=4,
    robot_time_table=None,
    robot_time_csv=ROBOT_TIME_DIR,
    robot_pick_counts=None,
    rebuild_P_each_candidate=False,
    case_1=False,
    skip_all_robot_block=True,
    max_eval_candidates=None,
    print_best_update=True,
):
    """
    等價排程去重後的全域窮舉版本。

    與 exhaustive_global_solution_by_kit_product 的差異：
    1. 每個 block 先由 48 種 raw variants 去重。
    2. 若兩個 variant 的 human 內部順序相同，robot 內部順序相同，
       則視為等價，只保留一個。
    3. 可排除 block 內 pick&place 全部分配給 robot 的結果。
    4. 再對各 block 的 unique variants 做 Cartesian product。
    """

    if robot_time_table is None:
        robot_time_table = load_robot_task_time_table(robot_time_csv)

    kits = extract_kits(schedule)
    kit_ids = sorted(kits.keys())
    kit_ids = kit_ids[:k_kits]

    kit_blocks = []
    for kid in kit_ids:
        idxs = kits[kid]
        block = [deepcopy(schedule[i]) for i in idxs]

        if block[0]["command"] != "replace":
            raise ValueError(f"Kit {kid} does not start with replace. Check schedule format.")

        kit_blocks.append(block)

    variant_lists = []
    block_stats = []

    raw_total_candidates = 1
    reduced_total_candidates = 1

    for kid, block in zip(kit_ids, kit_blocks):
        variants, stats = unique_kit_variants_by_agent_sequence(
            block,
            skip_all_robot=skip_all_robot_block,
        )

        if len(variants) == 0:
            raise ValueError(f"Kit {kid} has no valid variants after pruning.")

        variant_lists.append(variants)
        block_stats.append((kid, stats))

        raw_total_candidates *= stats["raw_count"]
        reduced_total_candidates *= len(variants)

    if max_eval_candidates is not None:
        progress_total = min(reduced_total_candidates, max_eval_candidates)
    else:
        progress_total = reduced_total_candidates

    reduction_ratio = (
        raw_total_candidates / reduced_total_candidates
        if reduced_total_candidates > 0
        else float("inf")
    )

    print("===================================")
    print("Global exhaustive search with equivalence pruning")
    print(f"Number of kits              : {len(kit_blocks)}")
    print(f"Kit IDs                     : {kit_ids}")
    print(f"Skip all-robot block         : {skip_all_robot_block}")
    print(f"Rebuild P each candidate     : {rebuild_P_each_candidate}")
    print("-----------------------------------")

    for kid, stats in block_stats:
        print(
            f"Kit {kid}: "
            f"raw={stats['raw_count']}, "
            f"unique={stats['unique_count']}, "
            f"duplicates={stats['duplicate_count']}, "
            f"all_robot_filtered={stats['all_robot_filtered_count']}"
        )

    print("-----------------------------------")
    print(f"Raw total candidates        : {raw_total_candidates:,}")
    print(f"Reduced total candidates    : {reduced_total_candidates:,}")
    print(f"Reduction ratio             : {reduction_ratio:.2f}x")
    print(f"Evaluation limit            : {progress_total:,}")
    print("===================================")

    best_schedule = None
    best_cost = float("inf")
    best_candidate_index = None

    start_time = time.time()
    evaluated_count = 0

    iterator = itertools.product(*variant_lists)

    for candidate_index, combo in enumerate(
        tqdm(
            iterator,
            total=progress_total,
            desc="Equivalence-pruned exhaustive search",
            unit="candidate",
        ),
        start=1,
    ):
        if max_eval_candidates is not None and evaluated_count >= max_eval_candidates:
            break

        cand = []
        for block_variant in combo:
            cand.extend(deepcopy(block_variant))

        if rebuild_P_each_candidate:
            cand_P = build_precedence_matrix(cand, case_1=case_1)
        else:
            cand_P = P

        c = get_makespan(
            cand,
            cand_P,
            robot_time_table=robot_time_table,
            robot_pick_counts=robot_pick_counts,
        )

        evaluated_count += 1

        if c < best_cost:
            best_cost = c
            best_schedule = deepcopy(cand)
            best_candidate_index = candidate_index

            if print_best_update:
                tqdm.write(
                    f"[BEST UPDATE] candidate={candidate_index:,}, makespan={best_cost:.3f}"
                )

    elapsed = time.time() - start_time
    speed = evaluated_count / elapsed if elapsed > 0 else 0

    estimated_full_time = (
        reduced_total_candidates / speed
        if speed > 0
        else None
    )

    print("===================================")
    print(f"Evaluated candidates         : {evaluated_count:,}")
    print(f"Best candidate index         : {best_candidate_index:,}")
    print(f"Best makespan                : {best_cost:.3f}")
    print(f"Elapsed time                 : {elapsed:.2f} sec")
    print(f"Speed                        : {speed:,.2f} candidates/sec")

    if estimated_full_time is not None:
        print(f"Estimated reduced full time  : {estimated_full_time / 3600:.4f} hours")

    print("===================================")

    return best_schedule, best_cost

def optimal_solution(
    schedule,
    P,
    case_3=False,
    robot_time_table=None,
    robot_time_csv=ROBOT_TIME_DIR,
    robot_pick_counts=None,
):
    """
    CASE 1 or 2 : 逐 kit 找最佳 variant（48 種），串起來得到全域最佳
    CASE 3 : CASE 1 2的最佳解 + VNS
    """
    if robot_time_table is None:
        robot_time_table = load_robot_task_time_table(robot_time_csv)

    kits = extract_kits(schedule)
    kit_ids = sorted(kits.keys())

    # 擷取每個 kit 的原始 block 
    kit_blocks = []
    for kid in kit_ids:
        idxs = kits[kid]
        block = [schedule[i] for i in idxs]
        if block[0]["command"] != "replace":
            raise ValueError(f"Kit {kid} does not start with replace. Check schedule format.")
        kit_blocks.append(block)

    best_schedule = []
    best_cost_so_far = float("inf")
    for kid, block in zip(kit_ids, kit_blocks):
        best_candidate_schedule = None
        best_cost = float("inf") #infinity

        for variant in kit_variants(block):
            cand = deepcopy(best_schedule) + deepcopy(variant)
            c = get_makespan(
                cand,
                P,
                robot_time_table=robot_time_table,
                robot_pick_counts=robot_pick_counts,
            )

            if c < best_cost:
                best_cost = c
                best_candidate_schedule = deepcopy(cand)

        best_schedule = deepcopy(best_candidate_schedule)
        best_cost_so_far = best_cost
        #print(f"[kit {kid}] best partial makespan = {best_cost_so_far:.3f}")
    
    # 目前的工時參數下，用block窮舉就可以
    if case_3:
        best_schedule, _, best_cost_so_far, _, _, _, _ = solver(
            best_schedule,
            P,
            start_kit_id=1,
            max_iter=100,
            robot_time_table=robot_time_table,
            robot_pick_counts=robot_pick_counts,
        )
    
    return best_schedule, best_cost_so_far


if __name__ == "__main__":
    schedule = load_schedule_csv()
    P = build_precedence_matrix(schedule, case_1=False) # CASE_1:case_1=True
    robot_time_table = load_robot_task_time_table(ROBOT_TIME_DIR)

    # best_schedule, best_cost = exhaustive_global_solution_by_kit_product(
    #     schedule,
    #     P,
    #     k_kits=4,
    #     robot_time_table=robot_time_table,
    #     rebuild_P_each_candidate=False,
    #     max_eval_candidates=None,
    #     print_best_update=True,
    # )

    # 目前的窮舉方法
    # best_schedule, best_cost = exhaustive_global_solution_by_unique_agent_sequences(
    #     schedule,
    #     P,
    #     k_kits=6,                         # 改成 6 個 kit
    #     robot_time_table=robot_time_table,
    #     rebuild_P_each_candidate=False,   # 先不要每組都重建 P，速度較快
    #     skip_all_robot_block=True,         # 排除每個 block 全 robot
    #     max_eval_candidates=None,          # None 表示完整跑完，不限制候選數
    #     print_best_update=True,
    # )

    # 用啟發式方法建立最佳解
    best_schedule, best_cost = optimal_solution(
        schedule,
        P,
        case_3=True,  # CASE_3:case_3=True
        robot_time_table=robot_time_table,
    )

    print("===================================")
    print(f"Optimal makespan : {best_cost:.3f}")
    print("===================================")

    '''
    kit_makespans = compute_kit_makespan(schedule, k_kits=6)
    print("Makespan for each kit initially :")
    for kit_id, kit_makespan in kit_makespans.items():
        print(f"Kit {kit_id}: {kit_makespan:.2f}")'''

    draw_gantt(best_schedule, P, title="Optimal schedule", ax=None, show=True)

    