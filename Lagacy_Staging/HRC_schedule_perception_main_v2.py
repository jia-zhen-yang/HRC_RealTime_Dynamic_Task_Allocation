"""
Server端 (URscript機械手臂控制端為Client)
Client端 (Unity顯示端為Server)
HRC 作業監控主程式
攝影機監控、YOLO/HOI 判斷、人員/robot 狀態更新、偵測失效、串接動態排程
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import socket
import select
import threading
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from HOI import TaskRecognitionModel
from read_schedule import load_schedule_csv
from precedence_matrix import build_precedence_matrix
from VNS_verification import optimal_solution
from VNS_dynamic_solver import (
    compute_start_times,
    get_task_duration,
    load_robot_task_time_table,
    ROBOT_TIME_DIR,
)
from HRC_perception_solver_bridge import (
    run_monitor_rescheduling,
    save_reschedule_gantt,
    make_monitor_summary_figure,
    save_monitor_summary_figure,
    show_scrollable_figure,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 執行程式的路徑

# YOLO 模型 
OBJ_PATH_DIR = os.path.join(BASE_DIR, 'Object Detection/yolov11x.pt') #精準x/輕量n
model_yolo = YOLO(OBJ_PATH_DIR)

# Zone定義(滑鼠點擊選取)
current_zone = "kit box"  # 開始取 kit box 的區域點
kitbox_zone = []
ZONE_ORDER = ["Top", "BottomLeft", "BottomRight"]
LAYOUT_ORDER = ["A", "B", "C"]
current_layout_id = "A"   # 正在定義哪個 layout
layouts_zones = {"A": {"Top": [], "BottomLeft": [], "BottomRight": []},   #不同layout存區域點位
           "B": {"Top": [], "BottomLeft": [], "BottomRight": []},
           "C": {"Top": [], "BottomLeft": [], "BottomRight": []}}
zone_points = []    # 取點暫存
active_layout_id = "A"     # 目前畫面要顯示哪個 layout
freeze_until = 0.0

# 預期排程
schedule = load_schedule_csv()
P = build_precedence_matrix(schedule)
robot_time_table = load_robot_task_time_table(ROBOT_TIME_DIR)
schedule, best_makespan = optimal_solution(schedule, P,case_3=True)
schedule_index = 0  # 用來追蹤當前人員的排程指令

def initialize_planned_times(
    schedule,
    P,
    case=3,
    frozen_task_ids=None,
    decision_time=0.0,
    fixed_start_by_id=None,
    agent_ready_times=None,
):
    start_times = compute_start_times(
        schedule,
        P,
        case=case,
        frozen_task_ids=frozen_task_ids,
        decision_time=decision_time,
        fixed_start_by_id=fixed_start_by_id,
        agent_ready_times=agent_ready_times,
    )

    for i, task in enumerate(schedule):
        dur = float(get_task_duration(task))

        task["planned_start_time"] = float(start_times[i])
        task["planned_finish_time"] = float(start_times[i] + dur)

        # 後面監控用
        task.setdefault("status", "pending")
        task.setdefault("actual_start_time", None)
        task.setdefault("actual_finish_time", None)
        task.setdefault("actual_time", None)
        task.setdefault("ur_dispatched", False)


initialize_planned_times(schedule, P, case=3)

# 離線最佳排程保留一份，用於最終 Planned Timeline 
offline_best_schedule = deepcopy(schedule)

# 每次動態排程圖的第一張「最佳化排程」：
# 第一次是離線最佳排程；之後會更新成前一次動態排程結果 
current_optimized_schedule = deepcopy(offline_best_schedule)

# 追蹤當前作業狀態 以及 人員/機械手臂目前執行狀況
operation_state = {
    "human": {"task_id": None, "status": "idle"},
    "robot": {"task_id": None, "status": "idle"},
    "task_status": {}
}

# 每個 task 的實際開始時間（human / robot 共用）
task_runtime_tracker = {}

# 用來顯示提示訊息的函數
def display_message(frame, message):
    cv2.putText(frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
# 定義滑鼠事件函數
def mouse_callback(event, x, y, flags, param):
    global zone_points, current_zone, kitbox_zone, current_layout_id, layouts_zones
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        if current_zone == "kit box":
            kitbox_zone.append((x, y))
            
            if len(kitbox_zone) == 4:
                kitbox_zone = np.array(kitbox_zone, np.int32)
                current_zone = "Top"
                print("KIt Box Zone定義完成，請取 Top Zone的點")
            return

        zone_points.append((x,y))
        if len(zone_points) == 4:
            poly = np.array(zone_points, np.int32)
            layouts_zones[current_layout_id][current_zone] = poly
            zone_points.clear()

            # 切下一個 zone
            zi = ZONE_ORDER.index(current_zone)
            if zi < len(ZONE_ORDER) - 1:
                current_zone = ZONE_ORDER[zi + 1]
                print(f"Layout {current_layout_id}：請選取 {current_zone}")
            else:
                global active_layout_id, freeze_until
                active_layout_id = current_layout_id
                freeze_until = time.time() + 1.0

                # 這個 layout 做完，切下一個 layout
                li = LAYOUT_ORDER.index(current_layout_id)
                if li < len(LAYOUT_ORDER) - 1:
                    current_layout_id = LAYOUT_ORDER[li + 1]
                    current_zone = ZONE_ORDER[0]
                    print(f"Layout {LAYOUT_ORDER[li]} 完成\n開始 Layout {current_layout_id}，請選取 {current_zone}")
                else:
                    # 全部完成
                    current_zone = ""
                    print("Layout A/B/C 全部完成！按下s開始任務追蹤")


def update_operation_state(schedule, human_task, robot_task):
    """更新目前 human / robot 的執行狀態，以及各 task 的簡易狀態摘要 """
    global operation_state

    task_status = {}
    for task in schedule:
        tid = task.get("ID")
        if tid is None:
            continue
        task_status[tid] = task.get("status")

    operation_state["task_status"] = task_status

    for agent, task in (("human", human_task), ("robot", robot_task)):
        if task is None:
            operation_state[agent] = {
                "task_id": None,
                "status": "idle",
            }
            continue

        tid = task.get("ID")

        task_status = task.get("status")
        if agent == "human":
            if task_status in {"pending", "executing"}:
                status = "human executing"
            elif task_status == "completed":
                status = "completed"
            else:
                status = "idle"
        else:
            # 任務送出後才是 executing / 完成後才是 completed
            if task_status == "dispatched":
                status = "robot executing"
            elif task_status == "completed":
                status = "completed"
            else:
                status = "idle"

        operation_state[agent] = {
            "task_id": tid,
            "status": status,
        }
        operation_state["task_status"][tid] = status # 紀錄執行中的任務


def ensure_task_started(task_idx, started_at=None, now_from_start=None):
    """第一次進入任務時記錄開始時間，避免重複初始化"""
    global schedule, task_runtime_tracker

    if started_at is None:
        started_at = time.time()

    task = schedule[task_idx]
    tid = task.get("ID")
    if tid not in task_runtime_tracker:
        task_runtime_tracker[tid] = started_at

        if now_from_start is not None and task.get("actual_start_time") is None:
            task["actual_start_time"] = now_from_start



def get_elapsed_task_time(task_idx, now=None):
    """計算任務經過的時間"""
    global schedule, task_runtime_tracker
    if now is None:
        now = time.time()
    tid = schedule[task_idx].get("ID")
    started_at = task_runtime_tracker.get(tid)
    if started_at is None:
        return 0.0
    return now - started_at


def finalize_task(task_idx, source, completed_at=None, actual_time=None):
    """統一完成任務時要做的事情"""
    global schedule, completion_log

    if completed_at is None:
        completed_at = time.time()

    task = schedule[task_idx]

    if task.get("status") == "completed":
        return

    task["status"] = "completed"

    if start_time is not None:
        task["actual_finish_time"] = completed_at - start_time

    if actual_time is None:
        actual_time = get_elapsed_task_time(task_idx, now=completed_at)

    task["actual_time"] = actual_time
    if task.get("actual_start_time") is not None and task.get("actual_finish_time") is not None:
        task["actual_time"] = max(
            0.0,
            task["actual_finish_time"] - task["actual_start_time"]
        )

    completion_log.append({
        "schedule": task.copy(),
        "source": source
    })


def get_previous_kit_range(schedule, replace_idx):
    """
    對於某個 kit replace 任務，回傳上一個 replace 之後，到這個 replace 之前的內容物範圍
    """
    start = 0
    for i in range(replace_idx - 1, -1, -1):
        if schedule[i]["command"] == "replace":
            start = i + 1
            break

    end = replace_idx
    return start, end


def all_pick_place_done(schedule, replace_idx):
    """
    replace 前一箱的人機 pick&place 是否都完成
    """
    start, end = get_previous_kit_range(schedule, replace_idx)

    for i in range(start, end):
        task = schedule[i]
        if task.get("command") != "pick and place":
            continue
        if task.get("status") != "completed":
            return False
    return True

def is_robot_waiting_place_object(label):
    """
    判斷目前偵測到的物件 label 是否為：
    robot 已經 PICK_DONE，但尚未 DONE，正在等待 place 的物件。

    這類物件可能被機械手臂拿著，中心點落在 kitbox_zone 裡，
    不應該讓 sth_on_box=True，否則會誤判 kit box 裡有物件
    """
    global schedule, robot_waiting_kit_ready_set

    for task_id in robot_waiting_kit_ready_set:
        _, task = get_index_by_id(schedule, task_id)

        if task is None:
            continue

        if task.get("agent") != "robot":
            continue

        if task.get("command") != "pick and place":
            continue

        if task.get("status") != "dispatched":
            continue

        if task.get("object") == label:
            return True

    return False

def check_kitbox_replace(kitboxes, replace_idx):
    global kitbox_zone, required_stay_time, schedule, sth_on_box
    obj = "kit box"
    now = time.time()

    if not all_pick_place_done(schedule, replace_idx):
        if obj in zone_stay_start:
            del zone_stay_start[obj]
        return False

    inside_any = False
    for box in kitboxes:
        if cv2.pointPolygonTest(kitbox_zone, box, False) >= 0:
            inside_any = True
            break

    if inside_any and not sth_on_box:
        if obj not in zone_stay_start:
            zone_stay_start[obj] = now
        elif now - zone_stay_start[obj] >= required_stay_time + 0.5:
            print(f"New {obj} has been placed.")
            del zone_stay_start[obj]
            return True
    elif obj in zone_stay_start:
        del zone_stay_start[obj]

    return False


def get_kit_range(schedule, anchor_idx):
    """用 replace 當 kit 分界，回傳目前 kit 的 [start, end) index 範圍"""
    start = 0
    for i in range(anchor_idx, -1, -1):
        if schedule[i]["command"] == "replace":
            start = i + 1
            break

    end = len(schedule)
    for i in range(anchor_idx + 1, len(schedule)):
        if schedule[i]["command"] == "replace":
            end = i
            break
    return start, end

def objects_for_current_box(schedule, start_idx):
    objs = set()
    for t in schedule[start_idx:]:
        if t["command"] == "replace":
            break
        if t["command"] == "pick and place":
            objs.add(t["object"])
    return objs

def count_robot_completed_items(schedule, object_type: str) -> int:
    """
    計算 schedule 中，某種類物件已由 robot 完成幾個 
    """
    count = 0
    for t in schedule:
        if t.get("command") != "pick and place":
            continue
        if t.get("agent") != "robot":
            continue
        if t.get("status") != "completed":
            continue

        obj = t.get("object")
        if not obj:
            continue
        if obj == object_type:
            count += 1

    return count

def compute_stack_for_task(task, schedule) -> int:
    """
    根據目前 task 與整體 schedule，動態計算 stack

    - stack 最高為 total_kits
    - 已完成同種類物件數量 = completed_count
    - stack = total_kits - completed_count

    """
    obj = task.get("object")
    if not obj:
        raise ValueError("task has no object field")

    robot_completed_count = count_robot_completed_items(schedule, obj)
    stack = robot_completed_count + 1

    return stack

def get_current_task(schedule, agent):
    if agent == "robot":
        # pending：代表排程上存在，但還沒真的派給 robot
        # dispatched：代表主程式已經把它送出去，robot 正在執行
        # completed：已完成
        return next(
            ((i, task) for i, task in enumerate(schedule)
             if task["agent"] == "robot" and task["status"] == "dispatched"),
            (None, None)
        )
    
    executing = next(
        ((i, task) for i, task in enumerate(schedule)
         if task.get("agent") == agent and task.get("status") == "executing"),
        (None, None)
    )
    if executing[0] is not None:
        return executing

    return next(
        ((i, task) for i, task in enumerate(schedule)
         if task["agent"] == agent and task["status"] == "pending"),
        (None, None)
    )

def has_unfinished_tasks(schedule):
    """是否仍有尚未完成或正在執行的任務 """
    return any(
        task.get("status") in {"pending", "executing", "dispatched"}
        for task in schedule
    )

def get_delay_check_time(task, standard_time):
    """
    回傳此 human 任務下一次 delay 應該觸發的時間點

    規則：
    1. 若已有 _expected_finish_time：
       表示此任務已經發生過 delay，下一次 delay 檢查時間就是 _expected_finish_time

    2. 若沒有 _expected_finish_time：
       表示尚未發生 delay，第一次 delay 檢查時間 =
       actual_start_time + standard_time

    3. 若 actual_start_time 沒有，就退回 planned_start_time + standard_time

    4. 若都沒有，回傳 None
    """
    if task is None:
        return None

    if task.get("_expected_finish_time") is not None:
        return float(task["_expected_finish_time"])

    start_t = task.get("actual_start_time")
    if start_t is None:
        start_t = task.get("planned_start_time")

    if start_t is None:
        return None

    return float(start_t) + float(standard_time)

def should_trigger_delay(task, standard_time, now_from_start, cooldown=0.1):
    """
    判斷目前 task 是否應該觸發 delay rescheduling

    使用統一規則：
    - 第一次：now_from_start >= actual_start_time + standard_time
    - 第二次以後：now_from_start >= _expected_finish_time

    cooldown 用來避免同一個時間點附近連續每幀觸發 (防呆)
    """
    global delay_reschedule_state

    delay_check_time = get_delay_check_time(task, standard_time)
    if delay_check_time is None:
        return False

    last_trigger_time = delay_reschedule_state.get("last_trigger_time")
    recently_triggered = (
        last_trigger_time is not None
        and float(now_from_start) - float(last_trigger_time) < cooldown
    )

    return (
        float(now_from_start) >= float(delay_check_time)
        and not recently_triggered
    )

def should_trigger_replace_delay(
    task,
    schedule_index,
    standard_time,
    now_from_start,
    replace_ready_start,
    start_time,
    cooldown=0.1,
):
    """
    replace 專用 delay 判斷

    規則：
    1. 第一次 delay：
       用 replace_ready_start + standard_time 判斷
       因為等待上一箱裝完不算 replace 實際作業時間

    2. 第二次以後：
       用 task["_expected_finish_time"] 判斷
       因為每次 delay rescheduling 後，handle_delay_failure()
       會更新新的預期完成時間

    3. cooldown：
       避免同一個時間點附近連續觸發。
    """
    global delay_reschedule_state

    if task is None:
        return False

    if task.get("_expected_finish_time") is not None:
        delay_check_time = float(task["_expected_finish_time"])
    else:
        ready_abs = replace_ready_start.get(schedule_index)
        if ready_abs is None or start_time is None:
            return False

        ready_from_start = float(ready_abs) - float(start_time)
        delay_check_time = ready_from_start + float(standard_time)

    last_trigger_time = delay_reschedule_state.get("last_trigger_time")
    if last_trigger_time is not None:
        if float(now_from_start) - float(last_trigger_time) < cooldown:
            return False

    return float(now_from_start) >= float(delay_check_time)

def send_robot_task(task, schedule, client_socket):
    """
    將任務資訊透過 TCP/IP 傳給 UR client
    """
    def convert_stack_for_ur(obj, stack):
        """
        將任務中的實際 stack 轉換成 UR 運動規劃查表用的 stack
        """

        object_stack_count = {
            "A_pink": 3,
            "B_green": 4,
            "C_mint": 5,
            "D_skin": 6,
        }

        if obj not in object_stack_count:
            raise ValueError(f"Unknown object type: {obj}")

        stack = int(stack)
        count = object_stack_count[obj]

        if stack < 1 or stack > count:
            raise ValueError(
                f"Invalid stack {stack} for object {obj}. "
                f"{obj} only has stack range 1~{count}"
            )

        return stack + (6 - count)

    if client_socket is None:
        print("[SEND TO UR] client_socket is None, skip sending")
        return
    
    stack = task.get("robot_stack")
    if stack is None:
        stack = compute_stack_for_task(task, schedule)
    
    # 轉換成 UR 查表用的 stack
    stack = convert_stack_for_ur(task["object"], stack)

    msg = (
        f"ID={task['ID']};"
        f"command={task['command']};"
        f"object={task['object']};"
        f"zone={task['zone']};"
        f"layout={task['layout']};"
        f"stack={stack}\n"
    )

    try:
        client_socket.sendall(msg.encode("utf-8"))
        print(f"[SEND TO UR] {msg.strip()}")
    except OSError as e:
        print(f"[SEND TO UR] socket send failed: {e}")


def send_kit_ready_to_robot(task_id, kit_ready):
    global client_socket

    msg = f"ID={task_id};kit_ready={kit_ready}\n"

    try:
        client_socket.sendall(msg.encode("utf-8"))
        # print(f"[SEND TO UR] {msg.strip()}")
    except OSError as e:
        print(f"[SEND KIT_READY TO UR] socket send failed: {e}")


def get_index_by_id(schedule, task_id):
    for i, task in enumerate(schedule):
        if task.get("ID") == task_id:
            return i, task
    return None, None

def  is_kit_ready_for_robot(schedule, task_id):
    """
    判斷 robot task 所屬 kit 是否 ready
    """
    task_idx, task = get_index_by_id(schedule, task_id)

    if task_idx is None:
        return False

    prev_replace_idx = None

    for i in range(task_idx - 1, -1, -1):
        if schedule[i].get("command") == "replace":
            prev_replace_idx = i
            break

    if prev_replace_idx is None:
        return True

    return schedule[prev_replace_idx].get("status") == "completed"


def get_bot_task_done(task_id):
    # """
    # TODO:
    # 暫時用時間模擬 BOT 完成：
    # - 任務一旦被 dispatch，就會在 task_runtime_tracker 記錄送出當下時間
    # - 經過該任務的 robot_standard_time 後，視為bot任務回傳完成
    # """
    # global schedule, task_runtime_tracker

    # # 先找到對應 task
    # task = next((t for t in schedule if t.get("ID") == task_id), None)
    # if task is None:
    #     return False

    # started_at = task_runtime_tracker.get(task_id)
    # if started_at is None:
    #     return False

    # now = time.time()
    # elapsed = now - started_at
    # required = float(task.get("robot_standard_time", 0.0))

    # return elapsed >= required

    return task_id in done_set


def dispatch_ready_robot_task(now_from_start):
    """
    若 robot 目前閒置，且有任務已達 planned start time，
    就發送給 robot，並記錄 actual start time
    """
    global schedule, task_runtime_tracker, P

    # robot 一次只做一個任務
    active_robot_idx, _ = get_current_task(schedule, "robot")
    if active_robot_idx is not None:
        return

    for i, task in enumerate(schedule):
        if task.get("agent") != "robot":
            continue
        if task.get("status") != "pending":
            continue

        planned_start = float(task.get("planned_start_time", 0.0))

        # 還沒到預定開始時間
        if now_from_start < planned_start:
            continue

        # 發送給任務
        send_robot_task(task, schedule, client_socket)

        task["status"] = "dispatched"
        task["ur_dispatched"] = True
        task["actual_start_time"] = now_from_start

        tid = task.get("ID")
        task_runtime_tracker[tid] = time.time()

        print(
            f"[ROBOT START] ID={task.get('ID')} "
            f"planned={planned_start:.2f}s "
            f"actual={task['actual_start_time']:.2f}s"
        )
        return

def update_robot_progress(robot_idx, robot_task, now_from_start):
    """
    機械手臂完成判定：
    - 任務開始：由 dispatch_ready_robot_task() 在 planned start time 到時送出
    - 任務完成：由 UR 回傳成功訊號後才 finalize
    """
    global completed_objects

    if robot_idx is None or robot_task is None:
        return

    if robot_task.get("status") != "dispatched":
        return
    bot_done = get_bot_task_done(robot_task["ID"])
    if not bot_done:
        return

    completed_at_abs = time.time()
    actual_start = robot_task.get("actual_start_time", now_from_start)
    actual_finish = now_from_start
    actual_time = actual_finish - actual_start

    robot_task["actual_finish_time"] = actual_finish

    finalize_task(
        robot_idx,
        source="robo_ur",
        completed_at=completed_at_abs,
        actual_time=actual_time
    )

    if robot_task.get("command") == "pick and place":
        completed_objects.add(robot_task.get("object"))

    notify_current_human_task_to_unity(force=True)

    print(
        f"[ROBOT DONE] ID={robot_task.get('ID')} "
        f"object={robot_task.get('object')} "
        f"actual_time={actual_time:.2f}s"
    )


def find_task_in_current_kit_by_object(schedule, anchor_idx, obj):
    """在目前 kit 中找指定物件對應的 pick and place task"""
    if anchor_idx is None:
        return None, None

    kit_start, kit_end = get_kit_range(schedule, anchor_idx)
    for i in range(kit_start, kit_end):
        task = schedule[i]
        if task.get("command") != "pick and place":
            continue
        if task.get("object") == obj:
            return i, task
    return None, None


def classify_error(schedule, schedule_index, obj, human_current_task):
    """
    以「目前 kit」為單位分類錯誤拿取物件 

    回傳類別：
    - placed           : 此 kit 中該物件已完成放置，不管由 human 或 robot 完成 
    - robot_executing  : 該物件是當前 kit 的 robot task，且 robot 已經開始執行
    - robot_task       : 該物件是當前 kit 的 robot task，但 robot 尚未開始
    - later_task       : 該物件屬於此 kit 後續 human task
    - not_belong       : 該物件不屬於目前 kit 
    - current_task     : 目前 human 正確任務
    """
    kit_start, kit_end = get_kit_range(schedule, schedule_index)

    match_idx, match = find_task_in_current_kit_by_object(schedule, schedule_index, obj)

    if match is None:
        return "not_belong", match_idx
    
    is_current_kit_task = (
        match_idx is not None
        and kit_start <= match_idx < kit_end
    )

    status = match.get("status", "pending")

    if status == "completed":
        return "placed", match_idx

    if human_current_task and match.get("ID") == human_current_task.get("ID"):
        return "current_task", match_idx

    if match.get("agent") == "robot":
        if status == "dispatched" and is_current_kit_task:
            return "robot_executing", match_idx
        
        if is_current_kit_task:
            return "robot_task", match_idx
        
        return "not_belong", match_idx

    return "later_task", match_idx


def wrong_item_message(error_class):
    messages = {
        "placed": "Already placed in this kit box",
        "robot executing": "This task is being executed by robot",
        "robot task": "This is a robot task item",
        "later task": "Belongs to a later task",
        "not belong": "Not belongs to this kit box",
    }
    return messages.get(error_class, "Wrong item")


def rebuild_completed_objects(anchor_idx):
    """依照目前 schedule 狀態重建 completed_objects，避免 clear 後與實際 status 不一致"""
    global completed_objects
    completed_objects.clear()
    if anchor_idx is None:
        return

    kit_start, kit_end = get_kit_range(schedule, anchor_idx)
    for i in range(kit_start, kit_end):
        task = schedule[i]
        if task.get("command") != "pick and place":
            continue
        if task.get("status") == "completed":
            completed_objects.add(task.get("object"))

def opportunistic_complete(anchor_idx, zone_objects, required_stay_time):
    """
    掃描目前 kit 範圍內所有 pending pick&place 任務（包含 human/robot）
    若任務物件出現在正確 zone 且停留 >= required_stay_time，則直接完成該任務
    """
    global schedule, frame_stay_start, completed_objects, done_set

    kit_start, kit_end = get_kit_range(schedule, anchor_idx)

    now = time.time()
    completed_opportunistic = []

    for i in range(kit_start, kit_end):
        if i == anchor_idx:
            continue

        if schedule[i].get("command") != "pick and place":
            continue

        tid = schedule[i].get("ID")
        agent = schedule[i].get("agent")
        status = schedule[i].get("status")

        # robot 已經執行中或已回傳完成，不能被影像判成 opportunistic
        if agent == "robot":
            if status == "dispatched" or tid in done_set:
                if i in frame_stay_start:
                    del frame_stay_start[i]
                continue

        # 只有尚未開始的任務，才可能是人員提前完成
        if status != "pending":
            continue

        obj = schedule[i]["object"]
        zone = schedule[i]["zone"]

        if obj in zone_objects.get(zone, []):
            if i not in frame_stay_start:
                frame_stay_start[i] = now
            elif now - frame_stay_start[i] >= required_stay_time:
                completed_objects.add(obj)
                print(
                    "[OPP DONE]",
                    schedule[i].get("ID"),
                    schedule[i]["object"],
                    schedule[i]["zone"],
                    schedule[i].get("agent")
                )
                completed_opportunistic.append(i)
                del frame_stay_start[i]
        else:
            if i in frame_stay_start:
                del frame_stay_start[i]

    return completed_opportunistic

def reset_interrupted_delay_task(task_idx):
    """
    當原本 delay / executing 中的 human task 被其他失效打斷時，
    將該任務恢復成 pending，讓後續 VNS 重新排程

    - current task 原本是 human 正在執行的任務
    - 它曾經 delay，因此 status = executing，且有 _duration_override / _expected_finish_time
    - 但人員實際完成了其他任務，觸發 order mistake / agent mistake / opportunistic completion
    - 此時 current task 不應繼續被視為正在執行，也不應保留 delay 後的長 duration
    """
    global schedule, task_runtime_tracker, delay_reschedule_state

    if task_idx is None or task_idx < 0 or task_idx >= len(schedule):
        return

    task = schedule[task_idx]
    tid = task.get("ID")
    if tid is None:
        return

    # 只重置 human delay task，避免誤清 robot dispatched 的 _duration_override。
    is_delay_executing = (
        task.get("agent") == "human"
        and (
            task.get("status") == "executing"
            or task.get("_delay_failure") is True
        )
    )

    if not is_delay_executing:
        return

    print(
        f"[RESET INTERRUPTED DELAY TASK] "
        f"task_id={tid}, "
    )

    # 恢復為未完成、可重新排程狀態
    task["status"] = "pending"

    # 清除此次被打斷的執行紀錄
    # 因為人員沒有完成這個 task，所以這段執行不應該成為 actual_time
    task["actual_finish_time"] = None
    task["actual_time"] = None

    # 清除 delay / executing 暫存資訊，讓 solver 回到標準時間估計
    task.pop("_duration_override", None)
    task.pop("_expected_finish_time", None)
    task.pop("_delay_time", None)
    task.pop("_delay_failure", None)

    # human delay task 不需要 robot place compensation
    task.pop("_place_wait_compensation", None)

    # 如果 delay 狀態正在追蹤這個 task，也一起清掉
    if delay_reschedule_state.get("task_id") == tid:
        delay_reschedule_state["task_id"] = None
        delay_reschedule_state["last_trigger_time"] = None
        delay_reschedule_state["next_trigger_time"] = None

def finalize_opportunistic(current_idx, opp_done, now=None, now_from_start=None):
    """
    1. 計算 opportunistic 任務的實際時間並完成它
    2. 重設目前 current task 的開始時間
    """
    global schedule, task_runtime_tracker
    current_tid = schedule[current_idx].get("ID")

    if now is None:
        now = time.time()

    for task_idx in opp_done:
        start_t = task_runtime_tracker[current_tid]
        actual_time = max(0.0, now - start_t)

        # opportunistic 完成的任務，起點等同於目前任務的起點
        if schedule[task_idx].get("actual_start_time") is None:
            schedule[task_idx]["actual_start_time"] = schedule[current_idx].get("actual_start_time")

        finalize_task(
            task_idx,
            source="opportunistic",
            completed_at=now,
            actual_time=actual_time
        )

    task_runtime_tracker[current_tid] = now
    if now_from_start is not None:
        schedule[current_idx]["actual_start_time"] = now_from_start

    return now

def  infer_failure_type_from_opp(schedule, opp_done):
    """
    根據 opportunistic 完成的任務判斷失效類型

    規則：
    1. 若 opp_done 中有原本 agent == robot 的任務：
       表示人員完成了 robot 任務，視為 agent mistake
    2. 若 opp_done 都是 human 任務：
       表示人員提前做了後面順序的任務，視為 order mistake
    3. 其他情況退回 opp_complete
    """
    if not opp_done:
        return "opp_complete"

    for idx in opp_done:
        if idx is None or idx < 0 or idx >= len(schedule):
            continue
        task = schedule[idx]
        if task.get("agent") == "robot":
            return "agent_mistake"

    return "order_mistake"


def  trigger_dynamic_reschedule_opp(current_idx, opp_done, now_from_start):
    """
    opp complete 完成後觸發 VNS 動態重排

    - 已完成任務與正在執行中的 robot 任務會 frozen
    - human 的下一次可用時間為 now_from_start
    - robot 若正在執行，下一次可用時間會估到該 robot task 完成
    - 甘特圖直接存到 dynamic_gantt_outputs，不會 plt.show()
    """
    global schedule, dynamic_reschedule_count, last_dynamic_reschedule_info
    global current_optimized_schedule, dynamic_summary_history, deferred_gantt_items
    global frame_stay_start, zone_stay_start, wrong_zone_stay_start

    if not opp_done:
        return None

    dynamic_reschedule_count += 1

    current_task_id = None
    if current_idx is not None and 0 <= current_idx < len(schedule):
        current_task_id = schedule[current_idx].get("ID")

    completed_task_ids = [
        schedule[i].get("ID")
        for i in opp_done
        if i is not None and 0 <= i < len(schedule)
    ]

    # current task 原本是 delay / executing 狀態
    # 此時要把 current task 恢復成 pending，清掉延遲後 duration，讓 VNS 重新排程
    if current_idx is not None and current_idx not in opp_done:
        reset_interrupted_delay_task(current_idx)

    failure_type =  infer_failure_type_from_opp(
        schedule=schedule,
        opp_done=opp_done,
    )

    failure_info = {
        "type": failure_type,
        "current_task_id": current_task_id,
        "completed_task_ids": completed_task_ids,
        "affected_task_ids": [tid for tid in [current_task_id, *completed_task_ids] if tid is not None],
    }

    new_schedule, info = run_monitor_rescheduling(
        schedule=schedule,
        P=P,
        now_from_start=now_from_start,
        failure_info=failure_info,
        output_dir=os.path.join(BASE_DIR, "dynamic_gantt_outputs"),
        step_idx=dynamic_reschedule_count,
        max_iter=200,
        optimized_reference_schedule=current_optimized_schedule,
        robot_time_table=robot_time_table,
        save_gantt_immediately=False,
    )

    schedule = new_schedule
    current_optimized_schedule = deepcopy(new_schedule)
    dynamic_summary_history.append(info["summary_item"])
    deferred_gantt_items.append(info["deferred_gantt_item"])
    last_dynamic_reschedule_info = info

    frame_stay_start.clear()
    zone_stay_start.clear()
    wrong_zone_stay_start.clear()

    new_human_idx, _ = get_current_task(schedule, "human")
    new_robot_idx, _ = get_current_task(schedule, "robot")
    rebuild_completed_objects(new_human_idx if new_human_idx is not None else new_robot_idx)

    print("[DYNAMIC RESCHEDULE] opp_complete triggered")

    return info

DELAY_FACTORS = [0.5, 1.0, 1.5]
def handle_delay_failure(current_idx, now_from_start, standard_time):
    global schedule, dynamic_reschedule_count, last_dynamic_reschedule_info
    global current_optimized_schedule, dynamic_summary_history, deferred_gantt_items
    global frame_stay_start, zone_stay_start, wrong_zone_stay_start
    global delay_reschedule_state
    global DELAY_FACTORS

    if current_idx is None or current_idx < 0 or current_idx >= len(schedule):
        return None

    task = schedule[current_idx]
    task_id = task.get("ID")
    if task_id is None:
        return None

    standard_time = float(standard_time)
    
    count_by_task = delay_reschedule_state.setdefault("count_by_task", {})
    delay_count = count_by_task.get(task_id, 0)

    if delay_count == 0:
        delay_factor = DELAY_FACTORS[0]   # 第一次 delay：0.5
    elif delay_count == 1:
        delay_factor = DELAY_FACTORS[1]   # 第二次 delay：1.0
    else:
        delay_factor = DELAY_FACTORS[2]   # 第三次以上 delay：1.5

    delay_time = delay_factor * standard_time

    # 優先使用真正開始時間
    start_t = task.get("actual_start_time")
    if start_t is None:
        start_t = task.get("planned_start_time")
    if start_t is None:
        start_t = max(0.0, float(now_from_start) - standard_time)

    start_t = float(start_t)

    # 取得前一次預估完成時間
    # 若此任務已經 delay 過，則_expected_finish_time 存在
    previous_expected_finish = task.get("_expected_finish_time")
    previous_duration = task.get("_duration_override")

    if previous_expected_finish is not None:
        # 第二次以上 delay：從上一個 expected_finish 再往後加 delay_time
        expected_finish = float(previous_expected_finish) + delay_time

    elif previous_duration is not None:
        # 防呆：
        # 若有 duration override 但沒有 expected finish，則用 start_t + previous_duration 再加 delay_time
        expected_finish = start_t + float(previous_duration) + delay_time

    else:
        # 第一次 delay：
        # 原本預期完成時間 = start_t + standard_time
        # 第一次延遲後，預期完成時間 = start_t + standard_time + delay_time
        expected_finish = start_t + standard_time + delay_time

    # 如果現在已經超過新的 expected_finish， (防呆)
    # 代表人員已經延遲更多，至少要從現在再往後加一段 delay_time
    if float(now_from_start) >= expected_finish:
        expected_finish = float(now_from_start) + delay_time

    estimated_duration = expected_finish - start_t

    # human delay 任務：正在執行，不可被重排
    task["status"] = "executing"
    task["actual_start_time"] = start_t
    task["actual_finish_time"] = None
    task["actual_time"] = None

    # 給 solver / gantt 使用
    task["_duration_override"] = estimated_duration
    task["_expected_finish_time"] = expected_finish
    task["_delay_time"] = delay_time
    task["_delay_failure"] = True

    # 同步 planned 欄位，方便主程式與繪圖看
    task["planned_start_time"] = start_t
    task["planned_finish_time"] = expected_finish

    failure_info = {
        "type": "delay",
        "current_task_id": task_id,
        "completed_task_ids": [],
        "affected_task_ids": [task_id],
        "delay_task_id": task_id,
        "estimated_delay": delay_time,
        "expected_finish": expected_finish,
    }

    new_schedule, info = run_monitor_rescheduling(
        schedule=schedule,
        P=P,
        now_from_start=now_from_start,
        failure_info=failure_info,
        output_dir=os.path.join(BASE_DIR, "dynamic_gantt_outputs"),
        step_idx=dynamic_reschedule_count + 1,
        max_iter=200,
        optimized_reference_schedule=current_optimized_schedule,
        robot_time_table=robot_time_table,
        save_gantt_immediately=False,
    )

    schedule[:] = new_schedule
    dynamic_reschedule_count += 1
    current_optimized_schedule = deepcopy(new_schedule)
    dynamic_summary_history.append(info["summary_item"])
    deferred_gantt_items.append(info["deferred_gantt_item"])
    last_dynamic_reschedule_info = info

    frame_stay_start.clear()
    zone_stay_start.clear()
    wrong_zone_stay_start.clear()

    new_human_idx, _ = get_current_task(schedule, "human")
    new_robot_idx, _ = get_current_task(schedule, "robot")
    rebuild_completed_objects(new_human_idx if new_human_idx is not None else new_robot_idx)

    delay_reschedule_state["task_id"] = task_id
    delay_reschedule_state["last_trigger_time"] = now_from_start
    delay_reschedule_state["next_trigger_time"] = expected_finish

    count_by_task[task_id] = delay_count + 1

    print("[DELAY FAILURE HANDLED]")
    print(
        f"[DELAY FAILURE] task_id={task_id}, "
        f"delay_count={delay_count + 1}, "
        f"delay_factor={delay_factor:.1f}, "
        f"delay_time={delay_time:.2f}s"
    )
    # print(f"[DELAY FAILURE] expected_finish={expected_finish:.2f}s")
    # print(f"[DELAY FAILURE] frozen={info['frozen_task_ids']}")
    # print(f"[DELAY FAILURE] agent_ready={info['agent_ready_times']}")

    return info


# def draw_gantt(
#     schedule,
#     title="Schedule",
#     start_getter=None,
#     duration_getter=None,
#     ax=None,
#     show=True,
#     skip_actual_time_zero=True,
# ):
#     """
#     通用 Gantt 繪圖函式 

#     固定 y 軸顯示順序：
#     - robot：上方
#     - human：下方

#     若 skip_actual_time_zero=True，actual_time == 0 的任務不畫 
#     """
#     colors = {"replace": "tab:blue", "pick and place": "tab:orange"}
#     agent_y = {"human": 0, "robot": 1}

#     created_fig = False
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(12, 3))
#         created_fig = True

#     for task in schedule:
#         actual_time = task.get("actual_time")
#         try:
#             if skip_actual_time_zero and actual_time is not None and float(actual_time) == 0.0:
#                 continue
#         except (TypeError, ValueError):
#             pass

#         agent = task.get("agent", "")
#         y = agent_y.get(agent)
#         if y is None:
#             continue

#         start = start_getter(task) if start_getter else None
#         dur = duration_getter(task) if duration_getter else None

#         if start is None or dur is None:
#             continue
#         if dur <= 0:
#             continue

#         ax.barh(
#             y,
#             dur,
#             left=start,
#             height=0.55,
#             color=colors.get(task.get("command"), "gray"),
#             edgecolor="black",
#         )

#         ax.text(
#             start + dur / 2,
#             y,
#             str(task.get("ID", "")),
#             va="center",
#             ha="center",
#             color="white",
#             fontsize=9,
#             fontweight="bold",
#         )

#     ax.set_yticks([0, 1])
#     ax.set_yticklabels(["human", "robot"])
#     ax.set_ylim(-0.5, 1.5)
#     ax.set_title(title)
#     ax.set_xlabel("Time")
#     ax.grid(axis="x", linestyle="--", alpha=0.3)

#     if created_fig:
#         plt.tight_layout()
#         if show:
#             plt.show()

# def draw_planned_gantt(schedule, ax=None, show=True):
#     draw_gantt(
#         schedule,
#         title="Planned Timeline",
#         start_getter=lambda task: task.get("planned_start_time"),
#         duration_getter=lambda task: (
#             task.get("planned_finish_time") - task.get("planned_start_time")
#             if task.get("planned_start_time") is not None and task.get("planned_finish_time") is not None
#             else None
#         ),
#         ax=ax,
#         show=show
#     )

# def draw_actual_gantt(schedule, ax=None, show=True):
#     """
#     只畫已完成任務：使用 actual_start_time + actual_time / actual_finish_time
#     """

#     def _start(task):
#         if task.get("status") != "completed":
#             return None

#         return task.get("actual_start_time")

#     def _duration(task):
#         if task.get("status") != "completed":
#             return None

#         if task.get("actual_time") is not None:
#             return task.get("actual_time")

#         if task.get("actual_start_time") is not None and task.get("actual_finish_time") is not None:
#             return task.get("actual_finish_time") - task.get("actual_start_time")

#         return None

#     draw_gantt(
#         schedule,
#         title="Actual / Final Timeline",
#         start_getter=_start,
#         duration_getter=_duration,
#         ax=ax,
#         show=show,
#         skip_actual_time_zero=True,
#     )


#有沒有東西在box裡面
sth_on_box = False

# 追蹤排程進度&失效紀錄
task_model = TaskRecognitionModel()
task_start = False
delay_reschedule_state = {
    "task_id": None,
    "last_trigger_time": None,
    "next_trigger_time": None,
    "count_by_task": {},
}
start_time = None  # 用來計算整體的開始時間
subtask_start_time = None #每個子任務的開始時間
all_done_reported = False
zone_stay_start = {} # 正確物件放置時間偵測
frame_stay_start = {} # 每一帧監控偏離排程的任務被執行
wrong_zone_stay_start = {}
kitbox_replace = False
required_stay_time = 1.0  # 物件至少停留特定秒數才算完成任務
completed_objects = set()
error_times = []
completion_log = []  # 記錄實際完成順序（依完成時間）
replace_waiting_message = "Waiting for all items in this kit box"
robot_started_ids = set()
replace_ready_start = {}
dynamic_reschedule_count = 0
last_dynamic_reschedule_info = None
# 每次動態排程結果會依序放到這裡，最後插入 Planned 與 Actual 之間形成總表 
dynamic_summary_history = []
# 每次失效先只記錄甘特圖資料，不立刻存圖片
deferred_gantt_items = []

# TCP/IP(client) to Unity Hint Server
UNITY_HOST = "192.168.0.216"   # HoloLens / Unity IP
# UNITY_HOST = "192.168.50.180"   # HoloLens / Unity IP
# UNITY_HOST = "127.0.0.1"   # Unity 和 Python 在同一台電腦時使用
UNITY_PORT = 5005               # Unity TcpHintServer 的 port

unity_socket = None
last_tcp_retry_time = 0
TCP_RETRY_INTERVAL = 1.0
unity_all_completed_sent = False
last_unity_hint_task_id = None

unity_rx_buffer = ""
monitor_start = False
pending_unity_start = False

# 錯誤提示傳送控制
ERROR_SEND_COOLDOWN = 2.0
last_unity_error_key = None
last_unity_error_time = 0.0
UNITY_ERROR_TYPES = {
    "placed",
    "robot_executing",
    "not_belong",
    "wrong_zone",
}
UNITY_REPLACE_WAITING_COMMAND = "REPLACE_WAITING"

def connect_unity():
    """
    嘗試連線到 Unity TCP Server
    """
    global unity_socket, last_tcp_retry_time

    if unity_socket is not None:
        return True

    # 防止連線失敗時太頻繁重試
    now = time.time()
    if now - last_tcp_retry_time < TCP_RETRY_INTERVAL:
        return False

    last_tcp_retry_time = now

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect((UNITY_HOST, UNITY_PORT))
        s.settimeout(None)

        unity_socket = s
        print(f"[TCP] Connected to Unity at {UNITY_HOST}:{UNITY_PORT}")
        return True

    except OSError as e:
        unity_socket = None
        print(f"[TCP] Unity connection failed: {e}")
        return False


def close_unity_connection():
    """
    關閉 Unity TCP 連線
    """
    global unity_socket

    if unity_socket is not None:
        try:
            unity_socket.close()
        except OSError:
            pass

        unity_socket = None
        print("[TCP] Unity connection closed.")


def send_to_unity(message):
    global unity_socket

    if not message.endswith("\n"):
        message += "\n"

    data = message.encode("utf-8")

    try:
        if unity_socket is None:
            if not connect_unity():
                return False

        unity_socket.sendall(data)
        print(f"[TCP] Sent to Unity: {message.strip()}")
        return True

    except OSError as e:
        print(f"[TCP] Send failed: {e}")
        close_unity_connection()
        return False
    
if unity_socket is None:
    connect_unity()

def sanitize_unity_message_text(text):
    """
    清理要送給 Unity 的文字，避免破壞 TCP 指令格式

    Unity 端預期格式：
    ERROR|error_type|message

    """
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("|", "/")
    return text.strip()


def send_error_to_unity(error_type, message, force=False):
    """
    傳送錯誤提示給 Unity

    格式：
    ERROR|error_type|message

    Unity 顯示規則：
    - error_type == wrong_zone：只顯示錯誤文字
    - 其他 error_type：顯示紅色光暈 + 錯誤文字
    """
    global last_unity_error_key, last_unity_error_time

    error_type = sanitize_unity_message_text(error_type)
    message = sanitize_unity_message_text(message)

    # 只處理指定的四種錯誤
    if error_type not in UNITY_ERROR_TYPES:
        return False

    now = time.time()
    error_key = (error_type, message)

    # 避免同一個錯誤在短時間內每一幀重複送到 Unity
    if (
        not force
        and last_unity_error_key == error_key
        and now - last_unity_error_time < ERROR_SEND_COOLDOWN
    ):
        return False

    unity_message = f"ERROR|{error_type}|{message}"
    success = send_to_unity(unity_message)

    if success:
        last_unity_error_key = error_key
        last_unity_error_time = now

    return success

def send_replace_waiting_to_unity(force=False):
    """
    通知 Unity 顯示 ReplaceWaitingMessage

    Unity 端會一直顯示到：
    1. 下一次 replace hint 出現
    2. 收到 COMPLETE
    """
    global last_unity_hint_task_id, unity_all_completed_sent

    if not force and last_unity_hint_task_id == UNITY_REPLACE_WAITING_COMMAND:
        return False

    send_to_unity("HIDE_ALL")
    success = send_to_unity(UNITY_REPLACE_WAITING_COMMAND)

    if success:
        last_unity_hint_task_id = UNITY_REPLACE_WAITING_COMMAND
        unity_all_completed_sent = False

    return success

def notify_current_human_task_to_unity(force=False):
    """
    將目前可以執行的人員任務 ID 傳給 Unity

    規則：
    1. 若目前沒有 human task：
       - 若全部任務完成，送 COMPLETE
       - 若仍有 robot 或 future pending task，送 HIDE_ALL

    2. 若目前 human task 是 replace：
       - 必須等同一 kit 內所有 pick and place 任務完成
       - 也就是 all_pick_place_done(schedule, human_idx) == True
       - 才可以送 replace 的 ID 給 Unity
       - 否則送 HIDE_ALL，避免太早顯示 replace 提示

    3. 若目前 human task 是 pick and place：
       - 可以直接送該 task ID

    4. force=False 時，避免同一個 ID 重複傳送
    """
    global unity_all_completed_sent, last_unity_hint_task_id

    human_idx, human_task = get_current_task(schedule, "human")

    if human_task is None:
        if not has_unfinished_tasks(schedule):
            if not unity_all_completed_sent:
                success = send_to_unity("COMPLETE")
                if success:
                    unity_all_completed_sent = True
                    last_unity_hint_task_id = None
        else:
            send_replace_waiting_to_unity(force=force)

        return

    task_id = human_task.get("ID")
    command = human_task.get("command")

    if task_id is None:
        return

    if command == "replace":
        replace_ready = all_pick_place_done(schedule, human_idx)

        if not replace_ready:
            send_replace_waiting_to_unity(force=force)
            return

    if not force and last_unity_hint_task_id == task_id:
        return

    success = send_to_unity(str(task_id))

    if success:
        last_unity_hint_task_id = task_id
        unity_all_completed_sent = False

def start_task_monitoring(source="Unity"):
    """
    啟動任務監控
    原本按下 s 做的事情，改成由 Unity 傳 START 後呼叫
    """
    global task_start, start_time, subtask_start_time
    global pending_unity_start, unity_all_completed_sent, last_unity_hint_task_id

    if task_start:
        print("[TASK] Task monitoring already started.")
        return

    # 如果區域點還沒取完，不要立刻開始
    if current_zone:
        pending_unity_start = True
        print("[TASK] START received, but zones are not completed yet.")
        return

    task_start = True
    pending_unity_start = False
    unity_all_completed_sent = False
    last_unity_hint_task_id = None

    print(f"[TASK] Start task monitoring by {source}.")
    start_time = time.time()
    subtask_start_time = start_time

    # 啟動後立刻送第一個目前可執行的人員任務提示
    notify_current_human_task_to_unity(force=True)

def handle_unity_message(message):
    """
    處理 Unity 傳來的文字訊息。
    Unity Start Button 會傳：START
    """
    global monitor_start

    msg = message.strip()
    print(f"[UNITY TCP] Received from Unity: {msg}")

    if msg.lower() == "start":
        if monitor_start:
            print("[UNITY TCP] START message already received. Ignored.")
            return

        monitor_start = True
        start_task_monitoring(source="Unity")

def poll_unity_messages():
    """
    非阻塞方式檢查 Unity 是否有傳訊息給 Python
    """
    global unity_socket, unity_rx_buffer

    if unity_socket is None:
        return

    try:
        readable, _, _ = select.select([unity_socket], [], [], 0)

        if not readable:
            return

        data = unity_socket.recv(1024)

        if not data:
            print("[UNITY TCP] Unity disconnected.")
            close_unity_connection()
            return

        unity_rx_buffer += data.decode("utf-8", errors="ignore")

        while "\n" in unity_rx_buffer:
            line, unity_rx_buffer = unity_rx_buffer.split("\n", 1)
            line = line.strip()

            if line:
                handle_unity_message(line)

    except OSError as e:
        print(f"[UNITY TCP] Receive failed: {e}")
        close_unity_connection()

# TCP/IP (server)
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5050

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(1)

print(f"[TCP SERVER] Listening on {SERVER_HOST}:{SERVER_PORT} ...")

client_socket, client_addr = server_socket.accept()
print(f"[TCP SERVER] UR client connected from {client_addr}")

done_set = set()
robot_waiting_kit_ready_set = set()
def receive_robot_done():
    global done_set, robot_waiting_kit_ready_set, client_socket

    buffer = ""

    while True:
        try:
            data = client_socket.recv(1024)
            if not data:
                break

            buffer += data.decode("utf-8", errors="ignore")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                msg = line.strip()

                if not msg:
                    continue

                if msg.startswith("PICK_DONE|"):
                    task_id = int(msg.split("|")[1])
                    robot_waiting_kit_ready_set.add(task_id)
                    print(f"[RECV FROM UR CLIENT] PICK_DONE task_id={task_id}")

                elif msg.startswith("DONE|"):
                    task_id = int(msg.split("|")[1])
                    done_set.add(task_id)
                    robot_waiting_kit_ready_set.discard(task_id)
                    print(f"[RECV FROM UR CLIENT] DONE task_id={task_id}")

        except OSError as e:
            print(f"[RECV FROM UR CLIENT] socket error: {e}")
            break

threading.Thread(target=receive_robot_done, daemon=True).start()


# 初始化攝影機串流
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
'''
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
'''
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

# 設置滑鼠事件回調函數
cv2.namedWindow("Task Monitor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Task Monitor", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    
    now = time.time()
    if not ret:
        break
    
    # 檢查 Unity 是否傳來 Start
    poll_unity_messages()

    # 如果 Unity 太早按 Start，等區域取點完成後自動開始
    if pending_unity_start and not current_zone and not task_start:
        start_task_monitoring(source="Unity pending Start")

    frame_height, frame_width = frame.shape[:2]

    if task_start and start_time is not None:
        now_from_start = now - start_time
    else:
        now_from_start = 0.0

    schedule_index, human_current_task = get_current_task(schedule, "human")
    robot_index, robot_current_task = get_current_task(schedule, "robot")
    update_operation_state(schedule, human_current_task, robot_current_task)
    
    # 現在畫面上要顯示哪一組 layout 的 zone
    if not task_start:
        if now < freeze_until:
            layout_id = active_layout_id
        else:
            layout_id = current_layout_id
            active_layout_id = current_layout_id
    else:
        if human_current_task:
            layout_id = human_current_task["layout"]
            active_layout_id = layout_id
        else:
            layout_id = active_layout_id
    
    zones = layouts_zones[layout_id]

    zone_objects = {"Top": [], "BottomLeft": [], "BottomRight": []}     # 區域內的物件統計表
    kitboxes = []
    sth_on_box = False

    # YOLO 推論
    results = model_yolo(frame, verbose=False)
    detections = results[0].boxes

    #HOI辨識
    interaction_object, pred_action = task_model.grab_object_recognition(frame, model_yolo, detections, zones)

    # 顯示區域取點訊息
    if current_zone:
        display_message(frame, f"Select points for {current_zone}")
    
    # 繪製zone多邊形
    if len(kitbox_zone) == 4:
        cv2.polylines(frame, [kitbox_zone], True, (255, 255, 255), 1)

    for zone_name, poly in zones.items():
        if len(poly) == 4:
            cv2.polylines(frame, [poly], True, (0, 255, 255), 2)

            x0, y0 = int(poly[0][0]), int(poly[0][1])
            cv2.putText(frame, zone_name, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 遍歷偵測到的物件
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0]
        cls_id = int(box.cls[0])
        label = model_yolo.names[cls_id]
        conf = float(box.conf[0])

        # 計算框中心點
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        point = (cx, cy)

        
        if len(kitbox_zone)==4 and label == "empty_kit_box":
            kitboxes.append(point)

        else:
            # 判斷是否有物料在kit box裡(防止model誤判的機制)
            # 排除 robot 已 pick 完、正在等待 place 的物件
            if len(kitbox_zone)==4 and cv2.pointPolygonTest(kitbox_zone, point, False) >= 0:
                if not is_robot_waiting_place_object(label):
                    sth_on_box = True

            # 判斷物料屬於哪個區域
            for zone_name, poly in zones.items():
                if len(poly) == 4:  # 確保區域已經被定義
                    inside = cv2.pointPolygonTest(poly, point, False)
                    if inside >= 0:  # >= 0 表示在區域內或邊界上
                        zone_objects[zone_name].append(label)

        # 繪製bounding box與中心點
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{label}", (int(x1), int(y1) - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


    # 開始追蹤排程(已取區域點完畢且按下開始鍵)
    if not current_zone and task_start:
        human_state_text = f"Human: {operation_state['human']['status']}"
        robot_state_text = f"Robot: {operation_state['robot']['status']}"
        cv2.putText(frame, human_state_text, (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, robot_state_text, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        dispatch_ready_robot_task(now_from_start)
        update_robot_progress(robot_index, robot_current_task, now_from_start)

        for waiting_task_id in list(robot_waiting_kit_ready_set):
            kit_ready =  is_kit_ready_for_robot(schedule, waiting_task_id)

            send_kit_ready_to_robot(waiting_task_id, kit_ready)

            if kit_ready:
                robot_waiting_kit_ready_set.remove(waiting_task_id)

        if human_current_task :
            command = human_current_task["command"]
            standard_time = human_current_task["human_standard_time"]  # 取得標準時間

            ensure_task_started(schedule_index,started_at=now, now_from_start=now_from_start)
            subtask_start_time = task_runtime_tracker.get(human_current_task.get("ID"))
            
            if command == "replace":
                # kit box replace 必須等同一 kit 內的人機物件都完成後才顯示
                replace_ready = all_pick_place_done(schedule, schedule_index)
                now = time.time()

                if replace_ready:
                    display_message(frame, "Current task: Replace new kit box")

                    # 第一次進入 ready 狀態，才開始計時
                    if schedule_index not in replace_ready_start:
                        replace_ready_start[schedule_index] = now

                else:
                    display_message(frame, replace_waiting_message)
                    send_replace_waiting_to_unity()

                    # 還沒 ready 前，不應保留 replace 計時起點
                    replace_ready_start.pop(schedule_index, None)

                if kitboxes:
                    kitbox_replace = check_kitbox_replace(kitboxes, schedule_index)
                else:
                    kitbox_replace = False

                if kitbox_replace:
                    start_t = replace_ready_start.get(schedule_index, now)
                    if start_time is not None:
                        schedule[schedule_index]["actual_start_time"] = start_t - start_time

                    tid = schedule[schedule_index].get("ID")
                    if tid is not None:
                        task_runtime_tracker[tid] = start_t

                    actual_time = now - start_t

                    finalize_task(
                        schedule_index,
                        source="replace",
                        completed_at=now,
                        actual_time=actual_time
                    )

                    notify_current_human_task_to_unity(force=True)

                    delay_reschedule_state["task_id"] = None
                    delay_reschedule_state["last_trigger_time"] = None
                    delay_reschedule_state["next_trigger_time"] = None

                    completed_objects.clear()
                    kitbox_replace = False
                    replace_ready_start.pop(schedule_index, None)

                    print(f"actual_time:{actual_time:.2f}, standard_time:{standard_time:.2f}")
                    delay_time = actual_time - standard_time
                    if delay_time > 0:
                        print(f"Task is delayed by {delay_time:.2f}s! ")

                elif replace_ready and schedule_index in replace_ready_start:

                    now_from_start = now - start_time if start_time is not None else now

                    if should_trigger_replace_delay(
                        task=human_current_task,
                        schedule_index=schedule_index,
                        standard_time=standard_time,
                        now_from_start=now_from_start,
                        replace_ready_start=replace_ready_start,
                        start_time=start_time,
                    ):
                        cv2.putText(frame, "Delayed !!!", (1000, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                        # replace 的實際作業開始時間應該是 replace_ready_start，而不是進入 replace 任務但還在等待前一箱完成的時間
                        if start_time is not None:
                            schedule[schedule_index]["actual_start_time"] = (
                                replace_ready_start[schedule_index] - start_time
                            )

                        tid = schedule[schedule_index].get("ID")
                        if tid is not None:
                            task_runtime_tracker[tid] = replace_ready_start[schedule_index]

                        print("Delay detected (replace) → trigger rescheduling")
                        info = handle_delay_failure(
                            current_idx=schedule_index,
                            now_from_start=now_from_start,
                            standard_time=standard_time,
                        )

                        notify_current_human_task_to_unity(force=True)

            elif command == "pick and place":
                # 顯示排程指示
                display_message(frame, f"Current task: Place {human_current_task['object']} in {human_current_task['zone']}")
                
                target_object = human_current_task["object"]
                target_zone = human_current_task["zone"]

                # 手部拿取錯誤的物件            
                if pred_action and interaction_object and interaction_object != target_object:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0,0), (frame_width, frame_height), (0,0,255), -1)
                    alpha = 0.2 #想把濾鏡調淡就調低 alpha
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    error_time = time.time()
                    cls, match_idx = classify_error(schedule, schedule_index, interaction_object, human_current_task)

                    if cls != "current_task":
                        error_message = wrong_item_message(cls)

                        cv2.putText(
                            frame,
                            error_message,
                            (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            3,
                        )

                        send_error_to_unity(cls, error_message)

                        error_times.append({
                            "time": round(error_time - start_time, 2),
                            "type": cls,
                            "object": interaction_object,
                            "matched_task_id": schedule[match_idx].get("ID") if match_idx is not None else None,
                        })
                
                opp_done = opportunistic_complete(schedule_index, zone_objects, required_stay_time)
                opp_now = time.time()
                if opp_done:
                    subtask_start_time = finalize_opportunistic(
                        current_idx=schedule_index,
                        opp_done=opp_done,
                        now=opp_now,
                        now_from_start=opp_now - start_time if start_time is not None else None
                    )
                    info =  trigger_dynamic_reschedule_opp(
                        current_idx=schedule_index,
                        opp_done=opp_done,
                        now_from_start=opp_now - start_time if start_time is not None else now_from_start,
                    )

                    notify_current_human_task_to_unity(force=True)
                
                # 檢查物件是否已經放置完成 or delay
                now = time.time()
                if target_object in zone_objects[target_zone]:
                    # 第一次發現物件進入區域記錄開始時間
                    if target_object not in zone_stay_start:
                        zone_stay_start[target_object] = now

                    elif now - zone_stay_start[target_object] >= required_stay_time:
                        print(f"{target_object} has been placed in {target_zone}")
                        completed_objects.add(target_object)
                        subtask_end_time = time.time()
                        actual_time = subtask_end_time - subtask_start_time
                        finalize_task(
                            schedule_index,
                            source="current",
                            completed_at=subtask_end_time,
                            actual_time=actual_time
                        )

                        notify_current_human_task_to_unity(force=True)

                        delay_time = actual_time - standard_time

                        print(f"actual_time:{actual_time:.2f}, standard_time:{standard_time:.2f}")

                        # 判斷是否有延遲
                        if delay_time > 0:
                            print(f"Task is delayed by {delay_time:.2f}s! ")
                        
                        del zone_stay_start[target_object]
                        wrong_zone_stay_start.clear()
                        # delay_reschedule_state["task_id"] = None
                        delay_reschedule_state["task_id"] = None
                        delay_reschedule_state["last_trigger_time"] = None
                        delay_reschedule_state["next_trigger_time"] = None
                else:
                    
                    # Delay：
                    # 統一使用「目前預期完成時間」判斷是否延遲
                    # 第一次 = actual_start_time + standard_time
                    # 第二次以後 = _expected_finish_time
                    now_from_start = now - start_time if start_time is not None else now

                    if should_trigger_delay(task=human_current_task, standard_time=standard_time, now_from_start=now_from_start):
                        cv2.putText(frame, f"Delayed !!!", (1500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                        print("Delay detected (pick and place) → trigger rescheduling")
                        info = handle_delay_failure(
                            current_idx=schedule_index,
                            now_from_start=now_from_start,
                            standard_time=standard_time,
                        )

                        notify_current_human_task_to_unity(force=True)

                    if target_object in zone_stay_start:
                        del zone_stay_start[target_object]

                    # 檢查正確物件位置擺置錯誤
                    for wrong_zone, objs in zone_objects.items():
                        if wrong_zone != target_zone and target_object in objs:
                            if wrong_zone not in wrong_zone_stay_start or wrong_zone_stay_start[wrong_zone]==0:
                                wrong_zone_stay_start[wrong_zone] = now
                            elif now - wrong_zone_stay_start[wrong_zone] >= required_stay_time:
                                error_message = "Object placed in wrong zone"
                                cv2.putText(frame, error_message, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                                send_error_to_unity("wrong_zone", error_message)
                                error_times.append({"time": round(time.time()-start_time,2),  "type": "wrong_zone"})         
                            break
                    for wrong_zone in wrong_zone_stay_start.keys():
                        if target_object not in zone_objects[wrong_zone]:
                            wrong_zone_stay_start[wrong_zone] = 0
        
        # 人員做完了，機械手臂還沒完成最後一個kit內容物
        elif robot_current_task is not None:
            display_message(frame, replace_waiting_message)
            send_replace_waiting_to_unity()

        # 沒有人員當前任務，也沒有 robot dispatched，但仍可能有 pending robot task 尚未到 planned_start_time
        elif has_unfinished_tasks(schedule):
            display_message(frame, "Waiting for next scheduled task")

        # 人機各別任務都做完了
        elif human_current_task is None and robot_current_task is None:
            if not all_done_reported:
                end_time = time.time()
                cycle_time = end_time - start_time
                print(f"All tasks completed, it totally took {cycle_time:.2f}s .")
                print(f"Initial Optimal Makespan: {best_makespan}")
                notify_current_human_task_to_unity(force=True)
                all_done_reported = True
            cv2.putText(frame, f"All tasks completed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # 顯示畫面
    cv2.imshow("Task Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if task_start == False:
            start_task_monitoring(source="keyboard")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_unity_connection()

planned_ids = [str(t.get("ID")) for t in offline_best_schedule]
actual_ids = [str(t["schedule"]["ID"]) for t in completion_log if t["schedule"]["ID"] is not None]
print("[Planned order]", " ".join(planned_ids))
print("[Actual order]", " ".join(actual_ids))

# 把所有失效當下記錄的甘特圖一次存檔
# tqdm找不出失敗原因
# from tqdm import tqdm
# if deferred_gantt_items:
#     print(f"Start saving {len(deferred_gantt_items)} failure gantt figure(s)...")

#     for item in tqdm(
#         deferred_gantt_items,
#         desc="Saving test failure gantt figures",
#         unit="fig"
#     ):
#         save_reschedule_gantt(**item)

#     print("All failure gantt figures saved.")
# else:
#     print("No failure gantt figures to save.")

# # 不使用 tqdm 的 failure gantt 存圖版本
def progress_bar(current, total, width=40, prefix="Progress"):
    percent = current / total if total else 1
    filled = int(width * percent)

    bar = "█" * filled + "-" * (width - filled)

    print(
        f"\r{prefix}: |{bar}| {current}/{total} {percent * 100:6.2f}%",
        end="",
        flush=True
    )

    if current >= total:
        print()

if deferred_gantt_items:
    total = len(deferred_gantt_items)
    print(f"Start saving {len(deferred_gantt_items)} failure gantt figure(s)...")

    for current, item in enumerate(deferred_gantt_items, start=1):
        save_reschedule_gantt(**item)

        progress_bar(
            current=current,
            total=total,
            prefix="Saving gantt figures"
        )

    print("All failure gantt figures saved.")
else:
    print("No failure gantt figures to save.")

# 最終總覽：Planned timeline -> 每輪 Dynamic Result -> Actual Timeline 
summary_rows = [
    {
        "title": "Planned Timeline - Offline optimal schedule",
        "schedule": deepcopy(offline_best_schedule),
    }
]
summary_rows.extend(deepcopy(dynamic_summary_history))
summary_rows.append(
    {
        "title": "Actual / Final Timeline",
        "schedule": deepcopy(schedule),
    }
)

summary_output_dir = os.path.join(BASE_DIR, "dynamic_gantt_outputs")
summary_path = save_monitor_summary_figure(
    summary_rows,
    output_dir=summary_output_dir,
)
print(f"[SUMMARY GANTT SAVED] {summary_path}")

summary_fig = make_monitor_summary_figure(summary_rows)
show_scrollable_figure(summary_fig, window_title="HRC Planned / Dynamic / Actual Timeline Summary")
