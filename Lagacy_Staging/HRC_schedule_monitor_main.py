#加入失效發生後的排程調整(尚未動態排程後續任務)

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import socket
from copy import deepcopy
import matplotlib.pyplot as plt
from HOI import TaskRecognitionModel
from read_schedule import load_schedule_csv
from precedence_matrix import build_precedence_matrix
from VNS_verification import optimal_solution
from VNS_dynamic_solver import compute_start_times
from VNS_dynamic_solver import (
    compute_start_times,
    solver as dynamic_solver,
    extract_kits,
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
schedule, best_makespan = optimal_solution(schedule, P,case_3=True)
schedule_index = 0  # 用來追蹤當前的排程指令

def initialize_planned_times(schedule, P, case=3):
    start_times = compute_start_times(schedule, P, case=case)

    for i, task in enumerate(schedule):
        if task["agent"] == "robot":
            dur = float(task.get("robot_standard_time", 0.0))
        else:
            dur = float(task.get("human_standard_time", 0.0))

        task["planned_start_time"] = float(start_times[i])
        task["planned_finish_time"] = float(start_times[i] + dur)

        # 後面監控用
        task.setdefault("actual_start_time", None)
        task.setdefault("actual_finish_time", None)
        task.setdefault("ur_dispatched", False)


initialize_planned_times(schedule, P, case=3) #TODO:每次動態排程都要更新 加入動態排程後內容可能也要改

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

'''
def reschedule_remaining_tasks(schedule,current_task):
    #隨機重新排序剩餘的任務(排除已完成和當前任務)
    completed = []
    current = []
    remaining = []

    for task in schedule:
        if task["status"] == "completed":
            completed.append(task)
        elif task is current_task:
            current.append(task)
        else:
            remaining.append(task)

    random.shuffle(remaining)

    return completed + current + remaining
'''


def update_operation_state(schedule, human_task, robot_task):
    """更新目前 human / robot 的執行狀態，以及各 task 的簡易狀態摘要。"""
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
            status = "human executing" if task_status == "pending" else "completed"
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

        # if task.get("actual_time") is None or task.get("actual_time") == 0:
        #     task["actual_time"] = 0



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
        elif now - zone_stay_start[obj] >= required_stay_time + 1.0:
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

def count_completed_items(schedule, object_type: str) -> int:
    """
    計算 schedule 中，某種類物件已完成幾個。
    只計算 pick and place 且 status == completed 的任務。
    """
    count = 0
    for t in schedule:
        if t.get("command") != "pick and place":
            continue
        if t.get("status") != "completed":
            continue

        obj = t.get("object")
        if not obj:
            continue
        if obj == object_type:
            count += 1

    return count

TOTAL_KITS = 6
def compute_stack_for_task(task, schedule, total_kits: int = TOTAL_KITS) -> int:
    """
    根據目前 task 與整體 schedule，動態計算 stack

    - stack 最高為 total_kits
    - 已完成同種類物件數量 = completed_count
    - stack = total_kits - completed_count

    """
    obj = task.get("object")
    if not obj:
        raise ValueError("task has no object field")

    completed_count = count_completed_items(schedule, obj)
    stack = total_kits - completed_count

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

    return next(
        ((i, task) for i, task in enumerate(schedule)
         if task["agent"] == agent and task["status"] == "pending"),
        (None, None)
    )

def send_robot_task(task, schedule, client_socket, total_kits: int = TOTAL_KITS):
    """
    將任務資訊透過 TCP/IP 傳給 UR client
    """

    if client_socket is None:
        print("[SEND TO UR] client_socket is None, skip sending")
        return
    
    stack = compute_stack_for_task(task, schedule, total_kits=total_kits)

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

    print(
        f"[SEND TO UR] "
        f"ID={task.get('ID')} "
        f"command={task.get('command')} "
        f"object={task.get('object')} "
        f"zone={task.get('zone')} "
        f"layout={task.get('layout')} "
        f"stack={stack}"
    )


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
        send_robot_task(task, schedule, client_socket, total_kits=3)

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

    print(
        f"[ROBOT DONE] ID={robot_task.get('ID')} "
        f"object={robot_task.get('object')} "
        f"actual_time={actual_time:.2f}s"
    )


def classify_error(schedule, schedule_index, obj, human_current_task):
    """
    分類失效：
    - not_belong   : 不屬於目前 kit
    - later_task   : 屬於目前 kit，但不是目前人員任務
    - robot_task   : 屬於目前 kit，且分配給 robot
    - current_task : 目前人員正在做的任務
    """
    kit_start, kit_end = get_kit_range(schedule, schedule_index)

    match = None
    match_idx = None

    for i in range(kit_start, kit_end):
        t = schedule[i]

        if t.get("command") != "pick and place":
            continue

        if t.get("object") == obj:
            match = t
            match_idx = i
            break

    if match is None:
        return "not_belong", match_idx

    # 目前人員任務
    if human_current_task and match.get("ID") == human_current_task.get("ID"):
        return "current_task", match_idx

    # robot 任務：pending / dispatched 都算 robot task
    if match.get("agent") == "robot":
        return "robot_task", match_idx

    # 剩下就是同 kit 內，但屬於後面的人員任務
    return "later_task", match_idx

def opportunistic_complete(anchor_idx, zone_objects, required_stay_time):
    """
    掃描目前 kit 範圍內所有 pending pick&place 任務（包含 human/robot）
    若任務物件出現在正確 zone 且停留 >= required_stay_time，則直接完成該任務
    """
    global schedule, frame_stay_start, completed_objects
    kit_start, kit_end = get_kit_range(schedule, anchor_idx)

    now = time.time()
    completed_opportunistic = []
    for i in range(kit_start, kit_end):
        if i == anchor_idx:
            continue

        if schedule[i]["status"] != "pending":
            continue
        if schedule[i]["command"] != "pick and place":
            continue

        obj = schedule[i]["object"]
        zone = schedule[i]["zone"]

        if obj in zone_objects.get(zone, []):
            if i not in frame_stay_start:
                frame_stay_start[i] = now
            elif now - frame_stay_start[i] >= required_stay_time:
                completed_objects.add(obj)
                print("[OPP DONE]", schedule[i].get("ID"), schedule[i]["object"], schedule[i]["zone"], schedule[i].get("agent"))
                completed_opportunistic.append(i)
                del frame_stay_start[i]
        else:
            if i in frame_stay_start:
                del frame_stay_start[i]
    return completed_opportunistic

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


def draw_gantt(
    schedule,
    title="Schedule",
    start_getter=None,
    duration_getter=None,
    ax=None,
    show=True
):
    """
    通用 Gantt 繪圖函式：
    - start_getter(task) -> 任務開始時間
    - duration_getter(task) -> 任務持續時間
    """
    colors = {"replace": "tab:blue", "pick and place": "tab:orange"}

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
        created_fig = True

    for task in schedule:
        tid = task.get("ID", "")
        agent = task.get("agent", "")
        start = start_getter(task)
        dur = duration_getter(task)

        # 沒資料就跳過
        if start is None or dur is None:
            continue
        if dur <= 0:
            continue

        ax.barh(
            agent,
            dur,
            left=start,
            color=colors.get(task.get("command"), "gray"),
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

def draw_planned_gantt(schedule, ax=None, show=True):
    draw_gantt(
        schedule,
        title="Planned Timeline",
        start_getter=lambda task: task.get("planned_start_time"),
        duration_getter=lambda task: (
            task.get("planned_finish_time") - task.get("planned_start_time")
            if task.get("planned_start_time") is not None and task.get("planned_finish_time") is not None
            else None
        ),
        ax=ax,
        show=show
    )

def draw_actual_gantt(schedule, ax=None, show=True):
    draw_gantt(
        schedule,
        title="Actual Timeline",
        start_getter=lambda task: task.get("actual_start_time"),
        duration_getter=lambda task: (
            task.get("actual_finish_time") - task.get("actual_start_time")
            if task.get("actual_start_time") is not None and task.get("actual_finish_time") is not None
            else None
        ),
        ax=ax,
        show=show
    )



#有沒有東西在box裡面
sth_on_box = False

# 追蹤排程進度&失效紀錄
task_model = TaskRecognitionModel()
task_start = False
rescheduled = False
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
replace_waiting_message = ""
robot_started_ids = set()
replace_ready_start = {}

# TCP/IP
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5050

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(1)

print(f"[TCP SERVER] Listening on {SERVER_HOST}:{SERVER_PORT} ...")

client_socket, client_addr = server_socket.accept()
print(f"[TCP SERVER UR client connected from {client_addr}")

done_set = set()
def receive_robot_done():
    global done_set, client_socket

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

                if msg.startswith("DONE|"):
                    task_id = int(msg.split("|")[1])
                    done_set.add(task_id)
                    print(f"[RECV FROM UR CLIENT] DONE task_id={task_id}")

        except OSError as e:
            print(f"[RECV FROM UR CLIENT] socket error: {e}")
            break

import threading
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
            if len(kitbox_zone)==4 and cv2.pointPolygonTest(kitbox_zone, point, False) >= 0:
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
                    display_message(frame, "Waiting for all items in this kit box")

                    # 還沒 ready 前，不應保留 replace 計時起點
                    replace_ready_start.pop(schedule_index, None)

                if kitboxes:
                    kitbox_replace = check_kitbox_replace(kitboxes, schedule_index)
                else:
                    kitbox_replace = False

                if kitbox_replace:
                    start_t = replace_ready_start.get(schedule_index, now)
                    actual_time = now - start_t

                    finalize_task(
                        schedule_index,
                        source="replace",
                        completed_at=now,
                        actual_time=actual_time
                    )

                    completed_objects.clear()
                    kitbox_replace = False
                    replace_ready_start.pop(schedule_index, None)

                    print(f"actual_time:{actual_time:.2f}, standard_time:{standard_time:.2f}")
                    delay_time = actual_time - standard_time
                    if delay_time > 0:
                        print(f"Task is delayed by {delay_time:.2f}s! ")

                elif replace_ready and schedule_index in replace_ready_start:
                    if now - replace_ready_start[schedule_index] >= standard_time:
                        cv2.putText(frame, "Delayed !!!", (1000, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

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
                    #拿到已放置於kit box的物件
                    if interaction_object in completed_objects:
                        cv2.putText(frame, f"Wrong item : Already placed in this kit box", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        error_times.append({"time": round(error_time-start_time,2),  "type": "placed"})

                    else:
                        cls, match_idx = classify_error(schedule, schedule_index, interaction_object, human_current_task)

                        #拿到不屬於該kit box的內容物
                        if cls == "not_belong":
                            cv2.putText(frame, "Wrong item : Not belongs to this kit box", (10, 125),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                            error_times.append({"time": round(error_time-start_time, 2), "type": "not belong"})

                        #拿到分配給機械手臂的物件
                        elif cls == "robot_task":
                            cv2.putText(frame, "Wrong item : This is a robot task item", (10, 125),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                            error_times.append({"time": round(error_time-start_time, 2), "type": "robot task"})

                        #拿到後面排程的物件
                        elif cls == "later_task":
                            cv2.putText(frame, "Wrong item : Belongs to a later task", (10, 125),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                            error_times.append({"time": round(error_time-start_time, 2), "type": "later task"})
                
                opp_done = opportunistic_complete(schedule_index, zone_objects, required_stay_time)
                opp_now = time.time()
                if opp_done:
                    subtask_start_time = finalize_opportunistic(
                        current_idx=schedule_index,
                        opp_done=opp_done,
                        now=opp_now,
                        now_from_start=opp_now - start_time if start_time is not None else None
                    )
                    zone_stay_start.pop(target_object, None)
                    wrong_zone_stay_start.clear()
                
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
                        finalize_task(schedule_index, source="current", completed_at=subtask_end_time, actual_time=actual_time)
                        delay_time = actual_time - standard_time

                        print(f"actual_time:{actual_time:.2f}, standard_time:{standard_time:.2f}")

                        # 判斷是否有延遲
                        if delay_time > 0:
                            print(f"Task is delayed by {delay_time:.2f}s! ")
                        
                        del zone_stay_start[target_object]
                        wrong_zone_stay_start.clear()
                        #rescheduled = False #TODO:排程啟動機制觸發 要refine
                else:
                    #Delay
                    if now - subtask_start_time >= standard_time:
                        cv2.putText(frame, f"Delayed !!!", (1500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                        '''
                        TODO:失效串接動態排程(還有其他失效也要)
                        if not rescheduled:
                            print("Delay detected → trigger rescheduling")

                            schedule = reschedule_remaining_tasks(schedule,human_current_task)
                            rescheduled = True
                        '''

                    if target_object in zone_stay_start:
                        del zone_stay_start[target_object]

                    # 檢查正確物件位置擺置錯誤
                    for wrong_zone, objs in zone_objects.items():
                        if wrong_zone != target_zone and target_object in objs:
                            if wrong_zone not in wrong_zone_stay_start or wrong_zone_stay_start[wrong_zone]==0:
                                wrong_zone_stay_start[wrong_zone] = now
                            elif now - wrong_zone_stay_start[wrong_zone] >= required_stay_time:
                                cv2.putText(frame, f"Object placed in wrong zone.", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                                error_times.append({"time": round(time.time()-start_time,2),  "type": "wrong zone"})         
                            break
                    for wrong_zone in wrong_zone_stay_start.keys():
                        if target_object not in zone_objects[wrong_zone]:
                            wrong_zone_stay_start[wrong_zone] = 0
        
        # 人員做完了，機械手臂還沒完成最後一個kit內容物
        elif robot_current_task is not None:
            display_message(frame, "Waiting for all items in this kit box")

        # 人機各別任務都做完了
        elif human_current_task is None and robot_current_task is None:
            if not all_done_reported:
                end_time = time.time()
                cycle_time = end_time - start_time
                print(f"All tasks completed, it totally took {cycle_time:.2f}s .")
                all_done_reported = True
            cv2.putText(frame, f"All tasks completed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # 顯示畫面
    cv2.imshow("Task Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if task_start == False:
            task_start = True
            print("Start task monitoring.")
            start_time = time.time()
            subtask_start_time = start_time
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

planned_ids = [str(t.get("ID")) for t in schedule]
actual_ids = [str(t["schedule"]["ID"]) for t in completion_log if t["schedule"]["ID"] is not None]
print("[Planned order]", " ".join(planned_ids))
print("[Actual order]", " ".join(actual_ids))

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

draw_planned_gantt(schedule, ax=axes[0], show=False)
draw_actual_gantt(schedule, ax=axes[1], show=False)

axes[0].tick_params(labelbottom=True)
axes[1].set_xlabel("Time")

plt.tight_layout()
plt.show()

print(f"sum of duration : {best_makespan:.2f}")
