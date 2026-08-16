#監控完整workflow

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import matplotlib.pyplot as plt
from HOI import TaskRecognitionModel
import random
from read_schedule import load_schedule_csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 執行程式的路徑

# YOLO 模型 
OBJ_PATH_DIR = os.path.join(BASE_DIR, 'Object Detection/yolov11x.pt') #精準/輕量
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
schedule_index = 0  # 用來追蹤當前的排程指令

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

def check_kitbox_replace(kitboxes):
    global kitbox_zone, required_stay_time, schedule, schedule_index, sth_on_box
    obj = "kit box"
    now = time.time()

    inside_any = False
    for box in kitboxes:
        if cv2.pointPolygonTest(kitbox_zone, box, False) >= 0:
            inside_any = True

    if inside_any and not sth_on_box:
        if obj not in zone_stay_start:
            zone_stay_start[obj] = now

        elif now - zone_stay_start[obj] >= required_stay_time:
            print(f"New {obj} has been placed.")
            schedule[schedule_index]["status"] = "completed"

            del zone_stay_start[obj]
            
            return True
    elif obj in zone_stay_start:
        del zone_stay_start[obj]
        
    return False

def objects_for_current_box(schedule, start_idx):
    objs = set()
    for t in schedule[start_idx:]:
        if t["command"] == "replace":
            break
        if t["command"] == "pick and place":
            objs.add(t["object"])
    return objs

def get_current_task(schedule, agent):
    return next(
        (task for task in schedule 
         if task["agent"] == agent and task["status"] == "pending"),
        None
    )

def gantt(results, category_names, error_times):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1) #累積和，計算每個堆疊的起點
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))
 
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
 
    bar_info = []
    handles = []  
    labels_legend = []
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'black'
        bar_info.append((rects, widths, text_color))

        handles.append(rects[0])
        labels_legend.append(colname)
    
    for rects, widths, color in bar_info:
        for rect, w in zip(rects, widths):
            text = f"{w:.1f}"
            x = rect.get_x() + rect.get_width()/2
            y = rect.get_y() + rect.get_height()/2
            ax.text(
                x, y, text,
                ha='center', va='center',
                color=color,
                fontsize=12,
                zorder=20 
            )
    legend1 = ax.legend(handles=handles, labels=labels_legend, 
                        loc='lower left', bbox_to_anchor=(0, 1), ncol=len(category_names), fontsize='small')

    #在甘特圖中加入錯誤標示
    labels = list(results.keys())
    actual_idx = labels.index("Actual workflow")
    bar_h = 0.5
    y_bottom = actual_idx - bar_h/2
    y_top = actual_idx + bar_h/2
    
    # 顏色依錯誤類型
    error_color_mapping = {
        "placed": "royalblue",
        "not belong": "purple",
        "later task": "lightgray",
        "wrong zone": "green"
    }

    for err in error_times:
        t = err["time"]
        etype = err["type"]
        color = error_color_mapping.get(etype)

        # 垂直線（ Actual workflow 列）
        ax.vlines(
            x=t,
            ymin=y_bottom,
            ymax=y_top,
            color=color,
            linestyle='--',
            linewidth=1.5,
            zorder=10
        )

    error_handles = []
    error_labels_legend = []
    
    for etype, color in error_color_mapping.items():
        error_handles.append(plt.Line2D([0], [0], color=color, lw=4))
        error_labels_legend.append(etype)

    ax.add_artist(legend1)
    ax.legend(handles=error_handles, labels=error_labels_legend, ncol=len(error_labels_legend)//2,
              bbox_to_anchor=(1, 1), loc='lower right', fontsize='small')

    return fig, ax

def get_task_duration(task, duration_key="actual_time"):
    """
    取得任務時間長度。

    duration_key:
    - "actual_time"：畫實際時間
    - "standard_time"：畫標準時間
    """

    if duration_key == "actual_time":
        dur = task.get("actual_time")

    elif duration_key == "standard_time":
        agent = task.get("agent")

        if agent == "human":
            dur = task.get("human_standard_time")
        elif agent == "robot":
            dur = task.get("robot_standard_time")
        else:
            dur = task.get("human_standard_time", task.get("robot_standard_time"))

    else:
        dur = task.get(duration_key)

    try:
        if dur is None:
            return 0.0
        return float(dur)
    except (TypeError, ValueError):
        return 0.0


def compute_start_times(schedule, duration_key="actual_time"):
    """
    依照 agent 分別累加任務開始時間。
    例如 human 和 robot 各自從 0 開始排。
    """

    agent_elapsed = {}
    start_times = []

    for task in schedule:
        agent = task.get("agent", "unknown")
        start = agent_elapsed.get(agent, 0.0)
        start_times.append(start)

        dur = get_task_duration(task, duration_key=duration_key)
        agent_elapsed[agent] = start + dur

    return start_times


def draw_gantt(
    schedule,
    title="Schedule",
    duration_key="actual_time",
    error_times=None,
    error_agent="human",
    ax=None,
    show=True
):
    """
    新版甘特圖。

    duration_key:
    - "standard_time"：畫預期排程
    - "actual_time"：畫實際流程
    """

    start_times = compute_start_times(schedule, duration_key=duration_key)
    colors = {
        "replace": "tab:blue",
        "pick and place": "tab:orange"
    }

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        created_fig = True
    else:
        fig = ax.figure

    for task, start in zip(schedule, start_times):
        dur = get_task_duration(task, duration_key=duration_key)

        if dur <= 0:
            continue

        agent = task.get("agent", "unknown")
        tid = task.get("ID", "")

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

    # 錯誤標示，只建議畫在 Actual workflow
    if error_times:
        error_color_mapping = {
            "placed": "royalblue",
            "not belong": "purple",
            "later task": "lightgray",
            "wrong zone": "green"
        }

        for err in error_times:
            t = err.get("time")
            etype = err.get("type")
            color = error_color_mapping.get(etype, "red")

            if t is None:
                continue

            ax.axvline(
                x=t,
                color=color,
                linestyle="--",
                linewidth=1.5,
                zorder=10
            )

        error_handles = [
            plt.Line2D([0], [0], color=color, lw=3, linestyle="--")
            for color in error_color_mapping.values()
        ]

        ax.legend(
            handles=error_handles,
            labels=list(error_color_mapping.keys()),
            loc="upper right",
            fontsize="small"
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    if created_fig:
        plt.tight_layout()
        if show:
            plt.show()

    return fig, ax
            
# 當前kit box物件
kitbox_objects = objects_for_current_box(schedule,schedule_index)

#有沒有東西在box裡面
sth_on_box = False

# 追蹤排程進度&失效紀錄
task_model = TaskRecognitionModel()
task_start = False
rescheduled = False
start_time = time.time()  # 用來計算整體的開始時間
zone_stay_start = {}
wrong_zone_stay_start = {}
kitbox_replace = False
required_stay_time = 0.8  # 物件至少停留特定秒數才算完成任務
completed_objects = set()
error_times = []

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

    human_current_task = get_current_task(schedule, "human")
    robot_current_task = get_current_task(schedule, "robot")
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
    #results = model_yolo.track(frame, persist=True, conf=0.5, verbose=False, tracker="botsort.yaml")
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
        if human_current_task:
            command = human_current_task["command"]
            standard_time = human_current_task["human_standard_time"]  # 取得標準時間

            if human_current_task["actual_time"] is None:
                subtask_start_time = time.time()  # 記錄任務開始時間
                human_current_task["actual_time"] = 0

            if command == "replace":
                # 顯示排程指示
                display_message(frame, "Current task: Replace new kit box")

                completed_objects.clear()
                now = time.time()

                if kitboxes:
                    kitbox_replace = check_kitbox_replace(kitboxes)
                
                if kitbox_replace:
                    kitbox_replace = False
                    subtask_end_time = time.time()  # 記錄任務結束時間
                    actual_time = (subtask_end_time - subtask_start_time)  # 計算實際完成時間
                    schedule[schedule_index]["actual_time"] = round(actual_time,2)
                    print(f"actual_time:{actual_time:.2f}, standard_time:{standard_time:.2f}")
                    delay_time = actual_time - standard_time  # 計算延遲時間
                    # 判斷是否有延遲
                    if delay_time > 0:
                        print(f"Task is delayed by {delay_time:.2f}s! ")
                    
                    schedule_index += 1
                    kitbox_objects = objects_for_current_box(schedule,schedule_index)

                #Delay
                elif now - subtask_start_time >= standard_time:
                    cv2.putText(frame, f"Delayed !!!", (1500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

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

                    #拿到不屬於該kit box的內容物
                    elif interaction_object not in kitbox_objects:
                        cv2.putText(frame, f"Wrong item : Not belongs to this kit box", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        error_times.append({"time": round(error_time-start_time,2),  "type": "not belong"})
                    
                    #拿到後面排程的物件
                    else:
                        cv2.putText(frame, f"Wrong item : Belongs to a later task", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        error_times.append({"time": round(error_time-start_time,2),  "type": "later task"})
                
                # 檢查物件是否已經放置完成 or delay
                now = time.time()
                if target_object in zone_objects[target_zone]:
                    # 第一次發現物件進入區域記錄開始時間
                    if target_object not in zone_stay_start:
                        zone_stay_start[target_object] = now

                    elif now - zone_stay_start[target_object] >= required_stay_time:
                        print(f"{target_object} has been placed in {target_zone}")
                        schedule[schedule_index]["status"] = "completed"
                        completed_objects.add(target_object)

                        subtask_end_time = time.time()  # 記錄任務結束時間
                        actual_time = (subtask_end_time - subtask_start_time)  # 計算實際完成時間
                        schedule[schedule_index]["actual_time"] = round(actual_time,2)
                        delay_time = actual_time - standard_time  # 計算延遲時間

                        print(f"actual_time:{actual_time:.2f}, standard_time:{standard_time:.2f}")

                        # 判斷是否有延遲
                        if delay_time > 0:
                            print(f"Task is delayed by {delay_time:.2f}s! ")
                        
                        del zone_stay_start[target_object]
                        wrong_zone_stay_start.clear()

                        schedule_index += 1  # 跳到下一個排程
                        #rescheduled = False
                else:
                    #Delay
                    if now - subtask_start_time >= standard_time:
                        cv2.putText(frame, f"Delayed !!!", (1500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                        '''
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

        # 人機各別任務都做完了
        elif not (human_current_task and robot_current_task):
            if schedule_index == len(schedule):
                end_time = time.time()
                cycle_time = end_time - start_time
                print(f"All tasks completed, it totally took {cycle_time:.2f}s .")
                schedule_index += 1
            cv2.putText(frame, f"All tasks completed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # 顯示畫面
    cv2.imshow("Task Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if task_start == False:
            task_start = True
            print("Start task monitoring.")
            start_time = time.time()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 華麗甘特圖
# category_names = [f"{task['object']}_{task['zone']}" for task in schedule]
# results = {
#     'Scheduled plan': [task["human_standard_time"] for task in schedule],
#     'Actual workflow': [task["actual_time"] for task in schedule],
# }
# gantt(results, category_names, error_times)
# plt.show()

# 簡明甘特圖
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

draw_gantt(
    schedule,
    title="Scheduled plan",
    duration_key="standard_time",
    ax=axes[0],
    show=False
)

draw_gantt(
    schedule,
    title="Actual workflow",
    duration_key="actual_time",
    error_times=error_times,
    error_agent="human",
    ax=axes[1],
    show=False
)

plt.tight_layout()
plt.show()




