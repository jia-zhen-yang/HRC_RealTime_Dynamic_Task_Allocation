#加入失效發生後的排程調整(尚未動態排程後續任務)

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

        elif now - zone_stay_start[obj] >= required_stay_time+1.0:
            print(f"New {obj} has been placed.")
            schedule[schedule_index]["status"] = "completed"
            log_completion(schedule_index, source="replace")


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

def get_current_task(schedule, agent):
    return next(
        ((i,task) for i,task in enumerate(schedule) 
         if task["agent"] == agent and task["status"] == "pending"),
        (None,None)
    )

def classify_error(schedule, schedule_index, obj, human_current_task):
    """
    分類失效 "not_belong" | "later_task" | "robot_task" | "current_task"
    """
    kit_start, kit_end = get_kit_range(schedule, schedule_index)

    match = None
    match_idx = None
    for i in range(kit_start, kit_end):
        t = schedule[i]
        if t["status"] != "pending":
            continue
        if t["command"] != "pick and place":
            continue
        if t["object"] == obj:
            match = t
            match_idx = i
            break

    if match is None:
        return "not_belong", match_idx
    if human_current_task and match.get("ID") == human_current_task.get("ID"):
        return "current_task", match_idx
    if match.get("agent") == "robot":
        return "robot_task", match_idx
    return "later_task", match_idx

def opportunistic_complete(anchor_idx, zone_objects, required_stay_time):
    """
    掃描目前 kit 範圍內所有 pending pick&place 任務（包含 human/robot）
    若任務物件出現在正確 zone 且停留 >= required_stay_time，則直接完成該任務。
    回傳完成的 task_idx list (schedule index)
    """
    global schedule,  frame_stay_start, completed_objects, subtask_start_time
    kit_start, kit_end = get_kit_range(schedule, anchor_idx)

    now = time.time()
    for i in range(kit_start, kit_end):
        if i == anchor_idx:
                continue

        if schedule[i]["status"] != "pending":
            continue
        if schedule[i]["command"] != "pick and place":
            continue

        obj = schedule[i]["object"]
        z = schedule[i]["zone"]

        if obj in zone_objects.get(z, []):
            if i not in frame_stay_start:
                frame_stay_start[i] = now
            elif now - frame_stay_start[i] >= required_stay_time:
                # 假設人員下好離手
                schedule[i]["status"] = "completed"
                completed_objects.add(obj)
                print("[OPP DONE]", schedule[i].get("ID"), schedule[i]["object"], schedule[i]["zone"], schedule[i].get("agent"))

                if schedule[i].get("actual_time", None) is None:
                    schedule[i]["actual_time"] = round(now - subtask_start_time, 2)

                # 若這筆不是當前 human 任務，則變更實際執行順序
                '''
                尚未加入:robot_index更新, robot路徑規劃處理
                '''
                log_completion(i, source="opportunistic")
                subtask_start_time = time.time() #重置作業時間計算

                del frame_stay_start[i]
        else:
            if i in frame_stay_start:
                del frame_stay_start[i]

def log_completion(task_idx, source):
    """
    source: "current" | "opportunistic" | "replace"
    """
    global schedule, completion_log
    completion_log.append({
        "schedule": schedule[task_idx],
        "source": source
    })
    mark("SUB_END", t=now, task_id=schedule[task_idx].get("ID"), extra=source)

def print_planned_timeline(schedule):
    parts = []
    for t in schedule:
        tid = t.get("ID")
        dur = t.get("human_standard_time")
        if tid is None:
            continue
        parts.append(f"{tid}:{dur:.2f}")
    print("[Planned timeline] " + "| " + " | ".join(parts) + " |")

def print_actual_timeline(completion_log):
    parts = []
    sum_time = 0
    for rec in completion_log:
        t = rec["schedule"]
        tid = t.get("ID")
        dur = t.get("actual_time")
        if tid is None:
            continue
        if dur is None:
            parts.append(f"{tid}:NA")
        else:
            parts.append(f"{tid}:{dur:.2f}")
            sum_time += dur
    print("[Actual timeline]  " + "| " + " | ".join(parts) + " |")
    print(f"sum of duration : {sum_time:.2f}")



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
required_stay_time = 1.5  # 物件至少停留特定秒數才算完成任務
completed_objects = set()
error_times = []
completion_log = []  # 記錄實際完成順序（依完成時間）

# ==============處理「作業總時間」與 「子任務時間和」不一致問題==================
events = [] #start/subtask_start/subtask_end/end 時間點記錄
sum_infer = 0
infer_count = 0
def mark(event, t=None, task_id=None, extra=None):
    """紀錄事件時間點（絕對時間），之後會轉成相對 start_time 繪圖"""
    global events
    if t is None:
        t = time.time()
    events.append({
        "event": event,      # e.g., 'START', 'SUB_START', 'SUB_END', 'END'
        "t": float(t),
        "task_id": task_id,
        "extra": extra
    })
def plot_timeline(events, start_time):
    # 轉成相對時間
    ev = []
    for e in events:
        if start_time is None:
            continue
        ev.append({
            **e,
            "t_rel": e["t"] - start_time
        })
    ev.sort(key=lambda x: x["t_rel"])

    # 顏色/層級（y 軸位置）
    style = {
        "START":    {"y": 3, "ls": "-",  "lw": 2.5},
        "SUB_START":{"y": 2, "ls": "--", "lw": 1.5},
        "SUB_END":  {"y": 1, "ls": "-",  "lw": 1.5},
        "END":      {"y": 0, "ls": "-",  "lw": 2.5},
    }

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_title("Timeline of START / SUB_START / SUB_END / END")
    ax.set_xlabel("Time (s, relative to START)")
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(["END", "SUB_END", "SUB_START", "START"])
    ax.set_ylim(-0.5, 3.5)

    # 畫每個事件的垂直線
    for e in ev:
        s = style.get(e["event"], {"y": 0, "ls": ":", "lw": 1})
        x = e["t_rel"]
        y = s["y"]
        ax.vlines(x, y-0.35, y+0.35, linestyles=s["ls"], linewidth=s["lw"])
        
        # 加標籤（只對 SUB_END 標 task_id）
        if e["event"] in ("SUB_END", "SUB_START"):
            tid = e.get("task_id")
            extra = e.get("extra")
            label = f"{e['event']}({tid})" if tid is not None else e["event"]
            if extra:
                label += f"\n{extra}"
            ax.text(x, y+0.38, label, rotation=90, va="bottom", ha="center", fontsize=8)

    ax.grid(True, axis="x", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.show()
# =======================================

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

    schedule_index, human_current_task = get_current_task(schedule, "human")
    robot_index, robot_current_task = get_current_task(schedule, "robot")
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

    #before_infer = time.time()
    # YOLO 推論
    results = model_yolo(frame, verbose=False)
    detections = results[0].boxes

    #HOI辨識
    interaction_object, pred_action = task_model.grab_object_recognition(frame, model_yolo, detections, zones)
    #after_infer = time.time()

    '''if task_start and not all_done_reported:
        infer_count += 1
        sum_infer += (after_infer - before_infer)'''

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
                mark("SUB_START", t=subtask_start_time, task_id=human_current_task.get("ID"), extra="current_task_init")

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

                    else:
                        cls, match_idx = classify_error(schedule, schedule_index, interaction_object, human_current_task)

                        #拿到不屬於該kit box的內容物
                        if cls == "not_belong":
                            cv2.putText(frame, "Wrong item : Not belongs to this kit box", (10, 125),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                            error_times.append({"time": round(error_time-start_time, 2), "type": "not belong"})

                        #拿到已放置於kit box的物件
                        elif cls == "robot_task":
                            cv2.putText(frame, "Wrong item : This is a robot task item", (10, 125),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                            error_times.append({"time": round(error_time-start_time, 2), "type": "robot task"})

                        #拿到後面排程的物件
                        elif cls == "later_task":
                            cv2.putText(frame, "Wrong item : Belongs to a later task", (10, 125),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                            error_times.append({"time": round(error_time-start_time, 2), "type": "later task"})

                        else:
                            # cls == "current_task" 理論上不會進來
                            pass
                
                opportunistic_complete(schedule_index, zone_objects, required_stay_time)
                
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
                        log_completion(schedule_index, source="current")

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
        elif human_current_task is None and robot_current_task is None:
            if not all_done_reported:
                end_time = time.time()
                mark("END", t=end_time)
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
            mark("START", t=start_time)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

planned_ids = [str(t.get("ID")) for t in schedule]
actual_ids = [str(t["schedule"]["ID"]) for t in completion_log if t["schedule"]["ID"] is not None]
print("[Planned order]", " ".join(planned_ids))
print("[Actual order]", " ".join(actual_ids))
print_planned_timeline(schedule)
print_actual_timeline(completion_log)

'''
兩種任務總時間算方式(作業總時間 與 子任務時間和)不一致
模型推論時間大約為0.55s(727電腦)
if start_time is not None and len(events) > 0:
    plot_timeline(events, start_time)
print(f"sum_infer / infer_count : {sum_infer:.2f} / {infer_count}")
'''

