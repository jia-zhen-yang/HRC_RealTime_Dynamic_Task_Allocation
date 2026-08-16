#給定靜態排程，可偵測任務執行是否偏離(delay)

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# 載入YOLO 模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑
PATH_DIR = os.path.join(BASE_DIR, 'yolov11x.pt') #精準
#PATH_DIR = os.path.join(BASE_DIR, 'yolov11n.pt') #輕量
model = YOLO(PATH_DIR)

# 排程指令（物件類別與目標區域的先後順序）
schedule = [
    {"object": "A_pink", "zone": "Zone A", "status": "pending", "standard_time": 6, "actual_time":None},
    {"object": "B_green", "zone": "Zone B", "status": "pending", "standard_time": 5, "actual_time":None},
    {"object": "C_mint", "zone": "Zone B", "status": "pending", "standard_time": 4, "actual_time":None},
    {"object": "D_skin", "zone": "Zone A", "status": "pending", "standard_time": 4, "actual_time":None},
]

# 初始化滑鼠點擊事件的全局變數
zone_points = []
current_zone = "Zone A"  # 開始取 Zone A 的點
zones = {"Zone A": [], "Zone B": []} #存區域點位

# 用來顯示提示訊息的函數
def display_message(frame, message):
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
# 定義滑鼠事件函數
def mouse_callback(event, x, y, flags, param):
    global zone_points, current_zone
    
    # 左鍵點擊事件
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(zone_points) < 4:
            zone_points.append((x, y))
            print(f"Point {len(zone_points)}: ({x}, {y})")

            # 每取完4個點，則切換區域
            if len(zone_points) == 4:
                if current_zone == "Zone A":
                    zones["Zone A"] = np.array(zone_points, np.int32)
                    zone_points = []  # 清空點，準備選取 Zone B
                    current_zone = "Zone B"
                    print("Zone A 完成，請取 Zone B 的點")
                elif current_zone == "Zone B":
                    zones["Zone B"] = np.array(zone_points, np.int32)
                    print("Zone B 完成，開始物件位置辨識")
                    current_zone = ""
        else:
            print("已選取完區域點")    
            

# 初始化攝影機串流
cap = cv2.VideoCapture(0)

# 設置滑鼠事件回調函數
cv2.namedWindow("Task Monitor")
cv2.setMouseCallback("Task Monitor", mouse_callback)

# 追蹤排程進度
schedule_index = 0  # 用來追蹤當前的排程指令
start_time = time.time()  # 用來計算整體的開始時間


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_height, frame_width = frame.shape[:2]

    # YOLO 推論
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # 顯示區域取點訊息
    if current_zone:
        display_message(frame, f"Select points for {current_zone}")
    
    # 如果已經選擇了區域的四個點，繪製多邊形
    if len(zones["Zone A"]) == 4:
        cv2.polylines(frame, [zones["Zone A"]], True, (0, 255, 255), 2)
    if len(zones["Zone B"]) == 4:
        cv2.polylines(frame, [zones["Zone B"]], True, (0, 255, 255), 2)
    for zone_name, poly in zones.items():
        if len(poly) == 4:  # 確保區域已經定義
            cv2.putText(frame, f"{zone_name}", (poly[0][0], poly[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # 區域內的物件統計表
    zone_objects = {"Zone A": [], "Zone B": []}
    

    # 遍歷偵測到的物件
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0]
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        # 計算框中心點
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        point = (cx, cy)

        # 判斷屬於哪個區域
        for zone_name, poly in zones.items():
            if len(poly) == 4:  # 確保區域已經被定義
                inside = cv2.pointPolygonTest(poly, point, False)
                if inside >= 0:  # >= 0 表示在區域內或邊界上
                    zone_objects[zone_name].append(label)

        # 繪製bounding box與中心點
        #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
        cv2.circle(frame, point, 3, (0, 0, 255), -1)
        cv2.putText(frame, f"{label}", (int(x1), int(y1) - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 物件進出判斷，依照排程驗證(已取區域點完畢)
    if schedule_index < len(schedule) and not current_zone:
        # 顯示排程指示
        display_message(frame, f"Current task: Place {schedule[schedule_index]['object']} in {schedule[schedule_index]['zone']}")
        
        task = schedule[schedule_index]
        target_object = task["object"]
        target_zone = task["zone"]
        standard_time = task["standard_time"]  # 取得標準時間

        if schedule[schedule_index]["actual_time"] is None:
            subtask_start_time = time.time()  # 記錄任務開始時間
            schedule[schedule_index]["actual_time"] = 0

        # 檢查物件是否已經放置完成
        if target_object in zone_objects[target_zone]:
            print(f"{target_object} has been placed in {target_zone}")
            schedule[schedule_index]["status"] = "completed"

            subtask_end_time = time.time()  # 記錄任務結束時間
            actual_time = (subtask_end_time - subtask_start_time)  # 計算實際完成時間
            delay_time = actual_time - standard_time  # 計算延遲時間

            print(f"actual_time:{actual_time:.2f}, standard_time:{standard_time:.2f}")

            # 判斷是否有延遲
            if delay_time > 0:
                print(f"Task{schedule_index} is delayed by {delay_time:.2f}s! ")

            schedule_index += 1  # 跳到下一個排程
        else:
            now = time.time()
            if now - subtask_start_time >= standard_time:
                cv2.putText(frame, f"Task{schedule_index} delayed !!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elif not current_zone:
        if schedule_index == len(schedule):
            end_time = time.time()
            cycle_time = end_time - start_time
            print(f"All tasks completed, it totally took {cycle_time:.2f}s .")
            schedule_index += 1
        cv2.putText(frame, f"All tasks completed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        


    # 顯示畫面
    cv2.imshow("Task Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
