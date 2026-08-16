#可偵測各區域不同種類物件個數

import cv2
import numpy as np
from ultralytics import YOLO
import os

# 載入YOLO 模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑
PATH_DIR = os.path.join(BASE_DIR, 'yolov11x.pt') #精準
#PATH_DIR = os.path.join(BASE_DIR, 'yolov11n.pt') #輕量
model = YOLO(PATH_DIR)


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
cv2.namedWindow("YOLO Zone Monitor")
cv2.setMouseCallback("YOLO Zone Monitor", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_height, frame_width = frame.shape[:2]

    # YOLO 推論
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # 顯示提示訊息
    if current_zone:
        display_message(frame, f"Select points for {current_zone}")
    
    # 如果已經選擇了區域的四個點，繪製多邊形
    if len(zones["Zone A"]) == 4:
        cv2.polylines(frame, [zones["Zone A"]], True, (0, 255, 255), 2)
    if len(zones["Zone B"]) == 4:
        cv2.polylines(frame, [zones["Zone B"]], True, (0, 255, 255), 2)

    # 區域內的物件統計表
    zone_objects = {"Zone A": [], "Zone B": []}
    '''
    zone_objects = {
    "Zone A": [],  # "Zone A" 這個區域的物件會被存儲在這個列表中
    "Zone B": [],  # "Zone B" 這個區域的物件會被存儲在這個列表中
    }  
    '''

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
                '''
                用來檢查某個點（在這裡是物件的中心點）是否位於一個多邊形區域內:
                cv2.pointPolygonTest(contour, point, measureDist)
                
                contour：這是表示多邊形的座標點，通常是一個 np.array，其元素是多邊形的頂點座標
                point：這是想檢查是否在多邊形內的點，通常是物件框的中心點
                measureDist：這個布林值決定函數是否計算點到多邊形邊界的距離
                當measureDist設定為true時，傳回實際距離值。若傳回值為正/負/0，表示點在多邊形內部/外部/上；
                設定為false時，會傳回-1/0/1三個固定值，表示點在多邊形內部/外部/上
                '''
                if inside >= 0:  # >= 0 表示在區域內或邊界上
                    zone_objects[zone_name].append(label)

        # 繪製bounding box與中心點
        #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
        cv2.circle(frame, point, 3, (0, 0, 255), -1)
        #cv2.putText(frame, f"{label}", (int(x1), int(y1) - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 繪製區域資訊
    for zone_name, poly in zones.items():
        if len(poly) == 4:  # 確保區域已經定義
            objs = zone_objects[zone_name]
            counts = {}
            for obj in objs:
                counts[obj] = counts.get(obj, 0) + 1
            info = f"{zone_name}: " + ", ".join([f"{k}*{v}" for k, v in counts.items()]) if counts else f"{zone_name}: (empty)"
            cv2.putText(frame, info, (poly[0][0], poly[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 顯示畫面
    cv2.imshow("YOLO Zone Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
