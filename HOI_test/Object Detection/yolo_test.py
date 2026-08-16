import cv2
from ultralytics import YOLO
import os

# 載入你訓練好的 YOLO 模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑
PATH_DIR = os.path.join(BASE_DIR, 'yolo.pt')
model = YOLO(PATH_DIR)

# 開啟 webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 模型進行預測
    results = model.predict(source=frame, conf=0.7, verbose=False)

    # 取得預測結果中的 box 資訊
    # 遍歷每一張預測結果（通常只會有一張）
    for result in results:
        #遍歷這張圖中所有的預測框
        for box in result.boxes:
            # 框預測的「類別編號」
            cls_id = int(box.cls)
            # 記錄每個類別對應的名稱
            label = model.names[cls_id]
            conf = box.conf.item()
            #取得預測框的四個角座標（左上 x1, y1；右下 x2, y2）
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 取得框座標

            # 計算中心座標
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 畫出預測框(包含中心點)與標籤
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 顯示影像
    cv2.imshow("YOLOv11n Webcam Detection", frame)

    # 按 q 鍵離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
