import cv2
import numpy as np
from ultralytics import YOLO
import os
from HOI_analysis import TaskRecognitionModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑

# YOLO 模型 
OBJ_PATH_DIR = os.path.join(BASE_DIR, 'Object Detection/yolov11x.pt') #精準
model_yolo = YOLO(OBJ_PATH_DIR)

task_model = TaskRecognitionModel()

# 初始化攝影機串流
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FPS, 60)

task_start = False
frame_task = 0
frame_grab = 0
frame_near = 0
frame_interaction = 0
frame_object = 0
frame_hands = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_height, frame_width = frame.shape[:2]

    # YOLO 推論
    # results = model_yolo.track(frame, persist=True, conf=0.7, verbose=False, tracker="botsort.yaml")
    results = model_yolo(source=frame, conf=0.7, verbose=False)
    detections = results[0].boxes

    #HOI辨識
    interaction_object, grab, hand_is_near, hands_detection = task_model.task_recognition(frame, model_yolo, detections)

    if task_start :
        frame_task += 1
        if detections:
            frame_object += 1

        if hands_detection:
            frame_hands += 1

        if grab:
            frame_grab += 1

        if hand_is_near:
            frame_near += 1
            
        if interaction_object:
            frame_interaction += 1
        
        '''if obj_moved:
            cv2.putText(frame, "moving", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "still", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)'''

    
    cv2.imshow("Task Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if task_start == False:
            print("task start")
            task_start = True
    elif key == ord('a'):
            print(f"recognized_object_frame_rate {frame_object/frame_task:.2f}")
            print(f"recognized_hands_frame_rate {frame_hands/frame_task:.2f}")
            print(f"recognized_interaction_frame_rate {frame_interaction/frame_task:.2f}")
            print(f"recognized_grab_frame_rate {frame_grab/frame_task:.2f}")
            print(f"------------------------------------------------------------------")
            print(f"hands_near:")
            print(f"recognized_hands_near_rate {frame_near/frame_task:.2f}")
            print(f"recognized_interaction_frame_rate {frame_interaction/frame_near:.2f}")
            print(f"------------------------------------------------------------------")
            frame_task = 0
            frame_near = 0
            frame_interaction = 0
            frame_object = 0
            frame_hands = 0
            frame_grab = 0
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()