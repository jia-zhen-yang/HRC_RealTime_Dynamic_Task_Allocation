#物件互動的距離判定考慮mediapipe pose的關鍵點
import cv2
import os
import numpy as np
from collections import deque
import mediapipe as mp
import torch
from HAR.utils import extract_joints, build_adjacency, SkeletonBuffer
from HAR.model import STGCN
from ultralytics import YOLO
import math

# === 設定 ===
height = 480
width = 640
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑
HAR_PATH_DIR = os.path.join(BASE_DIR, 'HAR/results')
YOLO_PATH_DIR = os.path.join(BASE_DIR, 'Object Detection/yolo_test.pt')

action_label_name = ['grab', 'assemble', 'release']

# === Mediapipe 初始化 ===
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

buffer = SkeletonBuffer()
A = build_adjacency()

# === 初始化 STGCN 模型 ===
model_stgcn = STGCN(in_channels=3, num_class=len(action_label_name), A=torch.tensor(A))
model_stgcn.load_state_dict(torch.load(os.path.join(HAR_PATH_DIR, 'stgcn_best.pth')))  # 若已訓練可取消註解
model_stgcn.eval()

# === 初始化 YOLO 模型 ===
model_yolo = YOLO(YOLO_PATH_DIR)

def is_hand_near_object(hx, hy, cx, cy, threshold=50):
    distance = math.sqrt((hx - cx)**2 + (hy - cy)**2)
    return distance < threshold

def distance_hand_object(hx, hy, cx, cy):
    distance = math.sqrt((hx - cx)**2 + (hy - cy)**2)
    return distance

# === 主程式：webcam 採集與緩衝輸出 ===
cap = cv2.VideoCapture(0)
print("Webcam initialized. Press 'q' to quit.")

pred_action = ""  #存放預測的動作(包含沒有動作)
pred_object = dict() #存放偵測到的物件和對應的中心位置
hand_joints = [] #存放手部關節點，搭配物件辨識實現HOI
interaction_object = [] #存放互動的物件(一次可能多個)
#interaction_object = "" #存放互動的物件(假設一次一個)
frame_counter = 0
while cap.isOpened(): 
    success, frame = cap.read()
    if not success:
        print("無法獲取相機畫面")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image_rgb)
    '''
    mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_style
            .get_default_pose_landmarks_style())
    '''
    results_hands = hands.process(image_rgb)
    '''
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))
    '''

    if results_pose.pose_landmarks:
        for idx in [19,20,21,22]:
            hx, hy = results_pose.pose_landmarks.landmark[idx].x * width, results_pose.pose_landmarks.landmark[idx].y * height
            hand_joints.append((int(hx), int(hy)))
    
    joints = extract_joints(image_rgb, results_pose, results_hands)
    buffer.add_frame(joints)

    #動作推論長度夠、moving window的stride=3
    if buffer.is_full() and frame_counter % 3 == 0:
    #if buffer.is_full():
        tensor = buffer.get_tensor()  # [C, T, V, M]
        #print("Tensor ready:", tensor.shape)  # 可送入 STGCN 推論
        
        # STGCN 模型推論
        input_tensor = torch.tensor(tensor.squeeze(-1)).unsqueeze(0)  # [1, C, T, V]
        with torch.no_grad():
            output = model_stgcn(input_tensor)  # [1, num_class]
            probs = torch.softmax(output, dim=1)
            pred_idx = probs.argmax(dim=1).item()

            #低於閾值的預測啥也不是
            if probs[0, pred_idx] > 0.7:
                pred_action = f'{action_label_name[pred_idx]}'
                print(f'{action_label_name[pred_idx]} {probs[0, pred_idx]:.2f}')
            else:
                pred_action = ""
            #print(pred_action)

        #buffer.clear()  # 清空緩衝區再收集下一段 (註解掉就是moving window)


    # 使用 YOLO 模型進行預測
    results = model_yolo.predict(source=frame, conf=0.5, verbose=False)
    for result in results:
        pred_object.clear()
        for box in result.boxes:
            cls_id = int(box.cls)
            label = model_yolo.names[cls_id]
            conf = box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 取得框座標
            
            # 計算中心座標
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            pred_object[label] = (cx, cy)

            # 如果手靠進則畫出不同顏色預測框(包含中心點)
            for joint in hand_joints:
                hx,hy = joint
                if is_hand_near_object(hx, hy, cx, cy, threshold=80):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    interaction_object.append(label)
                    #interaction_object = label
                    break #至少有一隻手靠近
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # 疊加分類結果在畫面上
    '''
    cv2.putText(frame, f'Action: {pred_action}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    #cv2.putText(frame, f'Object: {list(pred_object.keys())}', (20, 80),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  #畫面中所有物件
    cv2.putText(frame, f'Object: {interaction_object}', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)   #互動的物件
    '''
    if pred_action and interaction_object:
        cv2.putText(frame, f'{pred_action} {interaction_object}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('Live Feed', frame)

    hand_joints.clear()
    interaction_object.clear()
    #interaction_object = "" 
    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
