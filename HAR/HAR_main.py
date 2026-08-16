import cv2
import os
import numpy as np
from collections import deque
import mediapipe as mp
import torch
from utils import extract_joints, build_adjacency, SkeletonBuffer
from model import STGCN  # 你之前的模型定義

# === 設定 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑
HAR_PATH_DIR = os.path.join(BASE_DIR, 'results')

label_name = ['grab', 'release']

# === Mediapipe 初始化 ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

buffer = SkeletonBuffer()
A = build_adjacency()
# === 初始化 STGCN 模型 ===
model = STGCN(in_channels=3, num_class=len(label_name), A=torch.tensor(A))
model.load_state_dict(torch.load(os.path.join(HAR_PATH_DIR, 'stgcn_best_0.001_200.pth')))  # 若已訓練可取消註解
model.eval()

# === 主程式：webcam 採集與緩衝輸出 ===
cap = cv2.VideoCapture(0)
print("Webcam initialized. Press 'q' to quit.")

pred_label = ""
frame_counter = 0
#window = []
while cap.isOpened(): 
    success, frame = cap.read()
    if not success:
        print("無法獲取相機畫面")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    joints = extract_joints(image_rgb, results_hands)
    buffer.add_frame(joints)

    #動作推論長度夠、moving window的stride=3
    if buffer.is_full() and frame_counter % 3 == 0:
        tensor = buffer.get_tensor()  # [C, T, V, M]
        #print("Tensor ready:", tensor.shape)  # 可送入 STGCN 推論
        
        # [在此處接入 STGCN 模型推論程式]
        input_tensor = torch.tensor(tensor.squeeze(-1)).unsqueeze(0)  # [1, C, T, V]
        with torch.no_grad():
            output = model(input_tensor)  # [1, num_class]
            probs = torch.softmax(output, dim=1)
            pred_idx = probs.argmax(dim=1).item()

            #低於閾值的預測啥也不是
            if probs[0, pred_idx] > 0.85:
                #pred_label = f'{label_name[pred_idx]} ({probs[0, pred_idx]:.2f})'
                pred_label = f'{label_name[pred_idx]}'
                
            else:
                pred_label = ""
            
            #window.append(pred_label)
            print(pred_label)

        #buffer.clear()  # 清空緩衝區再收集下一段 (註解掉就是moving window)
                
    frame_counter += 1
    #if len(window) >= 3:
    #    print(window)
    #    pred_label = max(window,key=window.count) #平滑取眾數
    #    window.pop(0)
    
    # 疊加分類結果在畫面上
    cv2.putText(frame, f'Action: {pred_label}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Live Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
