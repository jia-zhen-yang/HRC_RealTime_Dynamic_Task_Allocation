#物件互動的距離判定考慮mediapipe pose + mediapipe hands的關鍵點
import cv2
import os
import numpy as np
from collections import deque
import mediapipe as mp
import torch
import math
import time
from collections import deque
from hand_curled import HandGraspDetector

DISTANCE_THRESHOLD = 70

class TaskRecognitionModel:
    def __init__(self):
        # === 設定 ===
        #1280 / 720
        #1920 / 1080
        self.height = 1080 
        self.width = 1920 
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑
        self.HAR_PATH_DIR = os.path.join(self.BASE_DIR, 'HAR/results')

        self.action_label_name = ['grab', 'release']

        # === Mediapipe 初始化 ===
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                         min_detection_confidence=0.6,min_tracking_confidence=0.6)
        self.mp_drawing = mp.solutions.drawing_utils

        # === 抓取辨識初始化 ===
        self.grasp_detector = HandGraspDetector(self.hands,self.mp_drawing)

        # === 初始化其他變數 ===
        self.pred_action = ""  # 存放預測的動作(包含沒有動作)
        self.hand_joints = []  # 存放手部關節點，搭配物件辨識實現HOI
        #self.interaction_object = []  # 存放互動的物件(一次可能多個)
        self.interaction_object = "" #存放互動的物件(假設一次一個)
        self.last_positions = {} # 每個物件上次的位置

        self.hand_is_near = False
        self.nearobj = {"label":None,"point":None}
        self.near_distance = None
        self.near_object_start_time = {}   # {label: timestamp}
        self.required_time = 0.5  # seconds 要維持特定時長才算真正靠近

        self.interaction_history = deque(maxlen=7)
    
    @staticmethod
    def is_hand_near_object(hx, hy, cx, cy, threshold=50):
        distance = math.sqrt((hx - cx)**2 + (hy - cy)**2)
        return distance < threshold,distance
    
    def object_moved(self, current_position, object_id, movement_threshold=20):
        if object_id not in self.last_positions:
            # 如果是物件第一次出現，儲存當前位置並返回 False（無位移）
            self.last_positions[object_id] = current_position
            return False
        
        # 取得物件上次的位置
        last_position = self.last_positions[object_id]
        
        # 計算物件上次位置與當前位置之間的距離
        distance = math.sqrt((last_position[0] - current_position[0])**2 + (last_position[1] - current_position[1])**2)
        
        # 如果距離超過閾值，則認為物件有位移
        if distance > movement_threshold:
            # 更新物件的上次位置為當前位置
            self.last_positions[object_id] = current_position
            return True
        
        # 如果距離沒有超過閾值，則認為物件沒有位移
        return False


    def task_recognition(self, frame, model_yolo, yolo_result):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = self.hands.process(image_rgb)
        
        #偵測手部抓取
        states = self.grasp_detector.detect_grasp(results_hands)

        #取得手掌節點位置
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks,
                                                results_hands.multi_handedness):
                # 提取五支手指指尖
                fingertips = [4, 8, 12, 16, 20]  # thumb tip, index tip, middle tip
                points = [(hand_landmarks.landmark[i].x * self.width,
                        hand_landmarks.landmark[i].y * self.height) for i in fingertips]
                
                # 計算手指尖平均位置
                hx = int(sum(p[0] for p in points) / len(points))
                hy = int(sum(p[1] for p in points) / len(points))

                # 加入 hand_joints，支援左右手
                self.hand_joints.append((hx, hy))

                self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))
    

        self.pred_action = ""
        #self.interaction_object.clear()
        self.interaction_object = ""
        object_moved = False
        self.nearobj.update({"label": None, "point": None})
        self.near_distance = None
        for box in yolo_result:
            cls_id = int(box.cls)
            label = model_yolo.names[cls_id]
            conf = box.conf.item()         
            track_id = box.id
            if track_id is not None:
                track_id = int(track_id)
            else:
                track_id = -1  # 或者設定為其他適合的預設值
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 取得框座標
            
            # 計算中心座標
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            point = (cx, cy)

            cv2.circle(frame, point, 7, (0, 0, 255), -1)
          
            # 手部靠近
            obj_uid = f"{label}_{track_id}"
            self.hand_is_near = False
            for hx,hy in self.hand_joints:
                is_near, distance = self.is_hand_near_object(hx, hy, cx, cy, threshold=DISTANCE_THRESHOLD)
                if is_near:
                    self.hand_is_near = True
                    self.nearobj.update({"label": label, "point": point})
                    self.near_distance = distance
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  #綠
                    break

            '''now = time.time()
            if self.hand_is_near:
                #用id判斷
                # 如果第一次靠近 → 記錄時間
                if obj_uid not in self.near_object_start_time:
                    self.near_object_start_time[obj_uid] = now

                # 檢查是否超過 required_time 秒
                if now - self.near_object_start_time[obj_uid] >= self.required_time:
                    # 手持續靠近超過設定時間才視為真正互動
                    self.interaction_object = label
                    #物件移動
                    object_moved = self.object_moved(point, track_id)
            else:
                # 若手離開則清除時間紀錄
                if obj_uid in self.near_object_start_time:
                    del self.near_object_start_time[obj_uid]
                
                #用label判斷
                if label not in self.near_object_start_time:
                    self.near_object_start_time[label] = now

                # 檢查是否超過 required_time 秒
                if now - self.near_object_start_time[label] >= self.required_time:
                    # 手持續靠近超過設定時間才視為真正互動
                    self.interaction_object = label
                    #物件移動
                    object_moved = self.object_moved(point, label)
            else:
                # 若手離開則清除時間紀錄
                if label in self.near_object_start_time:
                    del self.near_object_start_time[label]'''
            
        nearobj_label, nearobj_point = (self.nearobj["label"], self.nearobj["point"])
        # moving mode
        self.interaction_history.append(nearobj_label)      
        counts = {}
        weights = [(i+1) for i,item in enumerate(self.interaction_history)]
        for obj,w in zip(self.interaction_history,weights):
            counts[obj] = counts.get(obj, 0) + w
        if counts:
            self.interaction_object = max(counts, key=counts.get)
        

        for grasp in states.values():
            if grasp:
                self.pred_action = "grab"
                break
        
        # if nearobj_label:
        #     object_moved = self.object_moved(nearobj_point, nearobj_label)

        # HOI結果
        '''
        if self.pred_action and self.interaction_object:
            print(f'{self.pred_action} {self.interaction_object}')
        '''
        cv2.putText(frame, f'{self.pred_action} {self.interaction_object}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        if self.pred_action and self.interaction_object and self.near_distance is not None:
            cv2.putText(
                frame,
                f'Distance: {self.near_distance:.1f}px',
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3
            )

        self.hand_joints.clear()
        
        return self.interaction_object, self.pred_action, self.hand_is_near, results_hands.multi_hand_landmarks


