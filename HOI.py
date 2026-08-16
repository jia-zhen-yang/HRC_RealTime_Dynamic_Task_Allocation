#物件互動的距離判定考慮mediapipe pose + mediapipe hands的關鍵點
import cv2
import os
import numpy as np
from collections import deque, Counter
import mediapipe as mp
import torch
#from HAR.utils import extract_joints, build_adjacency, SkeletonBuffer
#from HAR.model import STGCN
import math
#import time
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
                                         min_detection_confidence=0.8,min_tracking_confidence=0.6)

        # === 初始化 STGCN 模型 ===
        '''
        self.model_stgcn = STGCN(in_channels=3, num_class=len(self.action_label_name), A=torch.tensor(self.A))
        self.model_stgcn.load_state_dict(torch.load(os.path.join(self.HAR_PATH_DIR, 'stgcn_best_0.001_200.pth'))) 
        self.model_stgcn.eval()
        self.buffer = SkeletonBuffer()
        self.A = build_adjacency()
        '''

        # === 抓取辨識初始化 ===
        self.grasp_detector = HandGraspDetector(self.hands,drawer=None)

        # === 初始化其他變數 ===
        self.pred_action = ""  # 存放預測的動作(包含沒有動作)
        self.hand_joints = []  # 存放手部關節點，搭配物件辨識實現HOI
        #self.interaction_object = []  # 存放互動的物件(一次可能多個)
        self.interaction_object = "" #存放互動的物件(假設一次一個)

        self.hand_is_near = False
        self.nearobj = {"label":None,"point":None}
        
        #self.near_object_start_time = {}   # {label: timestamp}
        #self.required_time = 0.5  # seconds 要維持特定時長才算真正靠近

        self.interaction_history = deque(maxlen=7)
    
    @staticmethod
    def is_hand_near_object(hx, hy, cx, cy, threshold=200):
        distance = math.sqrt((hx - cx)**2 + (hy - cy)**2)
        return distance < threshold
    
    def object_moved(self, current_position, object_id, movement_threshold=5):
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


    def grab_object_recognition(self, frame, model_yolo, yolo_result, zones):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = self.hands.process(image_rgb)

        #偵測手部抓取動作
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
        
        '''STGCN動作辨識
        joints = extract_joints(image_rgb, results_hands)
        self.buffer.add_frame(joints)
        #動作推論長度夠、moving window的stride=3
        #if self.buffer.is_full() and self.frame_counter % 3 == 0:
        if self.buffer.is_full():
            tensor = self.buffer.get_tensor()  # [C, T, V, M]
            #print("Tensor ready:", tensor.shape)  # 可送入 STGCN 推論
            
            # STGCN 模型推論
            input_tensor = torch.tensor(tensor.squeeze(-1)).unsqueeze(0)  # [1, C, T, V]
            with torch.no_grad():
                output = self.model_stgcn(input_tensor)  # [1, num_class]
                probs = torch.softmax(output, dim=1)
                pred_idx = probs.argmax(dim=1).item()

                #低於閾值的預測啥也不是
                if probs[0, pred_idx] > 0.7:
                    self.pred_action = f'{self.action_label_name[pred_idx]}'
                else:
                    self.pred_action = ""

            #buffer.clear()  # 清空緩衝區再收集下一段 (註解掉就是moving window)
        '''

        self.pred_action = ""
        #self.interaction_object.clear()
        self.interaction_object = ""
        object_moved = False
        self.nearobj.update({"label": None, "point": None})
        
        for box in yolo_result:
            cls_id = int(box.cls)
            label = model_yolo.names[cls_id]
            conf = box.conf.item()         
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 取得框座標
            
            # 計算中心座標
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            point = (cx, cy)


            # 在區域外的物料才進行HOI辨識
            if label != "empty_kit_box":
                in_box = False
                for zone_name, poly in zones.items():
                    if len(poly) == 4 and cv2.pointPolygonTest(poly, point, False)>=0:
                        in_box = True
                        break
                
                if in_box == False:            
                    # 手部靠近
                    self.hand_is_near = False
                    for hx,hy in self.hand_joints:
                        if self.is_hand_near_object(hx, hy, cx, cy, threshold=DISTANCE_THRESHOLD):
                            self.hand_is_near = True
                            self.nearobj.update({"label": label, "point": point})
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  #綠
                            break

                    '''時間判斷抓取避免掃過誤判
                    now = time.time()
                    if hand_is_near:
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

                        # moving mode
                        self.interaction_history.append(label)      
                        counts = {}
                        weights = [(i+1) for i,item in enumerate(self.interaction_history)]
                        for obj,w in zip(self.interaction_history,weights):
                            counts[obj] = counts.get(obj, 0) + w
                        if counts:
                            label = max(counts, key=counts.get)

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
                        
                    '''碰到加移動視為抓取
                    if object_moved:
                        self.interaction_object = label'''
                
        # moving mode
        nearobj_label, nearobj_point = (self.nearobj["label"], self.nearobj["point"])
        self.interaction_history.append(nearobj_label)
        counts = Counter(self.interaction_history) #出現次數
        smooth_counts = {obj: count for obj, count in counts.items()}
        self.interaction_object = max(smooth_counts, key=smooth_counts.get)

        # HOI結果
        '''
        if self.pred_action and self.interaction_object:
            print(f'{self.pred_action} {self.interaction_object}')
        '''
        
        for grasp in states.values():
            if grasp:
                self.pred_action = "grab"
                break

        self.hand_joints.clear()
        
        return self.interaction_object, self.pred_action
