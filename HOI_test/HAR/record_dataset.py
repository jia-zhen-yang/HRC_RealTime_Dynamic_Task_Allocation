import cv2
import numpy as np
import time
import os
import mediapipe as mp
import utils
from utils import extract_joints

# === 設定 ===
WINDOW_SIZE = utils.WINDOW_SIZE
JOINT_NUM =utils.JOINTS
CHANNEL = utils.CHANNELS
STRIDE = 2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑
SAVE_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(SAVE_DIR, exist_ok=True)

label_map = {
    'g': 'grab',
    'a': 'assemble',
    'r': 'release'
}
label_count = {k: 0 for k in label_map.values()}

# === Mediapipe 初始化 ===
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
print("錄製中，請按下 G / A / R 開始錄製動作，按 Q 結束")

is_recording = False
current_label = None
frame_accumulator = []

def save_clip(clip_frames, label):
    num_frames = len(clip_frames)
    for start in range(0, num_frames - WINDOW_SIZE + 1, STRIDE):
        clip = frame_accumulator[start:start + WINDOW_SIZE]
        tensor = np.stack(clip, axis=1)  # [3, T, V]
        save_name = f"{label}_{label_count[label]:03d}.npy"
        np.save(os.path.join(SAVE_DIR, save_name), tensor)
        label_count[label] += 1
        print(f"儲存 {save_name}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("讀取失敗")
        break

    key = cv2.waitKey(1) & 0xFF
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image_rgb)
    mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_style
            .get_default_pose_landmarks_style())
    results_hands = hands.process(image_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))
    joints = extract_joints(image_rgb, results_pose, results_hands)  # shape: [49, 3]
    joints = np.array(joints).T  # → [3, 49]

    # 鍵盤操作切換錄製的動作類別
    if key in [ord('g'), ord('a'), ord('r')] and not is_recording:
        current_label = label_map[chr(key)]
        is_recording = True
        frame_accumulator = []
        print(f"將開始錄製「{current_label}」動作，直到按下 X 停止")

    if key == ord('x') and is_recording:
        is_recording = False
        if len(frame_accumulator) >= WINDOW_SIZE:
            save_clip(frame_accumulator, current_label)
            print(f"完成錄製 {current_label}，共儲存 {label_count[current_label]} 段")
        else:
            print("錄製太短，未儲存")
        frame_accumulator = []

    # 錄製進行中
    if is_recording:
        frame_accumulator.append(joints)
        cv2.putText(frame, f"Recording: {current_label} ({len(frame_accumulator)} frames)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        cv2.putText(frame, f"Press G/A/R to start, X to stop", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

    if key == ord('q'):
        break

    cv2.imshow('Dataset Recorder', frame)

cap.release()
cv2.destroyAllWindows()

# 統計
print("\n錄製統計：")
for label, count in label_count.items():
    print(f" - {label}: {count} 段儲存")
