import numpy as np
import matplotlib.pyplot as plt
from collections import deque

CHANNELS = 3 # x, y, confidence
WINDOW_SIZE = 15
JOINTS = 49 

# === Skeleton 緩衝區 ===
class SkeletonBuffer:
    def __init__(self, window_size=WINDOW_SIZE, joint_num=JOINTS, channel=CHANNELS):
        self.T = window_size
        self.V = joint_num
        self.C = channel
        self.M = 1
        self.buffer = deque(maxlen=window_size)

    #加入每一幀的骨架資料，shape 為 [49, 3]
    def add_frame(self, joints_frame):
        if joints_frame.shape != (self.V, self.C):
            raise ValueError(f"Expected ({self.V},{self.C}), got {joints_frame.shape}")
        self.buffer.append(joints_frame)

    #當緩衝區累積到 30 幀時，回傳 True
    def is_full(self):
        return len(self.buffer) == self.T

    def get_tensor(self):
        if not self.is_full():
            return None
        data = np.stack(self.buffer, axis=0)  # [T, V, C]
        data = np.nan_to_num(data, nan=0.0) #補值處理：若某些 joints 缺失（可能因為 Mediapipe 偵測不到），NaN 值會被轉為 0
        data = data.transpose(2, 0, 1)  # ➜ [C, T, V]
        data = data[:, :, :, np.newaxis]  # ➜ [C, T, V, M]
        return data.astype(np.float32)


# === 擷取 joints ===
def extract_joints(image, results_pose, results_hands):
    joints = np.zeros((JOINTS, CHANNELS), dtype=np.float32)  # x, y, conf

    # Pose joints: 0, 11~16
    pose_ids = [0, 11, 12, 13, 14, 15, 16]
    if results_pose.pose_landmarks:
        for idx_i, mp_id in enumerate(pose_ids):
            lm = results_pose.pose_landmarks.landmark[mp_id]
            joints[idx_i] = [lm.x, lm.y, lm.visibility]

    # Hand joints (最多2手)
    if results_hands.multi_hand_landmarks:
        hands_info = list(zip(results_hands.multi_handedness, results_hands.multi_hand_landmarks))
        #joints[0~6]為pose 的 7 個點，joints[7~27]為左手 21 個點，joints[28~48]為右手 21 個點
        for handedness, hand_landmarks in hands_info:
            label = handedness.classification[0].label  # 'Left' or 'Right'
            if label == 'Left':
                base = 7  # 左手從 index 7 開始
            elif label == 'Right':
                base = 28  # 右手從 index 28 開始
            else:
                continue

            for j in range(21):
                lm = hand_landmarks.landmark[j]
                joints[base + j] = [lm.x, lm.y, 1.0]  # mediapipe hands 無 confidence，設 1.0
    return joints

#在GCN中，會用一個矩陣 A（大小為 V × V）來描述關節點之間的連線關係
def build_adjacency(joint_num=JOINTS):
    A = np.zeros((joint_num, joint_num), dtype=np.float32)

    # Pose connections
    pose_edges = [
        (0, 1), (0, 2),
        (1, 3), (3, 5),
        (2, 4), (4, 6)
    ]

    # Hand connections
    def make_hand_edges(base):
        edges = []
        fingers = [
            [0, 1, 2, 3, 4],     # Thumb
            [0, 5, 6, 7, 8],     # Index
            [0, 9,10,11,12],     # Middle
            [0,13,14,15,16],     # Ring
            [0,17,18,19,20]      # Pinky
        ]
        for finger in fingers:
            for i in range(len(finger) - 1):
                a = base + finger[i]
                b = base + finger[i + 1]
                edges.append((a, b))
        return edges

    #左右手
    hand_edges = make_hand_edges(7) + make_hand_edges(28)

    all_edges = pose_edges + hand_edges

    # 填入 A[i][j] = 1
    for i, j in all_edges:
        A[i][j] = 1
        A[j][i] = 1  # 對稱

    return A

#視覺化關節點的連接
if __name__ == '__main__':
    A = build_adjacency()
    plt.imshow(A, cmap='Greys')
    plt.title("Adjacency Matrix")
    plt.show()