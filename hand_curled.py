import cv2
import mediapipe as mp
import math
from collections import deque

SMOOTH_FRAMES = 10
CURL_THRESHOLD = 165
CURLED_FINGER = 2

def angle(a, b, c):
    #內積公式計算角度
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag1 = math.sqrt(ba[0]**2 + ba[1]**2)
    mag2 = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag1 * mag2 == 0:
        return 180
    cos = dot / (mag1 * mag2)
    cos = max(-1, min(1, cos))
    return math.degrees(math.acos(cos))


class HandGraspDetector:
    def __init__(self, hands, drawer, smooth_frames=SMOOTH_FRAMES):

        self.hands = hands
        self.drawer = drawer

        self.history = {
            "Left": deque(maxlen=smooth_frames),
            "Right": deque(maxlen=smooth_frames)
        }

    def detect_grasp_show(self, frame, curl_threshold = CURL_THRESHOLD):
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        annotated = frame.copy()

        hand_states = {"Left": False, "Right": False}

        if not result.multi_hand_landmarks:
            # 若沒有偵測到手則平滑更新為 False
            for hand_label in ["Left", "Right"]:
                self.history[hand_label].append(False)
            for hand_label in hand_states:
                hand_states[hand_label] = (
                    sum(self.history[hand_label]) > len(self.history[hand_label]) // 2
                )
            return annotated, hand_states

        for hand, handedness in zip(result.multi_hand_landmarks,
                                    result.multi_handedness):

            hand_label = handedness.classification[0].label  # Left / Right
            lm = hand.landmark

            #每根手指的角度
            angles = {
                "index": round(angle(lm[5], lm[6], lm[8]),2),
                "middle": round(angle(lm[9], lm[10], lm[12]),2),
                "ring": round(angle(lm[13], lm[14], lm[16]),2),
                "pinky": round(angle(lm[17], lm[18], lm[20]),2),
                "thumb": round(angle(lm[2], lm[3], lm[4]),2)
            }

            #判斷grab條件
            curled_count = sum(a < curl_threshold for a in angles.values())
            grabbing = curled_count >= CURLED_FINGER

            # 各手獨立平滑
            self.history[hand_label].append(grabbing)
            grabbing_smooth = (
                sum(self.history[hand_label]) > len(self.history[hand_label]) // 2
            )
            hand_states[hand_label] = grabbing_smooth

            #畫圖
            self.drawer.draw_landmarks(
                annotated, hand, mp.solutions.hands.HAND_CONNECTIONS
            )

            cx = int(lm[9].x * w)
            cy = int(lm[9].y * h)

            cv2.putText(
                annotated,
                f"{hand_label}: {'GRAB' if grabbing_smooth else 'RELEASE'}",
                (cx-150 , cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0,255,0) if grabbing_smooth else (0,0,255),
                3
            )

        
        return annotated, hand_states
        
    def detect_grasp(self, hands_result, curl_threshold=CURL_THRESHOLD):

        hand_states = {"Left": False, "Right": False}

        if not hands_result.multi_hand_landmarks:
            for hand_label in ["Left", "Right"]:
                self.history[hand_label].append(False)
            for hand_label in hand_states:
                hand_states[hand_label] = (
                    sum(self.history[hand_label]) > len(self.history[hand_label]) // 2
                )

            return hand_states

        for hand, handedness in zip(hands_result.multi_hand_landmarks,
                                    hands_result.multi_handedness):

            hand_label = handedness.classification[0].label  # Left / Right
            lm = hand.landmark

            #每根手指的角度
            angles = {
                "index": round(angle(lm[5], lm[6], lm[8]),2),
                "middle": round(angle(lm[9], lm[10], lm[12]),2),
                "ring": round(angle(lm[13], lm[14], lm[16]),2),
                "pinky": round(angle(lm[17], lm[18], lm[20]),2),
                "thumb": round(angle(lm[2], lm[3], lm[4]),2)
            }

            #判斷grab條件
            curled_count = sum(a < curl_threshold for a in angles.values())
            grabbing = curled_count >= CURLED_FINGER

            # 各手獨立平滑
            self.history[hand_label].append(grabbing)
            grabbing_smooth = (
                sum(self.history[hand_label]) > len(self.history[hand_label]) // 2
            )
            hand_states[hand_label] = grabbing_smooth

        return hand_states


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    drawer = mp.solutions.drawing_utils
    detector = HandGraspDetector(hands, drawer, smooth_frames=7)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, states = detector.detect_grasp_show(frame,curl_threshold=165)

        cv2.imshow("Dual-Hand Grasp Detection (Smooth)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
