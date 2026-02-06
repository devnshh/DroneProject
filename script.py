import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


MAX_BUFFER = 7
CONFIRM_THRESHOLD = 5
DETECTION_CONF = 0.7
TRACKING_CONF = 0.7

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=DETECTION_CONF,
    min_tracking_confidence=TRACKING_CONF
)

gesture_buffer = deque(maxlen=MAX_BUFFER)
last_command_time = 0
COMMAND_DELAY = 0.5

def drone_up(confidence=0.0):
    print(f"UP ({confidence:.2f})")


def drone_down(confidence=0.0):
    print(f"DOWN ({confidence:.2f})")


def drone_left(confidence=0.0):
    print(f"LEFT ({confidence:.2f})")


def drone_right(confidence=0.0):
    print(f"RIGHT ({confidence:.2f})")


def drone_takeoff(confidence=0.0):
    print(f"TAKEOFF ({confidence:.2f})")


def drone_land(confidence=0.0):
    print(f"Drone LAND ({confidence:.2f})")


def drone_flip(confidence=0.0):

    print(f"Drone FLIP ({confidence:.2f})")
def get_finger_states(landmarks):

    fingers = []

    if landmarks[4].x < landmarks[3].x:
        fingers.append(True)
    else:
        fingers.append(False)

    tips = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]

    for tip, base in zip(tips, bases):
        if landmarks[tip].y < landmarks[base].y:
            fingers.append(True)
        else:
            fingers.append(False)

    return fingers


def classify_gesture(fingers, landmarks):

    thumb, index, middle, ring, pinky = fingers

    if not any(fingers):
        return "LAND"

    if all(fingers):
        return "TAKEOFF"

    if index and not middle and not ring and not pinky:
        
        tip = landmarks[8]
        base = landmarks[5] 

        dx = tip.x - base.x
        dy = tip.y - base.y

        if abs(dx) > abs(dy):
            if dx > 0.05:
                return "RIGHT"
            elif dx < -0.05:
                return "LEFT"
        else:
            if dy < -0.05:
                return "UP"
    

    if index and middle and not ring and not pinky:
        return "FLIP"

    if middle and ring and pinky and not index:
        return "DOWN"

    return "UNKNOWN"

def execute_command(gesture, confidence=0.0):
    global last_command_time

    current_time = time.time()

    if current_time - last_command_time < COMMAND_DELAY:
        return

    if gesture == "UP":
        drone_up(confidence)

    elif gesture == "DOWN":
        drone_down(confidence)

    elif gesture == "LEFT":
        drone_left(confidence)

    elif gesture == "RIGHT":
        drone_right(confidence)

    elif gesture == "TAKEOFF":
        drone_takeoff(confidence)

    elif gesture == "LAND":
        drone_land(confidence)

    elif gesture == "FLIP":
        drone_flip(confidence)

    last_command_time = current_time

def get_stable_gesture():

    if len(gesture_buffer) < CONFIRM_THRESHOLD:
        return None

    most_common = max(
        set(gesture_buffer),
        key=gesture_buffer.count
    )

    count = gesture_buffer.count(most_common)

    if count >= CONFIRM_THRESHOLD and most_common != "unknown":
        return most_common

    return None

def main():

    cap = cv2.VideoCapture(0)

    prev_time = 0

    print("\n Control Script Started")
    print("Show gestures to control drone\n")

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        gesture = "NONE"
        confidence = 0.0

        if results.multi_hand_landmarks:
            
            if results.multi_handedness:
                confidence = results.multi_handedness[0].classification[0].score

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                fingers = get_finger_states(hand_landmarks.landmark)

                gesture = classify_gesture(fingers, hand_landmarks.landmark)

                gesture_buffer.append(gesture)

        else:
            gesture_buffer.clear()
            confidence = 0.0

        stable_gesture = get_stable_gesture()

        if stable_gesture:
            execute_command(stable_gesture, confidence)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Stable: {stable_gesture}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Drone Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
