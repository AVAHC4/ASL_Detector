"""
TWO-HANDED SIGN LANGUAGE DATA COLLECTION TOOL
==============================================
Interactive tool for collecting two-handed sign language gesture data for model training.
Uses legacy MediaPipe API for compatibility.
"""

import cv2
import os
import time
import numpy as np
from pathlib import Path

import mediapipe as mp
from data import KeypointNormalizer

# ============================================================================
# CONFIGURATION
# ============================================================================
ROI_WIDTH_RATIO = 0.60
ROI_HEIGHT_RATIO = 0.70
SEQUENCE_LENGTH = 30
CAPTURE_DELAY_MS = 50
MAX_SEQUENCES = 100
CAM_INDEX = 0
FEATURE_DIM = 126

# MediaPipe (Legacy API)
mp_hands = mp.solutions.hands

# Sign vocabulary
SIGNS = [
    "HELLO", "THANK_YOU", "YES", "NO", "PLEASE",
    "SORRY", "HELP", "MORE", "FINISHED", "I_LOVE_YOU",
    "GOOD", "BAD", "WANT", "NEED", "LIKE",
    "EAT", "DRINK", "SLEEP", "WORK", "PLAY",
    "HAPPY", "SAD", "ANGRY", "SCARED", "EXCITED",
    "BATHROOM"
]

out_root = Path("MP_Data_TwoHands")
for sign in SIGNS:
    (out_root / sign).mkdir(parents=True, exist_ok=True)


def next_sequence_index(sign_dir: Path) -> int:
    existing = [int(p.name) for p in sign_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return (max(existing) + 1) if existing else 0


def extract_two_hand_keypoints(results):
    keypoints = np.zeros(FEATURE_DIM)
    
    if not results.multi_hand_landmarks:
        return keypoints, False, False
    
    hands_data = []
    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        pts = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        hand_array = np.array(pts).flatten()
        wrist_x = hand_landmarks.landmark[0].x
        
        handedness = "Unknown"
        if results.multi_handedness and i < len(results.multi_handedness):
            handedness = results.multi_handedness[i].classification[0].label
        
        hands_data.append({'array': hand_array, 'wrist_x': wrist_x, 'handedness': handedness, 'landmarks': hand_landmarks})
    
    left_hand = right_hand = None
    for hand in hands_data:
        if hand['handedness'] == 'Left':
            left_hand = hand
        elif hand['handedness'] == 'Right':
            right_hand = hand
    
    if left_hand is None and right_hand is None and hands_data:
        sorted_hands = sorted(hands_data, key=lambda h: h['wrist_x'], reverse=True)
        left_hand = sorted_hands[0]
        if len(sorted_hands) > 1:
            right_hand = sorted_hands[1]
    
    if left_hand:
        keypoints[:63] = left_hand['array']
    if right_hand:
        keypoints[63:126] = right_hand['array']
    
    return keypoints, left_hand is not None, right_hand is not None


def put_text_center(img, text, y, scale=1.0, color=(255,255,255), thickness=2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(10, (img.shape[1] - w) // 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_hand_landmarks(img, results, roi_x, roi_y, roi_w, roi_h):
    if not results.multi_hand_landmarks:
        return
    
    connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    
    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness = results.multi_handedness[i].classification[0].label if results.multi_handedness else "Unknown"
        color = (255, 100, 100) if handedness == "Left" else (100, 255, 100)
        
        for lm in hand_landmarks.landmark:
            cv2.circle(img, (roi_x + int(lm.x * roi_w), roi_y + int(lm.y * roi_h)), 5, color, -1)
        
        for c in connections:
            s, e = hand_landmarks.landmark[c[0]], hand_landmarks.landmark[c[1]]
            cv2.line(img, (roi_x + int(s.x * roi_w), roi_y + int(s.y * roi_h)), (roi_x + int(e.x * roi_w), roi_y + int(e.y * roi_h)), color, 2)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cv2.namedWindow("Two-Hand Data Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Two-Hand Data Collector", 1280, 720)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    current_idx = 0
    current_sign = SIGNS[current_idx]
    recording = False
    sequence_buffer = []
    sequence_counts = {s: next_sequence_index(out_root / s) for s in SIGNS}
    session_counts = {s: 0 for s in SIGNS}
    last_capture_ms = 0

    print("=" * 50)
    print("TWO-HANDED SIGN DATA COLLECTION")
    print("=" * 50)
    print("Controls: -/+ Move | 0 Start | 1 Stop | Q Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        roi_w = int(w * ROI_WIDTH_RATIO)
        roi_h = int(h * ROI_HEIGHT_RATIO)
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        roi = frame[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_roi)

        keypoints, left_detected, right_detected = extract_two_hand_keypoints(results)
        keypoints = KeypointNormalizer.normalize_two_hands(keypoints)

        roi_color = (0, 255, 0) if (left_detected and right_detected) else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 3)
        draw_hand_landmarks(frame, results, x1, y1, roi_w, roi_h)

        # Header
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        put_text_center(frame, f"Sign: {current_sign}", 45, scale=1.4)
        status_color = (0, 255, 0) if recording else (0, 165, 255)
        put_text_center(frame, f"Status: {'RECORDING' if recording else 'IDLE'}", 85, scale=0.9, color=status_color)
        cv2.putText(frame, "L:●" if left_detected else "L:○", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,100) if left_detected else (100,100,100), 2)
        cv2.putText(frame, "R:●" if right_detected else "R:○", (w - 100, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100) if right_detected else (100,100,100), 2)

        # Footer
        cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 0), -1)
        put_text_center(frame, f"Sequences: {session_counts[current_sign]} / {MAX_SEQUENCES}", h - 70, scale=0.9)
        if recording:
            put_text_center(frame, f"Recording: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", h - 40, scale=0.7, color=(0,255,0))
        put_text_center(frame, "- / + : Move | 0: Start | 1: Stop | Q: Quit", h - 15, scale=0.6)

        # Capture
        now_ms = int(time.time() * 1000)
        if recording and (now_ms - last_capture_ms >= CAPTURE_DELAY_MS):
            if left_detected or right_detected:
                sequence_buffer.append(keypoints.copy())
                last_capture_ms = now_ms
                cv2.circle(frame, (w - 30, 30), 15, (0, 0, 255), -1)
                
                if len(sequence_buffer) >= SEQUENCE_LENGTH:
                    seq_idx = sequence_counts[current_sign]
                    seq_dir = out_root / current_sign / str(seq_idx)
                    seq_dir.mkdir(parents=True, exist_ok=True)
                    
                    for i, kp in enumerate(sequence_buffer):
                        np.save(seq_dir / f"{i}.npy", kp)
                    
                    sequence_counts[current_sign] += 1
                    session_counts[current_sign] += 1
                    sequence_buffer = []
                    print(f"✅ Saved sequence {seq_idx} for {current_sign}")
                    
                    if session_counts[current_sign] >= MAX_SEQUENCES:
                        recording = False

        cv2.imshow("Two-Hand Data Collector", frame)

        key = cv2.waitKey(1) & 0xFFFF
        if key in (ord('q'), ord('Q')):
            break
        if key == ord('0'):
            if session_counts[current_sign] < MAX_SEQUENCES:
                recording = True
                sequence_buffer = []
                last_capture_ms = 0
                print(f"🔴 Recording {current_sign}")
        elif key == ord('1'):
            recording = False
            sequence_buffer = []
            print("⏹️ Stopped")
        elif key in (ord('-'), 45):
            current_idx = (current_idx - 1) % len(SIGNS)
            current_sign = SIGNS[current_idx]
            recording = False
            sequence_buffer = []
        elif key in (ord('+'), ord('='), 43, 61):
            current_idx = (current_idx + 1) % len(SIGNS)
            current_sign = SIGNS[current_idx]
            recording = False
            sequence_buffer = []

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("✅ Done!")


if __name__ == "__main__":
    main()
