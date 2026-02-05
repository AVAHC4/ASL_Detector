"""
TWO-HANDED SIGN LANGUAGE DATA COLLECTION TOOL
==============================================
Interactive tool for collecting two-handed sign language gesture data for model training.
Captures sequences of hand keypoints from webcam for each configured sign.

Features:
- Full-screen capture interface
- Two-hand detection with ordering (left/right)
- Automatic sequence capture (30 frames per sequence)
- Organized folder structure (MP_Data_TwoHands/[SIGN]/[seq_num]/[frame].npy)
- Visual feedback during recording
- Configurable sign vocabulary

Controls:
- / or -    : Previous sign
- + or =    : Next sign
- 0         : Start recording sequence
- 1         : Stop recording
- Q         : Quit application
"""

import cv2
import os
import time
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from data import KeypointNormalizer

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================
ROI_WIDTH_RATIO = 0.60      # ROI width as fraction of frame width (60%)
ROI_HEIGHT_RATIO = 0.70     # ROI height as fraction of frame height (70%)
SEQUENCE_LENGTH = 30        # Number of frames per sequence
CAPTURE_DELAY_MS = 50       # Delay between frame captures (20 fps)
MAX_SEQUENCES = 100         # Maximum sequences per sign per session
CAM_INDEX = 0               # Camera device index
FEATURE_DIM = 126           # 2 hands × 21 landmarks × 3 coords

# MediaPipe setup
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# ============================================================================
# SIGN VOCABULARY - Customize these based on your needs
# ============================================================================
SIGNS = [
    "HELLO", "THANK_YOU", "YES", "NO", "PLEASE",
    "SORRY", "HELP", "MORE", "FINISHED", "I_LOVE_YOU",
    "GOOD", "BAD", "WANT", "NEED", "LIKE",
    "EAT", "DRINK", "SLEEP", "WORK", "PLAY",
    "HAPPY", "SAD", "ANGRY", "SCARED", "EXCITED",
    "BATHROOM"
]

# ============================================================================
# DATA STRUCTURE INITIALIZATION
# ============================================================================
out_root = Path("MP_Data_TwoHands")
for sign in SIGNS:
    (out_root / sign).mkdir(parents=True, exist_ok=True)


def next_sequence_index(sign_dir: Path) -> int:
    """Determine the next available sequence number for a sign."""
    existing = [int(p.name) for p in sign_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return (max(existing) + 1) if existing else 0


def extract_two_hand_keypoints(results):
    """
    Extract and order keypoints for two hands.
    Returns 126-feature vector: [left_hand (63), right_hand (63)]
    """
    keypoints = np.zeros(FEATURE_DIM)
    
    if not results.hand_landmarks:
        return keypoints, False, False
    
    hands_data = []
    for i, hand_landmarks in enumerate(results.hand_landmarks):
        pts = []
        for lm in hand_landmarks:
            pts.append([lm.x, lm.y, lm.z])
        hand_array = np.array(pts).flatten()
        wrist_x = hand_landmarks[0].x
        
        handedness = None
        if results.handedness and i < len(results.handedness):
            handedness = results.handedness[i][0].category_name
        
        hands_data.append({
            'array': hand_array,
            'wrist_x': wrist_x,
            'handedness': handedness,
            'landmarks': hand_landmarks
        })
    
    left_hand = None
    right_hand = None
    
    for hand in hands_data:
        if hand['handedness'] == 'Left':
            left_hand = hand
        elif hand['handedness'] == 'Right':
            right_hand = hand
    
    if left_hand is None and right_hand is None and len(hands_data) > 0:
        sorted_hands = sorted(hands_data, key=lambda h: h['wrist_x'], reverse=True)
        left_hand = sorted_hands[0]
        if len(sorted_hands) > 1:
            right_hand = sorted_hands[1]
    elif left_hand is None and len(hands_data) == 2:
        left_hand = [h for h in hands_data if h != right_hand][0]
    elif right_hand is None and len(hands_data) == 2:
        right_hand = [h for h in hands_data if h != left_hand][0]
    
    if left_hand:
        keypoints[:63] = left_hand['array']
    if right_hand:
        keypoints[63:126] = right_hand['array']
    
    return keypoints, left_hand is not None, right_hand is not None


def put_text_center(img, text, y, scale=1.0, color=(255,255,255), thickness=2):
    """Draw text centered horizontally on the image."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(10, (img.shape[1] - w) // 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                scale, color, thickness, cv2.LINE_AA)


def draw_hand_landmarks(img, results, roi_x, roi_y, roi_w, roi_h):
    """Draw hand landmarks on the frame."""
    if not results.hand_landmarks:
        return
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    
    for i, hand_landmarks in enumerate(results.hand_landmarks):
        handedness = "Unknown"
        if results.handedness and i < len(results.handedness):
            handedness = results.handedness[i][0].category_name
        
        color = (255, 100, 100) if handedness == "Left" else (100, 255, 100)
        
        for landmark in hand_landmarks:
            x_px = roi_x + int(landmark.x * roi_w)
            y_px = roi_y + int(landmark.y * roi_h)
            cv2.circle(img, (x_px, y_px), 5, color, -1)
        
        for connection in connections:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_point = (roi_x + int(start.x * roi_w), roi_y + int(start.y * roi_h))
            end_point = (roi_x + int(end.x * roi_w), roi_y + int(end.y * roi_h))
            cv2.line(img, start_point, end_point, color, 2)


def main():
    """Main data collection application loop."""
    
    # Initialize camera
    cap = cv2.VideoCapture(CAM_INDEX)
    cv2.namedWindow("Two-Hand Data Collector", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Two-Hand Data Collector", cv2.WND_PROP_FULLSCREEN, 
                         cv2.WINDOW_FULLSCREEN)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Initialize MediaPipe for two hands
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    landmarker = HandLandmarker.create_from_options(options)

    # State variables
    current_idx = 0
    current_sign = SIGNS[current_idx]
    recording = False
    sequence_buffer = []
    sequence_counts = {sign: next_sequence_index(out_root / sign) for sign in SIGNS}
    session_counts = {sign: 0 for sign in SIGNS}
    last_capture_ms = 0

    print("=" * 60)
    print("TWO-HANDED SIGN LANGUAGE DATA COLLECTION")
    print("=" * 60)
    print("Controls:")
    print("  - / +         : Move between signs")
    print("  0             : Start recording sequence")
    print("  1             : Stop recording")
    print("  Q             : Quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        h, w, _ = frame.shape

        # Calculate ROI
        roi_w = int(w * ROI_WIDTH_RATIO)
        roi_h = int(h * ROI_HEIGHT_RATIO)
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        # Extract ROI and detect hands
        roi = frame[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
        results = landmarker.detect(mp_image)

        # Extract keypoints
        keypoints, left_detected, right_detected = extract_two_hand_keypoints(results)
        
        # Normalize keypoints
        keypoints = KeypointNormalizer.normalize_two_hands(keypoints)

        # Draw ROI rectangle
        roi_color = (0, 255, 0) if (left_detected and right_detected) else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 3)

        # Draw hand landmarks
        draw_hand_landmarks(frame, results, x1, y1, roi_w, roi_h)

        # Draw header
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        put_text_center(frame, f"Sign: {current_sign}", 45, scale=1.4)
        
        status = "RECORDING" if recording else "IDLE"
        status_color = (0, 255, 0) if recording else (0, 165, 255)
        put_text_center(frame, f"Status: {status}", 85, scale=0.9, color=status_color)

        # Hand detection status
        left_txt = "LEFT: ●" if left_detected else "LEFT: ○"
        right_txt = "RIGHT: ●" if right_detected else "RIGHT: ○"
        cv2.putText(frame, left_txt, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 100, 100) if left_detected else (100, 100, 100), 2)
        cv2.putText(frame, right_txt, (w - 180, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (100, 255, 100) if right_detected else (100, 100, 100), 2)

        # Draw footer
        cv2.rectangle(frame, (0, h - 130), (w, h), (0, 0, 0), -1)
        
        counter_text = f"Sequences saved: {session_counts[current_sign]} / {MAX_SEQUENCES}"
        put_text_center(frame, counter_text, h - 90, scale=0.9)
        
        if recording:
            frame_count = len(sequence_buffer)
            put_text_center(frame, f"Recording: {frame_count}/{SEQUENCE_LENGTH} frames", 
                          h - 55, scale=0.8, color=(0, 255, 0))
        
        help_text = "- / + : Move | 0: Start | 1: Stop | Q: Quit"
        put_text_center(frame, help_text, h - 20, scale=0.7)

        # Automatic frame capture during recording
        now_ms = int(time.time() * 1000)
        
        if recording and (now_ms - last_capture_ms >= CAPTURE_DELAY_MS):
            # Only capture if at least one hand is detected
            if left_detected or right_detected:
                sequence_buffer.append(keypoints.copy())
                last_capture_ms = now_ms
                
                # Visual feedback
                cv2.circle(frame, (w - 30, 30), 15, (0, 0, 255), -1)
                
                # Check if sequence is complete
                if len(sequence_buffer) >= SEQUENCE_LENGTH:
                    # Save sequence
                    seq_idx = sequence_counts[current_sign]
                    seq_dir = out_root / current_sign / str(seq_idx)
                    seq_dir.mkdir(parents=True, exist_ok=True)
                    
                    for frame_idx, frame_keypoints in enumerate(sequence_buffer):
                        fp = seq_dir / f"{frame_idx}.npy"
                        np.save(fp, frame_keypoints)
                    
                    sequence_counts[current_sign] += 1
                    session_counts[current_sign] += 1
                    sequence_buffer = []
                    
                    print(f"✅ Saved sequence {seq_idx} for {current_sign}")
                    
                    if session_counts[current_sign] >= MAX_SEQUENCES:
                        recording = False
                        print(f"⚠️ Max sequences reached for {current_sign}")

        cv2.imshow("Two-Hand Data Collector", frame)

        # Keyboard input
        key = cv2.waitKey(1) & 0xFFFF
        
        if key in (ord('q'), ord('Q')):
            break
        
        if key == ord('0'):
            if session_counts[current_sign] < MAX_SEQUENCES:
                recording = True
                sequence_buffer = []
                last_capture_ms = 0
                print(f"🔴 Started recording for {current_sign}")
            else:
                print(f"⚠️ Limit reached for {current_sign}")
        
        elif key == ord('1'):
            recording = False
            sequence_buffer = []
            print(f"⏹️ Stopped recording")
        
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
    landmarker.close()
    cv2.destroyAllWindows()
    print("\n✅ Data collection complete!")


if __name__ == "__main__":
    main()
