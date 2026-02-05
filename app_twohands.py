"""
TWO-HANDED SIGN LANGUAGE RECOGNITION SYSTEM
=============================================
Real-time two-handed sign language recognition using MediaPipe hand tracking and LSTM neural network.
Supports detection of actual ASL signs that require both hands.

Features:
- Real-time two-hand detection and tracking
- Configurable sign vocabulary
- Confidence monitoring with visual graphs
- Word builder with keyboard controls
- Modern cyberpunk UI design
"""

import cv2
import numpy as np
import json
from collections import deque
import time

# Import mediapipe components for the new API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from data import KeypointNormalizer

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

import tensorflow as tf
from keras.layers import Layer
from keras.models import model_from_json

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', 
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[-1],),
                                 initializer='zeros', 
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        # Explicitly cast to float32 to avoid type issues with mixed precision
        return tf.cast(tf.reduce_sum(output, axis=1), dtype=tf.float32)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
# Define the action space (two-handed ASL signs)
# These can be customized based on what signs you want to detect
actions = np.array([
    "HELLO", "THANK_YOU", "YES", "NO", "PLEASE",
    "SORRY", "HELP", "MORE", "FINISHED", "I_LOVE_YOU",
    "GOOD", "BAD", "WANT", "NEED", "LIKE",
    "EAT", "DRINK", "SLEEP", "WORK", "PLAY",
    "HAPPY", "SAD", "ANGRY", "SCARED", "EXCITED",
    "BATHROOM"
])

# Number of consecutive frames needed for a prediction sequence
sequence_length = 30

# Feature dimension for two hands (21 landmarks × 3 coords × 2 hands = 126)
feature_dim = 126

# Minimum confidence threshold for accepting predictions (0-1 scale)
threshold = 0.85

# Number of consistent predictions required before accepting a sign
consistency_len = 12

# ============================================================================
# MODEL LOADING (Commented out until model is trained)
# ============================================================================
# Uncomment these lines after training a two-handed model:
# with open("model_twohands.json", "r") as f:
#     model = model_from_json(f.read(), custom_objects={'AttentionLayer': AttentionLayer})
# model.load_weights("best_model_twohands.h5")

# For testing without a trained model, set model to None
model = None
print("⚠️  No trained model loaded. Running in detection-only mode.")
print("   Train a model first using train_twohands.py with collected data.")

# ============================================================================
# COLOR SCHEME - CYBERPUNK NEON THEME
# ============================================================================
# All colors in BGR format (OpenCV standard)
COLOR_BG = (5, 10, 30)           # Deep navy background
COLOR_PRIMARY = (0, 150, 255)    # Bright blue - main highlights
COLOR_SECONDARY = (0, 90, 180)   # Deep blue - accents
COLOR_SUCCESS = (0, 255, 200)    # Cyan green - success states
COLOR_ACCENT = (0, 200, 255)     # Light cyan - secondary accents
COLOR_TEXT = (230, 240, 255)     # Light blue-white - text
COLOR_DIM = (100, 130, 170)      # Muted blue-gray - subtle elements
COLOR_ROI = (0, 180, 255)        # Cyan - Region of Interest border
COLOR_LEFT_HAND = (255, 100, 100)   # Red-ish for left hand
COLOR_RIGHT_HAND = (100, 255, 100)  # Green-ish for right hand

# ============================================================================
# VIDEO CAPTURE INITIALIZATION
# ============================================================================
# Initialize webcam with HD resolution
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("❌ Error: Cannot open camera. Please check:")
    print("   1. Camera is connected")
    print("   2. Camera permissions are granted to Terminal/Python")
    print("   3. No other application is using the camera")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Configure MediaPipe Hand Landmarker for TWO HANDS
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,  # CHANGED: Detect up to 2 hands
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

landmarker = HandLandmarker.create_from_options(options)


sequence = []                       # Rolling buffer of hand keypoints for sequence prediction
predictions = []                    # Recent prediction indices for consistency checking
current_sign = ""                   # Currently detected sign
current_confidence = 0.0            # Confidence score of current prediction
confidence_history = deque(maxlen=50)  # Historical confidence scores for graph visualization
fps_history = deque(maxlen=30)     # Frame rate history for averaging
recognized_sentence = ""            # Built-up sentence from recognized signs
last_sign = ""                      # Last sign added to prevent duplicates
last_recognition_time = 0           # Timestamp of last successful recognition


def extract_two_hand_keypoints(results, roi_w, roi_h):
    """
    Extract and order keypoints for two hands.
    Returns 126-feature vector: [left_hand (63), right_hand (63)]
    
    Hands are ordered by x-position of wrist (in mirrored view):
    - Right side of screen (smaller x after flip) = Right hand
    - Left side of screen (larger x after flip) = Left hand
    """
    keypoints = np.zeros(126)  # 2 hands × 21 landmarks × 3 coords
    
    if not results.hand_landmarks:
        return keypoints, [], []
    
    hands_data = []
    for i, hand_landmarks in enumerate(results.hand_landmarks):
        pts = []
        for lm in hand_landmarks:
            pts.append([lm.x, lm.y, lm.z])
        hand_array = np.array(pts).flatten()
        wrist_x = hand_landmarks[0].x  # Wrist x-position for ordering
        
        # Get handedness if available
        handedness = None
        if results.handedness and i < len(results.handedness):
            handedness = results.handedness[i][0].category_name  # "Left" or "Right"
        
        hands_data.append({
            'array': hand_array,
            'wrist_x': wrist_x,
            'handedness': handedness,
            'landmarks': hand_landmarks
        })
    
    # Order hands: use handedness if available, otherwise use wrist x-position
    left_hand = None
    right_hand = None
    
    for hand in hands_data:
        if hand['handedness'] == 'Left':
            left_hand = hand
        elif hand['handedness'] == 'Right':
            right_hand = hand
    
    # If handedness not available, order by wrist x (larger x = left side in mirrored view)
    if left_hand is None and right_hand is None and len(hands_data) > 0:
        sorted_hands = sorted(hands_data, key=lambda h: h['wrist_x'], reverse=True)
        left_hand = sorted_hands[0]  # Larger x = left hand
        if len(sorted_hands) > 1:
            right_hand = sorted_hands[1]
    elif left_hand is None and len(hands_data) == 2:
        right_hand = [h for h in hands_data if h != right_hand][0] if right_hand else hands_data[0]
        left_hand = [h for h in hands_data if h != right_hand][0]
    elif right_hand is None and len(hands_data) == 2:
        left_hand = [h for h in hands_data if h != left_hand][0] if left_hand else hands_data[0]
        right_hand = [h for h in hands_data if h != left_hand][0]
    
    # Assign to keypoints array
    if left_hand:
        keypoints[:63] = left_hand['array']
    if right_hand:
        keypoints[63:126] = right_hand['array']
    
    left_landmarks = left_hand['landmarks'] if left_hand else None
    right_landmarks = right_hand['landmarks'] if right_hand else None
    
    return keypoints, left_landmarks, right_landmarks


def draw_gradient_rect(img, pt1, pt2, color1, color2, alpha=0.3):
    """Draw a smooth vertical gradient rectangle overlay."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2

    for i in range(y1, y2):
        ratio = (i - y1) / (y2 - y1)
        color = tuple([int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2)])
        cv2.line(overlay, (x1, i), (x2, i), color, 1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hud_element(img, x, y, w, h, color, alpha=0.15):
    """Draw a HUD panel with semi-transparent background and corner accents."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

    corner_len = 15
    corner_thick = 3

    # Top-left corner
    cv2.line(img, (x, y), (x + corner_len, y), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x, y), (x, y + corner_len), color, corner_thick, cv2.LINE_AA)

    # Top-right corner
    cv2.line(img, (x + w, y), (x + w - corner_len, y), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x + w, y), (x + w, y + corner_len), color, corner_thick, cv2.LINE_AA)

    # Bottom-left corner
    cv2.line(img, (x, y + h), (x + corner_len, y + h), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x, y + h), (x, y + h - corner_len), color, corner_thick, cv2.LINE_AA)

    # Bottom-right corner
    cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, corner_thick, cv2.LINE_AA)


def draw_confidence_bar(img, x, y, w, h, confidence):
    """Draw an animated confidence meter with color-coded fill."""
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_DIM, -1)
    fill_w = int(w * confidence)

    if confidence >= threshold:
        bar_color = COLOR_SUCCESS
    elif confidence >= 0.6:
        bar_color = COLOR_ACCENT
    else:
        bar_color = (50, 100, 255)

    cv2.rectangle(img, (x, y), (x + fill_w, y + h), bar_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_PRIMARY, 2, cv2.LINE_AA)

    text = f"{int(confidence * 100)}%"
    cv2.putText(img, text, (x + w + 15, y + h - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)


def draw_line_graph(img, x, y, w, h, data, color, label=""):
    """Draw a real-time line graph for visualizing confidence history."""
    if len(data) < 2:
        return

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

    points = []
    max_val = max(data) if max(data) > 0 else 1
    
    for i, val in enumerate(data):
        px = x + int((i / len(data)) * w)
        py = y + h - int((val / max_val) * h)
        points.append((px, py))

    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, 2, cv2.LINE_AA)
        cv2.circle(img, points[i], 2, color, -1, cv2.LINE_AA)

    if label:
        cv2.putText(img, label, (x + 10, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)


def draw_glowing_text(img, text, pos, font_scale, color, thickness, glow_color=None):
    """Draw text with a glowing effect."""
    x, y = pos

    if glow_color:
        for offset in range(3, 0, -1):
            alpha = 0.2 - (offset * 0.05)
            glow_intensity = tuple(int(c * alpha) for c in glow_color)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, glow_intensity, thickness + offset, cv2.LINE_AA)

    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness, cv2.LINE_AA)


def draw_hand_landmarks(roi, landmarks, roi_w, roi_h, color):
    """Draw hand landmarks and connections on the ROI."""
    if landmarks is None:
        return
    
    # Draw landmarks
    for idx, landmark in enumerate(landmarks):
        x_px = int(landmark.x * roi_w)
        y_px = int(landmark.y * roi_h)
        cv2.circle(roi, (x_px, y_px), 5, color, -1)
    
    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    for connection in connections:
        start_idx, end_idx = connection
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_point = (int(start.x * roi_w), int(start.y * roi_h))
        end_point = (int(end.x * roi_w), int(end.y * roi_h))
        cv2.line(roi, start_point, end_point, color, 2)


# ============================================================================
# WINDOW SETUP
# ============================================================================
cv2.namedWindow("TWO-HANDED SIGN LANGUAGE RECOGNITION", cv2.WINDOW_NORMAL)
cv2.resizeWindow("TWO-HANDED SIGN LANGUAGE RECOGNITION", 1280, 720)

# ============================================================================
# STARTUP MESSAGE
# ============================================================================
print("🚀 Two-Handed Sign Language Recognition System Initialized")
print("🔹 Press 'Q' to quit | 'SPACE' to add sign to sentence | 'BACKSPACE' to delete | 'ENTER' to clear")
print("🎨 Color Scheme: CYBERPUNK NEON (Aesthetic Mode)")
print("👐 Two-Hand Mode: LEFT=Red, RIGHT=Green")

frame_time = time.time()

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)  # Mirror effect

    left_detected = False
    right_detected = False
    
    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - frame_time) if current_time - frame_time > 0 else 0
    frame_time = current_time
    fps_history.append(fps)

    h, w, _ = frame.shape

    # Create dark overlay
    hud = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)
    cv2.addWeighted(frame, 0.6, hud, 0.4, 0, frame)

    # Define larger ROI for two hands (60% width, 70% height)
    roi_w, roi_h = int(w * 0.60), int(h * 0.70)
    x1, y1 = (w - roi_w) // 2, (h - roi_h) // 2
    x2, y2 = x1 + roi_w, y1 + roi_h

    roi = frame[y1:y2, x1:x2]

    # Hand detection
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
    results = landmarker.detect(mp_image)

    # Extract two-hand keypoints
    keypoints, left_landmarks, right_landmarks = extract_two_hand_keypoints(results, roi_w, roi_h)
    
    left_detected = left_landmarks is not None
    right_detected = right_landmarks is not None
    hands_detected = left_detected or right_detected
    
    # Normalize keypoints
    keypoints = KeypointNormalizer.normalize_two_hands(keypoints)

    # Draw hand landmarks with different colors
    if left_landmarks:
        draw_hand_landmarks(roi, left_landmarks, roi_w, roi_h, COLOR_LEFT_HAND)
    if right_landmarks:
        draw_hand_landmarks(roi, right_landmarks, roi_w, roi_h, COLOR_RIGHT_HAND)

    sequence.append(keypoints)
    sequence = sequence[-sequence_length:]

    # Run prediction if model is loaded and sequence is complete
    if model is not None and len(sequence) == sequence_length:
        input_data = np.expand_dims(np.array(sequence), axis=0)
        res = model.predict(input_data, verbose=0)[0]
        
        pred_index = int(np.argmax(res))
        predictions.append(pred_index)
        predictions = predictions[-consistency_len:]

        confidence = float(np.max(res))
        confidence_history.append(confidence)

        if (len(predictions) == consistency_len and 
            len(np.unique(predictions)) == 1 and
            confidence >= threshold):
            current_sign = actions[pred_index]
            current_confidence = confidence
            last_recognition_time = current_time
    else:
        # Demo mode: just show detection status
        confidence_history.append(0.5 if hands_detected else 0.0)

    # Draw ROI boundary with status colors
    if left_detected and right_detected:
        roi_color = COLOR_SUCCESS  # Both hands
    elif hands_detected:
        roi_color = COLOR_ACCENT   # One hand
    else:
        roi_color = COLOR_ROI      # No hands

    # Draw corner brackets around ROI
    corner_len = 40
    corner_thick = 3

    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), roi_color, corner_thick, cv2.LINE_AA)

    # ========================================================================
    # TOP BAR - SYSTEM INFO
    # ========================================================================
    bar_h = 80
    draw_hud_element(frame, 20, 20, w - 40, bar_h, COLOR_PRIMARY, 0.15)

    draw_glowing_text(frame, "TWO-HANDED ASL RECOGNITION", (40, 55), 
                      1.0, COLOR_PRIMARY, 3, COLOR_PRIMARY)

    # Hand status indicators
    left_status = "L:●" if left_detected else "L:○"
    right_status = "R:●" if right_detected else "R:○"
    cv2.putText(frame, left_status, (w - 280, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LEFT_HAND if left_detected else COLOR_DIM, 2, cv2.LINE_AA)
    cv2.putText(frame, right_status, (w - 200, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RIGHT_HAND if right_detected else COLOR_DIM, 2, cv2.LINE_AA)

    # FPS display
    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
    cv2.putText(frame, f"FPS: {int(avg_fps)}", (w - 280, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)

    # ========================================================================
    # LEFT PANEL - CURRENT PREDICTION DISPLAY
    # ========================================================================
    panel_x, panel_y = 20, 120
    panel_w, panel_h = 350, 300
    draw_hud_element(frame, panel_x, panel_y, panel_w, panel_h, COLOR_SECONDARY, 0.12)

    cv2.putText(frame, "DETECTED SIGN", (panel_x + 20, panel_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DIM, 2, cv2.LINE_AA)

    if current_sign and (current_time - last_recognition_time < 2.0):
        # Adjust font size based on text length
        font_scale = 2.5 if len(current_sign) <= 6 else 1.5
        draw_glowing_text(frame, current_sign, 
                         (panel_x + 30, panel_y + 160), font_scale, 
                         COLOR_PRIMARY, 4, COLOR_PRIMARY)
    else:
        draw_glowing_text(frame, "---", (panel_x + 100, panel_y + 160), 
                         3.0, COLOR_DIM, 4, None)

    cv2.putText(frame, "CONFIDENCE", (panel_x + 20, panel_y + 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DIM, 2, cv2.LINE_AA)
    draw_confidence_bar(frame, panel_x + 20, panel_y + 230, 310, 30, current_confidence)

    # ========================================================================
    # RIGHT PANEL - CONFIDENCE HISTORY GRAPH
    # ========================================================================
    graph_x = w - 370
    graph_y = 120
    graph_w, graph_h = 350, 150
    draw_line_graph(frame, graph_x, graph_y, graph_w, graph_h, 
                    list(confidence_history), COLOR_PRIMARY, "CONFIDENCE HISTORY")

    # ========================================================================
    # RIGHT PANEL - SENTENCE BUILDER
    # ========================================================================
    word_y = graph_y + graph_h + 20
    word_h = 150
    draw_hud_element(frame, graph_x, word_y, graph_w, word_h, COLOR_SUCCESS, 0.12)

    cv2.putText(frame, "SENTENCE BUILDER", (graph_x + 20, word_y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DIM, 2, cv2.LINE_AA)

    display_sentence = recognized_sentence if recognized_sentence else "..."
    # Truncate if too long
    if len(display_sentence) > 25:
        display_sentence = "..." + display_sentence[-22:]
    cv2.putText(frame, display_sentence, (graph_x + 20, word_y + 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_SUCCESS, 2, cv2.LINE_AA)

    cv2.putText(frame, "[SPACE] Add | [BACK] Delete | [ENTER] Clear", 
                (graph_x + 20, word_y + 135), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DIM, 1, cv2.LINE_AA)

    # ========================================================================
    # BOTTOM INSTRUCTION BAR
    # ========================================================================
    bottom_y = h - 60
    draw_hud_element(frame, 20, bottom_y, w - 40, 40, COLOR_ACCENT, 0.12)
    
    cv2.putText(frame, "Place BOTH hands in the detection zone for two-handed signs", (40, bottom_y + 27), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)
    
    cv2.putText(frame, "[Q] QUIT", (w - 150, bottom_y + 27), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2, cv2.LINE_AA)

    # Display frame
    cv2.imshow("TWO-HANDED SIGN LANGUAGE RECOGNITION", frame)

    # Keyboard input handling
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord(' '):
        if current_sign and current_sign != last_sign:
            recognized_sentence += (" " + current_sign) if recognized_sentence else current_sign
            last_sign = current_sign
    elif key == 8:  # Backspace
        words = recognized_sentence.split()
        if words:
            recognized_sentence = " ".join(words[:-1])
    elif key == 13:  # Enter
        recognized_sentence = ""
        last_sign = ""

# ============================================================================
# CLEANUP
# ============================================================================
cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("✅ System shutdown complete")
