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

# Import mediapipe - using legacy solutions API for compatibility
import mediapipe as mp
from data import KeypointNormalizer

# ============================================================================
# MEDIAPIPE INITIALIZATION (Legacy API)
# ============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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
        return tf.cast(tf.reduce_sum(output, axis=1), dtype=tf.float32)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
# Define the action space (two-handed ASL signs)
actions = np.array([
    "HELLO", "THANK_YOU", "YES", "NO", "PLEASE",
    "SORRY", "HELP", "MORE", "FINISHED", "I_LOVE_YOU",
    "GOOD", "BAD", "WANT", "NEED", "LIKE",
    "EAT", "DRINK", "SLEEP", "WORK", "PLAY",
    "HAPPY", "SAD", "ANGRY", "SCARED", "EXCITED",
    "BATHROOM"
])

sequence_length = 30
feature_dim = 126  # 2 hands × 21 landmarks × 3 coords
threshold = 0.85
consistency_len = 12

# ============================================================================
# MODEL LOADING (Commented out until model is trained)
# ============================================================================
# Uncomment after training:
# with open("model_twohands.json", "r") as f:
#     model = model_from_json(f.read(), custom_objects={'AttentionLayer': AttentionLayer})
# model.load_weights("best_model_twohands.h5")
model = None
print("⚠️  No trained model loaded. Running in detection-only mode.")

# ============================================================================
# COLOR SCHEME
# ============================================================================
COLOR_BG = (5, 10, 30)
COLOR_PRIMARY = (0, 150, 255)
COLOR_SECONDARY = (0, 90, 180)
COLOR_SUCCESS = (0, 255, 200)
COLOR_ACCENT = (0, 200, 255)
COLOR_TEXT = (230, 240, 255)
COLOR_DIM = (100, 130, 170)
COLOR_ROI = (0, 180, 255)
COLOR_LEFT_HAND = (255, 100, 100)
COLOR_RIGHT_HAND = (100, 255, 100)

# ============================================================================
# VIDEO CAPTURE INITIALIZATION
# ============================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open camera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize MediaPipe Hands with 2 hands (Legacy API)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# State variables
sequence = []
predictions = []
current_sign = ""
current_confidence = 0.0
confidence_history = deque(maxlen=50)
fps_history = deque(maxlen=30)
recognized_sentence = ""
last_sign = ""
last_recognition_time = 0


def extract_two_hand_keypoints(results, frame_w, frame_h):
    """Extract and order keypoints for two hands."""
    keypoints = np.zeros(126)
    
    if not results.multi_hand_landmarks:
        return keypoints, None, None, False, False
    
    hands_data = []
    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        pts = []
        for lm in hand_landmarks.landmark:
            pts.append([lm.x, lm.y, lm.z])
        hand_array = np.array(pts).flatten()
        wrist_x = hand_landmarks.landmark[0].x
        
        # Get handedness
        handedness = "Unknown"
        if results.multi_handedness and i < len(results.multi_handedness):
            handedness = results.multi_handedness[i].classification[0].label
        
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
    
    # Fallback: order by position if handedness not detected
    if left_hand is None and right_hand is None and len(hands_data) > 0:
        sorted_hands = sorted(hands_data, key=lambda h: h['wrist_x'], reverse=True)
        left_hand = sorted_hands[0]
        if len(sorted_hands) > 1:
            right_hand = sorted_hands[1]
    
    if left_hand:
        keypoints[:63] = left_hand['array']
    if right_hand:
        keypoints[63:126] = right_hand['array']
    
    return keypoints, left_hand, right_hand, left_hand is not None, right_hand is not None


def draw_hud_element(img, x, y, w, h, color, alpha=0.15):
    """Draw a HUD panel."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
    
    corner_len = 15
    corner_thick = 3
    cv2.line(img, (x, y), (x + corner_len, y), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x, y), (x, y + corner_len), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x + w, y), (x + w - corner_len, y), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x + w, y), (x + w, y + corner_len), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x, y + h), (x + corner_len, y + h), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x, y + h), (x, y + h - corner_len), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, corner_thick, cv2.LINE_AA)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, corner_thick, cv2.LINE_AA)


def draw_confidence_bar(img, x, y, w, h, confidence):
    """Draw confidence meter."""
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_DIM, -1)
    fill_w = int(w * confidence)
    bar_color = COLOR_SUCCESS if confidence >= threshold else (COLOR_ACCENT if confidence >= 0.6 else (50, 100, 255))
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), bar_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_PRIMARY, 2, cv2.LINE_AA)
    cv2.putText(img, f"{int(confidence * 100)}%", (x + w + 15, y + h - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)


def draw_line_graph(img, x, y, w, h, data, color, label=""):
    """Draw confidence history graph."""
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
    if label:
        cv2.putText(img, label, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)


def draw_glowing_text(img, text, pos, font_scale, color, thickness, glow_color=None):
    """Draw text with glow effect."""
    x, y = pos
    if glow_color:
        for offset in range(3, 0, -1):
            alpha = 0.2 - (offset * 0.05)
            glow_intensity = tuple(int(c * alpha) for c in glow_color)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, glow_intensity, thickness + offset, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_hand_landmarks_custom(img, landmarks, color, x_offset=0, y_offset=0, scale_w=1, scale_h=1):
    """Draw hand landmarks with custom color."""
    if landmarks is None:
        return
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    
    for connection in connections:
        start = landmarks.landmark[connection[0]]
        end = landmarks.landmark[connection[1]]
        start_point = (int(start.x * scale_w) + x_offset, int(start.y * scale_h) + y_offset)
        end_point = (int(end.x * scale_w) + x_offset, int(end.y * scale_h) + y_offset)
        cv2.line(img, start_point, end_point, color, 2, cv2.LINE_AA)
    
    for lm in landmarks.landmark:
        x_px = int(lm.x * scale_w) + x_offset
        y_px = int(lm.y * scale_h) + y_offset
        cv2.circle(img, (x_px, y_px), 5, color, -1, cv2.LINE_AA)


# ============================================================================
# WINDOW SETUP
# ============================================================================
cv2.namedWindow("TWO-HANDED ASL RECOGNITION", cv2.WINDOW_NORMAL)
cv2.resizeWindow("TWO-HANDED ASL RECOGNITION", 1280, 720)

print("🚀 Two-Handed Sign Language Recognition System Initialized")
print("🔹 Press 'Q' to quit | 'SPACE' to add | 'BACKSPACE' to delete | 'ENTER' to clear")
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
    frame = cv2.flip(frame, 1)

    current_time = time.time()
    fps = 1 / (current_time - frame_time) if current_time - frame_time > 0 else 0
    frame_time = current_time
    fps_history.append(fps)

    # Dark overlay
    hud = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)
    cv2.addWeighted(frame, 0.6, hud, 0.4, 0, frame)

    # Larger ROI for two hands
    roi_w, roi_h = int(w * 0.60), int(h * 0.70)
    x1, y1 = (w - roi_w) // 2, (h - roi_h) // 2
    x2, y2 = x1 + roi_w, y1 + roi_h

    roi = frame[y1:y2, x1:x2]
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_roi)

    # Extract keypoints
    keypoints, left_hand, right_hand, left_detected, right_detected = extract_two_hand_keypoints(results, roi_w, roi_h)
    hands_detected = left_detected or right_detected
    
    # Normalize
    keypoints = KeypointNormalizer.normalize_two_hands(keypoints)

    # Draw hands with distinct colors
    if left_hand:
        draw_hand_landmarks_custom(frame, left_hand['landmarks'], COLOR_LEFT_HAND, x1, y1, roi_w, roi_h)
    if right_hand:
        draw_hand_landmarks_custom(frame, right_hand['landmarks'], COLOR_RIGHT_HAND, x1, y1, roi_w, roi_h)

    sequence.append(keypoints)
    sequence = sequence[-sequence_length:]

    # Prediction (if model loaded)
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
        confidence_history.append(0.5 if hands_detected else 0.0)

    # Draw ROI boundary
    if left_detected and right_detected:
        roi_color = COLOR_SUCCESS
    elif hands_detected:
        roi_color = COLOR_ACCENT
    else:
        roi_color = COLOR_ROI

    corner_len, corner_thick = 40, 3
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), roi_color, corner_thick, cv2.LINE_AA)

    # TOP BAR
    draw_hud_element(frame, 20, 20, w - 40, 80, COLOR_PRIMARY, 0.15)
    draw_glowing_text(frame, "TWO-HANDED ASL RECOGNITION", (40, 55), 1.0, COLOR_PRIMARY, 3, COLOR_PRIMARY)
    
    # Hand status
    cv2.putText(frame, "L:●" if left_detected else "L:○", (w - 280, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LEFT_HAND if left_detected else COLOR_DIM, 2, cv2.LINE_AA)
    cv2.putText(frame, "R:●" if right_detected else "R:○", (w - 200, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RIGHT_HAND if right_detected else COLOR_DIM, 2, cv2.LINE_AA)
    
    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
    cv2.putText(frame, f"FPS: {int(avg_fps)}", (w - 280, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)

    # LEFT PANEL - Current sign
    panel_x, panel_y = 20, 120
    panel_w, panel_h = 350, 300
    draw_hud_element(frame, panel_x, panel_y, panel_w, panel_h, COLOR_SECONDARY, 0.12)
    cv2.putText(frame, "DETECTED SIGN", (panel_x + 20, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DIM, 2, cv2.LINE_AA)

    if current_sign and (current_time - last_recognition_time < 2.0):
        font_scale = 2.5 if len(current_sign) <= 6 else 1.5
        draw_glowing_text(frame, current_sign, (panel_x + 30, panel_y + 160), font_scale, COLOR_PRIMARY, 4, COLOR_PRIMARY)
    else:
        draw_glowing_text(frame, "---", (panel_x + 100, panel_y + 160), 3.0, COLOR_DIM, 4, None)

    cv2.putText(frame, "CONFIDENCE", (panel_x + 20, panel_y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DIM, 2, cv2.LINE_AA)
    draw_confidence_bar(frame, panel_x + 20, panel_y + 230, 310, 30, current_confidence)

    # RIGHT PANEL - Graph
    graph_x = w - 370
    draw_line_graph(frame, graph_x, 120, 350, 150, list(confidence_history), COLOR_PRIMARY, "CONFIDENCE HISTORY")

    # RIGHT PANEL - Sentence
    word_y = 290
    draw_hud_element(frame, graph_x, word_y, 350, 150, COLOR_SUCCESS, 0.12)
    cv2.putText(frame, "SENTENCE BUILDER", (graph_x + 20, word_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DIM, 2, cv2.LINE_AA)
    display = recognized_sentence[-22:] if len(recognized_sentence) > 22 else recognized_sentence
    cv2.putText(frame, display if display else "...", (graph_x + 20, word_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_SUCCESS, 2, cv2.LINE_AA)
    cv2.putText(frame, "[SPACE] Add | [BACK] Delete | [ENTER] Clear", (graph_x + 20, word_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DIM, 1, cv2.LINE_AA)

    # BOTTOM BAR
    bottom_y = h - 60
    draw_hud_element(frame, 20, bottom_y, w - 40, 40, COLOR_ACCENT, 0.12)
    cv2.putText(frame, "Place BOTH hands in the detection zone", (40, bottom_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, "[Q] QUIT", (w - 150, bottom_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2, cv2.LINE_AA)

    cv2.imshow("TWO-HANDED ASL RECOGNITION", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord(' '):
        if current_sign and current_sign != last_sign:
            recognized_sentence += (" " + current_sign) if recognized_sentence else current_sign
            last_sign = current_sign
    elif key == 8:
        words = recognized_sentence.split()
        if words:
            recognized_sentence = " ".join(words[:-1])
    elif key == 13:
        recognized_sentence = ""
        last_sign = ""

cap.release()
hands.close()
cv2.destroyAllWindows()
print("✅ System shutdown complete")
