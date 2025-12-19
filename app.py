"""
SIGN LANGUAGE RECOGNITION SYSTEM - MAIN APPLICATION
====================================================
Real-time sign language recognition using MediaPipe hand tracking and LSTM neural network.
Displays a cyberpunk-themed HUD with confidence metrics and word building features.

Features:
- Real-time hand detection and tracking
- 26 ASL letter recognition (A-Z)
- Confidence monitoring with visual graphs
- Word builder with keyboard controls
- Modern cyberpunk UI design
"""
    
import cv2
import numpy as np
import json
from collections import deque
import time

# Import tf_keras first
from tf_keras.models import model_from_json  # type: ignore

# Import only the specific mediapipe components we need
import mediapipe.python.solutions.hands as mp_hands_module
import mediapipe.python.solutions.drawing_utils as mp_drawing_module

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================
# Initialize MediaPipe Hands solution for hand landmark detection
mp_hands = mp_hands_module
mp_drawing = mp_drawing_module

# ============================================================================
# MODEL LOADING
# ============================================================================
# Load pre-trained LSTM model architecture from JSON
with open("model(0.2).json", "r") as f:
    model = model_from_json(f.read())

# Load trained weights into the model
model.load_weights("newmodel(0.2).h5")

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
# Define the action space (26 letters A-Z)
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Number of consecutive frames needed for a prediction sequence
sequence_length = 30

# Minimum confidence threshold for accepting predictions (0-1 scale)
threshold = 0.9

# Number of consistent predictions required before accepting a letter
consistency_len = 12

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

# Configure MediaPipe Hands with optimized parameters
hands = mp_hands.Hands(
    static_image_mode=False,        # Process video stream (not static images)
    max_num_hands=1,                # Detect only one hand for better performance
    min_detection_confidence=0.6,   # Minimum confidence for initial detection
    min_tracking_confidence=0.6     # Minimum confidence for tracking across frames
)

# ============================================================================
# STATE VARIABLES
# ============================================================================
sequence = []                       # Rolling buffer of hand keypoints for sequence prediction
predictions = []                    # Recent prediction indices for consistency checking
current_letter = ""                 # Currently detected letter
current_confidence = 0.0            # Confidence score of current prediction
confidence_history = deque(maxlen=50)  # Historical confidence scores for graph visualization
fps_history = deque(maxlen=30)     # Frame rate history for averaging
recognized_word = ""                # Built-up word from recognized letters
last_letter = ""                    # Last letter added to prevent duplicates
last_recognition_time = 0           # Timestamp of last successful recognition


# ============================================================================
# UI DRAWING FUNCTIONS
# ============================================================================

def draw_gradient_rect(img, pt1, pt2, color1, color2, alpha=0.3):
    """
    Draw a smooth vertical gradient rectangle overlay.
    
    Args:
        img: Base image to draw on
        pt1: Top-left corner (x, y)
        pt2: Bottom-right corner (x, y)
        color1: Starting color (BGR tuple)
        color2: Ending color (BGR tuple)
        alpha: Transparency level (0-1)
    """
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw gradient line by line
    for i in range(y1, y2):
        ratio = (i - y1) / (y2 - y1)
        # Interpolate between color1 and color2
        color = tuple([int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2)])
        cv2.line(overlay, (x1, i), (x2, i), color, 1)

    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hud_element(img, x, y, w, h, color, alpha=0.15):
    """
    Draw a HUD panel with semi-transparent background and corner accents.
    Creates the cyberpunk aesthetic with glowing corners.
    
    Args:
        img: Base image
        x, y: Top-left corner position
        w, h: Width and height of panel
        color: Panel color (BGR)
        alpha: Background transparency
    """
    # Draw semi-transparent filled rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw panel border
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

    # Draw corner accents for cyberpunk style
    corner_len = 15      # Length of corner lines
    corner_thick = 3     # Thickness of corner lines

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


def draw_text_with_background(img, text, pos, font_scale, color, thickness, bg_color, padding=10):
    """
    Draw text with a semi-transparent background panel for better readability.
    
    Args:
        img: Base image
        text: Text string to display
        pos: Position (x, y) for text
        font_scale: Size of text
        color: Text color (BGR)
        thickness: Text line thickness
        bg_color: Background color (BGR)
        padding: Padding around text in pixels
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = pos

    # Draw semi-transparent background rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, 
                  (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + baseline + padding), 
                  bg_color, -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    # Draw text on top
    cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)


def draw_confidence_bar(img, x, y, w, h, confidence):
    """
    Draw an animated confidence meter with color-coded fill.
    Green: High confidence (>= threshold)
    Yellow: Medium confidence (>= 0.6)
    Red: Low confidence (< 0.6)
    
    Args:
        img: Base image
        x, y: Top-left position
        w, h: Bar dimensions
        confidence: Confidence value (0-1)
    """
    # Draw background (empty bar)
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_DIM, -1)

    # Calculate fill width based on confidence
    fill_w = int(w * confidence)

    # Determine fill color based on confidence level
    if confidence >= threshold:
        bar_color = COLOR_SUCCESS      # Green - good prediction
    elif confidence >= 0.6:
        bar_color = COLOR_ACCENT        # Yellow - medium prediction
    else:
        bar_color = (50, 100, 255)     # Red - low confidence

    # Draw filled portion
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), bar_color, -1)

    # Draw border around entire bar
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_PRIMARY, 2, cv2.LINE_AA)

    # Display percentage text next to bar
    text = f"{int(confidence * 100)}%"
    cv2.putText(img, text, (x + w + 15, y + h - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)


def draw_line_graph(img, x, y, w, h, data, color, label=""):
    """
    Draw a real-time line graph for visualizing confidence history.
    
    Args:
        img: Base image
        x, y: Top-left corner of graph
        w, h: Graph dimensions
        data: List/deque of data points to plot
        color: Line color (BGR)
        label: Optional label for graph
    """
    if len(data) < 2:
        return  # Need at least 2 points to draw a line

    # Draw semi-transparent background panel
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Draw border
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

    # Convert data to graph coordinates
    points = []
    max_val = max(data) if max(data) > 0 else 1  # Avoid division by zero
    
    for i, val in enumerate(data):
        # Calculate x position (spread evenly across width)
        px = x + int((i / len(data)) * w)
        # Calculate y position (inverted because y=0 is at top)
        py = y + h - int((val / max_val) * h)
        points.append((px, py))

    # Draw lines connecting points
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, 2, cv2.LINE_AA)
        # Draw small circles at data points
        cv2.circle(img, points[i], 2, color, -1, cv2.LINE_AA)

    # Draw label if provided
    if label:
        cv2.putText(img, label, (x + 10, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)


def draw_glowing_text(img, text, pos, font_scale, color, thickness, glow_color=None):
    """
    Draw text with a glowing effect by drawing multiple layers with decreasing opacity.
    
    Args:
        img: Base image
        text: Text to display
        pos: Position (x, y)
        font_scale: Text size
        color: Main text color
        thickness: Text line thickness
        glow_color: Optional glow color (uses main color if None)
    """
    x, y = pos

    if glow_color:
        # Draw glow layers (largest to smallest)
        for offset in range(3, 0, -1):
            alpha = 0.2 - (offset * 0.05)  # Decreasing opacity for outer layers
            glow_intensity = tuple(int(c * alpha) for c in glow_color)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, glow_intensity, thickness + offset, cv2.LINE_AA)

    # Draw main text on top
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness, cv2.LINE_AA)


# ============================================================================
# WINDOW SETUP
# ============================================================================
# Create resizable window for application (removed fullscreen to avoid macOS issues)
cv2.namedWindow("SIGN LANGUAGE RECOGNITION SYSTEM", cv2.WINDOW_NORMAL)
# Maximize window instead of fullscreen
cv2.resizeWindow("SIGN LANGUAGE RECOGNITION SYSTEM", 1280, 720)

# ============================================================================
# STARTUP MESSAGE
# ============================================================================
print("🚀 Advanced Sign Language Recognition System Initialized")
print("🔹 Press 'Q' to quit | 'SPACE' to add letter to word | 'BACKSPACE' to delete | 'ENTER' to clear word")
print("🎨 Color Scheme: CYBERPUNK NEON (Aesthetic Mode)")

# Initialize frame timing for FPS calculation
frame_time = time.time()

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break  # Exit if camera fails

    # ========================================================================
    # FPS CALCULATION
    # ========================================================================
    current_time = time.time()
    fps = 1 / (current_time - frame_time) if current_time - frame_time > 0 else 0
    frame_time = current_time
    fps_history.append(fps)

    h, w, _ = frame.shape  # Get frame dimensions

    # ========================================================================
    # CREATE DARK OVERLAY (CYBERPUNK AESTHETIC)
    # ========================================================================
    # Blend dark background with video feed for better UI visibility
    hud = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)
    cv2.addWeighted(frame, 0.6, hud, 0.4, 0, frame)

    # ========================================================================
    # DEFINE REGION OF INTEREST (ROI) FOR HAND DETECTION
    # ========================================================================
    # Create centered box where user should place their hand
    roi_w, roi_h = int(w * 0.35), int(h * 0.5)  # 35% width, 50% height
    x1, y1 = (w - roi_w) // 2, (h - roi_h) // 2  # Center the box
    x2, y2 = x1 + roi_w, y1 + roi_h

    # Extract ROI from frame for processing
    roi = frame[y1:y2, x1:x2]

    # ========================================================================
    # HAND DETECTION AND KEYPOINT EXTRACTION
    # ========================================================================
    # Convert BGR to RGB (MediaPipe requirement)
    image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Initialize keypoints array (21 landmarks × 3 coordinates = 63 values)
    keypoints = np.zeros(63)
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
        
        # Extract 3D coordinates (x, y, z) for all 21 landmarks
        pts = []
        for lm in hand_landmarks.landmark:
            pts.append([lm.x, lm.y, lm.z])
        keypoints = np.array(pts).flatten()  # Flatten to 1D array (63 values)

        # Draw hand landmarks on ROI with custom colors
        mp_drawing.draw_landmarks(
            roi, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_PRIMARY, thickness=2, circle_radius=3),  # Landmarks
            mp_drawing.DrawingSpec(color=COLOR_SECONDARY, thickness=2, circle_radius=2)  # Connections
        )

    # ========================================================================
    # SEQUENCE MANAGEMENT AND PREDICTION
    # ========================================================================
    # Add current keypoints to sequence buffer
    sequence.append(keypoints)
    sequence = sequence[-sequence_length:]  # Keep only last 30 frames

    # Make prediction when we have a full sequence
    if len(sequence) == sequence_length:
        # Prepare input for model (add batch dimension)
        input_data = np.expand_dims(np.array(sequence), axis=0)
        
        # Get model predictions (probability for each letter)
        res = model.predict(input_data, verbose=0)[0]
        
        # Get predicted letter index
        pred_index = int(np.argmax(res))
        predictions.append(pred_index)
        predictions = predictions[-consistency_len:]  # Keep last 12 predictions

        # Get confidence score
        confidence = float(np.max(res))
        confidence_history.append(confidence)

        # Accept prediction only if consistent and above threshold
        if (len(predictions) == consistency_len and 
            len(np.unique(predictions)) == 1 and  # All predictions are the same
            confidence >= threshold):
            current_letter = actions[pred_index]
            current_confidence = confidence
            last_recognition_time = current_time

    # ========================================================================
    # DRAW ROI BOUNDARY WITH ANIMATED CORNERS
    # ========================================================================
    # Change color based on hand detection status
    roi_color = COLOR_SUCCESS if hand_detected else COLOR_ROI
    thickness = 3 if hand_detected else 2

    # Draw corner brackets around ROI (cyberpunk style)
    corner_len = 40
    corner_thick = 3

    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), roi_color, corner_thick, cv2.LINE_AA)

    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), roi_color, corner_thick, cv2.LINE_AA)

    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), roi_color, corner_thick, cv2.LINE_AA)

    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), roi_color, corner_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), roi_color, corner_thick, cv2.LINE_AA)

    # ========================================================================
    # TOP BAR - SYSTEM INFO
    # ========================================================================
    bar_h = 80
    draw_hud_element(frame, 20, 20, w - 40, bar_h, COLOR_PRIMARY, 0.15)

    # Title with glowing effect
    draw_glowing_text(frame, "SIGN LANGUAGE RECOGNITION", (40, 55), 
                      1.2, COLOR_PRIMARY, 3, COLOR_PRIMARY)

    # Status indicator (Active/Idle)
    status = "● ACTIVE" if hand_detected else "● IDLE"
    status_color = COLOR_SUCCESS if hand_detected else COLOR_DIM
    cv2.putText(frame, status, (w - 280, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

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

    # Panel title
    cv2.putText(frame, "DETECTED SIGN", (panel_x + 20, panel_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DIM, 2, cv2.LINE_AA)

    # Large letter display with glow (shows detected letter for 2 seconds)
    if current_letter and (current_time - last_recognition_time < 2.0):
        draw_glowing_text(frame, current_letter, 
                         (panel_x + 100, panel_y + 170), 5.0, 
                         COLOR_PRIMARY, 8, COLOR_PRIMARY)
    else:
        # Show dash when no recent detection
        draw_glowing_text(frame, "-", (panel_x + 130, panel_y + 170), 
                         4.0, COLOR_DIM, 6, None)

    # Confidence label and bar
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
    # RIGHT PANEL - WORD BUILDER
    # ========================================================================
    word_y = graph_y + graph_h + 20
    word_h = 150
    draw_hud_element(frame, graph_x, word_y, graph_w, word_h, COLOR_SUCCESS, 0.12)

    # Panel title
    cv2.putText(frame, "WORD BUILDER", (graph_x + 20, word_y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DIM, 2, cv2.LINE_AA)

    # Display built word
    display_word = recognized_word if recognized_word else "..."
    cv2.putText(frame, display_word, (graph_x + 20, word_y + 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, COLOR_SUCCESS, 3, cv2.LINE_AA)

    # Control instructions
    cv2.putText(frame, "[SPACE] Add | [BACK] Delete | [ENTER] Clear", 
                (graph_x + 20, word_y + 135), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DIM, 1, cv2.LINE_AA)

    # ========================================================================
    # BOTTOM INSTRUCTION BAR
    # ========================================================================
    bottom_y = h - 60
    draw_hud_element(frame, 20, bottom_y, w - 40, 40, COLOR_ACCENT, 0.12)
    
    # Main instruction
    cv2.putText(frame, "Place your hand in the detection zone", (40, bottom_y + 27), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    
    # Quit instruction
    cv2.putText(frame, "[Q] QUIT", (w - 150, bottom_y + 27), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2, cv2.LINE_AA)

    # ========================================================================
    # DISPLAY FRAME
    # ========================================================================
    cv2.imshow("SIGN LANGUAGE RECOGNITION SYSTEM", frame)

    # ========================================================================
    # KEYBOARD INPUT HANDLING
    # ========================================================================
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        # Quit application
        break
    elif key == ord(' '):
        # Space: Add current letter to word (prevent duplicates)
        if current_letter and current_letter != last_letter:
            recognized_word += current_letter
            last_letter = current_letter
    elif key == 8:
        # Backspace: Delete last letter
        recognized_word = recognized_word[:-1]
    elif key == 13:
        # Enter: Clear entire word
        recognized_word = ""
        last_letter = ""

# ============================================================================
# CLEANUP
# ============================================================================
# Release camera and close all windows
cap.release()
hands.close()
cv2.destroyAllWindows()
print("✅ System shutdown complete")