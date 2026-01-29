"""
SIGN LANGUAGE DATA COLLECTION TOOL
===================================
Interactive tool for collecting sign language gesture images for model training.
Automatically captures images from webcam for each letter A-Z.

Features:
- Full-screen capture interface
- Automatic image capture at 90ms intervals
- 300 images per letter session limit
- Easy navigation between letters
- Visual feedback during recording
- Organized folder structure (Image/A, Image/B, etc.)

Controls:
- / or -    : Previous letter
- + or =    : Next letter
- A-Z keys  : Jump to specific letter
- 0         : Start recording
- 1         : Stop recording
- Q         : Quit application
"""

import cv2
import os
import time
import string
from pathlib import Path

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================
ROI_WIDTH_RATIO = 0.50      # ROI width as fraction of frame width (50%)
ROI_HEIGHT_RATIO = 0.60     # ROI height as fraction of frame height (60%)
SAVE_EVERY_MS = 90          # Capture interval in milliseconds (11 fps)
SESSION_LIMIT = 300         # Maximum images per letter per session
CAM_INDEX = 0               # Camera device index (0 = default camera)
IMG_EXT = ".png"            # Image file extension

# ============================================================================
# DATA STRUCTURE INITIALIZATION
# ============================================================================
# Create list of all uppercase letters A-Z
letters = list(string.ascii_uppercase)

# Create root output directory and subdirectories for each letter
out_root = Path("Image")
for c in letters:
    (out_root / c).mkdir(parents=True, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def next_index_for(letter_dir: Path) -> int:
    """
    Determine the next available index number for saving images.
    Scans existing files and returns max + 1, or 0 if no files exist.
    
    Args:
        letter_dir: Path to the letter's directory
        
    Returns:
        Next available index number
        
    Example:
        If folder contains 0.png, 1.png, 2.png, returns 3
    """
    # Find all PNG files with numeric names
    existing = [int(p.stem) for p in letter_dir.glob(f"*{IMG_EXT}") 
                if p.stem.isdigit()]
    
    # Return next number after the maximum, or 0 if no files exist
    return (max(existing) + 1) if existing else 0


def put_text_center(img, text, y, scale=1.0, color=(255,255,255), thickness=2):
    """
    Draw text centered horizontally on the image.
    
    Args:
        img: Image array to draw on
        text: Text string to display
        y: Vertical position (pixels from top)
        scale: Font scale multiplier
        color: Text color in BGR format
        thickness: Text line thickness
    """
    # Calculate text dimensions
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    
    # Calculate x position to center text (with minimum margin)
    x = max(10, (img.shape[1] - w) // 2)
    
    # Draw text with anti-aliasing
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                scale, color, thickness, cv2.LINE_AA)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main data collection application loop.
    Handles video capture, user input, and image saving.
    """
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    # Open video capture device
    cap = cv2.VideoCapture(CAM_INDEX)
    
    # Create fullscreen window
    cv2.namedWindow("Auto Collector", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Auto Collector", cv2.WND_PROP_FULLSCREEN, 
                         cv2.WINDOW_FULLSCREEN)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # ========================================================================
    # STATE VARIABLES
    # ========================================================================
    current_idx = 0                     # Index in letters list
    current_letter = letters[current_idx]  # Current letter being collected
    recording = False                   # Recording state flag
    last_saved_ms = 0                   # Timestamp of last saved image
    
    # Initialize counters for each letter
    counters = {c: next_index_for(out_root / c) for c in letters}
    session_counts = {c: 0 for c in letters}  # Images captured this session

    # ========================================================================
    # PRINT CONTROLS TO CONSOLE
    # ========================================================================
    print("Controls:")
    print("  - / +         : move between letters")
    print("  A—Z           : jump directly to letter")
    print("  0             : start recording (max 300 per letter)")
    print("  1             : stop recording")
    print("  Q             : quit")

    # ========================================================================
    # MAIN CAPTURE LOOP
    # ========================================================================
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            break  # Exit if frame capture fails

        h, w, _ = frame.shape  # Get frame dimensions

        # ====================================================================
        # CALCULATE ROI COORDINATES (CENTERED)
        # ====================================================================
        roi_w = int(w * ROI_WIDTH_RATIO)
        roi_h = int(h * ROI_HEIGHT_RATIO)
        x1 = (w - roi_w) // 2  # Center horizontally
        y1 = (h - roi_h) // 2  # Center vertically
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        # ====================================================================
        # DRAW ROI RECTANGLE
        # ====================================================================
        # Draw rectangle around region of interest
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 3)

        # ====================================================================
        # DRAW HEADER BAR
        # ====================================================================
        # Black background for header
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        
        # Display current letter
        put_text_center(frame, f"Letter: {current_letter}", 40, scale=1.4)
        
        # Display recording status with color coding
        status = "RECORDING" if recording else "IDLE"
        color = (0, 255, 0) if recording else (0, 165, 255)  # Green/Orange
        put_text_center(frame, f"Status: {status}", 80, scale=1.0, color=color)

        # ====================================================================
        # DRAW FOOTER BAR
        # ====================================================================
        # Black background for footer
        cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 0), -1)
        
        # Display image counter
        counter_text = f"Saved: {session_counts[current_letter]} / {SESSION_LIMIT}"
        put_text_center(frame, counter_text, h - 60, scale=1.0, 
                       color=(255, 255, 255))

        # Display controls help
        help_text = "- / + : Move | 0: Start | 1: Stop | Q: Quit"
        put_text_center(frame, help_text, h - 25, scale=0.8, 
                       color=(255, 255, 255))

        # ====================================================================
        # AUTOMATIC IMAGE CAPTURE DURING RECORDING
        # ====================================================================
        now_ms = int(time.time() * 1000)  # Current time in milliseconds
        
        # Check if we should save a frame
        if recording and (now_ms - last_saved_ms >= SAVE_EVERY_MS):
            # Check if we haven't exceeded session limit
            if session_counts[current_letter] < SESSION_LIMIT:
                # Extract ROI region from frame
                roi = frame[y1:y2, x1:x2]
                
                # Construct output path
                out_dir = out_root / current_letter
                idx = counters[current_letter]
                fp = out_dir / f"{idx}{IMG_EXT}"
                
                # Save image to disk
                cv2.imwrite(str(fp), roi)
                
                # Update counters
                counters[current_letter] += 1
                session_counts[current_letter] += 1
                last_saved_ms = now_ms
                
                # Draw visual feedback (red circle in corner)
                cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
            else:
                # Stop recording when limit reached
                recording = False
                print(f"✅ Session limit (300) reached for {current_letter}. "
                      f"Stopped recording.")

        # ====================================================================
        # DISPLAY FRAME
        # ====================================================================
        cv2.imshow("Auto Collector", frame)

        # ====================================================================
        # KEYBOARD INPUT HANDLING
        # ====================================================================
        key = cv2.waitKey(1) & 0xFFFF
        
        # Quit application
        if key in (ord('q'), ord('Q')):
            break

        # Start recording
        if key == ord('0'):
            if session_counts[current_letter] < SESSION_LIMIT:
                recording = True
                last_saved_ms = 0  # Reset timer to capture immediately
            else:
                print(f"⚠️ Limit reached for {current_letter}. "
                      f"Cannot record more.")
        
        # Stop recording
        elif key == ord('1'):
            recording = False

        # Jump to letter directly (A-Z or a-z key pressed)
        if ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
            ch = chr(key).upper()
            if ch in letters:
                current_idx = letters.index(ch)
                current_letter = ch
                recording = False  # Stop recording when switching letters

        # Move to previous letter (- or minus key)
        if key in (ord('-'), 45):
            current_idx = (current_idx - 1) % len(letters)  # Wrap around
            current_letter = letters[current_idx]
            recording = False

        # Move to next letter (+ or = key)
        elif key in (ord('+'), ord('='), 43, 61):
            current_idx = (current_idx + 1) % len(letters)  # Wrap around
            current_letter = letters[current_idx]
            recording = False

    # ========================================================================
    # CLEANUP
    # ========================================================================
    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()