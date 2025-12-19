import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from natsort import natsorted

# Initialize MediaPipe Hands for hand landmark detection
mp_hands = mp.solutions.hands

# Path to directory containing raw sign language images
images_root = Path("Image")

# Path to output directory for processed keypoint data
output_root = Path("MP_Data")
output_root.mkdir(parents=True, exist_ok=True)

# List of all letters A-Z to process
actions = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Number of frames required per sequence
sequence_length = 30

# Step size for sliding window (how many frames to skip between sequences)
STEP = 10

def list_image_paths(folder):
    """Get all image file paths from a folder, sorted naturally"""
    # Define valid image extensions
    exts = (".png", ".jpg", ".jpeg")
    
    # Get all files with valid image extensions
    paths = [p for p in Path(folder).glob("*") if p.suffix.lower() in exts]
    
    # Sort paths naturally (1, 2, 10 instead of 1, 10, 2)
    return natsorted(paths, key=lambda p: p.stem)

def extract_keypoints(image_bgr, hands):
    """Extract hand landmark keypoints from an image"""
    # Convert BGR image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Process image to detect hands
    results = hands.process(image_rgb)
    
    # Return None if no hand detected
    if not results.multi_hand_landmarks:
        return None
    
    # Get first detected hand
    hand = results.multi_hand_landmarks[0]
    
    # Extract x, y, z coordinates for all 21 landmarks
    pts = []
    for lm in hand.landmark:
        pts.append([lm.x, lm.y, lm.z])
    
    # Flatten to 1D array of 63 values
    return np.array(pts).flatten()

def build_sequences_for_action(paths, hands):
    """Create overlapping sequences of frames using sliding window"""
    sequences = []
    total = len(paths)
    
    # Need at least sequence_length images to create one sequence
    if total < sequence_length:
        return sequences
    
    # Sliding window: create sequences with STEP frame intervals
    for start in range(0, total - sequence_length + 1, STEP):
        # Get paths for this sequence
        seq_paths = paths[start:start + sequence_length]
        
        frames = []
        valid = True
        
        # Process each image in the sequence
        for p in seq_paths:
            # Load image
            img = cv2.imread(str(p))
            if img is None:
                valid = False
                break
            
            # Extract keypoints
            kps = extract_keypoints(img, hands)
            if kps is None:
                valid = False
                break
            
            # Add keypoints to frames list
            frames.append(kps)
        
        # Only add complete valid sequences
        if valid and len(frames) == sequence_length:
            sequences.append(np.array(frames))
    
    return sequences

def main():
    """Main function to process all images and create sequences"""
    # Create MediaPipe Hands instance
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6) as hands:
        
        # Process each letter
        for action in actions:
            # Get input directory for this letter
            in_dir = images_root / action
            
            # Skip if directory doesn't exist
            if not in_dir.exists():
                continue
            
            # Get all image paths for this letter
            paths = list_image_paths(in_dir)
            
            # Build sequences from images
            seqs = build_sequences_for_action(paths, hands)
            
            # Save each sequence
            for i, seq in enumerate(seqs):
                # Create output directory for this sequence
                out_dir = output_root / action / str(i)
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Save each frame's keypoints as separate .npy file
                for f_idx in range(sequence_length):
                    np.save(out_dir / f"{f_idx}.npy", seq[f_idx])

if _name_ == "_main_":
    main()