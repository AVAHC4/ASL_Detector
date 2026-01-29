import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil

try:
    from natsort import natsorted
except ImportError:
    def natsorted(paths, key=None):
        return sorted(paths, key=key)

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
SESSION_SIZE = 300
TRAIN_RATIO = 0.8
SEQUENCE_LENGTH = 30
STEP = 10

images_root = Path("Image")
train_output = Path("MP_Data_Train")
val_output = Path("MP_Data_Val")

actions = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

def detect_sessions(image_paths):
    if not image_paths:
        return []
    
    indices = []
    for p in image_paths:
        try:
            indices.append(int(p.stem))
        except ValueError:
            continue
    
    if not indices:
        return []
    
    indices.sort()
    min_idx = min(indices)
    max_idx = max(indices)
    
    sessions = []
    session_start = min_idx
    
    while session_start <= max_idx:
        session_end = min(session_start + SESSION_SIZE - 1, max_idx)
        sessions.append((session_start, session_end))
        session_start = session_end + 1
    
    return sessions

def split_sessions(sessions, train_ratio=0.8):
    if len(sessions) <= 1:
        split_point = int(len(sessions[0]) * train_ratio) if sessions else 0
        return sessions, []
    
    np.random.seed(42)
    np.random.shuffle(sessions)
    
    split_idx = max(1, int(len(sessions) * train_ratio))
    
    train_sessions = sessions[:split_idx]
    val_sessions = sessions[split_idx:]
    
    return train_sessions, val_sessions

def list_image_paths(folder):
    exts = (".png", ".jpg", ".jpeg")
    paths = [p for p in Path(folder).glob("*") if p.suffix.lower() in exts]
    return natsorted(paths, key=lambda p: p.stem)

def extract_keypoints(image_bgr, hands):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    hand = results.multi_hand_landmarks[0]
    pts = []
    for lm in hand.landmark:
        pts.append([lm.x, lm.y, lm.z])
    
    return np.array(pts).flatten()

def get_paths_in_session(all_paths, session_range):
    start, end = session_range
    return [p for p in all_paths if start <= int(p.stem) <= end]

def build_sequences_for_paths(paths, hands):
    sequences = []
    total = len(paths)
    
    if total < SEQUENCE_LENGTH:
        return sequences
    
    for start in range(0, total - SEQUENCE_LENGTH + 1, STEP):
        seq_paths = paths[start:start + SEQUENCE_LENGTH]
        
        frames = []
        valid = True
        
        for p in seq_paths:
            img = cv2.imread(str(p))
            if img is None:
                valid = False
                break
            
            kps = extract_keypoints(img, hands)
            if kps is None:
                valid = False
                break
            
            frames.append(kps)
        
        if valid and len(frames) == SEQUENCE_LENGTH:
            sequences.append(np.array(frames))
    
    return sequences

def save_sequences(sequences, output_dir, action, start_idx=0):
    action_dir = output_dir / action
    action_dir.mkdir(parents=True, exist_ok=True)
    
    for i, seq in enumerate(sequences):
        seq_dir = action_dir / str(start_idx + i)
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        for f_idx in range(SEQUENCE_LENGTH):
            np.save(seq_dir / f"{f_idx}.npy", seq[f_idx])
    
    return len(sequences)

def main():
    print("=" * 60)
    print("SESSION-BASED SIGNER-DISJOINT DATA SPLIT")
    print("=" * 60)
    
    for output_dir in [train_output, val_output]:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    train_total = 0
    val_total = 0
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6) as hands:
        for action in actions:
            in_dir = images_root / action
            
            if not in_dir.exists():
                print(f"⚠️  {action}: Directory not found, skipping")
                continue
            
            all_paths = list_image_paths(in_dir)
            print(f"\n📁 {action}: {len(all_paths)} images found")
            
            sessions = detect_sessions(all_paths)
            print(f"   └─ Detected {len(sessions)} sessions: {sessions}")
            
            train_sessions, val_sessions = split_sessions(sessions.copy())
            print(f"   └─ Train: {len(train_sessions)} sessions, Val: {len(val_sessions)} sessions")
            
            train_seqs = []
            for session in train_sessions:
                session_paths = get_paths_in_session(all_paths, session)
                seqs = build_sequences_for_paths(session_paths, hands)
                train_seqs.extend(seqs)
            
            val_seqs = []
            for session in val_sessions:
                session_paths = get_paths_in_session(all_paths, session)
                seqs = build_sequences_for_paths(session_paths, hands)
                val_seqs.extend(seqs)
            
            train_count = save_sequences(train_seqs, train_output, action)
            val_count = save_sequences(val_seqs, val_output, action)
            
            train_total += train_count
            val_total += val_count
            
            print(f"   └─ Created: {train_count} train sequences, {val_count} val sequences")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total training sequences:   {train_total}")
    print(f"Total validation sequences: {val_total}")
    print(f"Train/Val ratio:            {train_total/(train_total+val_total)*100:.1f}% / {val_total/(train_total+val_total)*100:.1f}%")
    print(f"\nOutput directories:")
    print(f"  Train: {train_output.absolute()}")
    print(f"  Val:   {val_output.absolute()}")
    print("\n✅ Session-based split complete!")

if __name__ == "__main__":
    main()
