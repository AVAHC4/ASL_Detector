import os
import numpy as np
from pathlib import Path
from data import KeypointNormalizer
from tqdm import tqdm

def process_directory(input_path, output_path):
    input_root = Path(input_path)
    output_root = Path(output_path)
    
    print("=" * 60)
    print(f"NORMALIZING: {input_root} -> {output_root}")
    print("=" * 60)
    
    if not input_root.exists():
        print(f"❌ Error: Input directory '{input_root}' not found.")
        return 0

    # Process each action (A, B, C...)
    actions = [d for d in input_root.iterdir() if d.is_dir()]
    actions.sort(key=lambda x: x.name)
    
    total_processed = 0
    total_files = 0
    
    for action_dir in tqdm(actions, desc=f"Processing {input_root.name}"):
        action = action_dir.name
        out_action_dir = output_root / action
        out_action_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each sequence folder (0, 1, 2...)
        sequences = [d for d in action_dir.iterdir() if d.is_dir()]
        
        for seq_dir in sequences:
            # Create corresponding output sequence folder
            out_seq_dir = out_action_dir / seq_dir.name
            out_seq_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each frame (.npy file)
            frames = list(seq_dir.glob("*.npy"))
            total_files += len(frames)
            
            for frame_path in frames:
                # Load raw keypoints
                try:
                    keypoints = np.load(frame_path)
                    
                    # Normalize
                    normalized_keypoints = KeypointNormalizer.normalize(keypoints)
                    
                    # Save normalized keypoints
                    out_path = out_seq_dir / frame_path.name
                    np.save(out_path, normalized_keypoints)
                    total_processed += 1
                    
                except Exception as e:
                    print(f"⚠️ Error processing {frame_path}: {e}")
    
    print(f"✅ Processed {total_processed} files in {input_root.name}")
    return total_processed

def normalize_dataset():
    # Normalize both Train and Val sets
    t = process_directory("MP_Data_Train", "MP_Data_Train_Normalized")
    v = process_directory("MP_Data_Val", "MP_Data_Val_Normalized")
    
    print("\n" + "=" * 60)
    print("NORMALIZATION COMPLETE")
    print("=" * 60)
    print(f"Total frames processed: {t + v}")

if __name__ == "__main__":
    normalize_dataset()
