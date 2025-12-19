"""
PRECISION-RECALL CURVE GENERATOR FOR ASL MODEL
===============================================
Generates precision-recall curves for multi-class sign language recognition model.
Creates both individual class curves and micro/macro-averaged curves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import argparse

# Import Keras from TensorFlow for compatibility
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

try:
    # Try TensorFlow 2.15+ style import
    from tensorflow.keras.models import model_from_json
    from tensorflow.keras.utils import to_categorical
except ImportError:
    # Fall back to tf_keras if available
    from tf_keras.models import model_from_json
    from tf_keras.utils import to_categorical

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Define all sign language letters (A-Z)
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Number of frames per sequence
sequence_length = 30

# Path to the dataset directory
data_root = Path("MP_Data")


def load_data():
    """Load preprocessed sequence data"""
    X, y_idx = [], []
    
    # Loop through each action/letter
    for idx, action in enumerate(actions):
        adir = data_root / action
        
        if not adir.exists():
            continue
        
        # Loop through each sequence folder
        for seq in sorted(adir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
            frames = []
            ok = True
            
            # Load all frames in the sequence
            for f in range(sequence_length):
                fp = seq / f"{f}.npy"
                
                if not fp.exists():
                    ok = False
                    break
                
                frames.append(np.load(fp))
            
            # Add sequence if all frames were loaded successfully
            if ok and len(frames) == sequence_length:
                X.append(frames)
                y_idx.append(idx)
    
    # Convert to numpy arrays
    X = np.array(X)
    # Convert to numpy arrays
    X = np.array(X)
    y_idx = np.array(y_idx)
    
    # Convert labels to one-hot encoded format
    y = to_categorical(y_idx, num_classes=len(actions)).astype(int)
    
    return X, y, y_idx
def load_trained_model(model_json_path, weights_path):
    """Load trained model architecture and weights"""
    with open(model_json_path, "r") as f:
        model = model_from_json(f.read())
    
    model.load_weights(weights_path)
    
    return model


def plot_precision_recall_curves(y_true, y_pred_proba, save_dir="evaluation_results"):
    """
    Generate comprehensive precision-recall curves for multi-class classification.
    Creates three visualizations:
    1. All classes together (overview)
    2. Classes A-M (detailed)
    3. Classes N-Z (detailed)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_classes = len(actions)
    
    # Binarize the labels for multi-class PR curve
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute precision-recall for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    print("\n" + "="*70)
    print("COMPUTING PRECISION-RECALL CURVES")
    print("="*70)
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_pred_proba[:, i]
        )
        average_precision[i] = average_precision_score(
            y_true_bin[:, i], y_pred_proba[:, i]
        )
        print(f"Class {actions[i]}: AP = {average_precision[i]:.4f}")
    
    # Compute micro-average precision-recall curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_pred_proba.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true_bin, y_pred_proba, average="micro"
    )
    
    # Compute macro-average precision-recall curve
    all_precision = np.unique(np.concatenate([precision[i] for i in range(n_classes)]))
    mean_recall = np.zeros_like(all_precision)
    for i in range(n_classes):
        mean_recall += np.interp(all_precision, precision[i][::-1], recall[i][::-1])
    mean_recall /= n_classes
    
    precision["macro"] = all_precision
    recall["macro"] = mean_recall
    average_precision["macro"] = average_precision_score(
        y_true_bin, y_pred_proba, average="macro"
    )
    
    print(f"\nMicro-average AP: {average_precision['micro']:.4f}")
    print(f"Macro-average AP: {average_precision['macro']:.4f}")
    
    # ========================================================================
    # PLOT 1: ALL CLASSES WITH MICRO/MACRO AVERAGE
    # ========================================================================
    plt.figure(figsize=(14, 10))
    
    # Plot micro-average curve (thick line)
    plt.plot(
        recall["micro"], precision["micro"],
        label=f'Micro-average (AP = {average_precision["micro"]:.2f})',
        color='deeppink', linestyle='--', linewidth=3
    )
    
    # Plot macro-average curve (thick line)
    plt.plot(
        recall["macro"], precision["macro"],
        label=f'Macro-average (AP = {average_precision["macro"]:.2f})',
        color='navy', linestyle='--', linewidth=3
    )
    
    # Use different colors for each class
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            recall[i], precision[i], color=color, lw=1.5, alpha=0.7,
            label=f'{actions[i]} (AP = {average_precision[i]:.2f})'
        )
    
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall Curves - All Classes', fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "precision_recall_all_classes.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {save_path}")
    plt.close()
    
    # ========================================================================
    # PLOT 2: CLASSES A-M (Detailed View)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot micro and macro averages as reference
    ax.plot(
        recall["micro"], precision["micro"],
        label=f'Micro-average (AP = {average_precision["micro"]:.2f})',
        color='gray', linestyle='--', linewidth=2, alpha=0.5
    )
    ax.plot(
        recall["macro"], precision["macro"],
        label=f'Macro-average (AP = {average_precision["macro"]:.2f})',
        color='black', linestyle='--', linewidth=2, alpha=0.5
    )
    
    # Plot first 13 classes (A-M)
    colors_am = plt.cm.Set3(np.linspace(0, 1, 13))
    
    for i in range(13):
        ax.plot(
            recall[i], precision[i], color=colors_am[i], lw=3,
            label=f'{actions[i]} (AP = {average_precision[i]:.2f})',
            marker='o', markersize=3, markevery=max(1, len(recall[i])//20)
        )
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Curves (A-M)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "precision_recall_A_M.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    # ========================================================================
    # PLOT 3: CLASSES N-Z (Detailed View)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot micro and macro averages as reference
    ax.plot(
        recall["micro"], precision["micro"],
        label=f'Micro-average (AP = {average_precision["micro"]:.2f})',
        color='gray', linestyle='--', linewidth=2, alpha=0.5
    )
    ax.plot(
        recall["macro"], precision["macro"],
        label=f'Macro-average (AP = {average_precision["macro"]:.2f})',
        color='black', linestyle='--', linewidth=2, alpha=0.5
    )
    
    # Plot last 13 classes (N-Z)
    colors_nz = plt.cm.Set3(np.linspace(0, 1, 13))
    
    for i in range(13, n_classes):
        ax.plot(
            recall[i], precision[i], color=colors_nz[i-13], lw=3,
            label=f'{actions[i]} (AP = {average_precision[i]:.2f})',
            marker='s', markersize=3, markevery=max(1, len(recall[i])//20)
        )
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Curves (N-Z)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "precision_recall_N_Z.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    # ========================================================================
    # PLOT 4: COMPARISON BAR CHART OF AVERAGE PRECISION SCORES
    # ========================================================================
    plt.figure(figsize=(16, 8))
    
    # Prepare data for bar chart
    class_names = list(actions)
    ap_scores = [average_precision[i] for i in range(n_classes)]
    
    # Create bar chart with color coding
    colors_bar = ['#2ecc71' if ap >= 0.9 else '#f39c12' if ap >= 0.7 else '#e74c3c' 
                  for ap in ap_scores]
    
    bars = plt.bar(class_names, ap_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
    
    # Add horizontal lines for micro and macro averages
    plt.axhline(y=average_precision["micro"], color='deeppink', 
                linestyle='--', linewidth=2, label=f'Micro-avg ({average_precision["micro"]:.3f})')
    plt.axhline(y=average_precision["macro"], color='navy', 
                linestyle='--', linewidth=2, label=f'Macro-avg ({average_precision["macro"]:.3f})')
    
    # Add value labels on bars
    for i, (bar, ap) in enumerate(zip(bars, ap_scores)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ap:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Sign Letter', fontsize=13, fontweight='bold')
    plt.ylabel('Average Precision (AP)', fontsize=13, fontweight='bold')
    plt.title('Average Precision Score per Class', fontsize=16, fontweight='bold', pad=20)
    plt.ylim([0, 1.1])
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "average_precision_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("PRECISION-RECALL ANALYSIS COMPLETE")
    print("="*70)
    
    # Return summary statistics
    return {
        "average_precision_per_class": {actions[i]: average_precision[i] for i in range(n_classes)},
        "micro_average_precision": average_precision["micro"],
        "macro_average_precision": average_precision["macro"]
    }


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Generate precision-recall curves for ASL model"
    )
    parser.add_argument(
        "--model_json", 
        default="model(0.35).json", 
        help="Path to model JSON file"
    )
    parser.add_argument(
        "--weights", 
        default="newmodel(0.35).h5", 
        help="Path to model weights file"
    )
    parser.add_argument(
        "--output_dir", 
        default="evaluation_results", 
        help="Directory to save plots"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2, 
        help="Validation split ratio"
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    X, y_onehot, y_idx = load_data()
    
    if len(X) == 0:
        print("❌ ERROR: No data found in MP_Data directory!")
        return
    
    print(f"✅ Loaded {len(X)} sequences across {len(actions)} classes")
    
    # Split data into training and validation sets (stratified)
    print("\nSplitting data (stratified)...")
    X_train, X_val, y_train_idx, y_val_idx = train_test_split(
        X, y_idx, 
        test_size=args.test_size, 
        stratify=y_idx, 
        random_state=args.random_state
    )
    
    print(f"✅ Train: {len(X_train)} | Validation: {len(X_val)}")
    
    # Load trained model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    print(f"Model architecture: {args.model_json}")
    print(f"Model weights: {args.weights}")
    
    model = load_trained_model(args.model_json, args.weights)
    print("✅ Model loaded successfully!")
    
    # Generate predictions for validation set
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    y_pred_proba = model.predict(X_val, verbose=1)
    print("✅ Predictions generated!")
    
    # Plot precision-recall curves
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    results = plot_precision_recall_curves(y_val_idx, y_pred_proba, args.output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Micro-average AP: {results['micro_average_precision']:.4f}")
    print(f"Macro-average AP: {results['macro_average_precision']:.4f}")
    
    # Find best and worst performing classes
    ap_per_class = results['average_precision_per_class']
    sorted_classes = sorted(ap_per_class.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Best Performing Classes:")
    for i, (cls, ap) in enumerate(sorted_classes[:5], 1):
        print(f"  {i}. {cls}: {ap:.4f}")
    
    print("\nTop 5 Worst Performing Classes:")
    for i, (cls, ap) in enumerate(sorted_classes[-5:][::-1], 1):
        print(f"  {i}. {cls}: {ap:.4f}")
    
    print("\n" + "="*70)
    print("✅ ALL PRECISION-RECALL CURVES GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
