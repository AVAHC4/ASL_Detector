"""
SIMPLE PRECISION-RECALL CURVE GENERATOR
========================================
Generates PR curves by first saving predictions, then plotting them.
This avoids TensorFlow/Keras compatibility issues during plotting.
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

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Define all sign language letters (A-Z)
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
sequence_length = 30
data_root = Path("MP_Data")


def load_data():
    """Load preprocessed sequence data"""
    X, y_idx = [], []
    
    for idx, action in enumerate(actions):
        adir = data_root / action
        if not adir.exists():
            continue
        
        for seq in sorted(adir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
            frames = []
            ok = True
            
            for f in range(sequence_length):
                fp = seq / f"{f}.npy"
                if not fp.exists():
                    ok = False
                    break
                frames.append(np.load(fp))
            
            if ok and len(frames) == sequence_length:
                X.append(frames)
                y_idx.append(idx)
    
    return np.array(X), np.array(y_idx)


def generate_predictions(weights_path, X_val, save_path="predictions.npz"):
    """Generate and save predictions"""
    print("Loading TensorFlow and building model...")
    
    from tf_keras.models import Sequential
    from tf_keras.layers import LSTM, Dense, Dropout, Bidirectional
    
    model = Sequential([
        Bidirectional(LSTM(96, return_sequences=True, activation='relu'), input_shape=(30, 63)),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
        Dropout(0.3),
        Bidirectional(LSTM(96, return_sequences=False, activation='relu')),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(26, activation='softmax')
    ])
    
    print(f"Loading weights from {weights_path}...")
    model.load_weights(weights_path)
    
    print("Generating predictions...")
    y_pred_proba = model.predict(X_val, verbose=1, batch_size=32)
    
    print(f"Saving predictions to {save_path}...")
    np.savez_compressed(save_path, predictions=y_pred_proba)
    
    print("✅ Predictions saved!")
    return y_pred_proba


def plot_precision_recall_curves(y_true, y_pred_proba, save_dir="evaluation_results"):
    """Generate comprehensive precision-recall curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    n_classes = len(actions)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    print("\n" + "="*70)
    print("COMPUTING PRECISION-RECALL CURVES")
    print("="*70)
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
        print(f"Class {actions[i]}: AP = {average_precision[i]:.4f}")
    
    # Micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_pred_proba.ravel()
    )
    average_precision["micro"] = average_precision_score(y_true_bin, y_pred_proba, average="micro")
    
    # Macro-average
    all_precision = np.unique(np.concatenate([precision[i] for i in range(n_classes)]))
    mean_recall = np.zeros_like(all_precision)
    for i in range(n_classes):
        mean_recall += np.interp(all_precision, precision[i][::-1], recall[i][::-1])
    mean_recall /= n_classes
    
    precision["macro"] = all_precision
    recall["macro"] = mean_recall
    average_precision["macro"] = average_precision_score(y_true_bin, y_pred_proba, average="macro")
    
    print(f"\n✨ Micro-average AP: {average_precision['micro']:.4f}")
    print(f"✨ Macro-average AP: {average_precision['macro']:.4f}")
    
    # === PLOT 1: ALL CLASSES ===
    plt.figure(figsize=(14, 10))
    
    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-avg (AP={average_precision["micro"]:.3f})',
             color='deeppink', linestyle='--', linewidth=3)
    
    plt.plot(recall["macro"], precision["macro"],
             label=f'Macro-avg (AP={average_precision["macro"]:.3f})',
             color='navy', linestyle='--', linewidth=3)
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=1.5, alpha=0.7,
                label=f'{actions[i]} ({average_precision[i]:.2f})')
    
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall Curves - All Classes', fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "precision_recall_all_classes.png"), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/precision_recall_all_classes.png")
    plt.close()
    
    # === PLOT 2: A-M ===
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.plot(recall["micro"], precision["micro"],
            label=f'Micro-avg ({average_precision["micro"]:.3f})',
            color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.plot(recall["macro"], precision["macro"],
            label=f'Macro-avg ({average_precision["macro"]:.3f})',
            color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    colors_am = plt.cm.Set3(np.linspace(0, 1, 13))
    for i in range(13):
        ax.plot(recall[i], precision[i], color=colors_am[i], lw=3,
                label=f'{actions[i]} ({average_precision[i]:.2f})',
                marker='o', markersize=3, markevery=max(1, len(recall[i])//20))
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Curves (A-M)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "precision_recall_A_M.png"), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/precision_recall_A_M.png")
    plt.close()
    
    # === PLOT 3: N-Z ===
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.plot(recall["micro"], precision["micro"],
            label=f'Micro-avg ({average_precision["micro"]:.3f})',
            color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.plot(recall["macro"], precision["macro"],
            label=f'Macro-avg ({average_precision["macro"]:.3f})',
            color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    colors_nz = plt.cm.Set3(np.linspace(0, 1, 13))
    for i in range(13, n_classes):
        ax.plot(recall[i], precision[i], color=colors_nz[i-13], lw=3,
                label=f'{actions[i]} ({average_precision[i]:.2f})',
                marker='s', markersize=3, markevery=max(1, len(recall[i])//20))
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Curves (N-Z)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "precision_recall_N_Z.png"), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/precision_recall_N_Z.png")
    plt.close()
    
    # === PLOT 4: BAR CHART ===
    plt.figure(figsize=(16, 8))
    
    ap_scores = [average_precision[i] for i in range(n_classes)]
    colors_bar = ['#2ecc71' if ap >= 0.9 else '#f39c12' if ap >= 0.7 else '#e74c3c' for ap in ap_scores]
    
    bars = plt.bar(actions, ap_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
    
    plt.axhline(y=average_precision["micro"], color='deeppink',
                linestyle='--', linewidth=2, label=f'Micro-avg ({average_precision["micro"]:.3f})')
    plt.axhline(y=average_precision["macro"], color='navy',
                linestyle='--', linewidth=2, label=f'Macro-avg ({average_precision["macro"]:.3f})')
    
    for bar, ap in zip(bars, ap_scores):
        plt.text(bar.get_x() + bar.get_width()/2., ap + 0.01,
                f'{ap:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Sign Letter', fontsize=13, fontweight='bold')
    plt.ylabel('Average Precision (AP)', fontsize=13, fontweight='bold')
    plt.title('Average Precision Score per Class', fontsize=16, fontweight='bold', pad=20)
    plt.ylim([0, 1.1])
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "average_precision_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/average_precision_comparison.png")
    plt.close()
    
    print("\n" + "="*70)
    print("✅ PRECISION-RECALL ANALYSIS COMPLETE!")
    print("="*70)
    
    return {
        "average_precision_per_class": {actions[i]: average_precision[i] for i in range(n_classes)},
        "micro_average_precision": average_precision["micro"],
        "macro_average_precision": average_precision["macro"]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate precision-recall curves")
    parser.add_argument("--weights", default="newmodel(0.35).h5", help="Model weights")
    parser.add_argument("--output_dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--use_saved_preds", action="store_true", help="Use saved predictions")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    X, y_idx = load_data()
    
    if len(X) == 0:
        print("❌ ERROR: No data in MP_Data!")
        return
    
    print(f"✅ Loaded {len(X)} sequences")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_idx, test_size=args.test_size, stratify=y_idx, random_state=args.random_state
    )
    
    print(f"✅ Train: {len(X_train)} | Val: {len(X_val)}")
    
    pred_file = "val_predictions.npz"
    
    if args.use_saved_preds and os.path.exists(pred_file):
        print(f"\nLoading saved predictions from {pred_file}...")
        y_pred_proba = np.load(pred_file)['predictions']
        print("✅ Predictions loaded!")
    else:
        print("\n" + "="*70)
        print("GENERATING PREDICTIONS")
        print("="*70)
        y_pred_proba = generate_predictions(args.weights, X_val, pred_file)
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    results = plot_precision_recall_curves(y_val, y_pred_proba, args.output_dir)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Micro-average AP: {results['micro_average_precision']:.4f}")
    print(f"Macro-average AP: {results['macro_average_precision']:.4f}")
    
    ap_sorted = sorted(results['average_precision_per_class'].items(), key=lambda x: x[1], reverse=True)
    
    print("\n🏆 Top 5 Best:")
    for i, (cls, ap) in enumerate(ap_sorted[:5], 1):
        print(f"  {i}. {cls}: {ap:.4f}")
    
    print("\n⚠️  Top 5 Worst:")
    for i, (cls, ap) in enumerate(ap_sorted[-5:][::-1], 1):
        print(f"  {i}. {cls}: {ap:.4f}")
    
    print(f"\n💾 Results saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main()
