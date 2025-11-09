import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, auc,
    top_k_accuracy_score
)
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical

from itertools import cycle

# Define all sign language letters (A-Z)
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Number of frames per sequence
sequence_length = 30

# Path to the dataset directory
data_root = Path("MP_Data")

# Configure matplotlib plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load preprocessed sequence data"""
    # Initialize empty lists for features and labels
    X, y_idx = [], []
    
    # Loop through each action/letter
    for idx, action in enumerate(actions):
        # Get directory for this letter
        adir = data_root / action
        
        # Skip if directory doesn't exist
        if not adir.exists():
            continue
        
        # Loop through each sequence folder
        for seq in sorted(adir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
            frames = []
            ok = True
            
            # Load all frames in the sequence
            for f in range(sequence_length):
                fp = seq / f"{f}.npy"
                
                # Check if frame file exists
                if not fp.exists():
                    ok = False
                    break
                
                # Load frame data
                frames.append(np.load(fp))
            
            # Add sequence if all frames were loaded successfully
            if ok and len(frames) == sequence_length:
                X.append(frames)
                y_idx.append(idx)
    
    # Convert to numpy arrays
    X = np.array(X)
    y_idx = np.array(y_idx)
    
    # Convert labels to one-hot encoded format
    y = to_categorical(y_idx, num_classes=len(actions)).astype(int)
    
    return X, y, y_idx

def load_trained_model(model_json_path="model(0.35).json", weights_path="newmodel(0.35).h5"):
    """Load trained model architecture and weights"""
    # Load model architecture from JSON file
    with open(model_json_path, "r") as f:
        model = model_from_json(f.read())
    
    # Load trained weights
    model.load_weights(weights_path)
    
    return model

def plot_confusion_matrix(cm, save_path, normalize=False):
    """Plot enhanced confusion matrix heatmap"""
    # Normalize confusion matrix if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Draw heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='YlOrRd', 
                xticklabels=actions, yticklabels=actions,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                linewidths=0.5, linecolor='gray')
    
    # Set labels and title
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_metrics(report_dict, save_path):
    """Plot per-class precision, recall, F1-score"""
    # Extract classes and their metrics
    classes = [k for k in report_dict.keys() if k in actions]
    precision = [report_dict[c]['precision'] for c in classes]
    recall = [report_dict[c]['recall'] for c in classes]
    f1 = [report_dict[c]['f1-score'] for c in classes]
    
    # Set up bar positions
    x = np.arange(len(classes))
    width = 0.25
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    # Configure axes
    ax.set_xlabel('Sign Letter', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, save_path):
    """Plot ROC curves for all classes (One-vs-Rest)"""
    n_classes = len(actions)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Convert labels to binary format
    y_true_bin = to_categorical(y_true, num_classes=n_classes)
    
    # Compute ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot ROC curves for A-M
    for i in range(13):
        ax1.plot(fpr[i], tpr[i], lw=2, 
                label=f'{actions[i]} (AUC = {roc_auc[i]:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curves (A-M)', fontsize=13)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(alpha=0.3)
    
    # Plot ROC curves for N-Z
    for i in range(13, n_classes):
        ax2.plot(fpr[i], tpr[i], lw=2,
                label=f'{actions[i]} (AUC = {roc_auc[i]:.2f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title('ROC Curves (N-Z)', fontsize=13)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and return average AUC
    avg_auc = np.mean(list(roc_auc.values()))
    return avg_auc

def plot_confidence_distribution(y_pred_proba, y_true, save_path):
    """Plot confidence distribution for correct vs incorrect predictions"""
    # Get predicted classes
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get maximum confidence for each prediction
    max_confidence = np.max(y_pred_proba, axis=1)
    
    # Separate confidence scores for correct and incorrect predictions
    correct_conf = max_confidence[y_pred == y_true]
    incorrect_conf = max_confidence[y_pred != y_true]
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(correct_conf, bins=50, alpha=0.7, label='Correct Predictions', 
             color='green', edgecolor='black')
    plt.hist(incorrect_conf, bins=50, alpha=0.7, label='Incorrect Predictions', 
             color='red', edgecolor='black')
    
    # Set labels and title
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_misclassifications(cm, save_path):
    """Identify and visualize most common misclassifications"""
    # Copy confusion matrix and remove diagonal (correct predictions)
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    
    # Find all misclassifications
    top_errors = []
    for i in range(len(actions)):
        for j in range(len(actions)):
            if cm_errors[i, j] > 0:
                top_errors.append((actions[i], actions[j], cm_errors[i, j]))
    
    # Sort by count and get top 15
    top_errors = sorted(top_errors, key=lambda x: x[2], reverse=True)[:15]
    
    # Create DataFrame
    df = pd.DataFrame(top_errors, columns=['True Label', 'Predicted As', 'Count'])
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(df)), df['Count'], color='salmon', edgecolor='black')
    plt.yticks(range(len(df)), 
              [f"{row['True Label']} → {row['Predicted As']}" for _, row in df.iterrows()])
    plt.xlabel('Number of Misclassifications', fontsize=12)
    plt.title('Top 15 Most Common Misclassifications', fontsize=14, pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f' {int(width)}', ha='left', va='center', fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def plot_class_support(y_true, save_path):
    """Plot number of samples per class in test set"""
    # Count samples per class
    unique, counts = np.unique(y_true, return_counts=True)
    class_counts = dict(zip([actions[i] for i in unique], counts))
    
    # Create bar chart
    plt.figure(figsize=(16, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), 
                   color='steelblue', edgecolor='black')
    plt.xlabel('Sign Letter', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Test Set Distribution', fontsize=14, pad=20)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_report(model, X_train, y_train_idx, X_val, y_val_idx, outdir):
    """Generate comprehensive evaluation with all metrics and visualizations"""
    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Generate predictions for validation set
    print("\n[1/9] Generating predictions...")
    y_pred_val_proba = model.predict(X_val, verbose=0)
    y_pred_val = np.argmax(y_pred_val_proba, axis=1)
    
    # Generate predictions for training set
    y_pred_train_proba = model.predict(X_train, verbose=0)
    y_pred_train = np.argmax(y_pred_train_proba, axis=1)
    
    # Calculate accuracy metrics
    print("[2/9] Calculating accuracy metrics...")
    acc_train = accuracy_score(y_train_idx, y_pred_train)
    acc_val = accuracy_score(y_val_idx, y_pred_val)
    top3_acc = top_k_accuracy_score(y_val_idx, y_pred_val_proba, k=3)
    top5_acc = top_k_accuracy_score(y_val_idx, y_pred_val_proba, k=5)
    
    # Print accuracy metrics
    print(f"   Training Accuracy: {acc_train:.4f}")
    print(f"   Validation Accuracy: {acc_val:.4f}")
    print(f"   Top-3 Accuracy: {top3_acc:.4f}")
    print(f"   Top-5 Accuracy: {top5_acc:.4f}")
    
    # Generate confusion matrices
    print("[3/9] Generating confusion matrices...")
    cm = confusion_matrix(y_val_idx, y_pred_val, labels=np.arange(len(actions)))
    plot_confusion_matrix(cm, os.path.join(outdir, "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(cm, os.path.join(outdir, "confusion_matrix_normalized.png"), normalize=True)
    
    # Generate per-class metrics
    print("[4/9] Analyzing per-class performance...")
    report = classification_report(y_val_idx, y_pred_val, 
                                   target_names=list(actions), 
                                   zero_division=0, output_dict=True)
    plot_per_class_metrics(report, os.path.join(outdir, "per_class_metrics.png"))
    
    # Generate ROC curves
    print("[5/9] Computing ROC curves...")
    avg_auc = plot_roc_curves(y_val_idx, y_pred_val_proba, 
                              os.path.join(outdir, "roc_curves.png"))
    
    # Analyze confidence distribution
    print("[6/9] Analyzing confidence distribution...")
    plot_confidence_distribution(y_pred_val_proba, y_val_idx,
                                os.path.join(outdir, "confidence_distribution.png"))
    
    # Analyze misclassifications
    print("[7/9] Identifying common misclassifications...")
    misclass_df = analyze_misclassifications(cm, 
                                            os.path.join(outdir, "top_misclassifications.png"))
    
    # Plot test set distribution
    print("[8/9] Plotting test set distribution...")
    plot_class_support(y_val_idx, os.path.join(outdir, "test_set_distribution.png"))
    
    # Save detailed metrics
    print("[9/9] Saving metrics and reports...")
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for i, a in enumerate(actions):
        mask = y_val_idx == i
        if mask.sum() == 0:
            per_class_acc[a] = None
        else:
            per_class_acc[a] = float((y_pred_val[mask] == i).mean())
    
    # Create metrics dictionary
    metrics = {
        "overall_metrics": {
            "train_accuracy": float(acc_train),
            "validation_accuracy": float(acc_val),
            "top3_accuracy": float(top3_acc),
            "top5_accuracy": float(top5_acc),
            "average_roc_auc": float(avg_auc)
        },
        "per_class_accuracy": per_class_acc,
        "dataset_info": {
            "total_sequences": len(X_train) + len(X_val),
            "train_sequences": len(X_train),
            "val_sequences": len(X_val),
            "sequence_length": sequence_length,
            "num_classes": len(actions),
            "features_per_frame": 63
        },
        "split_config": {
            "train_ratio": 0.8,
            "val_ratio": 0.2,
            "stratified": True,
            "random_state": 42
        }
    }
    
    # Save metrics as JSON
    with open(os.path.join(outdir, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save classification report as JSON
    with open(os.path.join(outdir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    # Save misclassifications as CSV
    misclass_df.to_csv(os.path.join(outdir, "top_misclassifications.csv"), index=False)
    
    # Generate text report
    with open(os.path.join(outdir, "evaluation_report.txt"), "w") as f:
        # Write header
        f.write("="*70 + "\n")
        f.write("SIGN LANGUAGE RECOGNITION - MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Write overall performance
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Training Accuracy:     {acc_train:.4f} ({acc_train*100:.2f}%)\n")
        f.write(f"Validation Accuracy:   {acc_val:.4f} ({acc_val*100:.2f}%)\n")
        f.write(f"Top-3 Accuracy:        {top3_acc:.4f} ({top3_acc*100:.2f}%)\n")
        f.write(f"Top-5 Accuracy:        {top5_acc:.4f} ({top5_acc*100:.2f}%)\n")
        f.write(f"Average ROC-AUC:       {avg_auc:.4f}\n\n")
        
        # Write dataset information
        f.write("DATASET INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Sequences:       {len(X_train) + len(X_val)}\n")
        f.write(f"Training Sequences:    {len(X_train)}\n")
        f.write(f"Validation Sequences:  {len(X_val)}\n")
        f.write(f"Classes:               {len(actions)}\n")
        f.write(f"Sequence Length:       {sequence_length} frames\n\n")
        
        # Write top 5 best performing classes
        f.write("TOP 5 BEST PERFORMING CLASSES\n")
        f.write("-"*70 + "\n")
        sorted_acc = sorted(per_class_acc.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)
        for i, (cls, acc) in enumerate(sorted_acc[:5], 1):
            f.write(f"{i}. {cls}: {acc:.4f} ({acc*100:.2f}%)\n")
        
        # Write top 5 worst performing classes
        f.write("\nTOP 5 WORST PERFORMING CLASSES\n")
        f.write("-"*70 + "\n")
        for i, (cls, acc) in enumerate(sorted_acc[-5:][::-1], 1):
            f.write(f"{i}. {cls}: {acc:.4f} ({acc*100:.2f}%)\n")
        
        # Write top 10 misclassifications
        f.write("\nTOP 10 MISCLASSIFICATIONS\n")
        f.write("-"*70 + "\n")
        for i, row in misclass_df.head(10).iterrows():
            f.write(f"{i+1}. {row['True Label']} misclassified as {row['Predicted As']}: "
                   f"{int(row['Count'])} times\n")
    
    # Print completion message
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {outdir}/")
    print("\nGenerated files:")
    print("  - metrics_summary.json")
    print("  - classification_report.json")
    print("  - evaluation_report.txt")
    print("  - confusion_matrix.png")
    print("  - confusion_matrix_normalized.png")
    print("  - per_class_metrics.png")
    print("  - roc_curves.png")
    print("  - confidence_distribution.png")
    print("  - top_misclassifications.png")
    print("  - top_misclassifications.csv")
    print("  - test_set_distribution.png")
    
    return metrics

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Enhanced model evaluation")
    parser.add_argument("--model_json", default="model(0.35).json", help="Path to model JSON")
    parser.add_argument("--weights", default="newmodel(0.35).h5", help="Path to model weights")
    parser.add_argument("--out", default="evaluation_results", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load dataset
    print("\nLoading dataset...")
    X, y_onehot, y_idx = load_data()
    
    # Check if data was loaded
    if len(X) == 0:
        print("ERROR: No data found in MP_Data directory!")
        return
    
    print(f"Loaded {len(X)} sequences across {len(actions)} classes")
    
    # Split data into training and validation sets (stratified to maintain class balance)
    print("\nSplitting data (stratified)...")
    X_train, X_val, y_train_idx, y_val_idx = train_test_split(
        X, y_idx, test_size=args.test_size, stratify=y_idx, 
        random_state=args.random_state
    )
    
    print(f"Train: {len(X_train)} | Validation: {len(X_val)}")
    
    # Load trained model
    print("\nLoading trained model...")
    model = load_trained_model(args.model_json, args.weights)
    print("Model loaded successfully!")
    
    # Run comprehensive evaluation
    print("\nStarting comprehensive evaluation...")
    metrics = generate_detailed_report(model, X_train, y_train_idx, 
                                      X_val, y_val_idx, args.out)
    
    print("\n✅ Evaluation completed successfully!")

if _name_ == "_main_":
    main()