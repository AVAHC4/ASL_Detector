import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical

# Config
DATA_ROOT = Path("MP_Data")
SEQUENCE_LENGTH = 30
ACTIONS = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
MODEL_JSON = "model_improved.json"
MODEL_WEIGHTS = "best_model_improved.h5"
HISTORY_FILE = "training_history.json"

# Plot formatting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data():
    sequences, labels = [], []
    for idx, action in enumerate(ACTIONS):
        action_dir = DATA_ROOT / action
        if not action_dir.exists():
            continue
        for seq_dir in sorted(action_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1e9):
            frames = []
            if not seq_dir.is_dir(): continue
            
            # fast check for completeness
            if not (seq_dir / f"{SEQUENCE_LENGTH-1}.npy").exists():
                continue

            for f in range(SEQUENCE_LENGTH):
                fp = seq_dir / f"{f}.npy"
                if fp.exists():
                    frames.append(np.load(fp))
                else:
                    break
            
            if len(frames) == SEQUENCE_LENGTH:
                sequences.append(frames)
                labels.append(idx)
                
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return X, y, np.array(labels)

def load_model():
    # Try different model names if improved not found, but fallback to provided names
    json_path = MODEL_JSON
    weights_path = MODEL_WEIGHTS
    
    if not os.path.exists(json_path):
        # Fallback to the one used in app.py if improved missing
        if os.path.exists("model(0.2).json"):
            json_path = "model(0.2).json"
            weights_path = "newmodel(0.2).h5"
    
    print(f"Loading model from {json_path} and {weights_path}")
    with open(json_path, "r") as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)
    return model

# 1. Training Curves (Figure 7)
def plot_training_curves(history, save_path="trainingcurves.png"):
    if not history: return
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    if 'categorical_accuracy' in history:
        plt.plot(history['categorical_accuracy'], label='Train Accuracy')
        plt.plot(history['val_categorical_accuracy'], label='Val Accuracy')
    elif 'accuracy' in history:
         plt.plot(history['accuracy'], label='Train Accuracy')
         plt.plot(history['val_accuracy'], label='Val Accuracy')       
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Generated {save_path}")

# 2. Top-K Accuracy (Figure 8)
def plot_top_k(history, save_path="topkaccuracy.png"):
    if not history or 'top_3_accuracy' not in history: return
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['top_3_accuracy'], label='Train Top-3')
    plt.plot(history['val_top_3_accuracy'], label='Val Top-3')
    plt.title('Top-3 Accuracy during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Generated {save_path}")

# 3. Confusion Matrix Normalized (Figure 3)
def plot_cm_norm(y_true, y_pred, save_path="confusionmatrixnormalized.png"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=ACTIONS, yticklabels=ACTIONS)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Generated {save_path}")

# 4. ROC Curves (Figure 4)
def plot_roc(y_true_onehot, y_pred_proba, save_path="roccurves.png"):
    n_classes = len(ACTIONS)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Plot all ROC curves
    plt.figure(figsize=(12, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan'])
    
    # We can't plot 26 lines legibly, let's plot micro-average or just line soup
    # The paper shows multiple lines. Let's plot them all thin.
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.5)
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (One-vs-Rest)')
    # Legend is too big for 26 classes, omitting or making it small? 
    # The paper figure 4 shows two subplots (A-M, N-Z). Let's do that.
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for i in range(13):
        ax1.plot(fpr[i], tpr[i], label=f'{ACTIONS[i]} (AUC={roc_auc[i]:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_title('ROC Curves (A-M)')
    ax1.legend(loc="lower right", fontsize='small')
    
    for i in range(13, 26):
        ax2.plot(fpr[i], tpr[i], label=f'{ACTIONS[i]} (AUC={roc_auc[i]:.2f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_title('ROC Curves (N-Z)')
    ax2.legend(loc="lower right", fontsize='small')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Generated {save_path}")

# 5. Per Class Metrics (Figure 5)
def plot_per_class(y_true, y_pred, save_path="perclassmetrics.png"):
    report = classification_report(y_true, y_pred, target_names=ACTIONS, output_dict=True)
    
    precision = [report[c]['precision'] for c in ACTIONS]
    recall = [report[c]['recall'] for c in ACTIONS]
    f1 = [report[c]['f1-score'] for c in ACTIONS]
    
    x = np.arange(len(ACTIONS))
    width = 0.25
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')
    
    plt.xticks(x, ACTIONS)
    plt.legend()
    plt.title('Per-class Precision, Recall, and F1-Score')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Generated {save_path}")

# 6. Confidence Distribution (Figure 6)
def plot_conf_dist(y_pred_proba, y_true, y_pred, save_path="confidencedistribution.png"):
    confidences = np.max(y_pred_proba, axis=1)
    
    correct_mask = (y_true == y_pred)
    incorrect_mask = ~correct_mask
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidences[correct_mask], bins=30, alpha=0.5, color='green', label='Correct')
    plt.hist(confidences[incorrect_mask], bins=30, alpha=0.5, color='red', label='Incorrect')
    
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Generated {save_path}")

from itertools import cycle

def main():
    # 1 & 2: From History
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        plot_training_curves(history)
        plot_top_k(history)
    else:
        print(f"Warning: {HISTORY_FILE} not found. Skipping training plots.")

    # Load Data & Model for 3-6
    print("Loading data...")
    X, y_onehot, y_idx = load_data()
    print(f"Loaded {len(X)} sequences.")
    
    # 80/20 Stratified Split
    X_train, X_val, y_train, y_val, y_train_idx, y_val_idx = train_test_split(
        X, y_onehot, y_idx, test_size=0.2, stratify=y_idx, random_state=42
    )
    
    print("Loading model...")
    model = load_model()
    
    print("Running inference on validation set...")
    y_pred_proba = model.predict(X_val)
    y_pred = np.argmax(y_pred_proba, axis=1)
    # y_val_idx is the true integer labels
    
    print("Generating plots...")
    plot_cm_norm(y_val_idx, y_pred)
    plot_roc(y_val, y_pred_proba) # y_val is onehot
    plot_per_class(y_val_idx, y_pred)
    plot_conf_dist(y_pred_proba, y_val_idx, y_pred)
    
    print("Done.")

if __name__ == "__main__":
    main()
