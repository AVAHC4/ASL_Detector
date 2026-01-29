import json
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

HISTORY_FILE = "training_history.json"
sns.set_style("whitegrid")

def main():
    if not os.path.exists(HISTORY_FILE):
        print(f"Error: {HISTORY_FILE} not found")
        return

    print(f"Loading {HISTORY_FILE}...")
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)

    # 1. Training Curves
    print("Generating trainingcurves.png...")
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
    plt.savefig("trainingcurves.png")
    plt.close()

    # 2. Top-K Accuracy
    if 'top_3_accuracy' in history:
        print("Generating topkaccuracy.png...")
        plt.figure(figsize=(10, 6))
        plt.plot(history['top_3_accuracy'], label='Train Top-3')
        plt.plot(history['val_top_3_accuracy'], label='Val Top-3')
        plt.title('Top-3 Accuracy during Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig("topkaccuracy.png")
        plt.close()
    else:
        print("Skipping topkaccuracy.png (key not found)")

if __name__ == "__main__":
    main()
