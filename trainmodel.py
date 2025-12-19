import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import json

# Path to directory containing processed keypoint sequences
data_root = Path("MP_Data")

# List of all letters A-Z (output classes)
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Number of frames per sequence
sequence_length = 30

def load_data():
    """Load all sequence data from MP_Data directory"""
    # Initialize lists for sequences and labels
    sequences = []
    labels = []
    
    # Loop through each letter
    for idx, action in enumerate(actions):
        # Get directory for this letter
        action_dir = data_root / action
        
        # Skip if directory doesn't exist
        if not action_dir.exists():
            continue
        
        # Loop through each sequence folder (sorted numerically)
        for seq_dir in sorted(action_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1e9):
            frames = []
            ok = True
            
            # Load all 30 frames in the sequence
            for f in range(sequence_length):
                fp = seq_dir / f"{f}.npy"
                
                # Check if frame file exists
                if not fp.exists():
                    ok = False
                    break
                
                # Load frame keypoints
                frames.append(np.load(fp))
            
            # Add to dataset if all frames loaded successfully
            if ok and len(frames) == sequence_length:
                sequences.append(frames)
                labels.append(idx)
    
    # Convert to numpy arrays
    X = np.array(sequences)
    
    # Convert labels to one-hot encoded format
    y = to_categorical(labels).astype(int)
    
    # Return features, one-hot labels, and integer labels
    return X, y, np.array(labels)

# Load the dataset
X, y, y_int = load_data()

# Split data into training (80%) and validation (20%) sets
# Stratified split maintains class distribution
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y_int, random_state=42
)

# Build the neural network model
model = Sequential([
    # First bidirectional LSTM layer (96 units in each direction)
    Bidirectional(LSTM(96, return_sequences=True, activation="relu"), input_shape=(sequence_length, 63)),
    Dropout(0.3),  # Dropout to prevent overfitting
    
    # Second bidirectional LSTM layer (128 units in each direction)
    Bidirectional(LSTM(128, return_sequences=True, activation="relu")),
    Dropout(0.3),
    
    # Third bidirectional LSTM layer (96 units in each direction, no return sequences)
    Bidirectional(LSTM(96, return_sequences=False, activation="relu")),
    Dropout(0.3),
    
    # Fully connected dense layers
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    
    # Output layer with softmax for 26 classes (A-Z)
    Dense(len(actions), activation="softmax")
])

# Compile model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

# Create directory for TensorBoard logs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Define training callbacks
callbacks = [
    # TensorBoard for visualizing training metrics
    TensorBoard(log_dir=log_dir),
    
    # Early stopping to prevent overfitting (stops if no improvement for 20 epochs)
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    
    # Save best model based on validation accuracy
    ModelCheckpoint("best_model(33k).h5", monitor="val_categorical_accuracy", save_best_only=True, mode="max"),
    
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1)
]

# Compute class weights to handle imbalanced data
cw = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_int), y=y_int)

# Convert to dictionary format for Keras
cw_dict = {i: cw[i] for i in range(len(cw))}

# Train the model
model.fit(
    X_train, y_train,  # Training data
    epochs=200,  # Maximum number of epochs
    batch_size=64,  # Number of samples per gradient update
    validation_data=(X_val, y_val),  # Validation data
    callbacks=callbacks,  # Training callbacks
    class_weight=cw_dict,  # Apply class weights
    verbose=2  # Print progress
)

# Save model architecture as JSON
with open("model33k.json", "w") as f:
    f.write(model.to_json())

# Save model weights
model.save("newmodel33k.h5")