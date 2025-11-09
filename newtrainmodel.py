import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential, Model                                                                                                               # type: ignore
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional,                                                                                           # type: ignore
                                      BatchNormalization, Input, Add, Layer,                                                                                        # type: ignore
                                      Attention, Concatenate)                                                                                                       # type: ignore
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler                                                           # type: ignore
from tensorflow.keras.utils import to_categorical                                                                                                                   # type: ignore
from tensorflow.keras.regularizers import l2                                                                                                                        # type: ignore
from tensorflow.keras.optimizers import Adam                                                                                                                        # type: ignore
import tensorflow as tf
import json

# ============= CONFIGURATION =============
data_root = Path("MP_Data")
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
sequence_length = 30
feature_dim = 63

# ============= DATA AUGMENTATION =============
class DataAugmentation:
    @staticmethod
    def add_noise(X, noise_factor=0.02):
        """Add Gaussian noise to sequences"""
        noise = np.random.normal(0, noise_factor, X.shape)
        return X + noise
    
    @staticmethod
    def time_warp(X, sigma=0.2):
        """Apply time warping augmentation"""
        seq_len = X.shape[1]
        warp = np.random.normal(1.0, sigma, seq_len)
        warp = np.cumsum(warp)
        warp = (warp - warp.min()) / (warp.max() - warp.min()) * (seq_len - 1)
        
        warped = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                warped[i, :, j] = np.interp(np.arange(seq_len), warp, X[i, :, j])
        return warped
    
    @staticmethod
    def magnitude_warp(X, sigma=0.2):
        """Apply magnitude warping"""
        warp = np.random.normal(1.0, sigma, (X.shape[0], 1, X.shape[2]))
        return X * warp
    
    @staticmethod
    def augment_batch(X, y):
        """Apply random augmentations to a batch"""
        X_aug = X.copy()
        
        # Randomly apply augmentations
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.add_noise(X_aug)
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.time_warp(X_aug)
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.magnitude_warp(X_aug)
            
        return X_aug, y

# ============= DATA GENERATOR =============
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, augment=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indexes = np.arange(len(X))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[indexes]
        y_batch = self.y[indexes]
        
        if self.augment:
            X_batch, y_batch = DataAugmentation.augment_batch(X_batch, y_batch)
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# ============= DATA LOADING =============
def load_data():
    sequences = []
    labels = []
    for idx, action in enumerate(actions):
        action_dir = data_root / action
        if not action_dir.exists():
            continue
        for seq_dir in sorted(action_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1e9):
            frames = []
            ok = True
            for f in range(sequence_length):
                fp = seq_dir / f"{f}.npy"
                if not fp.exists():
                    ok = False
                    break
                frames.append(np.load(fp))
            if ok and len(frames) == sequence_length:
                sequences.append(frames)
                labels.append(idx)
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return X, y, np.array(labels)

# ============= ATTENTION LAYER =============
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', 
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[-1],),
                                 initializer='zeros', 
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

# ============= IMPROVED MODEL =============
def create_improved_model():
    inputs = Input(shape=(sequence_length, feature_dim))
    
    # First BiLSTM block with residual
    x = BatchNormalization()(inputs)
    x1 = Bidirectional(LSTM(128, return_sequences=True, activation="tanh", 
                            recurrent_regularizer=l2(0.001)))(x)
    x1 = Dropout(0.4)(x1)
    x1 = BatchNormalization()(x1)
    
    # Second BiLSTM block with residual
    x2 = Bidirectional(LSTM(160, return_sequences=True, activation="tanh",
                            recurrent_regularizer=l2(0.001)))(x1)
    x2 = Dropout(0.4)(x2)
    x2 = BatchNormalization()(x2)
    
    # Third BiLSTM block
    x3 = Bidirectional(LSTM(128, return_sequences=True, activation="tanh",
                            recurrent_regularizer=l2(0.001)))(x2)
    x3 = Dropout(0.4)(x3)
    x3 = BatchNormalization()(x3)
    
    # Attention mechanism
    attention_output = AttentionLayer()(x3)
    
    # Dense layers with residual connections
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(attention_output)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(len(actions), activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# ============= LEARNING RATE SCHEDULE =============
def cosine_annealing_with_warmup(epoch, lr):
    """Cosine annealing with warmup"""
    warmup_epochs = 10
    total_epochs = 300
    max_lr = 0.001
    min_lr = 1e-6
    
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

# ============= MAIN TRAINING =============
print("Loading data...")
X, y, y_int = load_data()
print(f"Loaded {len(X)} sequences")

# Split data with stratification
X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
    X, y, y_int, test_size=0.2, stratify=y_int, random_state=42
)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Create data generators
train_gen = DataGenerator(X_train, y_train, batch_size=32, augment=True)
val_gen = DataGenerator(X_val, y_val, batch_size=32, augment=False)

# Create model
print("Creating improved model...")
model = create_improved_model()

# Compile with custom optimizer
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(
    optimizer=optimizer, 
    loss="categorical_crossentropy", 
    metrics=["categorical_accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

model.summary()

# Setup callbacks
log_dir = "logs_improved"
os.makedirs(log_dir, exist_ok=True)

callbacks = [
    TensorBoard(log_dir=log_dir, histogram_freq=1),
    EarlyStopping(
        monitor="val_categorical_accuracy", 
        patience=40, 
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        "best_model_improved.h5", 
        monitor="val_categorical_accuracy", 
        save_best_only=True, 
        mode="max",
        verbose=1
    ),
    LearningRateScheduler(cosine_annealing_with_warmup, verbose=1)
]

# Compute class weights
cw = class_weight.compute_class_weight(
    class_weight="balanced", 
    classes=np.unique(y_train_int), 
    y=y_train_int
)
cw_dict = {i: cw[i] for i in range(len(cw))}

print("Starting training...")
history = model.fit(
    train_gen,
    epochs=300,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=cw_dict,
    verbose=2
)

# Save model
print("Saving model...")
with open("model_improved.json", "w") as f:
    f.write(model.to_json())
model.save("newmodel_improved.h5")

# Print final results
val_loss, val_acc, val_top3 = model.evaluate(val_gen, verbose=0)
print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
print(f"Final Top-3 Accuracy: {val_top3*100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")

# Save training history
with open("training_history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional, 
                                      BatchNormalization, Input, Add, Layer,
                                      Attention, Concatenate)
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json

# ============= CONFIGURATION =============
# Path to directory containing processed keypoint sequences
data_root = Path("MP_Data")

# List of all letters A-Z (output classes)
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Number of frames per sequence
sequence_length = 30

# Number of features per frame (21 landmarks × 3 coordinates)
feature_dim = 63

# ============= DATA AUGMENTATION =============
class DataAugmentation:
    """Class containing various data augmentation techniques"""
    
    @staticmethod
    def add_noise(X, noise_factor=0.02):
        """Add Gaussian noise to sequences to improve robustness"""
        # Generate random noise with same shape as input
        noise = np.random.normal(0, noise_factor, X.shape)
        
        # Add noise to original data
        return X + noise
    
    @staticmethod
    def time_warp(X, sigma=0.2):
        """Apply time warping augmentation - stretches/compresses time axis"""
        seq_len = X.shape[1]
        
        # Generate random warp curve
        warp = np.random.normal(1.0, sigma, seq_len)
        
        # Create cumulative warp function
        warp = np.cumsum(warp)
        
        # Normalize warp to sequence length
        warp = (warp - warp.min()) / (warp.max() - warp.min()) * (seq_len - 1)
        
        # Apply warping to each sequence
        warped = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                # Interpolate values according to warp function
                warped[i, :, j] = np.interp(np.arange(seq_len), warp, X[i, :, j])
        
        return warped
    
    @staticmethod
    def magnitude_warp(X, sigma=0.2):
        """Apply magnitude warping - scales feature magnitudes randomly"""
        # Generate random scaling factors for each feature
        warp = np.random.normal(1.0, sigma, (X.shape[0], 1, X.shape[2]))
        
        # Apply scaling
        return X * warp
    
    @staticmethod
    def augment_batch(X, y):
        """Apply random augmentations to a batch"""
        # Create copy to avoid modifying original
        X_aug = X.copy()
        
        # Randomly apply each augmentation with 50% probability
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.add_noise(X_aug)
        
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.time_warp(X_aug)
        
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.magnitude_warp(X_aug)
            
        return X_aug, y

# ============= DATA GENERATOR =============
class DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for training with augmentation"""
    
    def _init_(self, X, y, batch_size=32, augment=True):
        """Initialize generator with data and settings"""
        self.X = X                      # Input sequences
        self.y = y                      # Labels
        self.batch_size = batch_size    # Samples per batch
        self.augment = augment          # Whether to apply augmentation
        self.indexes = np.arange(len(X)) # Sample indices
        self.on_epoch_end()             # Shuffle initially
    
    def _len_(self):
        """Return number of batches per epoch"""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def _getitem_(self, index):
        """Generate one batch of data"""
        # Get indices for this batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get batch data
        X_batch = self.X[indexes]
        y_batch = self.y[indexes]
        
        # Apply augmentation if enabled
        if self.augment:
            X_batch, y_batch = DataAugmentation.augment_batch(X_batch, y_batch)
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """Shuffle indices at end of each epoch"""
        np.random.shuffle(self.indexes)

# ============= DATA LOADING =============
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
    
    return X, y, np.array(labels)

# ============= ATTENTION LAYER =============
class AttentionLayer(Layer):
    """Custom attention mechanism to focus on important frames"""
    
    def _init_(self, **kwargs):
        """Initialize attention layer"""
        super(AttentionLayer, self)._init_(**kwargs)
    
    def build(self, input_shape):
        """Build layer weights"""
        # Create weight matrix for attention computation
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', 
                                 trainable=True)
        
        # Create bias vector
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[-1],),
                                 initializer='zeros', 
                                 trainable=True)
        
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        """Forward pass - compute attention-weighted output"""
        # Compute attention scores
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        
        # Normalize scores to attention weights using softmax
        a = tf.keras.activations.softmax(e, axis=1)
        
        # Apply attention weights to input
        output = x * a
        
        # Sum over time dimension to get final output
        return tf.reduce_sum(output, axis=1)

# ============= IMPROVED MODEL =============
def create_improved_model():
    """Create advanced model with attention and residual connections"""
    # Input layer
    inputs = Input(shape=(sequence_length, feature_dim))
    
    # Normalize input
    x = BatchNormalization()(inputs)
    
    # First BiLSTM block - processes sequence in both directions
    x1 = Bidirectional(LSTM(128, return_sequences=True, activation="tanh", 
                            recurrent_regularizer=l2(0.001)))(x)
    x1 = Dropout(0.4)(x1)  # Dropout to prevent overfitting
    x1 = BatchNormalization()(x1)  # Normalize activations
    
    # Second BiLSTM block - deeper representation
    x2 = Bidirectional(LSTM(160, return_sequences=True, activation="tanh",
                            recurrent_regularizer=l2(0.001)))(x1)
    x2 = Dropout(0.4)(x2)
    x2 = BatchNormalization()(x2)
    
    # Third BiLSTM block - final temporal features
    x3 = Bidirectional(LSTM(128, return_sequences=True, activation="tanh",
                            recurrent_regularizer=l2(0.001)))(x2)
    x3 = Dropout(0.4)(x3)
    x3 = BatchNormalization()(x3)
    
    # Attention mechanism - focuses on most important frames
    attention_output = AttentionLayer()(x3)
    
    # First dense layer - high-level feature extraction
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(attention_output)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Second dense layer - further refinement
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer - softmax for 26 classes (A-Z)
    outputs = Dense(len(actions), activation="softmax")(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# ============= LEARNING RATE SCHEDULE =============
def cosine_annealing_with_warmup(epoch, lr):
    """Cosine annealing with warmup for learning rate"""
    # Configuration
    warmup_epochs = 10      # Number of warmup epochs
    total_epochs = 300      # Total training epochs
    max_lr = 0.001          # Maximum learning rate
    min_lr = 1e-6           # Minimum learning rate
    
    # Warmup phase - gradually increase learning rate
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    
    # Annealing phase - cosine decay
    else:
        # Calculate progress through annealing phase
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        
        # Apply cosine annealing formula
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

# ============= MAIN TRAINING =============
# Load dataset
print("Loading data...")
X, y, y_int = load_data()
print(f"Loaded {len(X)} sequences")

# Split data into train (85%) and validation (15%) sets
# Stratified split maintains class distribution
X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
    X, y, y_int, test_size=0.15, stratify=y_int, random_state=42
)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Create data generators
train_gen = DataGenerator(X_train, y_train, batch_size=32, augment=True)   # With augmentation
val_gen = DataGenerator(X_val, y_val, batch_size=32, augment=False)        # Without augmentation

# Create model
print("Creating improved model...")
model = create_improved_model()

# Compile with custom optimizer settings
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)  # Gradient clipping to prevent exploding gradients

# Compile model with metrics
model.compile(
    optimizer=optimizer, 
    loss="categorical_crossentropy",  # Standard loss for classification
    metrics=[
        "categorical_accuracy",  # Standard accuracy
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')  # Top-3 accuracy
    ]
)

# Display model architecture
model.summary()

# Setup logging directory
log_dir = "logs_improved"
os.makedirs(log_dir, exist_ok=True)

# Define training callbacks
callbacks = [
    # TensorBoard for visualization
    TensorBoard(log_dir=log_dir, histogram_freq=1),
    
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor="val_categorical_accuracy",  # Monitor validation accuracy
        patience=40,                         # Stop after 40 epochs without improvement
        restore_best_weights=True,           # Restore weights from best epoch
        mode='max',                          # Higher is better
        verbose=1
    ),
    
    # Save best model during training
    ModelCheckpoint(
        "best_model_improved.h5", 
        monitor="val_categorical_accuracy",  # Save based on validation accuracy
        save_best_only=True,                 # Only save if better than previous best
        mode="max",
        verbose=1
    ),
    
    # Dynamic learning rate adjustment
    LearningRateScheduler(cosine_annealing_with_warmup, verbose=1)
]

# Compute class weights to handle imbalanced data
cw = class_weight.compute_class_weight(
    class_weight="balanced",          # Use balanced weighting
    classes=np.unique(y_train_int),   # All unique classes
    y=y_train_int                     # Training labels
)

# Convert to dictionary format for Keras
cw_dict = {i: cw[i] for i in range(len(cw))}

# Train the model
print("Starting training...")
history = model.fit(
    train_gen,                      # Training data generator
    epochs=300,                     # Maximum number of epochs
    validation_data=val_gen,        # Validation data generator
    callbacks=callbacks,            # Training callbacks
    class_weight=cw_dict,           # Class weights for imbalanced data
    verbose=2                       # Print progress
)

# Save model architecture and weights
print("Saving model...")
with open("model_improved.json", "w") as f:
    f.write(model.to_json())

# Save complete model
model.save("newmodel_improved.h5")

# Evaluate final model performance
val_loss, val_acc, val_top3 = model.evaluate(val_gen, verbose=0)

# Print final results
print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
print(f"Final Top-3 Accuracy: {val_top3*100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")

# Save training history for analysis
with open("training_history.json", "w") as f:
    # Convert numpy arrays to lists for JSON serialization
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)