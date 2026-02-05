"""
TWO-HANDED SIGN LANGUAGE MODEL TRAINING
========================================
Training script for two-handed ASL sign recognition model.
Uses BiLSTM with attention mechanism for sequence classification.

Usage:
    python train_twohands.py
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional,
                                      BatchNormalization, Input, Layer)
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
data_root = Path("MP_Data_TwoHands")

# Two-handed ASL signs vocabulary
actions = np.array([
    "HELLO", "THANK_YOU", "YES", "NO", "PLEASE",
    "SORRY", "HELP", "MORE", "FINISHED", "I_LOVE_YOU",
    "GOOD", "BAD", "WANT", "NEED", "LIKE",
    "EAT", "DRINK", "SLEEP", "WORK", "PLAY",
    "HAPPY", "SAD", "ANGRY", "SCARED", "EXCITED",
    "BATHROOM"
])

sequence_length = 30
feature_dim = 126  # 2 hands × 21 landmarks × 3 coords


# ============================================================================
# DATA AUGMENTATION
# ============================================================================
class DataAugmentation:
    """Data augmentation techniques for sequence data."""
    
    @staticmethod
    def add_noise(X, noise_factor=0.02):
        """Add Gaussian noise to sequences."""
        noise = np.random.normal(0, noise_factor, X.shape)
        return X + noise
    
    @staticmethod
    def time_warp(X, sigma=0.2):
        """Apply time warping augmentation."""
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
        """Apply magnitude warping."""
        warp = np.random.normal(1.0, sigma, (X.shape[0], 1, X.shape[2]))
        return X * warp
    
    @staticmethod
    def augment_batch(X, y):
        """Apply random augmentations to a batch."""
        X_aug = X.copy()
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.add_noise(X_aug)
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.time_warp(X_aug)
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.magnitude_warp(X_aug)
        return X_aug, y


class DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator with augmentation."""
    
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


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load all sequence data from MP_Data_TwoHands directory."""
    sequences = []
    labels = []
    
    for idx, action in enumerate(actions):
        action_dir = data_root / action
        
        if not action_dir.exists():
            print(f"⚠️  No data found for: {action}")
            continue
        
        seq_count = 0
        for seq_dir in sorted(action_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1e9):
            frames = []
            ok = True
            
            for f in range(sequence_length):
                fp = seq_dir / f"{f}.npy"
                
                if not fp.exists():
                    ok = False
                    break
                
                frame_data = np.load(fp)
                if len(frame_data) != feature_dim:
                    print(f"⚠️  Invalid feature dim in {fp}: expected {feature_dim}, got {len(frame_data)}")
                    ok = False
                    break
                
                frames.append(frame_data)
            
            if ok and len(frames) == sequence_length:
                sequences.append(frames)
                labels.append(idx)
                seq_count += 1
        
        print(f"✅ Loaded {seq_count} sequences for: {action}")
    
    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)
    
    return X, y, np.array(labels)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class AttentionLayer(Layer):
    """Custom attention mechanism for sequence data."""
    
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


def create_model():
    """Create BiLSTM model with attention for two-handed sign recognition."""
    inputs = Input(shape=(sequence_length, feature_dim))
    
    # Initial normalization
    x = BatchNormalization()(inputs)
    
    # BiLSTM layers
    x1 = Bidirectional(LSTM(128, return_sequences=True, activation="tanh", 
                            recurrent_regularizer=l2(0.001)))(x)
    x1 = Dropout(0.4)(x1)
    x1 = BatchNormalization()(x1)
    
    x2 = Bidirectional(LSTM(160, return_sequences=True, activation="tanh",
                            recurrent_regularizer=l2(0.001)))(x1)
    x2 = Dropout(0.4)(x2)
    x2 = BatchNormalization()(x2)
    
    x3 = Bidirectional(LSTM(128, return_sequences=True, activation="tanh",
                            recurrent_regularizer=l2(0.001)))(x2)
    x3 = Dropout(0.4)(x3)
    x3 = BatchNormalization()(x3)
    
    # Attention mechanism
    attention_output = AttentionLayer()(x3)
    
    # Dense layers
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(attention_output)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(len(actions), activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def cosine_annealing_with_warmup(epoch, lr):
    """Cosine annealing learning rate schedule with warmup."""
    warmup_epochs = 10
    total_epochs = 300
    max_lr = 0.001
    min_lr = 1e-6
    
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


# ============================================================================
# MAIN TRAINING
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TWO-HANDED SIGN LANGUAGE MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data...")
    X, y, y_int = load_data()
    
    if len(X) == 0:
        print("\n❌ No training data found!")
        print("   Please collect data first using: python collectdata_twohands.py")
        exit(1)
    
    print(f"\n📊 Loaded {len(X)} total sequences")
    print(f"   Feature dimension: {feature_dim}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Number of classes: {len(np.unique(y_int))}")
    
    # Split data
    X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
        X, y, y_int, test_size=0.2, stratify=y_int, random_state=42
    )
    
    print(f"\n📈 Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Create data generators
    train_gen = DataGenerator(X_train, y_train, batch_size=32, augment=True)
    val_gen = DataGenerator(X_val, y_val, batch_size=32, augment=False)
    
    # Create model
    print("\n🔧 Creating model...")
    model = create_model()
    
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer, 
        loss="categorical_crossentropy", 
        metrics=[
            "categorical_accuracy", 
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )
    
    model.summary()
    
    # Setup callbacks
    log_dir = "logs_twohands"
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
            "best_model_twohands.h5", 
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
    
    # Train
    print("\n🚀 Starting training...")
    history = model.fit(
        train_gen,
        epochs=300,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=cw_dict,
        verbose=2
    )
    
    # Save model
    print("\n💾 Saving model...")
    with open("model_twohands.json", "w") as f:
        f.write(model.to_json())
    model.save("newmodel_twohands.h5")
    
    # Evaluate
    val_loss, val_acc, val_top3 = model.evaluate(val_gen, verbose=0)
    print(f"\n📊 Final Results:")
    print(f"   Validation Accuracy: {val_acc*100:.2f}%")
    print(f"   Top-3 Accuracy: {val_top3*100:.2f}%")
    print(f"   Validation Loss: {val_loss:.4f}")
    
    # Save training history
    with open("training_history_twohands.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    
    print("\n✅ Training complete!")
