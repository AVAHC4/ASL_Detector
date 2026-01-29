import os
import numpy as np
from pathlib import Path
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
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================
# Use the normalized dataset we just created
# Note: For simplicity, we are pointing to the root.
# However, train_signer_disjoint.py expected MP_Data_Train / MP_Data_Val.
# Since we normalized the full MP_Data, we should re-split it properly.
# BUT, to match the rigorous evaluation, we should ideally use the exact same split.
#
# Let's check session_split.py again. It generates MP_Data_Train and MP_Data_Val.
# We should probably normalize THOSE directories if they exist, OR normalize MP_Data and re-run session_split logic.
#
# A better approach: 
# 1. Normalize MP_Data -> MP_Data_Normalized
# 2. Modify this script to use MP_Data_Normalized and perform the session split internally OR
#    Just update the Data Paths to point to MP_Data_Train_Normalized if we create that.
#
# Let's keep it simple: normalize EVERYTHING first (already done by normalize_existing_data.py).
# Then run the session splitting logic ON THE NORMALIZED DATA to create split directories.
# OR, modify this script to accept the root and split on the fly.
#
# Let's adapt the session_split logic simply here to separate Train/Val based on the normalized data.
# Actually, the user already has MP_Data_Train and MP_Data_Val.
# I should normalize THOSE directly to preserve the exact same split.
# I'll update the normalize script to normalize those two folders specifically.
# But for now, let's assume we will normalize MP_Data_Train -> MP_Data_Train_Normalized
# and MP_Data_Val -> MP_Data_Val_Normalized.

TRAIN_DATA_ROOT = Path("MP_Data_Train_Normalized")
VAL_DATA_ROOT = Path("MP_Data_Val_Normalized")

actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
sequence_length = 30
feature_dim = 63

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
        
        # Note: Augmentations like rotation/scaling might conflict with normalization slightly,
        # but noise and time warp are fine.
        # Since we force normalization, magnitude warp is less effective but still adds noise.
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

def load_data_from_dir(data_root):
    sequences = []
    labels = []
    print(f"Loading from {data_root}...")
    
    if not data_root.exists():
        print(f"❌ Error: {data_root} does not exist.")
        return np.array([]), np.array([]), np.array([])

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
    if len(X) == 0:
        return np.array([]), np.array([]), np.array([])
        
    y = to_categorical(labels, num_classes=len(actions)).astype(int)
    return X, y, np.array(labels)

def create_improved_model():
    inputs = Input(shape=(sequence_length, feature_dim))
    
    # Batch Normalization is still useful even with normalized inputs
    x = BatchNormalization()(inputs)
    
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

    attention_output = AttentionLayer()(x3)

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
    warmup_epochs = 10
    total_epochs = 200
    max_lr = 0.001
    min_lr = 1e-6
    
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

if __name__ == "__main__":
    print("=" * 60)
    print("NORMALIZED TRAINING (Robust for All Hands/ROIs)")
    print("=" * 60)
    
    X_train, y_train, y_train_int = load_data_from_dir(TRAIN_DATA_ROOT)
    X_val, y_val, y_val_int = load_data_from_dir(VAL_DATA_ROOT)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("\n❌ Error: No normalized data found. Run normalize_existing_data.py first!")
        exit(1)
    
    # Augmentation is still good for noise robustness
    train_gen = DataGenerator(X_train, y_train, batch_size=32, augment=True)
    val_gen = DataGenerator(X_val, y_val, batch_size=32, augment=False)
    
    print("\nCreating model...")
    model = create_improved_model()
    
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer, 
        loss="categorical_crossentropy", 
        metrics=["categorical_accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    model.summary()
    
    # Use a new log directory
    log_dir = "logs_normalized"
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        EarlyStopping(
            monitor="val_categorical_accuracy", 
            patience=30, 
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ModelCheckpoint(
            "best_model_normalized.h5", 
            monitor="val_categorical_accuracy", 
            save_best_only=True, 
            mode="max",
            verbose=1
        ),
        LearningRateScheduler(cosine_annealing_with_warmup, verbose=1)
    ]
    
    cw = class_weight.compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(y_train_int), 
        y=y_train_int
    )
    cw_dict = {i: cw[i] for i in range(len(cw))}
    
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        epochs=200,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=cw_dict,
        verbose=2
    )
    
    print("\nSaving model...")
    with open("model_normalized.json", "w") as f:
        f.write(model.to_json())
    model.save("newmodel_normalized.h5")
    
    val_loss, val_acc, val_top3 = model.evaluate(val_gen, verbose=0)
    print(f"\n{'='*60}")
    print("FINAL RESULTS (NORMALIZED MODEL)")
    print(f"{'='*60}")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Top-3 Accuracy:      {val_top3*100:.2f}%")
    print(f"Validation Loss:     {val_loss:.4f}")
    
    with open("training_history_normalized.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    
    print("\n✅ Normalized training complete!")
