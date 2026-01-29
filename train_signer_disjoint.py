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

train_data_root = Path("MP_Data_Train")
val_data_root = Path("MP_Data_Val")
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

class DataAugmentation:
    @staticmethod
    def add_noise(X, noise_factor=0.02):
        noise = np.random.normal(0, noise_factor, X.shape)
        return X + noise
    
    @staticmethod
    def time_warp(X, sigma=0.2):
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
        warp = np.random.normal(1.0, sigma, (X.shape[0], 1, X.shape[2]))
        return X * warp
    
    @staticmethod
    def augment_batch(X, y):
        X_aug = X.copy()
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.add_noise(X_aug)
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.time_warp(X_aug)
        if np.random.random() > 0.5:
            X_aug = DataAugmentation.magnitude_warp(X_aug)
        return X_aug, y

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

def load_data_from_dir(data_root):
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
    y = to_categorical(labels, num_classes=len(actions)).astype(int)
    return X, y, np.array(labels)

def create_improved_model():
    inputs = Input(shape=(sequence_length, feature_dim))
    
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
    print("SIGNER-DISJOINT TRAINING")
    print("=" * 60)
    
    print("\nLoading training data...")
    X_train, y_train, y_train_int = load_data_from_dir(train_data_root)
    print(f"Training samples: {len(X_train)}")
    
    print("\nLoading validation data...")
    X_val, y_val, y_val_int = load_data_from_dir(val_data_root)
    print(f"Validation samples: {len(X_val)}")
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("\n❌ Error: No data found. Run session_split.py first!")
        exit(1)
    
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
    
    log_dir = "logs_signer_disjoint"
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
            "best_model_signer_disjoint.h5", 
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
    with open("model_signer_disjoint.json", "w") as f:
        f.write(model.to_json())
    model.save("newmodel_signer_disjoint.h5")
    
    val_loss, val_acc, val_top3 = model.evaluate(val_gen, verbose=0)
    print(f"\n{'='*60}")
    print("FINAL RESULTS (SIGNER-DISJOINT SPLIT)")
    print(f"{'='*60}")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Top-3 Accuracy:      {val_top3*100:.2f}%")
    print(f"Validation Loss:     {val_loss:.4f}")
    
    with open("training_history_signer_disjoint.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    
    print("\n✅ Training complete!")
