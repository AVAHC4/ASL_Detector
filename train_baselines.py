import os
import numpy as np
from pathlib import Path
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional,
                                      BatchNormalization, Input, Flatten, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json
import time

train_data_root = Path("MP_Data_Train")
val_data_root = Path("MP_Data_Val")
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
sequence_length = 30
feature_dim = 63

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

def create_simple_lstm():
    inputs = Input(shape=(sequence_length, feature_dim))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(actions), activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)

def create_frame_mlp():
    inputs = Input(shape=(sequence_length, feature_dim))
    x = Flatten()(inputs)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(actions), activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)

def create_reduced_bilstm():
    inputs = Input(shape=(sequence_length, feature_dim))
    x = BatchNormalization()(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(actions), activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)

def train_baseline(model, name, X_train, y_train, X_val, y_val, y_train_int, epochs=100):
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )
    
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor="val_categorical_accuracy", patience=20, 
                      restore_best_weights=True, mode='max', verbose=1),
        ModelCheckpoint(f"baseline_{name.lower().replace(' ', '_')}.h5", 
                       monitor="val_categorical_accuracy", 
                       save_best_only=True, mode="max", verbose=0)
    ]
    
    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    cw_dict = {i: cw[i] for i in range(len(cw))}
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=cw_dict,
        verbose=2
    )
    train_time = time.time() - start_time
    
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    return {
        "name": name,
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "train_time": train_time,
        "params": model.count_params(),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }

if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE MODELS TRAINING")
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
    
    baselines = [
        ("Simple LSTM", create_simple_lstm),
        ("Frame MLP", create_frame_mlp),
        ("Reduced BiLSTM", create_reduced_bilstm),
    ]
    
    results = []
    
    for name, create_fn in baselines:
        model = create_fn()
        result = train_baseline(model, name, X_train, y_train, X_val, y_val, y_train_int)
        results.append(result)
    
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} {'Val Accuracy':<15} {'Parameters':<15} {'Train Time':<15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<20} {r['val_accuracy']*100:>10.2f}%     {r['params']:>12,}    {r['train_time']:>10.1f}s")
    
    print("-" * 80)
    
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Baseline training complete! Results saved to baseline_results.json")
