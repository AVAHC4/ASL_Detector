import cv2
import numpy as np
import time
import json
from pathlib import Path
from statistics import mean, stdev

from tf_keras.models import model_from_json

import mediapipe as mp

mp_hands = mp.solutions.hands

WARMUP_ITERATIONS = 20
BENCHMARK_ITERATIONS = 200
SEQUENCE_LENGTH = 30

def percentile(data, p):
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]

def load_model(model_json_path, model_weights_path):
    with open(model_json_path, "r") as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_path)
    return model

def create_synthetic_frame(width=640, height=480):
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return frame

def create_synthetic_sequence(seq_len=30, feature_dim=63):
    return np.random.rand(1, seq_len, feature_dim).astype(np.float32)

def benchmark_mediapipe(hands, iterations):
    frame = create_synthetic_frame()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    for _ in range(WARMUP_ITERATIONS):
        hands.process(rgb_frame)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        hands.process(rgb_frame)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return times

def benchmark_model_inference(model, iterations):
    sequence = create_synthetic_sequence()
    
    for _ in range(WARMUP_ITERATIONS):
        model.predict(sequence, verbose=0)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.predict(sequence, verbose=0)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return times

def benchmark_full_pipeline(hands, model, iterations):
    frame = create_synthetic_frame()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sequence = create_synthetic_sequence()
    
    for _ in range(WARMUP_ITERATIONS):
        hands.process(rgb_frame)
        model.predict(sequence, verbose=0)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        hands.process(rgb_frame)
        model.predict(sequence, verbose=0)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return times

def print_stats(name, times):
    avg = mean(times)
    std = stdev(times) if len(times) > 1 else 0
    min_t = min(times)
    max_t = max(times)
    p95 = percentile(times, 95)
    fps = 1000 / avg if avg > 0 else 0
    
    return {
        "name": name,
        "mean_ms": round(avg, 2),
        "std_ms": round(std, 2),
        "min_ms": round(min_t, 2),
        "max_ms": round(max_t, 2),
        "p95_ms": round(p95, 2),
        "fps": round(fps, 1)
    }

def main():
    print("=" * 70)
    print("FPS / LATENCY BENCHMARK")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Warmup iterations:    {WARMUP_ITERATIONS}")
    print(f"  Benchmark iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Sequence length:      {SEQUENCE_LENGTH}")
    
    print("\nInitializing MediaPipe Hands...")
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6
    )
    
    print("Loading LSTM model...")
    
    model_configs = []
    
    if Path("model(0.2).json").exists() and Path("newmodel(0.2).h5").exists():
        model_configs.append(("BiLSTM+Attention", "model(0.2).json", "newmodel(0.2).h5"))
    
    if Path("model_signer_disjoint.json").exists() and Path("newmodel_signer_disjoint.h5").exists():
        model_configs.append(("Signer-Disjoint", "model_signer_disjoint.json", "newmodel_signer_disjoint.h5"))
    
    if Path("baseline_simple_lstm.h5").exists():
        model_configs.append(("Simple LSTM", "model(0.2).json", "baseline_simple_lstm.h5"))
    
    if Path("baseline_reduced_bilstm.h5").exists():
        model_configs.append(("Reduced BiLSTM", "model(0.2).json", "baseline_reduced_bilstm.h5"))
    
    results = []
    
    print("\n" + "-" * 70)
    print("MEDIAPIPE BENCHMARK")
    print("-" * 70)
    mp_times = benchmark_mediapipe(hands, BENCHMARK_ITERATIONS)
    mp_stats = print_stats("MediaPipe Hand Detection", mp_times)
    results.append(mp_stats)
    print(f"  Mean: {mp_stats['mean_ms']:.2f}ms | P95: {mp_stats['p95_ms']:.2f}ms | FPS: {mp_stats['fps']:.1f}")
    
    for model_name, json_path, weights_path in model_configs:
        if not Path(json_path).exists() or not Path(weights_path).exists():
            print(f"\nSkipping {model_name} (files not found)")
            continue
            
        print(f"\n{'-'*70}")
        print(f"MODEL BENCHMARK: {model_name}")
        print(f"{'-'*70}")
        
        model = load_model(json_path, weights_path)
        
        model_times = benchmark_model_inference(model, BENCHMARK_ITERATIONS)
        model_stats = print_stats(f"{model_name} - Inference", model_times)
        results.append(model_stats)
        print(f"  Mean: {model_stats['mean_ms']:.2f}ms | P95: {model_stats['p95_ms']:.2f}ms | FPS: {model_stats['fps']:.1f}")
        
        pipeline_times = benchmark_full_pipeline(hands, model, BENCHMARK_ITERATIONS)
        pipeline_stats = print_stats(f"{model_name} - Full Pipeline", pipeline_times)
        results.append(pipeline_stats)
        print(f"  Full Pipeline Mean: {pipeline_stats['mean_ms']:.2f}ms | FPS: {pipeline_stats['fps']:.1f}")
    
    hands.close()
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Component':<40} {'Mean (ms)':<12} {'P95 (ms)':<12} {'FPS':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<40} {r['mean_ms']:>8.2f}    {r['p95_ms']:>8.2f}    {r['fps']:>7.1f}")
    print("-" * 70)
    
    with open("latency_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Benchmark complete! Results saved to latency_benchmark.json")
    
    print("\n" + "=" * 70)
    print("RUNTIME TABLE (for paper)")
    print("=" * 70)
    print("""
| Component                | Mean (ms) | P95 (ms) | FPS    |
|--------------------------|-----------|----------|--------|""")
    for r in results:
        print(f"| {r['name']:<24} | {r['mean_ms']:>9.2f} | {r['p95_ms']:>8.2f} | {r['fps']:>6.1f} |")

if __name__ == "__main__":
    main()
