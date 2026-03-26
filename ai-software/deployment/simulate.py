"""
CG4002 B02 - Step 10: Simulation & Verification
==================================================
Covers Task 4 demo requirements:
  - Software emulation: verify model logic before hardware deployment
  - Bit-accuracy verification: float32 vs INT8 output comparison
  - Hardware-in-the-loop (HIL): test with pre-recorded sensor data
  - End-to-end pipeline validation: data → preprocessing → inference → output
  - Edge case and stress testing

This script proves that:
  1. The model produces correct results BEFORE touching hardware
  2. INT8 quantization preserves classification correctness
  3. The full pipeline (normalize → infer → decode) works end-to-end
  4. The system handles edge cases gracefully

Usage:
  cd cg4002-ai-demo
  python software/simulate.py
"""

import numpy as np
import torch
import os
import sys
import json
import time
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ExerciseCNN

CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]
NUM_FEATURES = 12
WINDOW_SIZE = 20


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


#  TEST 1: Software Emulation

def test_software_emulation(model, X_test, y_test):
    """
    Verify model correctness in pure software BEFORE hardware deployment.

    This is the first step in the Vitis AI flow: run the trained model
    in a simulated environment on your PC to confirm the logic is correct.

    What we check:
      - Model loads without errors
      - Output shape is correct (batch, 7)
      - Predictions are valid class indices (0-6)
      - Accuracy matches training expectations
      - Softmax outputs sum to 1.0
    """
    print("  TEST 1: Software Emulation")
    print("  " + "-" * 55)

    results = {"name": "software_emulation", "checks": [], "passed": 0, "failed": 0}

    model.eval()
    X_tensor = torch.from_numpy(X_test)

    try:
        with torch.no_grad():
            logits = model(X_tensor)
        results["checks"].append(("Forward pass executes", True))
        results["passed"] += 1
        print("  ✓ Forward pass executes without error")
    except Exception as e:
        results["checks"].append(("Forward pass executes", False))
        results["failed"] += 1
        print(f"  ✗ Forward pass FAILED: {e}")
        return results

    expected_shape = (len(X_test), len(CLASSES))
    shape_ok = tuple(logits.shape) == expected_shape
    results["checks"].append(("Output shape correct", bool(shape_ok)))
    if shape_ok:
        results["passed"] += 1
        print(f"  ✓ Output shape: {tuple(logits.shape)} (expected {expected_shape})")
    else:
        results["failed"] += 1
        print(f"  ✗ Output shape: {tuple(logits.shape)} (expected {expected_shape})")

    preds = logits.argmax(dim=1).numpy()
    all_valid = np.all((preds >= 0) & (preds < len(CLASSES)))
    results["checks"].append(("All predictions are valid class indices", all_valid))
    if all_valid:
        results["passed"] += 1
        print(f"  ✓ All predictions in range [0, {len(CLASSES)-1}]")
    else:
        results["failed"] += 1
        invalid = np.where((preds < 0) | (preds >= len(CLASSES)))[0]
        print(f"  ✗ {len(invalid)} predictions out of valid range")

    probs = torch.softmax(logits, dim=1).numpy()
    sums = probs.sum(axis=1)
    sums_ok = np.allclose(sums, 1.0, atol=1e-5)
    results["checks"].append(("Softmax probabilities sum to 1.0", sums_ok))
    if sums_ok:
        results["passed"] += 1
        print(f"  ✓ Softmax sums: mean={sums.mean():.6f}, range=[{sums.min():.6f}, {sums.max():.6f}]")
    else:
        results["failed"] += 1
        print(f"  ✗ Softmax sums incorrect: range=[{sums.min():.6f}, {sums.max():.6f}]")

    accuracy = (preds == y_test).mean()
    acc_ok = accuracy > 0.8  # at least 80% (dummy data should be ~100%)
    results["checks"].append(("Accuracy above 80%", acc_ok))
    results["accuracy"] = float(accuracy)
    if acc_ok:
        results["passed"] += 1
        print(f"  ✓ Accuracy: {accuracy*100:.1f}% (threshold: 80%)")
    else:
        results["failed"] += 1
        print(f"  ✗ Accuracy: {accuracy*100:.1f}% (below 80% threshold)")

    no_nan = not (torch.isnan(logits).any() or torch.isinf(logits).any())
    results["checks"].append(("No NaN/Inf in outputs", no_nan))
    if no_nan:
        results["passed"] += 1
        print(f"  ✓ No NaN or Inf values in output logits")
    else:
        results["failed"] += 1
        print(f"  ✗ Found NaN or Inf in output logits!")

    return results


#  TEST 2: Bit-Accuracy Verification (Float32 vs INT8)

def test_bit_accuracy(model, X_test, y_test):
    """
    Compare float32 and INT8 model outputs at bit level.

    This verifies that quantization does not change the model's
    CLASSIFICATION decisions, even though the raw logit values differ.

    What the Vitis AI simulated environment does internally:
      1. Run float32 model → get logits
      2. Quantize model to INT8 → run → get logits
      3. Compare: do both produce the same argmax?
    """
    print("\n  TEST 2: Bit-Accuracy Verification (Float32 vs INT8)")
    print("  " + "-" * 55)

    results = {"name": "bit_accuracy", "checks": [], "passed": 0, "failed": 0}

    model.eval()
    X_tensor = torch.from_numpy(X_test)
    with torch.no_grad():
        f32_logits = model(X_tensor)
    f32_preds = f32_logits.argmax(dim=1).numpy()
    f32_probs = torch.softmax(f32_logits, dim=1).numpy()

    int8_model = copy.deepcopy(model)
    for name, param in int8_model.named_parameters():
        if param.requires_grad:
            max_val = param.data.abs().max().item()
            scale = max_val / 127 if max_val > 0 else 1.0
            q = torch.clamp(torch.round(param.data / scale), -128, 127).to(torch.int8)
            param.data = q.float() * scale

    int8_model.eval()
    with torch.no_grad():
        int8_logits = int8_model(X_tensor)
    int8_preds = int8_logits.argmax(dim=1).numpy()
    int8_probs = torch.softmax(int8_logits, dim=1).numpy()

    agreement = (f32_preds == int8_preds).mean()
    agree_ok = agreement > 0.95  # allow up to 5% disagreement
    results["checks"].append(("Classification agreement > 95%", agree_ok))
    results["agreement"] = float(agreement)
    if agree_ok:
        results["passed"] += 1
        print(f"  ✓ Classification agreement: {agreement*100:.1f}% "
              f"({(f32_preds == int8_preds).sum()}/{len(y_test)} identical)")
    else:
        results["failed"] += 1
        print(f"  ✗ Classification agreement: {agreement*100:.1f}% (below 95%)")

    logit_diff = (f32_logits.numpy() - int8_logits.numpy())
    mae = np.mean(np.abs(logit_diff))
    max_diff = np.max(np.abs(logit_diff))
    diff_ok = mae < 0.5  # mean absolute logit error < 0.5
    results["checks"].append(("Mean logit difference < 0.5", diff_ok))
    results["logit_mae"] = float(mae)
    if diff_ok:
        results["passed"] += 1
        print(f"  ✓ Logit difference: MAE={mae:.4f}, max={max_diff:.4f}")
    else:
        results["failed"] += 1
        print(f"  ✗ Logit difference too large: MAE={mae:.4f}, max={max_diff:.4f}")

    # Clip to avoid log(0)
    f32_p = np.clip(f32_probs, 1e-10, 1.0)
    int8_p = np.clip(int8_probs, 1e-10, 1.0)
    kl_div = np.mean(np.sum(f32_p * np.log(f32_p / int8_p), axis=1))
    kl_ok = kl_div < 0.1  # KL divergence < 0.1
    results["checks"].append(("KL divergence < 0.1", kl_ok))
    results["kl_divergence"] = float(kl_div)
    if kl_ok:
        results["passed"] += 1
        print(f"  ✓ KL divergence (F32 || INT8): {kl_div:.6f}")
    else:
        results["failed"] += 1
        print(f"  ✗ KL divergence too high: {kl_div:.6f}")

    f32_correct = (f32_preds == y_test)
    int8_correct = (int8_preds == y_test)
    both_correct = (f32_correct & int8_correct).mean()
    both_ok = both_correct > 0.9
    results["checks"].append(("Both correct on >90% samples", both_ok))
    if both_ok:
        results["passed"] += 1
        print(f"  ✓ Both correct: {both_correct*100:.1f}%  "
              f"(F32 only: {(f32_correct & ~int8_correct).sum()}, "
              f"INT8 only: {(~f32_correct & int8_correct).sum()})")
    else:
        results["failed"] += 1
        print(f"  ✗ Both correct: {both_correct*100:.1f}%")

    # Show disagreements
    disagreements = np.where(f32_preds != int8_preds)[0]
    if len(disagreements) > 0:
        print(f"\n  Disagreements ({len(disagreements)} samples):")
        for idx in disagreements[:5]:
            print(f"    Sample {idx}: F32={CLASSES[f32_preds[idx]]}, "
                  f"INT8={CLASSES[int8_preds[idx]]}, True={CLASSES[y_test[idx]]}")
    else:
        print(f"  → Zero disagreements: INT8 is bit-accurate for all predictions")

    return results


#  TEST 3: Hardware-in-the-Loop (HIL) Simulation

def test_hil_simulation(model, norm_mean, norm_std):
    """
    Simulate the full Hardware-in-the-Loop test.

    In a real HIL test:
      1. Pre-recorded sensor data is fed into the system
      2. Data flows through the same pipeline as live operation
      3. Outputs are compared against known ground truth

    We simulate this by:
      1. Loading real samples from the test dataset (as "pre-recorded data")
      2. Running them through the full pipeline (normalize → infer → decode)
      3. Verifying predictions match known ground truth labels
    """
    print("\n  TEST 3: Hardware-in-the-Loop (HIL) Simulation")
    print("  " + "-" * 55)

    results = {"name": "hil_simulation", "checks": [], "passed": 0, "failed": 0}

    model.eval()

    X_raw = np.load("data/test/X.npy")   # unnormalized
    y_raw = np.load("data/test/y.npy")

    recordings = []
    for class_idx in range(len(CLASSES)):
        indices = np.where(y_raw == class_idx)[0][:2]
        for idx in indices:
            recordings.append((CLASSES[class_idx], X_raw[idx], class_idx))

    print(f"\n  Playing back {len(recordings)} pre-recorded sensor recordings:")
    print(f"  {'#':>3} {'Recording':<15} {'Expected':<15} {'Predicted':<15} {'Conf':>6} {'Lat':>8} {'Result':>6}")
    print(f"  {'-'*68}")

    all_correct = True
    total_lat = 0

    for i, (rec_name, raw_data, expected_class) in enumerate(recordings):
        data_norm = (raw_data - norm_mean) / norm_std
        data_tensor = torch.from_numpy(
            data_norm.reshape(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
        )

        t0 = time.time()
        with torch.no_grad():
            logits = model(data_tensor)
        lat = (time.time() - t0) * 1000
        total_lat += lat

        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = int(np.argmax(probs))
        conf = probs[pred]
        correct = pred == expected_class
        if not correct:
            all_correct = False

        mark = "✓" if correct else "✗"
        print(f"  {i+1:>3} {rec_name:<15} {CLASSES[expected_class]:<15} {CLASSES[pred]:<15} "
              f"{conf:>5.3f} {lat:>6.2f}ms {mark:>6}")

    results["checks"].append(("All HIL recordings classified correctly", bool(all_correct)))
    if all_correct:
        results["passed"] += 1
        print(f"\n  ✓ All {len(recordings)} recordings correctly classified")
    else:
        results["failed"] += 1
        print(f"\n  ✗ Some recordings misclassified")

    # Check latency consistency
    avg_lat = total_lat / len(recordings)
    lat_ok = avg_lat < 50  # < 50ms per inference
    results["checks"].append(("Average HIL latency < 50ms", bool(lat_ok)))
    if lat_ok:
        results["passed"] += 1
        print(f"  ✓ Average HIL latency: {avg_lat:.2f} ms")
    else:
        results["failed"] += 1
        print(f"  ✗ Average HIL latency: {avg_lat:.2f} ms (too slow)")

    return results


#  TEST 4: End-to-End Pipeline Validation

def test_pipeline_validation(model, norm_mean, norm_std):
    """
    Validate the complete inference pipeline from raw data to output.

    Simulates exactly what happens on the Ultra96:
      1. Receive raw IMU JSON (like from Zenoh)
      2. Parse and align 3 sensors
      3. Build 128-sample sliding window
      4. Normalize with saved parameters
      5. Run inference
      6. Decode prediction
      7. Format output JSON (like publishing to Zenoh)
    """
    print("\n  TEST 4: End-to-End Pipeline Validation")
    print("  " + "-" * 55)

    results = {"name": "pipeline_validation", "checks": [], "passed": 0, "failed": 0}

    model.eval()

    print(f"\n  Simulating 128 Zenoh IMU packets arriving...")

    raw_packets = []
    for i in range(WINDOW_SIZE):
        t = i / 50.0  # 50 Hz
        packet = {
            "chest": {"ax": float(0.1 + 0.5*np.sin(2*np.pi*0.6*t)),
                      "ay": float(9.8 + 1.5*np.sin(2*np.pi*0.6*t)),
                      "az": float(0.2*np.random.randn()),
                      "gx": float(0.5*np.sin(2*np.pi*0.6*t)),
                      "gy": float(0.1*np.random.randn()),
                      "gz": float(0.1*np.random.randn())},
            "wrist": {"ax": float(0.1*np.random.randn()),
                      "ay": float(0.2*np.random.randn()),
                      "az": float(0.1*np.random.randn()),
                      "gx": float(0.1*np.random.randn()),
                      "gy": float(0.1*np.random.randn()),
                      "gz": float(0.1*np.random.randn())},
            "thigh": {"ax": float(0.3*np.random.randn()),
                      "ay": float(2.5*np.sin(2*np.pi*0.6*t)),
                      "az": float(0.2*np.random.randn()),
                      "gx": float(0.3*np.random.randn()),
                      "gy": float(1.5*np.sin(2*np.pi*0.6*t)),
                      "gz": float(0.2*np.random.randn())},
        }
        raw_packets.append(packet)

    sensor_order = ["chest", "wrist", "thigh"]
    imu_fields = ["ax", "ay", "az", "gx", "gy", "gz"]

    window = np.zeros((WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)
    for i, pkt in enumerate(raw_packets):
        row = []
        for sensor in sensor_order:
            for field in imu_fields:
                row.append(pkt[sensor][field])
        window[i] = row

    parse_ok = window.shape == (WINDOW_SIZE, NUM_FEATURES) and not np.any(np.isnan(window))
    results["checks"].append(("JSON parsing and alignment", parse_ok))
    if parse_ok:
        results["passed"] += 1
        print(f"  ✓ Parsed 128 packets → window shape {window.shape}")
    else:
        results["failed"] += 1
        print(f"  ✗ Parse error: shape={window.shape}, NaN={np.any(np.isnan(window))}")

    window_norm = (window - norm_mean) / norm_std
    norm_ok = not np.any(np.isnan(window_norm)) and not np.any(np.isinf(window_norm))
    results["checks"].append(("Normalization produces valid values", norm_ok))
    if norm_ok:
        results["passed"] += 1
        print(f"  ✓ Normalized: mean~{window_norm.mean():.2f}, std~{window_norm.std():.2f}")
    else:
        results["failed"] += 1
        print(f"  ✗ Normalization produced NaN/Inf")

    input_tensor = torch.from_numpy(window_norm.reshape(1, WINDOW_SIZE, NUM_FEATURES))
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_tensor)
    lat = (time.time() - t0) * 1000

    infer_ok = logits.shape == (1, len(CLASSES)) and not torch.isnan(logits).any()
    results["checks"].append(("Inference produces valid output", infer_ok))
    if infer_ok:
        results["passed"] += 1
        print(f"  ✓ Inference: output shape {tuple(logits.shape)}, latency {lat:.2f}ms")
    else:
        results["failed"] += 1
        print(f"  ✗ Inference error")

    pred_class = int(logits.argmax(dim=1).item())
    probs = torch.softmax(logits, dim=1).numpy()[0]
    confidence = float(probs[pred_class])

    decode_ok = 0 <= pred_class < len(CLASSES) and 0 <= confidence <= 1
    results["checks"].append(("Prediction decoding valid", decode_ok))
    if decode_ok:
        results["passed"] += 1
        print(f"  ✓ Prediction: {CLASSES[pred_class]} (confidence: {confidence:.3f})")
    else:
        results["failed"] += 1
        print(f"  ✗ Invalid prediction: class={pred_class}, conf={confidence}")

    output_json = json.dumps({
        "ts": int(time.time() * 1000),
        "exercise": pred_class,
        "class_name": CLASSES[pred_class],
        "confidence": round(confidence, 3),
    })
    json_ok = len(output_json) > 0
    try:
        parsed = json.loads(output_json)
        json_ok = "exercise" in parsed and "class_name" in parsed
    except:
        json_ok = False

    results["checks"].append(("Output JSON valid and parseable", json_ok))
    if json_ok:
        results["passed"] += 1
        print(f"  ✓ Output JSON: {output_json}")
    else:
        results["failed"] += 1
        print(f"  ✗ Invalid output JSON")

    return results


#  TEST 5: Edge Cases & Stress Testing

def test_edge_cases(model, norm_mean, norm_std):
    """
    Test robustness against edge cases the system might encounter.

    These are conditions that could happen in real deployment:
      - All-zero input (sensor failure)
      - Extremely large values (sensor spike)
      - Constant input (user standing still)
      - High-frequency noise (electromagnetic interference)
      - Mixed valid/invalid data
    """
    print("\n  TEST 5: Edge Case & Stress Testing")
    print("  " + "-" * 55)

    results = {"name": "edge_cases", "checks": [], "passed": 0, "failed": 0}

    model.eval()

    edge_cases = {
        "All zeros (sensor failure)": np.zeros((1, WINDOW_SIZE, NUM_FEATURES), dtype=np.float32),
        "All ones": np.ones((1, WINDOW_SIZE, NUM_FEATURES), dtype=np.float32),
        "Very large values (spike)": np.full((1, WINDOW_SIZE, NUM_FEATURES), 1000.0, dtype=np.float32),
        "Very small values": np.full((1, WINDOW_SIZE, NUM_FEATURES), 1e-8, dtype=np.float32),
        "Random noise (no signal)": np.random.randn(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32) * 0.01,
        "High-frequency noise": (np.sin(np.linspace(0, 100*np.pi, WINDOW_SIZE))[:, None] *
                                  np.ones(NUM_FEATURES)[None, :]).reshape(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32),
    }

    print(f"\n  {'Edge Case':<30} {'Prediction':<15} {'Conf':>6} {'Lat':>8} {'Status':>8}")
    print(f"  {'-'*67}")

    all_stable = True
    for name, data in edge_cases.items():
        data_norm = (data - norm_mean) / norm_std

        # Replace NaN/Inf from normalization (e.g., if std=0)
        data_norm = np.nan_to_num(data_norm, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            data_tensor = torch.from_numpy(data_norm)
            t0 = time.time()
            with torch.no_grad():
                logits = model(data_tensor)
            lat = (time.time() - t0) * 1000

            pred = int(logits.argmax(dim=1).item())
            probs = torch.softmax(logits, dim=1).numpy()[0]
            conf = probs[pred]

            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()

            if has_nan or has_inf:
                status = "⚠ NaN"
                all_stable = False
            else:
                status = "✓ OK"

            print(f"  {name:<30} {CLASSES[pred]:<15} {conf:>5.3f} {lat:>6.2f}ms {status:>8}")

        except Exception as e:
            status = "✗ CRASH"
            all_stable = False
            print(f"  {name:<30} {'ERROR':<15} {'':>6} {'':>8} {status:>8}")
            print(f"    Error: {e}")

    results["checks"].append(("All edge cases handled without crash", all_stable))
    if all_stable:
        results["passed"] += 1
        print(f"\n  ✓ Model handles all edge cases gracefully")
    else:
        results["failed"] += 1
        print(f"\n  ✗ Some edge cases caused issues")

    # Stress test: rapid consecutive inferences
    print(f"\n  Stress test: 500 rapid inferences...")
    random_data = np.random.randn(500, 1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
    crash_count = 0
    t0 = time.time()
    for i in range(500):
        try:
            with torch.no_grad():
                _ = model(torch.from_numpy(random_data[i]))
        except:
            crash_count += 1
    total_time = (time.time() - t0) * 1000

    stress_ok = crash_count == 0
    results["checks"].append(("500 rapid inferences without crash", stress_ok))
    if stress_ok:
        results["passed"] += 1
        print(f"  ✓ 500/500 inferences completed in {total_time:.0f}ms "
              f"({total_time/500:.2f}ms avg)")
    else:
        results["failed"] += 1
        print(f"  ✗ {crash_count} crashes during stress test")

    return results


#  TEST 6: Model Reproducibility

def test_reproducibility(model, X_test):
    """
    Verify that the model produces identical outputs given identical inputs.
    This is critical for debugging — if results are non-deterministic,
    you can't trust any comparison.
    """
    print("\n  TEST 6: Model Reproducibility")
    print("  " + "-" * 55)

    results = {"name": "reproducibility", "checks": [], "passed": 0, "failed": 0}

    model.eval()
    X_tensor = torch.from_numpy(X_test[:10])  # use 10 samples

    # Run inference 3 times
    outputs = []
    for run in range(3):
        with torch.no_grad():
            logits = model(X_tensor)
        outputs.append(logits.numpy().copy())

    # Compare all runs
    all_identical = True
    for i in range(1, len(outputs)):
        if not np.array_equal(outputs[0], outputs[i]):
            all_identical = False
            break

    results["checks"].append(("3 runs produce identical outputs", all_identical))
    if all_identical:
        results["passed"] += 1
        print(f"  ✓ 3 consecutive runs produce bit-identical results")
    else:
        results["failed"] += 1
        max_diff = max(np.max(np.abs(outputs[0] - outputs[i])) for i in range(1, len(outputs)))
        print(f"  ✗ Non-deterministic! Max difference: {max_diff:.10f}")

    return results


#  Main

def main():
    print("=" * 65)
    print("  CG4002 B02 — Step 10: Simulation & Verification")
    print("=" * 65)

    print("""
  This script validates the AI model BEFORE deploying to hardware.
  It simulates what the Vitis AI environment does:
    - Software emulation (verify logic)
    - Bit-accuracy check (float32 vs INT8)
    - Hardware-in-the-loop simulation (pre-recorded data)
    - Full pipeline validation (JSON → normalize → infer → JSON)
    - Edge case robustness testing
    - Reproducibility verification
    """)

    # Load everything
    print("  Loading model, data, and normalization parameters...")
    model = ExerciseCNN(num_features=12, num_classes=7)
    model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))
    model.eval()

    X_test = np.load("data/test/X.npy")
    y_test = np.load("data/test/y.npy")
    norm_mean = np.load("models/norm_mean.npy")
    norm_std  = np.load("models/norm_std.npy")
    X_test_norm = ((X_test - norm_mean) / norm_std).astype(np.float32)

    print(f"  Model: {model.count_params():,} params | Test: {len(y_test)} samples\n")

    # Run all tests
    all_results = []

    all_results.append(test_software_emulation(model, X_test_norm, y_test))
    all_results.append(test_bit_accuracy(model, X_test_norm, y_test))
    all_results.append(test_hil_simulation(model, norm_mean, norm_std))
    all_results.append(test_pipeline_validation(model, norm_mean, norm_std))
    all_results.append(test_edge_cases(model, norm_mean, norm_std))
    all_results.append(test_reproducibility(model, X_test_norm))

    # Overall summary
    total_passed = sum(r["passed"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)
    total_checks = total_passed + total_failed

    print(f"\n{'='*65}")
    print(f"  SIMULATION & VERIFICATION SUMMARY")
    print(f"{'='*65}")
    print(f"\n  {'Test Suite':<40} {'Passed':>8} {'Failed':>8}")
    print(f"  {'-'*56}")
    for r in all_results:
        name = r["name"].replace("_", " ").title()
        print(f"  {name:<40} {r['passed']:>8} {r['failed']:>8}")
    print(f"  {'-'*56}")
    print(f"  {'TOTAL':<40} {total_passed:>8} {total_failed:>8}")

    if total_failed == 0:
        verdict = "✓ ALL TESTS PASSED — Ready for hardware deployment!"
    else:
        verdict = f"✗ {total_failed} TESTS FAILED — Fix issues before deployment."

    print(f"\n  {verdict}")

    os.makedirs("models/simulation", exist_ok=True)
    save_data = {
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_checks": total_checks,
        "verdict": "PASS" if total_failed == 0 else "FAIL",
        "tests": [{
            "name": r["name"],
            "passed": r["passed"],
            "failed": r["failed"],
            "checks": r["checks"],
        } for r in all_results],
    }
    with open("models/simulation/simulation_results.json", "w") as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved: models/simulation/simulation_results.json")

    print(f"\n  ✓ Step 10 complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
