"""
CG4002 B02 - Pipeline Health Check
====================================
Run this to verify all steps completed successfully.

Usage:
  cd cg4002-ai-demo
  python software/check_pipeline.py
"""

import os
import sys
import json
import numpy as np


def check(condition, msg_pass, msg_fail):
    """Print pass/fail check."""
    if condition:
        print(f"  ✓  {msg_pass}")
        return True
    else:
        print(f"  ✗  {msg_fail}")
        return False


def main():
    print("=" * 65)
    print("  CG4002 B02 — Pipeline Health Check")
    print("=" * 65)

    passed = 0
    failed = 0

    print("\n  Step 1: Dummy Data Generation")
    try:
        X_tr = np.load("data/train/X.npy")
        y_tr = np.load("data/train/y.npy")
        X_te = np.load("data/test/X.npy")
        y_te = np.load("data/test/y.npy")

        if check(len(X_tr.shape) == 3 and X_tr.shape[1] == 20 and X_tr.shape[2] == 12,
                 f"Train X shape: {X_tr.shape}",
                 f"Train X shape wrong: {X_tr.shape}, expected (N, 20, 12)"):
            passed += 1
        else: failed += 1

        if check(len(y_tr.shape) == 1 and y_tr.shape[0] == X_tr.shape[0],
                 f"Train y shape: {y_tr.shape}",
                 f"Train y shape wrong: {y_tr.shape}"):
            passed += 1
        else: failed += 1

        if check(len(X_te.shape) == 3 and X_te.shape[1] == 20 and X_te.shape[2] == 12,
                 f"Test X shape: {X_te.shape}",
                 f"Test X shape wrong: {X_te.shape}, expected (N, 20, 12)"):
            passed += 1
        else: failed += 1

        if check(len(np.unique(y_tr)) == 7,
                 f"Classes found: {len(np.unique(y_tr))} (expected 7)",
                 f"Wrong number of classes: {len(np.unique(y_tr))}"):
            passed += 1
        else: failed += 1

    except FileNotFoundError as e:
        print(f"  ✗  Data files not found: {e}")
        print(f"     Run: python software/generate_data.py")
        failed += 4

    print("\n  Step 2: Model Definition")
    try:
        sys.path.insert(0, "software")
        from model import ExerciseCNN
        import torch

        m = ExerciseCNN(num_features=12, num_classes=7)
        dummy = torch.randn(1, 20, 12)
        out = m(dummy)

        if check(out.shape == (1, 7),
                 f"Model output shape: {tuple(out.shape)}",
                 f"Wrong output shape: {tuple(out.shape)}"):
            passed += 1
        else: failed += 1

        if check(m.count_params() == 16295,
                 f"Parameter count: {m.count_params():,}",
                 f"Wrong param count: {m.count_params():,}, expected 16,295"):
            passed += 1
        else: failed += 1

    except Exception as e:
        print(f"  ✗  Model error: {e}")
        failed += 2

    print("\n  Step 3: Training")
    for f, desc in [
        ("models/best_model.pth", "Best model weights"),
        ("models/final_model.pth", "Final model weights"),
        ("models/exercise_cnn.onnx", "ONNX export"),
        ("models/training_curves.png", "Training curves plot"),
        ("models/norm_mean.npy", "Normalization mean"),
        ("models/norm_std.npy", "Normalization std"),
    ]:
        if check(os.path.exists(f), f"{desc}: {f}", f"MISSING: {f}"):
            passed += 1
        else: failed += 1

    try:
        with open("models/training_history.json") as fh:
            hist = json.load(fh)
        final_acc = hist["val_acc"][-1]
        if check(final_acc > 0.9,
                 f"Final val accuracy: {final_acc*100:.1f}%",
                 f"Low val accuracy: {final_acc*100:.1f}% (expected >90%)"):
            passed += 1
        else: failed += 1

        loss_decreased = hist["train_loss"][-1] < hist["train_loss"][0]
        if check(loss_decreased,
                 f"Loss decreased: {hist['train_loss'][0]:.4f} → {hist['train_loss'][-1]:.4f}",
                 f"Loss did NOT decrease!"):
            passed += 1
        else: failed += 1

    except FileNotFoundError:
        print(f"  ✗  MISSING: models/training_history.json")
        failed += 2

    print("\n  Step 4: Evaluation")
    for f, desc in [
        ("models/confusion_matrix.png", "Confusion matrix plot"),
        ("models/per_class_accuracy.png", "Per-class accuracy plot"),
        ("models/evaluation_results.json", "Evaluation results JSON"),
    ]:
        if check(os.path.exists(f), f"{desc}: {f}", f"MISSING: {f}"):
            passed += 1
        else: failed += 1

    try:
        with open("models/evaluation_results.json") as fh:
            res = json.load(fh)
        acc = res["overall_accuracy"]
        if check(acc > 0.9,
                 f"Overall accuracy: {acc*100:.1f}%",
                 f"Low accuracy: {acc*100:.1f}%"):
            passed += 1
        else: failed += 1

        cm = np.array(res["confusion_matrix"])
        diagonal_pct = cm.diagonal().sum() / cm.sum()
        if check(diagonal_pct > 0.9,
                 f"Confusion matrix diagonal: {diagonal_pct*100:.1f}%",
                 f"Confusion matrix off-diagonal too high"):
            passed += 1
        else: failed += 1

    except (FileNotFoundError, KeyError):
        print(f"  ✗  Cannot read evaluation results")
        failed += 2

    print("\n  Step 5: Quantization")
    quant_file = "models/quantized/quantization_results.json"
    if os.path.exists(quant_file):
        with open(quant_file) as fh:
            qr = json.load(fh)
        if check(qr["compression_ratio"] >= 3.5,
                 f"Compression ratio: {qr['compression_ratio']:.1f}× (float32 → INT8)",
                 f"Low compression: {qr['compression_ratio']:.1f}×"):
            passed += 1
        else: failed += 1

        drop = qr["accuracy_drop"] * 100
        if check(drop < 5.0,
                 f"Accuracy drop from quantization: {drop:.2f}%",
                 f"High accuracy drop: {drop:.2f}% (expected < 5%)"):
            passed += 1
        else: failed += 1
    else:
        print(f"  ⚠  {quant_file} not found (run quantize_demo.py)")
        print(f"     Not critical — check your .xmodel files from Docker instead")

    for f, desc in [
        ("models/quantized/weight_distributions.png", "Weight distribution plot"),
        ("models/quantized/quantization_error.png", "Quantization error plot"),
    ]:
        if check(os.path.exists(f), f"{desc}", f"MISSING: {f}"):
            passed += 1
        else: failed += 1

    print("\n  Step 6: Compilation")
    comp_file = "models/compiled/compilation_analysis.json"
    if os.path.exists(comp_file):
        with open(comp_file) as fh:
            ca = json.load(fh)
        if check(ca["dpu_coverage_pct"] > 80,
                 f"DPU coverage: {ca['dpu_coverage_pct']:.0f}%",
                 f"Low DPU coverage: {ca['dpu_coverage_pct']:.0f}%"):
            passed += 1
        else: failed += 1

        if check(ca["total_macs"] > 0,
                 f"Total MACs: {ca['total_macs']:,}",
                 f"Zero MACs — something wrong"):
            passed += 1
        else: failed += 1
    else:
        print(f"  ⚠  {comp_file} not found (run compile_demo.py)")

    print("\n  Vitis AI Docker Outputs (if applicable)")
    xmodel_locations = [
        "models/quantized/ExerciseCNN_int.xmodel",
        "models/compiled/exercise_cnn.xmodel",
    ]
    for f in xmodel_locations:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024
            print(f"  ✓  Found: {f} ({size:.1f} KB)")
            passed += 1
        else:
            print(f"  ⚠  Not found: {f} (expected if using Docker separately)")

    total = passed + failed
    print(f"\n  {'='*50}")
    print(f"  Results: {passed}/{total} checks passed", end="")
    if failed == 0:
        print(f"  ✓ ALL GOOD!")
    else:
        print(f"  ({failed} failed)")
    print(f"  {'='*50}")

    if failed == 0:
        print(f"\n  Pipeline is healthy! Ready for deployment.")
    else:
        print(f"\n  Fix the failed checks above before proceeding.")

    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
