"""
CG4002 B02 - Step 5a: Vitis AI Quantization (INT8)
====================================================
Converts the float32 PyTorch model to INT8 using Vitis AI Quantizer.
This is required because the DPU on Ultra96 only operates on INT8.

What this does:
  float32 weights (32 bits per value) → INT8 weights (8 bits per value)
  - 4x smaller model
  - 4x faster MAC operations on DPU
  - Slight accuracy loss (typically < 1-2%)

Prerequisites:
  Must run inside Vitis AI Docker container:
    $ docker pull xilinx/vitis-ai-pytorch-cpu:latest
    $ docker run -it -v $(pwd):/workspace xilinx/vitis-ai-pytorch-cpu:latest
    (inside container)$ conda activate vitis-ai-pytorch
    (inside container)$ python software/quantize.py

How Vitis AI Quantizer works:
  1. Takes your trained float32 model
  2. Runs a "calibration" dataset through the model
  3. Measures the actual range of activations at each layer
  4. Maps float32 ranges to INT8 [-128, 127] with minimal loss
  5. Outputs a quantized model ready for the Vitis AI Compiler

Usage:
  python software/quantize.py
"""

import os
import sys
import textwrap
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import ExerciseCNN

CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]


def get_calibration_dataloader(num_samples=100, batch_size=1):
    """
    Load a small subset of training data for calibration.
    The quantizer needs to see real-ish data to determine
    the activation ranges at each layer.
    """
    X = np.load("data/train/X.npy")
    y = np.load("data/train/y.npy")

    mean = np.load("models/norm_mean.npy")
    std  = np.load("models/norm_std.npy")
    X = ((X - mean) / std).astype(np.float32)

    # Take a small balanced subset for calibration
    indices = []
    per_class = num_samples // len(CLASSES)
    for cls_idx in range(len(CLASSES)):
        cls_indices = np.where(y == cls_idx)[0][:per_class]
        indices.extend(cls_indices)

    X_cal = X[indices]
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_cal))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def quantize_pytorch():
    """
    Quantize using Vitis AI PyTorch Quantizer (vai_q_pytorch).
    This is the recommended method for PyTorch models.
    """
    from pytorch_nndct.apis import torch_quantizer

    print("  Loading trained float32 model...")
    model = ExerciseCNN(num_features=12, num_classes=7)
    model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))
    model.eval()

    # Dummy input for tracing (must match real input shape)
    dummy_input = torch.randn(1, 20, 12)

    print("  Initializing Vitis AI Quantizer...")
    print("    Mode: 'calib' (calibration)")
    print("    Target: DPUCZDX8G (Ultra96 DPU)")

    # Create quantizer
    # quant_mode='calib' → calibration pass to determine ranges
    quantizer = torch_quantizer(
        quant_mode='calib',
        module=model,
        input_args=(dummy_input,),
        output_dir='models/quantized',
        bitwidth=8,                              # INT8
        device=torch.device('cpu'),
    )

    # Get the quantized (wrapped) model
    quant_model = quantizer.quant_model

    print("\n  Running calibration (100 samples)...")
    cal_loader = get_calibration_dataloader(num_samples=100, batch_size=1)

    quant_model.eval()
    with torch.no_grad():
        for i, (batch,) in enumerate(cal_loader):
            quant_model(batch)
            if (i + 1) % 20 == 0:
                print(f"    Calibrated {i+1} samples...")

    print("\n  Exporting calibration results...")
    quantizer.export_quant_config()
    print("    → models/quantized/quant_info.json")

    print("\n  Running quantized inference test...")
    quantizer_test = torch_quantizer(
        quant_mode='test',
        module=model,
        input_args=(dummy_input,),
        output_dir='models/quantized',
        bitwidth=8,
        device=torch.device('cpu'),
    )

    quant_model_test = quantizer_test.quant_model
    quant_model_test.eval()

    X_test = np.load("data/test/X.npy")
    y_test = np.load("data/test/y.npy")
    mean = np.load("models/norm_mean.npy")
    std  = np.load("models/norm_std.npy")
    X_test = ((X_test - mean) / std).astype(np.float32)

    X_tensor = torch.from_numpy(X_test)
    correct = 0
    with torch.no_grad():
        for i in range(len(X_test)):
            out = quant_model_test(X_tensor[i:i+1])
            pred = out.argmax(dim=1).item()
            if pred == y_test[i]:
                correct += 1

    quant_acc = correct / len(y_test)
    print(f"    Quantized (INT8) accuracy: {quant_acc*100:.1f}%")

    print("\n  Exporting quantized model for Vitis AI Compiler...")
    quantizer_test.export_xmodel(output_dir='models/quantized', deploy_check=False)
    print("    → models/quantized/ExerciseCNN_int.xmodel")

    return quant_acc


def quantize_onnx():
    """
    Alternative: Quantize the ONNX model using vai_q_onnx.
    Use this if you prefer working with ONNX format.
    """
    from vaitrace_nndct.apis import vai_q_onnx

    print("  Loading ONNX model...")
    import onnx

    model_path = "models/exercise_cnn.onnx"

    print("  Running Vitis AI ONNX Quantizer...")

    X = np.load("data/train/X.npy")[:100]
    mean = np.load("models/norm_mean.npy")
    std  = np.load("models/norm_std.npy")
    X = ((X - mean) / std).astype(np.float32)

    vai_q_onnx.quantize_static(
        model_input=model_path,
        model_output="models/quantized/exercise_cnn_int8.onnx",
        calibration_data_reader=X,
        quant_format="QDQ",           # Quantize-Dequantize format
        calibrate_method="MinMax",
        activation_type="int8",
        weight_type="int8",
    )

    print("    → models/quantized/exercise_cnn_int8.onnx")


def main():
    print("=" * 65)
    print("  CG4002 B02 — Step 5a: Vitis AI Quantization (INT8)")
    print("=" * 65)
    print()
    print("  Pipeline: float32 (.pth) → INT8 (.xmodel input)")
    print("  Target:   DPUCZDX8G on Ultra96-V2 (ZU3EG)")
    print()

    os.makedirs("models/quantized", exist_ok=True)

    try:
        print("  Attempting PyTorch quantization (vai_q_pytorch)...")
        quant_acc = quantize_pytorch()
        print(f"\n  ✓ Quantization complete!")
        print(f"    Float32 accuracy: see models/evaluation_results.json")
        print(f"    INT8 accuracy:    {quant_acc*100:.1f}%")

    except ImportError:
        print("  vai_q_pytorch not available.")
        print()
        try:
            print("  Attempting ONNX quantization (vai_q_onnx)...")
            quantize_onnx()
            print("\n  ✓ ONNX quantization complete!")

        except ImportError:
            sys.exit(textwrap.dedent("""\
                Vitis AI not detected. To run this script, use the Vitis AI Docker:

                  docker pull xilinx/vitis-ai-pytorch-cpu
                  docker run -it -v $(pwd):/workspace xilinx/vitis-ai-pytorch-cpu:latest
                  conda activate vitis-ai-pytorch
                  cd /workspace
                  python software/quantize.py

                Or run the standalone demo (no Docker needed):
                  python software/quantize_demo.py
            """))

    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
