"""
CG4002 B02 - Step 5b: Quantization Demo (Standalone, No Docker)
================================================================
Demonstrates INT8 quantization concepts WITHOUT needing Vitis AI Docker.
Run this for your demo video to show:
  - How float32 → INT8 conversion works mathematically
  - The accuracy impact of quantization
  - Weight distribution before/after
  - Why this is needed for the DPU

This simulates what Vitis AI Quantizer does internally:
  1. Analyze weight/activation ranges (calibration)
  2. Compute scale factors per layer
  3. Quantize: int8_val = round(float_val / scale)
  4. Dequantize: float_val ≈ int8_val * scale
  5. Measure accuracy loss

Usage:
  cd cg4002-ai-demo
  python software/quantize_demo.py
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import ExerciseCNN

CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]


def compute_scale(tensor, num_bits=8):
    """
    Compute the quantization scale factor for a tensor.

    This is what the Vitis AI Quantizer does during calibration:
      scale = max(|tensor|) / (2^(bits-1) - 1)

    For INT8: range is [-128, 127], so max_int = 127
      scale = max(|tensor|) / 127

    Example: if weights range from -0.5 to +0.5
      scale = 0.5 / 127 = 0.00394
      A weight of 0.3 becomes: round(0.3 / 0.00394) = round(76.14) = 76
      Reconstructed: 76 * 0.00394 = 0.2994 (error = 0.0006)
    """
    max_val = tensor.abs().max().item()
    max_int = (2 ** (num_bits - 1)) - 1  # 127 for INT8
    if max_val == 0:
        return 1.0
    scale = max_val / max_int
    return scale


def quantize_tensor(tensor, scale, num_bits=8):
    """
    Quantize a float tensor to INT8.
      int8_val = clamp(round(float_val / scale), -128, 127)
    """
    max_int = (2 ** (num_bits - 1)) - 1    # 127
    min_int = -(2 ** (num_bits - 1))        # -128
    quantized = torch.clamp(torch.round(tensor / scale), min_int, max_int).to(torch.int8)
    return quantized


def dequantize_tensor(quantized, scale):
    """
    Dequantize INT8 back to float (with quantization error).
      float_val ≈ int8_val * scale
    """
    return quantized.float() * scale


def quantize_model(model):
    """
    Quantize all weights in the model to INT8.
    Returns a new model with quantized-then-dequantized weights
    (simulates what happens on the DPU).
    """
    quant_model = copy.deepcopy(model)
    quant_info = {}

    for name, param in quant_model.named_parameters():
        if param.requires_grad:
            scale = compute_scale(param.data)

            q = quantize_tensor(param.data, scale)
            param.data = dequantize_tensor(q, scale)

            orig_range = (param.data.min().item(), param.data.max().item())
            quant_info[name] = {
                "scale": scale,
                "original_range": orig_range,
                "num_elements": param.numel(),
                "size_float32_bytes": param.numel() * 4,
                "size_int8_bytes": param.numel() * 1,
            }

    return quant_model, quant_info


def evaluate_model(model, X_test, y_test):
    """Run inference and return accuracy + predictions."""
    model.eval()
    X_tensor = torch.from_numpy(X_test)
    with torch.no_grad():
        logits = model(X_tensor)
    preds = logits.argmax(dim=1).numpy()
    acc = (preds == y_test).mean()
    return acc, preds


def plot_weight_distributions(float_model, quant_model, save_path):
    """Show weight distributions before/after quantization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    layers_to_plot = [
        ("conv1.weight", "Conv1 (32 filters, k=9)"),
        ("conv2.weight", "Conv2 (64 filters, k=5)"),
        ("fc1.weight", "FC1 (64→32)"),
        ("fc2.weight", "FC2 (32→6)"),
        ("conv1.bias", "Conv1 Bias"),
        ("conv2.bias", "Conv2 Bias"),
    ]

    float_params = dict(float_model.named_parameters())
    quant_params = dict(quant_model.named_parameters())

    for idx, (name, title) in enumerate(layers_to_plot):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]

        f_data = float_params[name].data.cpu().numpy().flatten()
        q_data = quant_params[name].data.cpu().numpy().flatten()

        ax.hist(f_data, bins=50, alpha=0.6, color='blue', label='Float32', density=True)
        ax.hist(q_data, bins=50, alpha=0.6, color='red', label='INT8 (dequant)', density=True)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Weight Distributions: Float32 vs INT8 Quantized', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Weight distribution plot saved: {save_path}")


def plot_quantization_error(float_model, quant_model, save_path):
    """Plot per-layer quantization error."""
    float_params = dict(float_model.named_parameters())
    quant_params = dict(quant_model.named_parameters())

    layer_names = []
    errors = []

    for name in float_params:
        f = float_params[name].data.cpu().numpy().flatten()
        q = quant_params[name].data.cpu().numpy().flatten()
        mae = np.mean(np.abs(f - q))
        rel_error = mae / (np.abs(f).mean() + 1e-10) * 100

        layer_names.append(name.replace('.weight', '\n.weight').replace('.bias', '\n.bias'))
        errors.append(rel_error)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.RdYlGn_r(np.array(errors) / max(errors))
    bars = ax.bar(range(len(layer_names)), errors, color=colors, edgecolor='white')

    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, fontsize=8, rotation=0, ha='center')
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('Per-Layer Quantization Error (Float32 → INT8)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{err:.2f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Quantization error plot saved: {save_path}")


def main():
    print("=" * 65)
    print("  CG4002 B02 — Step 5b: Quantization Demo (INT8)")
    print("=" * 65)

    print("""
  ┌─ What is Quantization? ─────────────────────────────────┐
  │                                                          │
  │  Float32 (software):  32 bits per weight                │
  │  INT8 (FPGA DPU):      8 bits per weight                │
  │                                                          │
  │  Math for each weight:                                   │
  │    scale = max(|weights|) / 127                          │
  │    int8_val = round(float_val / scale)                   │
  │    reconstructed = int8_val × scale                      │
  │                                                          │
  │  Example: weight = 0.3, max = 0.5                        │
  │    scale = 0.5 / 127 = 0.00394                           │
  │    int8  = round(0.3 / 0.00394) = 76                     │
  │    recon = 76 × 0.00394 = 0.2994                         │
  │    error = 0.0006 (0.2%)                                 │
  │                                                          │
  │  Benefits: 4× smaller, 4× faster MACs on DPU            │
  │  Cost:     Small accuracy loss (typically < 1-2%)        │
  └──────────────────────────────────────────────────────────┘
    """)

    print("  Loading float32 model...")
    float_model = ExerciseCNN(num_features=18, num_classes=7)
    float_model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))
    float_model.eval()

    X_test = np.load("data/test/X.npy")
    y_test = np.load("data/test/y.npy")
    mean = np.load("models/norm_mean.npy")
    std  = np.load("models/norm_std.npy")
    X_test = ((X_test - mean) / std).astype(np.float32)

    print("  Evaluating float32 model...")
    float_acc, float_preds = evaluate_model(float_model, X_test, y_test)
    print(f"    Float32 accuracy: {float_acc*100:.1f}%")

    print("\n  Quantizing model to INT8...")
    quant_model, quant_info = quantize_model(float_model)

    print("  Evaluating INT8 quantized model...")
    quant_acc, quant_preds = evaluate_model(quant_model, X_test, y_test)
    print(f"    INT8 accuracy:    {quant_acc*100:.1f}%")
    print(f"    Accuracy drop:    {(float_acc - quant_acc)*100:.2f}%")

    total_float = 0
    total_int8 = 0
    layer_rows = []
    for name, info in quant_info.items():
        f_size = info['size_float32_bytes']
        i_size = info['size_int8_bytes']
        total_float += f_size
        total_int8 += i_size
        saving = (1 - i_size / f_size) * 100
        layer_rows.append([name, f"{info['scale']:.6f}", info['num_elements'],
                           f"{f_size} B", f"{i_size} B", f"{saving:.0f}%"])
    layer_rows.append(["TOTAL", "", "", f"{total_float} B", f"{total_int8} B",
                       f"{(1 - total_int8/total_float)*100:.0f}%"])

    print("\n  Per-layer quantization details:")
    print(tabulate(layer_rows,
                   headers=["Layer", "Scale", "Elements", "Float32", "INT8", "Saving"],
                   tablefmt="simple"))
    print(f"\n  Model size: {total_float/1024:.1f} KB → {total_int8/1024:.1f} KB "
          f"({total_float/total_int8:.1f}× compression)")

    mismatches = np.where(float_preds != quant_preds)[0]
    if len(mismatches) > 0:
        print(f"\n  Predictions changed by quantization: {len(mismatches)}/{len(y_test)}")
        for idx in mismatches[:10]:
            print(f"    Sample {idx}: float={CLASSES[float_preds[idx]]}, "
                  f"int8={CLASSES[quant_preds[idx]]}, true={CLASSES[y_test[idx]]}")
    else:
        print(f"\n  All predictions identical (0 mismatches out of {len(y_test)})")

    print("\n  Generating plots...")
    os.makedirs("models/quantized", exist_ok=True)
    plot_weight_distributions(float_model, quant_model, "models/quantized/weight_distributions.png")
    plot_quantization_error(float_model, quant_model, "models/quantized/quantization_error.png")

    results = {
        "float32_accuracy": float(float_acc),
        "int8_accuracy": float(quant_acc),
        "accuracy_drop": float(float_acc - quant_acc),
        "model_size_float32_bytes": total_float,
        "model_size_int8_bytes": total_int8,
        "compression_ratio": float(total_float / total_int8),
        "per_layer": {name: {"scale": info["scale"]} for name, info in quant_info.items()},
    }
    with open("models/quantized/quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n  Quantization Summary:")
    print(tabulate([
        ["Float32 Accuracy", f"{float_acc*100:.1f}%"],
        ["INT8 Accuracy",    f"{quant_acc*100:.1f}%"],
        ["Accuracy Drop",    f"{(float_acc-quant_acc)*100:.2f}%"],
        ["Model Size (F32)", f"{total_float} bytes ({total_float/1024:.1f} KB)"],
        ["Model Size (INT8)", f"{total_int8} bytes ({total_int8/1024:.1f} KB)"],
        ["Compression",      f"{total_float/total_int8:.1f}×"],
        ["Vitis AI Flow",    ".pth → vai_q_pytorch → .xmodel → DPU"],
        ["DPU target",       "DPUCZDX8G (Ultra96-V2, ZU3EG)"],
    ], tablefmt="simple"))

    print("  ✓ Step 5 complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
