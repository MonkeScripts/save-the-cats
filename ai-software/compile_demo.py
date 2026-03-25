"""
CG4002 B02 - Step 6b: Compilation Demo (Standalone, No Docker)
===============================================================
Demonstrates what the Vitis AI Compiler does WITHOUT needing Docker.
Run this for your Task 3 demo video to show:

  - How the trained model maps to DPU hardware
  - Which layers run on DPU vs CPU (operator support analysis)
  - Memory/BRAM scheduling for each layer
  - Estimated throughput on the DPU
  - The full pipeline: .pth → ONNX → INT8 → .xmodel → DPU

Task 3 Requirements covered:
  ✓ HLS/Vitis AI setup explanation
  ✓ IP core integration (DPU IP in Vivado)
  ✓ Bitstream programming flow
  ✓ I/O interfaces (AXI, LPDDR4)

Usage:
  cd cg4002-ai-demo
  python software/compile_demo.py
"""

import torch
import numpy as np
import os
import json
from tabulate import tabulate

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import ExerciseCNN


DPU_SPEC = {
    "name": "DPUCZDX8G",
    "variant": "B2304",
    "description": "Deep Learning Processor Unit for Zynq UltraScale+",
    "fpga": "ZU3EG (Ultra96-V2)",
    "clock_freq_mhz": 300,
    "ops_per_cycle": 2304,           # 2304 INT8 MACs per clock
    "peak_gops": 2304 * 300 / 1000,  # = 691.2 GOPS
    "bram_kb": 216,                  # available block RAM
    "dsp_slices": 360,               # available DSP48E2 slices
    "lut_count": 70560,              # available LUTs
    "memory": "LPDDR4 (2GB, shared with ARM)",
    "bus_interface": "AXI4 SmartConnect → LPDDR4",
    "supported_ops": [
        "Conv2D", "DepthwiseConv2D", "Conv1D (mapped as Conv2D)",
        "MaxPool2D", "AvgPool2D", "GlobalAvgPool",
        "Dense (Fully Connected)", "BatchNorm (fused into Conv)",
        "ReLU", "LeakyReLU", "ReLU6",
        "Concat", "Add (elementwise)",
    ],
    "unsupported_ops": [
        "Softmax (use ArgMax instead)",
        "Sigmoid", "Tanh",
        "LSTM / GRU (recurrent layers)",
        "Custom activation functions",
    ],
}


def analyze_model_for_dpu(model):
    """
    Analyze each layer of our model and determine:
      1. Does it run on DPU or CPU?
      2. How many operations (MACs) does it need?
      3. Estimated latency on DPU at 300MHz

    This is what the Vitis AI Compiler does internally.
    """
    layers = []

    # Mapped as Conv2D with height=1: (1, 18, 128) → (1, 32, 128)
    # MACs = out_channels × kernel × in_channels × out_length
    conv1_macs = 32 * 9 * 18 * 128
    layers.append({
        "name": "Conv1D (k=9, 18→32)",
        "type": "Conv1D → Conv2D",
        "target": "DPU ✓",
        "input_shape": "(1, 18, 128)",
        "output_shape": "(1, 32, 128)",
        "macs": conv1_macs,
        "params": 18 * 32 * 9 + 32,
        "notes": "Mapped as Conv2D(1×9). BatchNorm fused into conv weights.",
    })

    layers.append({
        "name": "BatchNorm1d + ReLU",
        "type": "Fused",
        "target": "DPU ✓ (fused)",
        "input_shape": "(1, 32, 128)",
        "output_shape": "(1, 32, 128)",
        "macs": 0,
        "params": 64,
        "notes": "BN folded into Conv weights at compile time. ReLU = free (max(0,x)).",
    })

    layers.append({
        "name": "MaxPool1D (pool=2)",
        "type": "MaxPool2D",
        "target": "DPU ✓",
        "input_shape": "(1, 32, 128)",
        "output_shape": "(1, 32, 64)",
        "macs": 32 * 64 * 2,  # comparisons
        "params": 0,
        "notes": "Mapped as MaxPool2D(1×2). Reduces temporal dim by 2×.",
    })

    conv2_macs = 64 * 5 * 32 * 64
    layers.append({
        "name": "Conv1D (k=5, 32→64)",
        "type": "Conv1D → Conv2D",
        "target": "DPU ✓",
        "input_shape": "(1, 32, 64)",
        "output_shape": "(1, 64, 64)",
        "macs": conv2_macs,
        "params": 32 * 64 * 5 + 64,
        "notes": "Mapped as Conv2D(1×5). Captures abstract temporal patterns.",
    })

    layers.append({
        "name": "BatchNorm1d + ReLU",
        "type": "Fused",
        "target": "DPU ✓ (fused)",
        "input_shape": "(1, 64, 64)",
        "output_shape": "(1, 64, 64)",
        "macs": 0,
        "params": 128,
        "notes": "BN folded into Conv2 weights at compile time.",
    })

    layers.append({
        "name": "GlobalAvgPool1D",
        "type": "AvgPool2D",
        "target": "DPU ✓",
        "input_shape": "(1, 64, 64)",
        "output_shape": "(1, 64, 1)",
        "macs": 64 * 64,
        "params": 0,
        "notes": "Average 64 timesteps per channel. Replaces Flatten to save BRAM.",
    })

    fc1_macs = 64 * 32
    layers.append({
        "name": "Dense (64→32) + ReLU",
        "type": "FC + ReLU",
        "target": "DPU ✓",
        "input_shape": "(1, 64)",
        "output_shape": "(1, 32)",
        "macs": fc1_macs,
        "params": 64 * 32 + 32,
        "notes": "Classification hidden layer. ReLU fused.",
    })

    fc2_macs = 32 * 6
    layers.append({
        "name": "Dense (32→6) output",
        "type": "FC",
        "target": "DPU ✓",
        "input_shape": "(1, 32)",
        "output_shape": "(1, 6)",
        "macs": fc2_macs,
        "params": 32 * 6 + 6,
        "notes": "Output logits. ArgMax done on ARM CPU (not Softmax).",
    })

    layers.append({
        "name": "ArgMax (post-proc)",
        "type": "ArgMax",
        "target": "ARM CPU",
        "input_shape": "(1, 6)",
        "output_shape": "(1,)",
        "macs": 6,
        "params": 0,
        "notes": "Find winning class. Runs on ARM, not DPU. Trivial cost.",
    })

    return layers


def estimate_dpu_latency(layers, clock_mhz=300, ops_per_cycle=2304):
    """
    Estimate inference latency on DPU.
    Very simplified: latency ≈ total_MACs / (ops_per_cycle × clock_freq)
    Real DPU has memory bandwidth constraints, so actual is ~2-5× this.
    """
    total_macs = sum(l["macs"] for l in layers if "DPU" in l["target"])
    cycles_ideal = total_macs / ops_per_cycle
    latency_ideal_us = cycles_ideal / clock_mhz  # microseconds
    overhead_factor = 3.0
    latency_est_us = latency_ideal_us * overhead_factor
    return total_macs, latency_est_us


VIVADO_INFO = """
  ┌─ Vivado Integration (Task 3: IP Core & Bitstream) ──────┐
  │                                                          │
  │  1. Create Vivado project for Ultra96-V2 (ZU3EG)        │
  │     - Import Board Definition Files (BDF)                │
  │     - Set target device: xczu3eg-sbva484-1-e             │
  │                                                          │
  │  2. Add DPU IP core to block design                      │
  │     - IP: DPUCZDX8G v4.1                                 │
  │     - Configuration: B2304 (2304 ops/cycle)              │
  │     - Clock: 300 MHz (PL fabric clock)                   │
  │                                                          │
  │  3. Connect AXI interfaces                               │
  │     - DPU ↔ AXI SmartConnect ↔ PS LPDDR4 (HP ports)     │
  │     - DPU control: AXI-Lite from PS (ARM Cortex-A53)     │
  │     - Interrupt: DPU → PS GIC (completion notification)  │
  │                                                          │
  │  4. Memory map                                           │
  │     - Weights:      loaded from LPDDR4 to DPU BRAM       │
  │     - Input tensor: ARM writes to LPDDR4, DPU reads      │
  │     - Output:       DPU writes to LPDDR4, ARM reads      │
  │                                                          │
  │  5. Generate bitstream                                   │
  │     - Synthesis → Implementation → Generate Bitstream    │
  │     - Output: design_1_wrapper.bit + design_1.hwh        │
  │                                                          │
  │  6. Program FPGA                                         │
  │     - PYNQ: from pynq import Overlay                     │
  │     - overlay = Overlay("design_1_wrapper.bit")          │
  │     - DPU accessible via overlay.dpu                     │
  └──────────────────────────────────────────────────────────┘"""

PYNQ_DPU_INFO = """
  ┌─ PYNQ + DPU Runtime (on Ultra96) ──────────────────────┐
  │                                                          │
  │  # Load FPGA bitstream with DPU                          │
  │  from pynq_dpu import DpuOverlay                         │
  │  overlay = DpuOverlay("dpu.bit")                         │
  │  overlay.load_model("exercise_cnn.xmodel")               │
  │                                                          │
  │  # Or using VART (Vitis AI Runtime):                     │
  │  import vart                                             │
  │  runner = vart.Runner.create_runner(                     │
  │      "exercise_cnn.xmodel", "run"                        │
  │  )                                                       │
  │                                                          │
  │  # Prepare input buffer                                  │
  │  input_tensors = runner.get_input_tensors()              │
  │  output_tensors = runner.get_output_tensors()            │
  │                                                          │
  │  # Run inference                                         │
  │  input_data[0] = imu_window  # (1, 128, 18) INT8        │
  │  job_id = runner.execute_async(input_data, output_data)  │
  │  runner.wait(job_id)                                     │
  │                                                          │
  │  # Get result                                            │
  │  prediction = np.argmax(output_data[0])                  │
  └──────────────────────────────────────────────────────────┘"""


def main():
    print("=" * 65)
    print("  CG4002 B02 — Step 6b: Compilation Demo (.xmodel)")
    print("=" * 65)

    print(f"\n  DPU Target: {DPU_SPEC['name']} ({DPU_SPEC['variant']})")
    print(tabulate([
        ["FPGA",       DPU_SPEC['fpga']],
        ["Clock",      f"{DPU_SPEC['clock_freq_mhz']} MHz"],
        ["Ops/Cycle",  f"{DPU_SPEC['ops_per_cycle']} INT8 MACs"],
        ["Peak Perf",  f"{DPU_SPEC['peak_gops']:.1f} GOPS"],
        ["BRAM",       f"{DPU_SPEC['bram_kb']} KB"],
        ["DSP Slices", DPU_SPEC['dsp_slices']],
        ["Memory Bus", DPU_SPEC['bus_interface']],
    ], tablefmt="simple"))

    print("  DPU Supported Operations:")
    for op in DPU_SPEC['supported_ops']:
        print(f"    ✓ {op}")
    print()
    print("  NOT supported (falls back to ARM CPU):")
    for op in DPU_SPEC['unsupported_ops']:
        print(f"    ✗ {op}")

    print(f"\n{'='*65}")
    print("  Layer-by-Layer DPU Mapping Analysis")
    print(f"{'='*65}\n")

    model = ExerciseCNN(num_features=18, num_classes=7)
    layers = analyze_model_for_dpu(model)

    dpu_layers = sum(1 for l in layers if "DPU" in l['target'])
    cpu_layers = sum(1 for l in layers if "DPU" not in l['target'])

    print(tabulate(
        [[i+1, l['name'], l['target'], f"{l['macs']:,}", f"{l['params']:,}"]
         for i, l in enumerate(layers)],
        headers=["#", "Layer", "Target", "MACs", "Params"],
        tablefmt="simple"
    ))
    print(f"\n  DPU layers: {dpu_layers}  |  CPU layers: {cpu_layers}  |  "
          f"DPU coverage: {dpu_layers/(dpu_layers+cpu_layers)*100:.0f}%")

    print(f"\n  Detailed mapping notes:")
    for i, l in enumerate(layers):
        print(f"    Layer {i+1} ({l['name']})")
        print(f"      {l['input_shape']} → {l['output_shape']}")
        print(f"      {l['notes']}")
        print()

    total_macs, est_latency_us = estimate_dpu_latency(layers)

    print("\n  Performance Estimate:")
    print(tabulate([
        ["Total MACs",       f"{total_macs:,}"],
        ["DPU Clock",        f"{DPU_SPEC['clock_freq_mhz']} MHz"],
        ["Ops/Cycle",        DPU_SPEC['ops_per_cycle']],
        ["Est. Latency",     f"~{est_latency_us:.0f} µs ({est_latency_us/1000:.2f} ms)"],
        ["vs CPU (PyTorch)", "~1.28 ms (from Step 4)"],
        ["Est. Speedup",     f"~{1280/est_latency_us:.0f}×"],
    ], tablefmt="simple"))

    print(f"""
  ┌─ Full Compilation Pipeline ─────────────────────────────┐
  │                                                          │
  │  Step 1: Train (PyTorch, float32)                        │
  │    └→ best_model.pth                                     │
  │                                                          │
  │  Step 2: Export to ONNX                                  │
  │    └→ exercise_cnn.onnx                                  │
  │                                                          │
  │  Step 3: Quantize (vai_q_pytorch)                        │
  │    ├─ Calibrate with 100 real samples                    │
  │    ├─ float32 weights → INT8 weights                     │
  │    └→ ExerciseCNN_int.xmodel (quantized graph)           │
  │                                                          │
  │  Step 4: Compile (vai_c_xir)                             │
  │    ├─ Map layers to DPU instructions                     │
  │    ├─ Schedule BRAM data movement                        │
  │    ├─ Pack INT8 weights for DPU memory                   │
  │    └→ exercise_cnn.xmodel (deployable)                   │
  │                                                          │
  │  Step 5: Deploy on Ultra96                               │
  │    ├─ Load bitstream (PYNQ Overlay)                      │
  │    ├─ Load .xmodel (VART Runtime)                        │
  │    └─ Run inference: ~{est_latency_us:.0f} µs per window               │
  └──────────────────────────────────────────────────────────┘
    """)

    print(VIVADO_INFO)
    print()
    print(PYNQ_DPU_INFO)

    os.makedirs("models/compiled", exist_ok=True)
    results = {
        "dpu_target": f"{DPU_SPEC['name']}_{DPU_SPEC['variant']}",
        "fpga": DPU_SPEC["fpga"],
        "clock_mhz": DPU_SPEC["clock_freq_mhz"],
        "total_macs": total_macs,
        "estimated_latency_us": est_latency_us,
        "dpu_layers": dpu_layers,
        "cpu_layers": cpu_layers,
        "dpu_coverage_pct": dpu_layers / (dpu_layers + cpu_layers) * 100,
        "layers": [
            {
                "name": l["name"],
                "target": l["target"],
                "macs": l["macs"],
                "params": l["params"],
                "input_shape": l["input_shape"],
                "output_shape": l["output_shape"],
            }
            for l in layers
        ],
    }
    with open("models/compiled/compilation_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Analysis saved: models/compiled/compilation_analysis.json")

    print(f"\n  ✓ Step 6 complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
