"""
CG4002 B02 - Step 6a: Vitis AI Compilation (.xmodel)
=====================================================
Compiles the quantized INT8 model into a .xmodel for the Ultra96 DPU.

The .xmodel contains:
  - DPU instructions (how to schedule convolutions, pooling, FC layers)
  - Quantized INT8 weights packed for the DPU memory controller
  - Data flow schedule (how tensors move through the DPU's internal BRAM)

Pipeline position:
  float32 (.pth) → INT8 (quantize.py) → .xmodel (THIS STEP) → DPU execution

Prerequisites:
  Must run inside Vitis AI Docker:
    $ docker run -it -v $(pwd):/workspace xilinx/vitis-ai-pytorch-cpu:latest
    $ conda activate vitis-ai-pytorch
    $ python software/compile.py

  The quantization step (software/quantize.py) must have been run first.

DPU Target: DPUCZDX8G_ISA1_B2304
  - B2304 = 2304 operations per clock cycle
  - Designed for Zynq UltraScale+ ZU3EG (Ultra96-V2)
  - Supports: Conv2D, DepthwiseConv, MaxPool, AvgPool, FC, Concat, ReLU

Usage:
  python software/compile.py
"""

import os
import sys
import subprocess


# The DPU fingerprint / arch file tells the compiler which
# DPU variant is on your Ultra96. This determines:
#   - Number of parallel compute engines (B2304)
#   - Supported operators
#   - Memory bandwidth constraints

DPU_ARCH = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/Ultra96/arch.json"
DPU_ARCH_ALT = [
    "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json",
    "arch.json",  # local copy
]

QUANT_MODEL_DIR = "models/quantized"

# Output
OUTPUT_DIR = "models/compiled"
OUTPUT_NAME = "exercise_cnn"


def find_arch_file():
    """Find the DPU architecture file."""
    candidates = [DPU_ARCH] + DPU_ARCH_ALT
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def compile_xmodel():
    """
    Run the Vitis AI Compiler (vai_c_xilinx).

    The compiler does:
      1. Reads the quantized model graph
      2. Maps each layer to DPU instructions
      3. Schedules data movement through DPU BRAM
      4. Generates the .xmodel binary

    Layers the DPU handles natively (no CPU fallback):
      ✓ Conv1D/Conv2D  ✓ MaxPool  ✓ AvgPool  ✓ FC (Dense)
      ✓ BatchNorm (fused into Conv)  ✓ ReLU

    Our model uses ALL DPU-supported ops → 100% hardware acceleration.
    """
    print("=" * 65)
    print("  CG4002 B02 — Step 6a: Vitis AI Compilation")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    arch_file = find_arch_file()
    if arch_file:
        print(f"\n  DPU architecture file: {arch_file}")
    else:
        print("\n  WARNING: DPU arch file not found.")
        print("  Using default DPUCZDX8G target.")

    quant_files = os.listdir(QUANT_MODEL_DIR) if os.path.exists(QUANT_MODEL_DIR) else []
    print(f"  Quantized model dir: {QUANT_MODEL_DIR}")
    print(f"  Files: {quant_files}")

    print("\n  Compiling with vai_c_xir...")

    xmodel_path = os.path.join(QUANT_MODEL_DIR, "ExerciseCNN_int.xmodel")
    if not os.path.exists(xmodel_path):
        print(f"  ERROR: {xmodel_path} not found.")
        print("  Run software/quantize.py first (inside Vitis AI Docker).")
        return

    cmd = [
        "vai_c_xir",
        "--xmodel", xmodel_path,
        "--arch", arch_file or DPU_ARCH,
        "--output_dir", OUTPUT_DIR,
        "--net_name", OUTPUT_NAME,
    ]

    print(f"  Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr}")
            compile_alternative(arch_file)
        else:
            print(f"\n  ✓ Compiled: {OUTPUT_DIR}/{OUTPUT_NAME}.xmodel")
    except FileNotFoundError:
        print("  vai_c_xir not found. Trying vai_c_xilinx...")
        compile_alternative(arch_file)


def compile_alternative(arch_file):
    """Try the older vai_c_xilinx compiler."""
    cmd = [
        "vai_c_xilinx",
        "--xmodel", os.path.join(QUANT_MODEL_DIR, "ExerciseCNN_int.xmodel"),
        "--arch", arch_file or DPU_ARCH,
        "--output_dir", OUTPUT_DIR,
        "--net_name", OUTPUT_NAME,
    ]

    print(f"  Alternative command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            print(f"\n  ✓ Compiled: {OUTPUT_DIR}/{OUTPUT_NAME}.xmodel")
        else:
            print(f"  STDERR: {result.stderr}")
            print("\n  Compilation failed. Ensure you're inside Vitis AI Docker.")
    except FileNotFoundError:
        print("\n  Neither vai_c_xir nor vai_c_xilinx found.")
        print("  Make sure you're running inside the Vitis AI Docker container.")
        print("  Use software/compile_demo.py for a standalone demo instead.")


if __name__ == "__main__":
    compile_xmodel()
