# AI Software — Exercise Classification on Ultra96-V2

This directory contains the full machine learning pipeline for real-time exercise classification using IMU wearable sensors and an FPGA-based DPU accelerator.

The system classifies 7 exercises — `high_knees`, `pushup`, `situp`, `lunge`, `squat`, `overhead_hold`, `unknown` — from raw IMU data collected across 3 body-worn sensors (arm, chest, thigh), each reporting accelerometer and gyroscope readings.

---

## Prerequisites

### Local development (laptop)

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Ultra96-V2

The Ultra96 runs on its own Python environment managed by the PYNQ image. Do not use `requirements.txt` on the board — the required packages (`vart`, `pynq`, `zenoh`) are either pre-installed or installed separately. Use the `sudo -E` command shown in the [Runtime](#runtime) section to launch scripts with the correct library paths.

**Optional dependencies** (not in `requirements.txt`):

| Package | Where needed |
|---------|-------------|
| `zenoh` | `deploy.py`, `repub.py` — runtime on Ultra96 |
| `vai_q_pytorch` / `vai_q_onnx` | `quantize.py` — inside Vitis AI Docker only |
| `vart`, `pynq` | `deploy.py` — pre-installed on Ultra96 PYNQ image |

---

## Directory Structure

```
ai-software/
├── model.py                  # Neural network architecture (shared by all scripts)
├── data_pipeline/            # Data collection and preprocessing
├── training/                 # Model training and evaluation
└── deployment/               # Quantization, compilation, and runtime
```

---

## Pipeline Overview

The pipeline has 10 steps, from raw IMU data to a deployed DPU model running on Ultra96:

```
[Data Collection] → [Training] → [Evaluation] → [Quantization] → [Compilation] → [Deployment]
      ↓                  ↓              ↓               ↓                ↓               ↓
convert_real_data.py  train.py    evaluate.py      quantize.py      compile.py      deploy.py
generate_data.py                  benchmark.py     (Docker)         (Docker)        repub.py
generate_data.py                  simulate.py
```

---

## The Model

**File:** `model.py`

Defines `ExerciseCNN`, a lightweight 1D convolutional neural network with 16,295 parameters.

```
Input: (batch, window_size, features)
  → Conv1D(18→32, k=9) + BatchNorm + ReLU
  → MaxPool1d(2)
  → Conv1D(32→64, k=5) + BatchNorm + ReLU
  → GlobalAvgPool1d               ← replaces Flatten to save BRAM on FPGA
  → Linear(64→32) + ReLU + Dropout(0.3)
  → Linear(32→7)                  ← output logits for 7 classes
Output: (batch, 7)
```

The Conv1D layers are internally mapped to Conv2D with height=1 for DPU compatibility. BatchNorm layers are fused into the Conv weights at compile time.

---

## Data Pipeline (`data_pipeline/`)

### `generate_data.py`
Generates 1,400 synthetic training samples (200 per class) by simulating realistic IMU signals for each exercise. Each class has distinct signal characteristics — for example, `high_knees` produces 2–3 Hz rapid oscillations with strong thigh acceleration, while `overhead_hold` shows elevated wrist position with fatigue tremor.

Use this to test the pipeline end-to-end without real sensor hardware.

### `convert_log_to_npy.py`
Converts raw CSV recordings from `record_data.py` into windowed `.npy` arrays ready for training. Aligns the 3 sensors by timestamp (within 20ms tolerance), then segments the time series into sliding windows.

**Output:** `data/train/X.npy`, `data/test/X.npy`, and corresponding labels

### `convert_real_data.py`
Parses InfluxDB CSV exports from the real gym dataset (5 people, 7 exercises each). Handles JSON payloads per sensor, aligns readings by timestamp, and creates `(20, 12)` windows at 8Hz.

This is the primary data source for training the final deployed model. Also exports constants (`CLASSES`, `WINDOW_SIZE`, `NUM_FEATURES`, etc.) used by other scripts.

**Output:** `data/train/`, `data/val/`, `data/test/` splits (70/15/15)

---

## Training (`training/`)

### `train.py`
Trains `ExerciseCNN` on the prepared dataset. Applies z-score normalisation (fit on training data only, saved for use at inference time). Uses Adam optimiser with StepLR decay, CrossEntropyLoss, and saves the best checkpoint on validation accuracy.

**Key outputs:**
- `models/best_model.pth` — best model weights
- `models/exercise_cnn.onnx` — ONNX export for Vitis AI quantizer
- `models/norm_mean.npy`, `models/norm_std.npy` — normalisation parameters
- `models/training_curves.png`

### `evaluate.py`
Runs the saved model on the held-out test set. Generates a confusion matrix, per-class accuracy bar chart, and a full classification report (precision, recall, F1). Also measures inference latency (average, P50, P95, P99) on CPU.

**Key outputs:** `models/confusion_matrix.png`, `models/evaluation_results.json`

### `lopo_cv.py`
Leave-One-Person-Out cross-validation. Trains 5 folds, each time holding out one person's entire data as the test set. This validates that the model generalises to unseen users, not just unseen windows from people it has seen before. Applies 4 augmentation strategies (scaling, noise injection, time warping, rotation) to expand each training fold.

**Key outputs:** `models/lopo/lopo_results.json`, per-fold accuracy plots

### `benchmark.py`
Compares float32 CPU inference against simulated INT8 DPU inference. Reports accuracy drop (float32 → INT8), latency speedup (~6.7×), energy efficiency (~2.5× more efficient on DPU), and FPGA resource utilisation (LUT, BRAM, DSP).

**Key outputs:** `models/benchmark/benchmark_results.json`, comparison plots

### `visualize.py`
Diagnostic plots for the real IMU training data. Generates time-domain signal plots per class, cross-person variability comparisons, t-SNE/PCA embeddings to check class separability, and a 12×12 feature correlation matrix.

**Key outputs:** `models/plots/` (4 diagnostic PNGs)

---

## Deployment (`deployment/`)

### Quantization

**`quantize.py`** *(requires Vitis AI Docker)*

Converts the float32 model to INT8 using `vai_q_pytorch`. Runs a calibration pass on 100 training samples to measure activation ranges, then quantizes weights and activations per layer. The quantized model is exported as an `.xmodel` file.

```
float32 (.pth) → INT8 calibration → ExerciseCNN_int.xmodel
```

**`quantize_demo.py`** *(no Docker needed)*

Simulates the quantization process mathematically on a laptop. Shows per-layer weight distributions before and after quantization, measures logit MAE and KL divergence, and demonstrates the ~4× memory compression with typically <1% accuracy drop.

---

### Compilation

**`compile.py`** *(requires Vitis AI Docker)*

Compiles the quantized `.xmodel` into DPU instructions using `vai_c_xir`. Targets the DPUCZDX8G B2304 variant on the Ultra96-V2 (Zynq UltraScale+ ZU3EG). At this step, Conv1D is mapped to Conv2D with height=1, and BatchNorm is fused into Conv weights.

```
ExerciseCNN_int.xmodel → DPU instructions → exercise_cnn.xmodel
```

**`compile_demo.py`** *(no Docker needed)*

Explains what the compiler does without running it. Analyses each model layer, shows that all 11 layers are DPU-compatible (0% CPU fallback), and estimates ~150µs DPU latency based on 1.3M MACs at 300MHz.

---

### Runtime

**`deploy.py`** *(runs on Ultra96)*

The production runtime. Subscribes to 3 Zenoh IMU topics, maintains a 20-sample sliding window, normalises incoming data using the saved z-score parameters, and runs inference on the DPU every 0.5 seconds via the VART runtime. Publishes predictions to the `exercise` Zenoh topic. Falls back to CPU if DPU is unavailable.

**`deploy_demo.py`** *(no hardware needed)*

Simulates the full deployment pipeline on a laptop. Runs inference on test data, compares CPU vs simulated DPU latency, and prints the same output format you would see on Ultra96 via SSH.

**`repub.py`** *(runs on Ultra96)*

Alternative production runtime using Zenoh Advanced Subscriber. Uses a 20-sample IMU buffer, applies a confidence threshold (0.7 — below which the output is `unknown`), and publishes `{"action": predicted_class}` to `ultra/action1`. Includes history and recovery settings for reliable Zenoh delivery.

---

### Validation and Utilities

**`simulate.py`**

Runs 6 test suites before deploying to hardware:
1. **Software emulation** — forward pass, output shape, softmax sums, accuracy
2. **Bit-accuracy** — float32 vs INT8 agreement >95%, logit MAE <0.5
3. **Hardware-in-the-loop** — pre-recorded sensor playback with ground truth
4. **Pipeline validation** — full JSON → normalise → infer → JSON workflow
5. **Edge cases** — all-zeros, spikes, noise, high-frequency; stress test of 500 inferences
6. **Reproducibility** — 3 consecutive runs produce identical outputs

**`check_pipeline.py`**

Health check utility. Verifies that all expected output files exist, checks model parameter count (16,295 expected), and validates accuracy thresholds (>90% required). Run this anytime to identify which pipeline step is incomplete.

**`power_management.py`**

Analyses power consumption of the Ultra96 and wearable sensors. Simulates the 7 PMIC power rails, documents CPU DVFS options (300/600/1200 MHz), and calculates DPU duty cycle (active only 0.03% of the time — 15µs per 500ms cycle). Used for battery life estimation and design justification.

---

## Running the Pipeline

### 1. Prepare data

Using synthetic data (no hardware needed):
```bash
python data_pipeline/generate_data.py
```

Using real sensor recordings:
```bash
python data_pipeline/record_data.py        # record live data
python data_pipeline/convert_log_to_npy.py # convert to .npy
```

Using the real gym dataset:
```bash
python data_pipeline/convert_real_data.py
```

### 2. Train and evaluate

```bash
python training/train.py
python training/evaluate.py
python training/visualize.py   # optional: diagnostic plots
python training/lopo_cv.py     # optional: cross-validation
python training/benchmark.py   # optional: hardware benchmarks
```

### 3. Quantize and compile (inside Vitis AI Docker)

```bash
docker run -it -v $(pwd):/workspace xilinx/vitis-ai-pytorch-cpu:latest
conda activate vitis-ai-pytorch
cd /workspace

python deployment/quantize.py
python deployment/compile.py
```

Or run the standalone demos (no Docker):
```bash
python deployment/quantize_demo.py
python deployment/compile_demo.py
```

### 4. Validate and deploy

```bash
python deployment/simulate.py      # pre-deployment test suites
python deployment/deploy_demo.py   # simulate deployment on laptop
```

On Ultra96:
```bash
/home/xilinx/run.sh repub.py -c /home/xilinx/save-the-cats/zenoh/configs/ultra96/SESSION_CONFIG.json5
```

### 5. Utilities

```bash
python deployment/check_pipeline.py    # verify all steps complete
python deployment/power_management.py  # power analysis
```

---

## Ultra96 Setup

### Install dependencies on Ultra96

```bash
pip install eclipse-zenoh
pip install numpy
```

> `vart` and `pynq` are pre-installed on the PYNQ image — do not reinstall them.

### After every reboot

VART hardcodes the xclbin path to `/run/media/mmcblk0p1/` but the SD card is mounted at `/boot`. This symlink must be recreated after every reboot:

```bash
cd ~/ai-demo/dpu_overlay
bash load_dpu.sh

sudo mkdir -p /run/media/mmcblk0p1
sudo cp ~/ai-demo/dpu_overlay/dpu.xclbin /run/media/mmcblk0p1/
```

### Run inference on Ultra96

```bash
source /etc/profile.d/xrt_setup.sh
/home/xilinx/run.sh repub.py -c /home/xilinx/save-the-cats/zenoh/configs/ultra96/SESSION_CONFIG.json5
```

> **Note:** VART requires `/opt/python3.9/bin/python3.9` — the system `python3` (3.10.4) cannot load `vart.so`. The `run.sh` wrapper handles this automatically.

### Copy model files to Ultra96

```bash
scp models/compiled/exercise_cnn.xmodel xilinx@pynq:~/ai-demo/dpu_overlay/
scp models/norm_mean.npy models/norm_std.npy xilinx@pynq:~/ai-demo/dpu_overlay/
```

---

## Hardware Target

- **Board:** Ultra96-V2 (Xilinx Zynq UltraScale+ ZU3EG)
- **DPU variant:** DPUCZDX8G_ISA1_B2304
- **DPU throughput:** 691.2 GOPS (2304 ops/cycle @ 300MHz)
- **End-to-end latency target:** <100ms
- **Sensors:** 3× ESP32 IMU (arm, chest, thigh) over Zenoh/WiFi
