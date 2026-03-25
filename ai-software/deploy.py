"""
CG4002 B02 - Step 7a: VART Deployment on Ultra96
==================================================
This script runs ON THE ULTRA96 via SSH.
It loads the compiled .xmodel onto the DPU and performs real-time
inference on IMU data received via Zenoh.

Pipeline:
  ESP32 (IMU data) → Zenoh → Ultra96 (this script) → DPU inference → Zenoh → Phone

Prerequisites (on Ultra96):
  - PYNQ image with DPU overlay installed
  - .xmodel copied to Ultra96: scp exercise_cnn.xmodel xilinx@<ip>:~/
  - Zenoh Python: pip3 install eclipse-zenoh
  - numpy: pip3 install numpy

How to run:
  1. SSH into Ultra96:  ssh xilinx@<ultra96-ip>
  2. Run:  python3 deploy.py

  Or for testing without live sensors:
  python3 deploy.py --test
"""

import numpy as np
import time
import json
import os
import sys
import argparse
import asyncio
from collections import deque
import threading


CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]
NUM_CLASSES = len(CLASSES)
NUM_FEATURES = 12           # 3 IMUs × 4 accel values (ax, ay, az, avm)
WINDOW_SIZE = 20            # ~2.5s at 8Hz
SAMPLE_RATE = 8             # Hz (effective rate from ESPs)

IMU_TOPICS = {
    "esp/arm": "arm",
    "esp/chest": "chest",
    "esp/thigh": "thigh",
}
SENSOR_ORDER = ["arm", "chest", "thigh"]
IMU_FIELDS = ["ax", "ay", "az", "avm"]

OUTPUT_TOPIC = "exercise"

ZENOH_ENDPOINT = "tcp/127.0.0.1:7448"  # via SSH tunnel

XMODEL_PATH = "exercise_cnn.xmodel"
NORM_MEAN_PATH = "norm_mean.npy"
NORM_STD_PATH = "norm_std.npy"


class DPUInference:
    """
    Handles DPU inference using Vitis AI Runtime (VART).

    The DPU workflow:
      1. Load .xmodel (compiled model with DPU instructions)
      2. Create runner (allocates DPU resources)
      3. Get input/output tensor info (shapes, scales)
      4. For each inference:
         a. Write normalized INT8 data to input buffer
         b. Call execute_async() → DPU processes in hardware
         c. Read output buffer → argmax for prediction
    """

    def __init__(self, xmodel_path):
        self.xmodel_path = xmodel_path
        self.runners = []
        self.runner = None  # kept for cleanup compatibility

        self._load_model()

    def _load_model(self):
        """Load .xmodel and create DPU runners for all DPU subgraphs."""
        try:
            import vart
            import xir

            self.graph = xir.Graph.deserialize(self.xmodel_path)  # keep alive to prevent GC
            graph = self.graph
            subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

            dpu_subgraphs = [
                sg for sg in subgraphs
                if sg.has_attr("device") and sg.get_attr("device") == "DPU"
            ]

            if not dpu_subgraphs:
                raise RuntimeError("No DPU subgraph found in .xmodel!")

            self.runners = [vart.Runner.create_runner(sg, "run") for sg in dpu_subgraphs]
            self.runner = self.runners[0]

            # Pre-allocate buffers and cache fix_point values to avoid
            # repeated VART tensor calls during inference (causes segfault)
            self.in_bufs  = [np.zeros(tuple(r.get_input_tensors()[0].dims),  dtype=np.int8) for r in self.runners]
            self.out_bufs = [np.zeros(tuple(r.get_output_tensors()[0].dims), dtype=np.int8) for r in self.runners]
            self.in_fixes  = [r.get_input_tensors()[0].get_attr("fix_point")  for r in self.runners]
            self.out_fixes = [r.get_output_tensors()[0].get_attr("fix_point") for r in self.runners]

            print(f"  DPU model loaded: {self.xmodel_path}")
            print(f"  DPU subgraphs: {len(self.runners)}")
            for i in range(len(self.runners)):
                print(f"  Runner {i}: in={self.in_bufs[i].shape} fix={self.in_fixes[i]}  "
                      f"out={self.out_bufs[i].shape} fix={self.out_fixes[i]}")

        except ImportError:
            print("  ERROR: vart/xir not available.")
            print("  Make sure you're running on Ultra96 with PYNQ + Vitis AI Runtime.")
            self.runners = []
            self.runner = None

    def predict(self, input_data):
        """
        Run inference across all DPU subgraphs, with manual avg_pool
        for the CPU subgraph between them (AvgPool1d is not in XIR).

        Args:
            input_data: numpy array, shape (1, 20, 12), float32, normalized

        Returns:
            (predicted_class_id, confidence, latency_ms)
        """
        if not self.runners:
            return -1, 0.0, 0.0

        t_start = time.time()

        np.copyto(self.in_bufs[0],
                  np.clip(input_data * (2 ** self.in_fixes[0]), -128, 127).astype(np.int8))

        for i, runner in enumerate(self.runners):
            job = runner.execute_async([self.in_bufs[i]], [self.out_bufs[i]])
            runner.wait(job)

            if i < len(self.runners) - 1:
                # CPU op: global average pool over time dimension
                # out_bufs[i] shape: (1, T, C) → mean over T → (1, C)
                mid_float = self.out_bufs[i].astype(np.float32) / (2 ** self.out_fixes[i])
                pooled = mid_float.mean(axis=1)  # (1, C)
                np.copyto(self.in_bufs[i + 1],
                          np.clip(pooled.reshape(self.in_bufs[i + 1].shape) * (2 ** self.in_fixes[i + 1]),
                                  -128, 127).astype(np.int8))

        latency_ms = (time.time() - t_start) * 1000

        output_float = self.out_bufs[-1].astype(np.float32) / (2 ** self.out_fixes[-1])
        pred_class = int(np.argmax(output_float[0]))

        logits = output_float[0]
        exp_logits = np.exp(logits - np.max(logits))
        softmax = exp_logits / exp_logits.sum()
        confidence = float(softmax[pred_class])

        return pred_class, confidence, latency_ms

    def cleanup(self):
        """Release DPU resources."""
        for r in self.runners:
            del r


class CPUInference:
    """
    Fallback: run inference on ARM CPU using PyTorch/numpy.
    Used when VART is not available or for comparison testing.
    """

    def __init__(self, model_path="best_model.pth"):
        self.model = None
        try:
            import torch
            sys.path.insert(0, os.path.dirname(__file__))
            from model import ExerciseCNN

            self.model = ExerciseCNN(num_features=12, num_classes=7)
            self.model.load_state_dict(
                torch.load(model_path, map_location='cpu', weights_only=True)
            )
            self.model.eval()
            self.torch = torch
            print(f"  CPU model loaded: {model_path}")
        except Exception as e:
            print(f"  WARNING: CPU fallback not available: {e}")

    def predict(self, input_data):
        """Run inference on ARM CPU."""
        if self.model is None:
            return -1, 0.0, 0.0

        input_tensor = self.torch.from_numpy(input_data)

        t_start = time.time()
        with self.torch.no_grad():
            logits = self.model(input_tensor)
        latency_ms = (time.time() - t_start) * 1000

        probs = self.torch.softmax(logits, dim=1).numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return pred_class, confidence, latency_ms


class IMUBuffer:
    """
    Thread-safe sliding window buffer for 3 IMU sensors.
    Collects incoming IMU readings and produces aligned 12-feature windows.
    """

    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.lock = threading.Lock()

        self.latest = {s: [0.0] * len(IMU_FIELDS) for s in SENSOR_ORDER}
        self.ready = False

    def update_sensor(self, sensor_name, values):
        """
        Update one sensor's latest values.
        When all 3 sensors have been updated, push a combined row to the buffer.
        """
        if sensor_name not in self.latest:
            return

        self.latest[sensor_name] = values

        combined = []
        for s in SENSOR_ORDER:
            combined.extend(self.latest[s])

        with self.lock:
            self.buffer.append(combined)
            if len(self.buffer) >= self.window_size:
                self.ready = True

    def get_window(self):
        """
        Get the current window as numpy array (1, 20, 12).
        Returns None if buffer not full yet.
        """
        with self.lock:
            if not self.ready:
                return None
            window = np.array(list(self.buffer), dtype=np.float32)
        return window.reshape(1, self.window_size, NUM_FEATURES)

    def get_fill_level(self):
        """How full is the buffer (0.0 to 1.0)."""
        return len(self.buffer) / self.window_size


class ZenohComms:
    """Handle Zenoh pub/sub for IMU input and prediction output."""

    def __init__(self, endpoint, imu_buffer, on_prediction=None):
        self.endpoint = endpoint
        self.imu_buffer = imu_buffer
        self.on_prediction = on_prediction
        self.session = None
        self.publisher = None

    def connect(self):
        """Connect to Zenoh router."""
        try:
            import zenoh
            config = zenoh.Config()
            config.insert_json5("connect/endpoints", json.dumps([self.endpoint]))
            self.session = zenoh.open(config)

            for topic in IMU_TOPICS:
                self.session.declare_subscriber(topic, self._on_imu_sample)
                print(f"  Subscribed: {topic}")

            self.publisher = self.session.declare_publisher(OUTPUT_TOPIC)
            print(f"  Publisher:  {OUTPUT_TOPIC}")
            return True

        except ImportError:
            print("  ERROR: zenoh not installed. pip3 install eclipse-zenoh")
            return False
        except Exception as e:
            print(f"  ERROR connecting to Zenoh: {e}")
            return False

    def _on_imu_sample(self, sample):
        """Callback for incoming IMU data."""
        topic = str(sample.key_expr)
        if topic not in IMU_TOPICS:
            return

        sensor_name = IMU_TOPICS[topic]

        try:
            payload = json.loads(sample.payload.to_string())
        except:
            try:
                payload = json.loads(str(sample.payload))
            except:
                return

        values = [float(payload.get(f, 0.0)) for f in IMU_FIELDS]
        self.imu_buffer.update_sensor(sensor_name, values)

    def publish_prediction(self, pred_class, confidence, timestamp):
        """Publish prediction result to phone visualizer."""
        if self.publisher is None:
            return

        msg = json.dumps({
            "ts": int(timestamp * 1000),
            "exercise": pred_class,
            "class_name": CLASSES[pred_class] if 0 <= pred_class < NUM_CLASSES else "unknown",
            "confidence": round(confidence, 3),
        })
        self.publisher.put(msg)

    def close(self):
        if self.session:
            self.session.close()


def run_inference_loop(engine, imu_buffer, comms, norm_mean, norm_std,
                       inference_interval=0.5):
    """
    Main loop: collect IMU data → build window → DPU inference → publish result.

    Args:
        engine: DPUInference or CPUInference
        imu_buffer: IMUBuffer
        comms: ZenohComms (or None for test mode)
        norm_mean, norm_std: normalization parameters
        inference_interval: seconds between inferences
    """
    print(f"\n  Inference loop started (interval: {inference_interval}s)")
    print(f"  Waiting for IMU buffer to fill ({WINDOW_SIZE} samples)...")
    print(f"  Press Ctrl+C to stop.\n")

    inference_count = 0
    total_latency = 0.0

    try:
        while True:
            window = imu_buffer.get_window()
            if window is None:
                fill = imu_buffer.get_fill_level()
                print(f"\r  Buffer: {fill*100:.0f}%", end="", flush=True)
                time.sleep(0.05)
                continue

            window_norm = (window - norm_mean) / norm_std

            pred_class, confidence, latency_ms = engine.predict(window_norm)
            inference_count += 1
            total_latency += latency_ms

            if comms:
                comms.publish_prediction(pred_class, confidence, time.time())

            class_name = CLASSES[pred_class] if 0 <= pred_class < NUM_CLASSES else "?"
            avg_lat = total_latency / inference_count
            print(f"  [{inference_count:>4}] {class_name:<15} "
                  f"conf={confidence:.3f}  lat={latency_ms:.2f}ms  "
                  f"avg={avg_lat:.2f}ms")

            time.sleep(inference_interval)

    except KeyboardInterrupt:
        print(f"\n\n  Stopped after {inference_count} inferences.")
        if inference_count > 0:
            print(f"  Average latency: {total_latency/inference_count:.2f} ms")


def run_test_mode(engine, norm_mean, norm_std):
    """Test inference with dummy data (no sensors needed)."""
    print("\n  --- TEST MODE ---")
    print("  Running inference on dummy data for each class...\n")

    np.random.seed(42)

    for class_idx, class_name in enumerate(CLASSES):
        # Create a recognizable dummy window
        window = np.random.randn(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32) * 0.5
        window_norm = (window - norm_mean) / norm_std

        pred, conf, lat = engine.predict(window_norm)
        pred_name = CLASSES[pred] if 0 <= pred < NUM_CLASSES else "?"
        print(f"  Test {class_idx}: input=random  →  pred={pred_name}  "
              f"conf={conf:.3f}  latency={lat:.2f}ms")

    print(f"\n  Latency benchmark (100 inferences)...")
    latencies = []
    window = np.random.randn(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
    window_norm = (window - norm_mean) / norm_std

    for _ in range(100):
        _, _, lat = engine.predict(window_norm)
        latencies.append(lat)

    latencies = np.array(latencies)
    print(f"    Mean:   {latencies.mean():.3f} ms")
    print(f"    Std:    {latencies.std():.3f} ms")
    print(f"    Min:    {latencies.min():.3f} ms")
    print(f"    Max:    {latencies.max():.3f} ms")
    print(f"    P50:    {np.percentile(latencies, 50):.3f} ms")
    print(f"    P99:    {np.percentile(latencies, 99):.3f} ms")
    print(f"    Throughput: {1000/latencies.mean():.0f} inferences/sec")


def main():
    parser = argparse.ArgumentParser(description="CG4002 B02 Ultra96 Deployment")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode with dummy data (no Zenoh)")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU inference instead of DPU")
    parser.add_argument("--xmodel", default=XMODEL_PATH,
                        help=f"Path to .xmodel file (default: {XMODEL_PATH})")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between inferences (default: 0.5)")
    parser.add_argument("--endpoint", default=ZENOH_ENDPOINT,
                        help=f"Zenoh endpoint (default: {ZENOH_ENDPOINT})")
    args = parser.parse_args()

    print("=" * 65)
    print("  CG4002 B02 — Ultra96 DPU Deployment")
    print("=" * 65)

    print("\n  Loading normalization parameters...")
    try:
        norm_mean = np.load(NORM_MEAN_PATH).astype(np.float32)
        norm_std = np.load(NORM_STD_PATH).astype(np.float32)
        print(f"    Mean shape: {norm_mean.shape}")
        print(f"    Std shape:  {norm_std.shape}")
    except FileNotFoundError:
        print("  WARNING: Normalization files not found. Using zeros/ones.")
        norm_mean = np.zeros(NUM_FEATURES, dtype=np.float32)
        norm_std = np.ones(NUM_FEATURES, dtype=np.float32)

    print(f"\n  Initializing inference engine...")
    if args.cpu:
        print("  Mode: CPU (ARM Cortex-A53)")
        engine = CPUInference("best_model.pth")
    else:
        print("  Mode: DPU (DPUCZDX8G)")
        engine = DPUInference(args.xmodel)
        if engine.runner is None:
            print("  Falling back to CPU inference...")
            engine = CPUInference("best_model.pth")

    if args.test:
        run_test_mode(engine, norm_mean, norm_std)
    else:
        imu_buffer = IMUBuffer(WINDOW_SIZE)

        print(f"\n  Connecting to Zenoh at {args.endpoint}...")
        comms = ZenohComms(args.endpoint, imu_buffer)
        if comms.connect():
            run_inference_loop(engine, imu_buffer, comms, norm_mean, norm_std,
                               args.interval)
            comms.close()
        else:
            print("  Cannot connect to Zenoh. Use --test for test mode.")

    if hasattr(engine, 'cleanup'):
        engine.cleanup()

    print("\n  ✓ Deployment finished.")
    print("=" * 65)


if __name__ == "__main__":
    main()
