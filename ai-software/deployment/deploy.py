"""
CG4002 B02 - VART Deployment on Ultra96
=========================================
Runs on the Ultra96 via SSH. Loads the compiled .xmodel onto the DPU and
performs real-time inference on IMU data received via Zenoh.

Pipeline:
  ESP32 (IMU) -> Zenoh -> Ultra96 (this script) -> DPU inference -> Zenoh -> Phone

Prerequisites (on Ultra96):
  - PYNQ image with DPU overlay installed
  - .xmodel copied to Ultra96: scp exercise_cnn.xmodel xilinx@<ip>:~/
  - pip3 install eclipse-zenoh numpy

Usage:
  python3 deploy.py           # live mode with Zenoh
  python3 deploy.py --test    # test mode with dummy data, no sensors needed
  python3 deploy.py --cpu     # force CPU inference
"""

import numpy as np
import time
import json
import os
import sys
import argparse
from collections import deque
import threading


CLASSES     = ["high_knees", "lunge", "squat", "overhead_arm", "push_up", "sit_up", "unknown"]
NUM_CLASSES = len(CLASSES)

# 3 sensors x 4 features (avm, ax, ay, az) = 12 raw features
SENSOR_ORDER  = ["arm", "chest", "thigh"]
IMU_FIELDS    = ["avm", "ax", "ay", "az"]
NUM_FEATURES  = len(SENSOR_ORDER) * len(IMU_FIELDS)  # 12

WINDOW_SIZE   = 20      # ~2.5s at 8 Hz
SAMPLE_RATE   = 8       # Hz

IMU_TOPICS = {
    "esp/arm":   "arm",
    "esp/chest": "chest",
    "esp/thigh": "thigh",
}

OUTPUT_TOPIC   = "exercise"
ZENOH_ENDPOINT = "tcp/127.0.0.1:7448"  # via SSH tunnel

XMODEL_PATH   = "exercise_cnn.xmodel"
NORM_MEAN_PATH = "norm_mean.npy"
NORM_STD_PATH  = "norm_std.npy"


class DPUInference:
    """
    DPU inference using Vitis AI Runtime (VART).

    Workflow:
      1. Load .xmodel (compiled model with DPU instructions)
      2. Create runner (allocates DPU resources)
      3. Pre-allocate input/output INT8 buffers
      4. Per inference: quantize input -> execute_async -> dequantize output -> argmax
    """

    def __init__(self, xmodel_path):
        self.xmodel_path = xmodel_path
        self.runners = []
        self.runner = None

        self._load_model()

    def _load_model(self):
        try:
            import vart
            import xir

            self.graph = xir.Graph.deserialize(self.xmodel_path)
            subgraphs = self.graph.get_root_subgraph().toposort_child_subgraph()
            dpu_subgraphs = [
                sg for sg in subgraphs
                if sg.has_attr("device") and sg.get_attr("device") == "DPU"
            ]

            if not dpu_subgraphs:
                raise RuntimeError("No DPU subgraph found in .xmodel")

            self.runners = [vart.Runner.create_runner(sg, "run") for sg in dpu_subgraphs]
            self.runner  = self.runners[0]

            # Pre-allocate buffers and cache fix_point values
            self.in_bufs  = [np.zeros(tuple(r.get_input_tensors()[0].dims),  dtype=np.int8) for r in self.runners]
            self.out_bufs = [np.zeros(tuple(r.get_output_tensors()[0].dims), dtype=np.int8) for r in self.runners]
            self.in_fixes  = [r.get_input_tensors()[0].get_attr("fix_point")  for r in self.runners]
            self.out_fixes = [r.get_output_tensors()[0].get_attr("fix_point") for r in self.runners]

            print(f"  DPU model loaded: {self.xmodel_path}")
            print(f"  DPU subgraphs: {len(self.runners)}")
            for i in range(len(self.runners)):
                print(f"    Runner {i}: in={self.in_bufs[i].shape}  out={self.out_bufs[i].shape}")

        except ImportError:
            print("  ERROR: vart/xir not available.")
            print("  Run on Ultra96 with PYNQ + Vitis AI Runtime installed.")
            self.runners = []
            self.runner  = None

    def predict(self, input_data, return_stage_times=False):
        """
        Run inference across all DPU subgraphs.

        Args:
            input_data: numpy array, shape (1, 20, 12), float32, normalized
            return_stage_times: if True, also return list of per-stage latencies (ms)

        Returns:
            (pred_class, confidence, latency_ms)
            or (pred_class, confidence, latency_ms, stage_times) if return_stage_times
        """
        if not self.runners:
            return (-1, 0.0, 0.0, []) if return_stage_times else (-1, 0.0, 0.0)

        stage_times = []
        t_start = time.time()

        np.copyto(self.in_bufs[0],
                  np.clip(input_data * (2 ** self.in_fixes[0]), -128, 127).astype(np.int8))

        for i, runner in enumerate(self.runners):
            t0 = time.time()
            job = runner.execute_async([self.in_bufs[i]], [self.out_bufs[i]])
            runner.wait(job)
            stage_times.append((time.time() - t0) * 1000)

            if i < len(self.runners) - 1:
                # GlobalAvgPool is a CPU subgraph between DPU runners
                t0 = time.time()
                mid_float = self.out_bufs[i].astype(np.float32) / (2 ** self.out_fixes[i])
                pooled = mid_float.mean(axis=1)
                np.copyto(self.in_bufs[i + 1],
                          np.clip(pooled.reshape(self.in_bufs[i + 1].shape) * (2 ** self.in_fixes[i + 1]),
                                  -128, 127).astype(np.int8))
                stage_times.append((time.time() - t0) * 1000)

        latency_ms = (time.time() - t_start) * 1000

        output_float = self.out_bufs[-1].astype(np.float32) / (2 ** self.out_fixes[-1])
        pred_class   = int(np.argmax(output_float[0]))

        logits = output_float[0]
        exp_l  = np.exp(logits - np.max(logits))
        softmax = exp_l / exp_l.sum()
        confidence = float(softmax[pred_class])

        if return_stage_times:
            return pred_class, confidence, latency_ms, stage_times
        return pred_class, confidence, latency_ms

    def cleanup(self):
        for r in self.runners:
            del r


class CPUInference:
    """Fallback: ARM CPU inference using PyTorch. Used when VART is unavailable."""

    def __init__(self, model_path="best_model.pth"):
        self.model = None
        try:
            import torch
            sys.path.insert(0, os.path.dirname(__file__))
            from model import ExerciseCNN

            self.model = ExerciseCNN(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            self.model.eval()
            self.torch = torch
            print(f"  CPU model loaded: {model_path}")
        except Exception as e:
            print(f"  WARNING: CPU fallback not available: {e}")

    def predict(self, input_data):
        if self.model is None:
            return -1, 0.0, 0.0

        t_start = time.time()
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(input_data))
        latency_ms = (time.time() - t_start) * 1000

        probs      = self.torch.softmax(logits, dim=1).numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return pred_class, confidence, latency_ms


class IMUBuffer:
    """Thread-safe sliding window buffer for 3 IMU sensors."""

    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.buffer      = deque(maxlen=window_size)
        self.lock        = threading.Lock()
        self.latest      = {s: [0.0] * len(IMU_FIELDS) for s in SENSOR_ORDER}
        self.ready       = False

    def update_sensor(self, sensor_name, values):
        """
        Update one sensor's latest reading. When all sensors have been seen,
        push a combined 12-feature row to the sliding window buffer.
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
        """Return current window as (1, window_size, 12) float32, or None if not ready."""
        with self.lock:
            if not self.ready:
                return None
            return np.array(list(self.buffer), dtype=np.float32).reshape(
                1, self.window_size, NUM_FEATURES
            )

    def get_fill_level(self):
        return len(self.buffer) / self.window_size


class ZenohComms:
    """Zenoh pub/sub for IMU input and prediction output."""

    def __init__(self, endpoint, imu_buffer):
        self.endpoint   = endpoint
        self.imu_buffer = imu_buffer
        self.session    = None
        self.publisher  = None

    def connect(self):
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
        topic = str(sample.key_expr)
        if topic not in IMU_TOPICS:
            return

        sensor_name = IMU_TOPICS[topic]
        try:
            payload = json.loads(sample.payload.to_string())
        except Exception:
            try:
                payload = json.loads(str(sample.payload))
            except Exception:
                return

        values = [float(payload.get(f, 0.0)) for f in IMU_FIELDS]
        self.imu_buffer.update_sensor(sensor_name, values)

    def publish_prediction(self, pred_class, confidence, timestamp):
        if self.publisher is None:
            return

        msg = json.dumps({
            "ts":         int(timestamp * 1000),
            "exercise":   pred_class,
            "class_name": CLASSES[pred_class] if 0 <= pred_class < NUM_CLASSES else "unknown",
            "confidence": round(confidence, 3),
        })
        self.publisher.put(msg)

    def close(self):
        if self.session:
            self.session.close()


def run_inference_loop(engine, imu_buffer, comms, norm_mean, norm_std,
                       inference_interval=0.5):
    """Main loop: buffer IMU data -> normalize -> inference -> publish."""
    print(f"\n  Inference loop started (interval: {inference_interval}s)")
    print(f"  Waiting for buffer to fill ({WINDOW_SIZE} samples)...")
    print(f"  Press Ctrl+C to stop.\n")

    inference_count = 0
    total_latency   = 0.0

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
            total_latency   += latency_ms

            if comms:
                comms.publish_prediction(pred_class, confidence, time.time())

            class_name = CLASSES[pred_class] if 0 <= pred_class < NUM_CLASSES else "?"
            avg_lat    = total_latency / inference_count
            print(f"  [{inference_count:>4}] {class_name:<18}  "
                  f"conf={confidence:.3f}  lat={latency_ms:.2f}ms  avg={avg_lat:.2f}ms")

            time.sleep(inference_interval)

    except KeyboardInterrupt:
        print(f"\n\n  Stopped after {inference_count} inferences.")
        if inference_count > 0:
            print(f"  Average latency: {total_latency/inference_count:.2f} ms")


def run_test_mode(engine, norm_mean, norm_std):
    """Test inference with random input — no sensors needed."""
    print("\n  --- TEST MODE ---")
    print("  Running inference on random input for each class...\n")

    np.random.seed(42)

    for class_idx, class_name in enumerate(CLASSES):
        window = np.random.randn(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32) * 0.5
        window_norm = (window - norm_mean) / norm_std
        pred, conf, lat = engine.predict(window_norm)
        pred_name = CLASSES[pred] if 0 <= pred < NUM_CLASSES else "?"
        print(f"  Test {class_idx}: input=random  -> pred={pred_name}  "
              f"conf={conf:.3f}  latency={lat:.2f}ms")

    # Latency benchmark: 100 inferences
    print(f"\n  Benchmark: 100 inferences")
    N      = 100
    window = np.random.randn(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
    window_norm = (window - norm_mean) / norm_std
    is_dpu = isinstance(engine, DPUInference) and engine.runners

    latencies      = []
    all_stage_times = []

    for _ in range(N):
        if is_dpu:
            _, _, lat, stage_times = engine.predict(window_norm, return_stage_times=True)
            all_stage_times.append(stage_times)
        else:
            _, _, lat = engine.predict(window_norm)
        latencies.append(lat)

    latencies = np.array(latencies)
    print(f"  Mean: {latencies.mean():.3f} ms  Std: {latencies.std():.3f} ms  "
          f"P50: {np.percentile(latencies, 50):.3f} ms  P99: {np.percentile(latencies, 99):.3f} ms")
    print(f"  Throughput: {1000/latencies.mean():.0f} inferences/sec")

    # Per-stage breakdown (DPU only)
    if is_dpu and all_stage_times:
        stages = np.array(all_stage_times)
        n_runners = len(engine.runners)
        stage_labels = []
        for i in range(n_runners):
            stage_labels.append(f"Runner {i} (DPU)")
            if i < n_runners - 1:
                stage_labels.append("CPU avg pool")

        print(f"\n  Per-stage breakdown:")
        for i, label in enumerate(stage_labels):
            col = stages[:, i]
            print(f"    {label:<20}  mean={col.mean():.3f}ms  min={col.min():.3f}ms  max={col.max():.3f}ms")

    # DPU vs CPU comparison
    print(f"\n  DPU vs CPU comparison:")
    cpu_engine = CPUInference("best_model.pth")
    if cpu_engine.model is not None:
        cpu_lats = []
        for _ in range(N):
            _, _, lat = cpu_engine.predict(window_norm)
            cpu_lats.append(lat)
        cpu_lats = np.array(cpu_lats)
        speedup = cpu_lats.mean() / latencies.mean()
        print(f"    CPU (ARM):  {cpu_lats.mean():.3f} ms")
        print(f"    DPU:        {latencies.mean():.3f} ms")
        print(f"    Speedup:    {speedup:.1f}x")
    else:
        print(f"    CPU fallback unavailable (PyTorch not installed)")

    print(f"\n  Model info:")
    print(f"    Input:   (1, {WINDOW_SIZE}, {NUM_FEATURES})")
    print(f"    Output:  (1, {NUM_CLASSES})")
    print(f"    Classes: {', '.join(CLASSES)}")
    if is_dpu:
        xmodel_size = os.path.getsize(engine.xmodel_path) / 1024
        print(f"    xmodel:  {xmodel_size:.1f} KB (INT8)")


def main():
    parser = argparse.ArgumentParser(description="CG4002 B02 Ultra96 Deployment")
    parser.add_argument("--test",     action="store_true", help="Test mode with dummy data")
    parser.add_argument("--cpu",      action="store_true", help="Force CPU inference")
    parser.add_argument("--xmodel",   default=XMODEL_PATH, help="Path to .xmodel file")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between inferences")
    parser.add_argument("--endpoint", default=ZENOH_ENDPOINT, help="Zenoh endpoint")
    args = parser.parse_args()

    print("=" * 65)
    print("  CG4002 B02 - Ultra96 DPU Deployment")
    print("=" * 65)

    print("\n  Loading normalization parameters...")
    try:
        norm_mean = np.load(NORM_MEAN_PATH).astype(np.float32)
        norm_std  = np.load(NORM_STD_PATH).astype(np.float32)
        print(f"    Mean shape: {norm_mean.shape}")
    except FileNotFoundError:
        print("  WARNING: Normalization files not found. Using defaults.")
        norm_mean = np.zeros(NUM_FEATURES, dtype=np.float32)
        norm_std  = np.ones(NUM_FEATURES, dtype=np.float32)

    print(f"\n  Initializing inference engine...")
    if args.cpu:
        print("  Mode: CPU (ARM Cortex-A53)")
        engine = CPUInference("best_model.pth")
    else:
        print("  Mode: DPU (DPUCZDX8G)")
        engine = DPUInference(args.xmodel)
        if not engine.runners:
            print("  Falling back to CPU inference...")
            engine = CPUInference("best_model.pth")

    if args.test:
        run_test_mode(engine, norm_mean, norm_std)
    else:
        imu_buffer = IMUBuffer(WINDOW_SIZE)

        print(f"\n  Connecting to Zenoh at {args.endpoint}...")
        comms = ZenohComms(args.endpoint, imu_buffer)
        if comms.connect():
            run_inference_loop(engine, imu_buffer, comms, norm_mean, norm_std, args.interval)
            comms.close()
        else:
            print("  Cannot connect to Zenoh. Use --test for test mode.")

    if hasattr(engine, "cleanup"):
        engine.cleanup()

    print("\n  Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
