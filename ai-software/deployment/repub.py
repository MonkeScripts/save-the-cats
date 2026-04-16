"""
CG4002 B02 - Zenoh IMU Republisher with DPU Classification
===========================================================
Subscribes to IMU data from ESP32 sensors, runs exercise classification
on the Ultra96 DPU, and republishes the result to the visualizer.

Pipeline:
  ESP32 (esp/arm, esp/chest, esp/thigh) → [this script] → ultra/action1

Usage (on Ultra96):
  source /etc/profile.d/xrt_setup.sh
  sudo -E LD_LIBRARY_PATH=/opt/python3.9/lib:/usr/lib \\
          PYTHONPATH=/usr/lib/python3.9/site-packages \\
          /opt/python3.9/bin/python3.9 repub.py

  # CPU fallback (no DPU needed):
  python3 repub.py --cpu
"""

import time
import threading
import json
from collections import deque

import numpy as np
import zenoh
from zenoh import ZBytes
from zenoh.ext import HistoryConfig, Miss, RecoveryConfig, declare_advanced_subscriber


CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]
NUM_CLASSES   = len(CLASSES)
NUM_FEATURES  = 12   # 3 IMUs × 4 values (ax, ay, az, avm)
WINDOW_SIZE   = 20

IMU_TOPICS = {
    "esp/arm":   "arm",
    "esp/chest": "chest",
    "esp/thigh": "thigh",
}
SENSOR_ORDER = ["arm", "chest", "thigh"]
IMU_FIELDS   = ["ax", "ay", "az", "avm"]

CONFIDENCE_THRESHOLD = 0.7   # below this → classify as "unknown"

NORM_MEAN_PATH = "norm_mean.npy"
NORM_STD_PATH  = "norm_std.npy"
XMODEL_PATH    = "exercise_cnn.xmodel"


class IMUBuffer:
    """Thread-safe sliding window buffer for 3 IMU sensors."""

    def __init__(self):
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.lock = threading.Lock()
        self.latest = {s: [0.0] * len(IMU_FIELDS) for s in SENSOR_ORDER}
        self.ready = False

    def update_sensor(self, sensor_name, values):
        if sensor_name not in self.latest:
            return
        self.latest[sensor_name] = values
        combined = []
        for s in SENSOR_ORDER:
            combined.extend(self.latest[s])
        with self.lock:
            self.buffer.append(combined)
            if len(self.buffer) >= WINDOW_SIZE:
                self.ready = True

    def get_window(self):
        """Returns (1, 20, 12) numpy array or None if buffer not full."""
        with self.lock:
            if not self.ready:
                return None
            return np.array(list(self.buffer), dtype=np.float32).reshape(1, WINDOW_SIZE, NUM_FEATURES)

    def get_fill_level(self):
        return len(self.buffer) / WINDOW_SIZE


def load_engine(use_cpu=False, xmodel_path=XMODEL_PATH):
    """Load DPU or CPU inference engine. Returns engine tuple or None."""
    if not use_cpu:
        try:
            import vart
            import xir

            graph = xir.Graph.deserialize(xmodel_path)
            subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
            dpu_sgs = [sg for sg in subgraphs
                       if sg.has_attr("device") and sg.get_attr("device") == "DPU"]
            if not dpu_sgs:
                raise RuntimeError("No DPU subgraph found in .xmodel")

            runners  = [vart.Runner.create_runner(sg, "run") for sg in dpu_sgs]
            in_bufs  = [np.zeros(tuple(r.get_input_tensors()[0].dims),  dtype=np.int8) for r in runners]
            out_bufs = [np.zeros(tuple(r.get_output_tensors()[0].dims), dtype=np.int8) for r in runners]
            in_fixes  = [r.get_input_tensors()[0].get_attr("fix_point")  for r in runners]
            out_fixes = [r.get_output_tensors()[0].get_attr("fix_point") for r in runners]

            print(f"  DPU engine loaded: {xmodel_path} ({len(runners)} subgraph(s))")
            for i in range(len(runners)):
                print(f"    Runner {i}: in={in_bufs[i].shape} fix={in_fixes[i]}"
                      f"  out={out_bufs[i].shape} fix={out_fixes[i]}")

            # graph must stay alive (prevents GC segfault)
            return ("dpu", graph, runners, in_bufs, out_bufs, in_fixes, out_fixes)

        except Exception as e:
            print(f"  DPU not available ({e}), falling back to CPU.")

    try:
        import torch
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model import ExerciseCNN

        model = ExerciseCNN(num_features=12, num_classes=7)
        model.load_state_dict(torch.load("best_model.pth", map_location="cpu", weights_only=True))
        model.eval()
        print("  CPU engine loaded: best_model.pth")
        return ("cpu", model, torch)

    except Exception as e:
        print(f"  ERROR: No inference engine available: {e}")
        return None


def run_inference(engine, window_norm):
    """
    Run inference on a normalized window.

    Args:
        engine: result of load_engine()
        window_norm: numpy array (1, 20, 12), float32, normalized

    Returns:
        (pred_class: int, confidence: float)
    """
    if engine is None:
        return -1, 0.0

    if engine[0] == "dpu":
        _, graph, runners, in_bufs, out_bufs, in_fixes, out_fixes = engine

        np.copyto(in_bufs[0],
                  np.clip(window_norm * (2 ** in_fixes[0]), -128, 127).astype(np.int8))

        for i, runner in enumerate(runners):
            job = runner.execute_async([in_bufs[i]], [out_bufs[i]])
            runner.wait(job)

            if i < len(runners) - 1:
                # CPU op: global average pool (AvgPool1d not in XIR)
                mid_float = out_bufs[i].astype(np.float32) / (2 ** out_fixes[i])
                pooled = mid_float.mean(axis=1)
                np.copyto(in_bufs[i + 1],
                          np.clip(pooled.reshape(in_bufs[i + 1].shape) * (2 ** in_fixes[i + 1]),
                                  -128, 127).astype(np.int8))

        output_float = out_bufs[-1].astype(np.float32) / (2 ** out_fixes[-1])
        pred_class = int(np.argmax(output_float[0]))
        logits = output_float[0]
        exp_l = np.exp(logits - np.max(logits))
        confidence = float((exp_l / exp_l.sum())[pred_class])
        return pred_class, confidence

    else:  # cpu
        _, model, torch = engine
        tensor = torch.from_numpy(window_norm)
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_class = int(np.argmax(probs))
        return pred_class, float(probs[pred_class])


def main(conf, repub_key, sub_key, interval, add_matching_listener, use_cpu, xmodel_path):
    zenoh.init_log_from_env_or("error")
    print(f"Current Config: {conf}")

    try:
        norm_mean = np.load(NORM_MEAN_PATH).astype(np.float32)
        norm_std  = np.load(NORM_STD_PATH).astype(np.float32)
        print(f"  Normalization loaded: mean={norm_mean.shape}, std={norm_std.shape}")
    except FileNotFoundError:
        print("  WARNING: Normalization files not found. Using zeros/ones.")
        norm_mean = np.zeros(NUM_FEATURES, dtype=np.float32)
        norm_std  = np.ones(NUM_FEATURES, dtype=np.float32)

    print(f"  Loading {'CPU' if use_cpu else 'DPU'} inference engine...")
    engine = load_engine(use_cpu=use_cpu, xmodel_path=xmodel_path)

    imu_buffer = IMUBuffer()

    print("Opening session...")
    with zenoh.open(conf) as session:

        print(f"Declaring Publisher on '{repub_key}'...")
        pub = session.declare_publisher(repub_key)

        if add_matching_listener:
            def on_matching_status_update(status: zenoh.MatchingStatus):
                state = "has" if status.matching else "has NO MORE"
                print(f"Publisher {state} matching subscribers.")
            pub.declare_matching_listener(on_matching_status_update)

        def on_imu_sample(sample: zenoh.Sample):
            topic = str(sample.key_expr)
            sensor_name = IMU_TOPICS.get(topic)
            if sensor_name is None:
                return
            try:
                msg = json.loads(sample.payload.to_string())
                values = [float(msg.get(f, 0.0)) for f in IMU_FIELDS]
                imu_buffer.update_sensor(sensor_name, values)
            except Exception as e:
                print(f">> [Warning] Could not parse IMU sample from {topic}: {e}")

        print(f"Declaring Subscriber on '{sub_key}'...")
        sub = declare_advanced_subscriber(
            session,
            sub_key,
            on_imu_sample,
            history=HistoryConfig(detect_late_publishers=True),
            recovery=RecoveryConfig(heartbeat=True),
            subscriber_detection=True,
        )

        def miss_listener(miss: Miss):
            print(f">> [Subscriber] Missed {miss.nb} samples from {miss.source} !!!")
        sub.sample_miss_listener(miss_listener)

        print(f"Classifying every {interval}s. Press CTRL-C to quit...")
        inference_count = 0

        try:
            while True:
                time.sleep(interval)

                window = imu_buffer.get_window()
                if window is None:
                    fill = imu_buffer.get_fill_level()
                    print(f"\r  Buffer filling: {fill*100:.0f}%", end="", flush=True)
                    continue

                window_norm = (window - norm_mean) / norm_std
                pred_class, confidence = run_inference(engine, window_norm)
                inference_count += 1

                if confidence < CONFIDENCE_THRESHOLD:
                    pred_class = CLASSES.index("unknown")
                class_name = CLASSES[pred_class] if 0 <= pred_class < NUM_CLASSES else "unknown"
                payload = json.dumps({"action": pred_class})
                pub.put(ZBytes(payload))

                print(f">> [Repub #{inference_count}] "
                      f"{class_name} (id={pred_class}, conf={confidence:.3f})")

        except KeyboardInterrupt:
            print(f"\nShutting down after {inference_count} inferences.")


if __name__ == "__main__":
    import argparse

    try:
        import common
    except ImportError:
        common = None

    parser = argparse.ArgumentParser(
        prog="repub",
        description="IMU Zenoh subscriber → DPU classification → Visualizer publisher",
    )

    if common:
        common.add_config_arguments(parser)

    parser.add_argument(
        "--key", "-k", dest="repub_key", default="ultra/action1",
        help="Key to publish classifications onto (default: ultra/action1).",
    )
    parser.add_argument(
        "--sub-key", "-s", default="esp/**",
        help="Key expression to subscribe to for IMU data (default: esp/**).",
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=2.0,
        help="Seconds between inferences (default: 2.0).",
    )
    parser.add_argument(
        "--add-matching-listener", action="store_true",
        help="Print when publisher gains/loses subscribers.",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Use CPU (PyTorch) inference instead of DPU.",
    )
    parser.add_argument(
        "--xmodel", default=XMODEL_PATH,
        help=f"Path to compiled .xmodel file (default: {XMODEL_PATH}).",
    )

    args = parser.parse_args()
    conf = common.get_config_from_args(args) if common else zenoh.Config()

    main(conf, args.repub_key, args.sub_key, args.interval,
         args.add_matching_listener, args.cpu, args.xmodel)
